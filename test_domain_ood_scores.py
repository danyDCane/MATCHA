import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

import util
from pacs_dataset import PACSDataset
from dood.utils.diffusion import get_diffusion_model, get_diffusion_scores

PACS_ALL_DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']


def _infer_train_domains_from_checkpoints(checkpoint_dir: str, description: str, candidate_domains):
    """
    Infer which train domains were actually trained by checking which *_final.pth exist.
    Expected filename: {description}_{train_domain}_final.pth
    """
    train_domains = []
    for d in candidate_domains:
        ckpt = os.path.join(checkpoint_dir, f'{description}_{d}_final.pth')
        if os.path.exists(ckpt):
            train_domains.append(d)
    return train_domains


def _infer_leave_out_from_any_checkpoint(checkpoint_dir: str, description: str, candidate_domains, device):
    """
    Try to infer leave_out domain from checkpoint metadata.
    We look for:
      - checkpoint['leave_out']
      - checkpoint['args'].leave_out
    """
    for d in candidate_domains:
        ckpt = os.path.join(checkpoint_dir, f'{description}_{d}_final.pth')
        if not os.path.exists(ckpt):
            continue
        try:
            checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
        except Exception:
            continue
        leave_out = checkpoint.get('leave_out', None)
        if leave_out:
            return leave_out
        ckpt_args = checkpoint.get('args', None)
        if ckpt_args is not None and hasattr(ckpt_args, 'leave_out'):
            val = getattr(ckpt_args, 'leave_out')
            if val:
                return val
    return None


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate domain OOD scores for MATCHA trained models')
    
    # 模型相关
    parser.add_argument('--checkpoint_dir', type=str, required=True, 
                       help='Directory containing checkpoint files (e.g., exp_result_sinpro_ood_V1)')
    parser.add_argument('--description', type=str, required=True,
                       help='Experiment description used in checkpoint filename (e.g., test_sinpro_ood_V1)')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes (7 for PACS)')
    parser.add_argument('--model', default="res", type=str, help='model name: res/VGG/wrn')
    parser.add_argument('--resnet_type', default='simplified', type=str, 
                        choices=['simplified', 'standard'], 
                        help='ResNet type: simplified or standard (torchvision)')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pretrained ImageNet weights for ResNet')
    parser.add_argument('--dataset', default='pacs', type=str, 
                        choices=['pacs', 'cifar10'],
                        help='Dataset name (default: pacs)')
    
    # 数据相关
    parser.add_argument('--datasetRoot', type=str, required=True, help='Root directory for PACS dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Diffusion相关
    parser.add_argument('--diffusion_channels', type=int, default=512, help='Diffusion denoiser channels')
    parser.add_argument('--diffusion_steps', type=int, default=1000, help='Number of diffusion steps')
    parser.add_argument('--ood_eval_scores_type', type=str, default='eps_mse',
                       choices=['eps_mse', 'eps_cos', 'recon_mse', 'bpd'],
                       help='Type of OOD scoring function')
    parser.add_argument('--num_eval_steps', type=int, default=25, help='Number of diffusion steps for evaluation')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument(
        '--leave_out',
        type=str,
        default=None,
        help='Optional leave-out domain name (e.g., cartoon). If not set, try to infer from checkpoints.',
    )
    
    # 額外 OOD 評估（ID = 該 train_domain 的 PACS test；OOD = noise 或 SVHN）
    parser.add_argument(
        '--external_ood',
        type=str,
        default='none',
        choices=['none', 'noise', 'svhn', 'textures'],
        help="Extra OOD vs train-domain ID: 'noise'/'svhn'/'textures' (default: none)",
    )
    parser.add_argument('--use_noise_ood', action='store_true',
                       help='Deprecated: same as --external_ood noise')
    parser.add_argument('--noise_samples', type=int, default=10000, 
                       help='Number of noise samples when external_ood=noise')
    parser.add_argument(
        '--svhn_root',
        type=str,
        default=None,
        help='Folder containing test_32x32.mat for SVHN (default: <datasets>/SVHN next to PACS root)',
    )
    parser.add_argument(
        '--textures_root',
        type=str,
        default=None,
        help="Root for DTD textures. Can be <datasets>/dtd or <datasets>/dtd/images (default: <datasets>/dtd next to PACS root)",
    )
    parser.add_argument(
        '--id_source',
        type=str,
        default='train_domain',
        choices=['train_domain', 'target', 'both'],
        help=(
            "Which domain provides the ID set for the external-OOD AUROC: "
            "'train_domain' (node's own train domain, default/legacy), "
            "'target' (the leave-out/target domain, e.g. photo), "
            "or 'both' (compute one AUROC per ID source against the same OOD)."
        ),
    )
    parser.add_argument(
        '--skip_cross_domain',
        action='store_true',
        help='Skip the per-(train_domain x test_domain) PACS scoring sweep (only run external-OOD AUROC).',
    )

    args = parser.parse_args()
    if args.use_noise_ood:
        args.external_ood = 'noise'
    return args


def compute_auroc(id_scores, ood_scores):
    """计算AUROC"""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    auroc = roc_auc_score(labels, scores)
    return auroc


def compute_fpr_at_tpr(id_scores, ood_scores, tpr=0.95):
    """计算FPR@TPR (False Positive Rate at True Positive Rate)，使用 sklearn 的 roc_curve（更精确）"""
    # 1. 建立標籤：ID=0, OOD=1
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    y_scores = np.concatenate([id_scores, ood_scores])
    
    # 2. 處理分數方向：確保 OOD 分數比 ID 高
    if np.mean(id_scores) > np.mean(ood_scores):
        y_scores = -y_scores
    
    # 3. 計算 ROC curve
    fpr, tpr_array, thresholds = roc_curve(y_true, y_scores)
    
    # 4. 找到 TPR >= tpr 的第一個點
    idx = np.searchsorted(tpr_array, tpr)
    
    # 5. 處理邊界情況
    if idx == 0:
        # 如果第一個點的 TPR 就已經 >= tpr，返回該點的 FPR
        return fpr[0]
    elif idx >= len(fpr):
        # 如果所有點的 TPR 都 < tpr，返回最後一個點的 FPR
        return fpr[-1]
    
    # 6. 線性插值（更精確）
    # 如果 tpr_array[idx-1] < tpr < tpr_array[idx]，進行插值
    if idx > 0 and tpr_array[idx-1] < tpr < tpr_array[idx]:
        # 線性插值
        tpr_diff = tpr_array[idx] - tpr_array[idx-1]
        if tpr_diff > 0:
            weight = (tpr - tpr_array[idx-1]) / tpr_diff
            fpr_interpolated = fpr[idx-1] + weight * (fpr[idx] - fpr[idx-1])
            return fpr_interpolated
    
    # 7. 如果恰好等於，直接返回
    return fpr[idx]


def load_checkpoint(checkpoint_path, backbone, diffusion_model, device):
    """加载checkpoint（backbone_state和diffusion_state）"""
    print(f'Loading checkpoint from {checkpoint_path}...')
    # weights_only=False 允许加载包含 argparse.Namespace 等对象的 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 檢查創建的模型類型
    # StandardResNetWrapper 有 backbone 屬性，ResNet 沒有
    is_standard_resnet = hasattr(backbone, 'backbone')
    
    # MATCHA使用backbone_state和diffusion_state（不是backbone_state_dict）
    if 'backbone_state' in checkpoint:
        backbone_state = checkpoint['backbone_state']
        
        # 過濾掉 diffusion_model 相關的鍵（載入 backbone 時不要管 diffusion_model 的權重）
        backbone_state_filtered = {
            key: value 
            for key, value in backbone_state.items() 
            if not key.startswith('diffusion_model.')
        }
        
        # 檢查 checkpoint 中的鍵是否有 'backbone.' 前綴
        has_backbone_prefix = any(key.startswith('backbone.') for key in backbone_state_filtered.keys())
        
        if has_backbone_prefix and is_standard_resnet:
            # 兩者都是 StandardResNetWrapper，直接加載（保留前綴）
            print('  Both checkpoint and model are StandardResNetWrapper, loading directly...')
            missing_keys, unexpected_keys = backbone.load_state_dict(backbone_state_filtered, strict=False)
            if missing_keys:
                print(f'  Warning: Missing keys: {missing_keys[:5]}...' if len(missing_keys) > 5 else f'  Warning: Missing keys: {missing_keys}')
            if unexpected_keys:
                # 過濾掉 diffusion_model 相關的 unexpected keys（這些是正常的，因為我們已經過濾了）
                unexpected_keys_filtered = [k for k in unexpected_keys if not k.startswith('diffusion_model.')]
                if unexpected_keys_filtered:
                    print(f'  Warning: Unexpected keys: {unexpected_keys_filtered[:5]}...' if len(unexpected_keys_filtered) > 5 else f'  Warning: Unexpected keys: {unexpected_keys_filtered}')
            print('  Backbone loaded successfully')
        elif has_backbone_prefix and not is_standard_resnet:
            # checkpoint 是 StandardResNetWrapper，但模型是 ResNet，需要移除前綴
            print('  Checkpoint is StandardResNetWrapper, model is ResNet, extracting backbone submodule...')
            backbone_state_clean = {
                key[len('backbone.'):]: value 
                for key, value in backbone_state_filtered.items() 
                if key.startswith('backbone.')
            }
            missing_keys, unexpected_keys = backbone.load_state_dict(backbone_state_clean, strict=False)
            if missing_keys:
                print(f'  Warning: Missing keys: {missing_keys[:5]}...' if len(missing_keys) > 5 else f'  Warning: Missing keys: {missing_keys}')
            if unexpected_keys:
                print(f'  Warning: Unexpected keys: {unexpected_keys[:5]}...' if len(unexpected_keys) > 5 else f'  Warning: Unexpected keys: {unexpected_keys}')
            print('  Backbone loaded successfully')
        else:
            # 沒有 'backbone.' 前綴，直接加載
            missing_keys, unexpected_keys = backbone.load_state_dict(backbone_state_filtered, strict=False)
            if missing_keys:
                print(f'  Warning: Missing keys: {missing_keys[:5]}...' if len(missing_keys) > 5 else f'  Warning: Missing keys: {missing_keys}')
            if unexpected_keys:
                unexpected_keys_filtered = [k for k in unexpected_keys if not k.startswith('diffusion_model.')]
                if unexpected_keys_filtered:
                    print(f'  Warning: Unexpected keys: {unexpected_keys_filtered[:5]}...' if len(unexpected_keys_filtered) > 5 else f'  Warning: Unexpected keys: {unexpected_keys_filtered}')
            print('  Backbone loaded successfully')
    elif 'backbone_state_dict' in checkpoint:
        backbone_state_dict = checkpoint['backbone_state_dict']
        
        # 過濾掉 diffusion_model 相關的鍵
        backbone_state_dict_filtered = {
            key: value 
            for key, value in backbone_state_dict.items() 
            if not key.startswith('diffusion_model.')
        }
        
        has_backbone_prefix = any(key.startswith('backbone.') for key in backbone_state_dict_filtered.keys())
        
        if has_backbone_prefix and is_standard_resnet:
            print('  Both checkpoint and model are StandardResNetWrapper, loading directly...')
            missing_keys, unexpected_keys = backbone.load_state_dict(backbone_state_dict_filtered, strict=False)
            if missing_keys:
                print(f'  Warning: Missing keys: {missing_keys[:5]}...' if len(missing_keys) > 5 else f'  Warning: Missing keys: {missing_keys}')
            if unexpected_keys:
                unexpected_keys_filtered = [k for k in unexpected_keys if not k.startswith('diffusion_model.')]
                if unexpected_keys_filtered:
                    print(f'  Warning: Unexpected keys: {unexpected_keys_filtered[:5]}...' if len(unexpected_keys_filtered) > 5 else f'  Warning: Unexpected keys: {unexpected_keys_filtered}')
            print('  Backbone loaded successfully')
        elif has_backbone_prefix and not is_standard_resnet:
            print('  Checkpoint is StandardResNetWrapper, model is ResNet, extracting backbone submodule...')
            backbone_state_clean = {
                key[len('backbone.'):]: value 
                for key, value in backbone_state_dict_filtered.items() 
                if key.startswith('backbone.')
            }
            missing_keys, unexpected_keys = backbone.load_state_dict(backbone_state_clean, strict=False)
            if missing_keys:
                print(f'  Warning: Missing keys: {missing_keys[:5]}...' if len(missing_keys) > 5 else f'  Warning: Missing keys: {missing_keys}')
            if unexpected_keys:
                print(f'  Warning: Unexpected keys: {unexpected_keys[:5]}...' if len(unexpected_keys) > 5 else f'  Warning: Unexpected keys: {unexpected_keys}')
            print('  Backbone loaded successfully')
        else:
            missing_keys, unexpected_keys = backbone.load_state_dict(backbone_state_dict_filtered, strict=False)
            if missing_keys:
                print(f'  Warning: Missing keys: {missing_keys[:5]}...' if len(missing_keys) > 5 else f'  Warning: Missing keys: {missing_keys}')
            if unexpected_keys:
                unexpected_keys_filtered = [k for k in unexpected_keys if not k.startswith('diffusion_model.')]
                if unexpected_keys_filtered:
                    print(f'  Warning: Unexpected keys: {unexpected_keys_filtered[:5]}...' if len(unexpected_keys_filtered) > 5 else f'  Warning: Unexpected keys: {unexpected_keys_filtered}')
            print('  Backbone loaded successfully')
    else:
        raise ValueError(f'Checkpoint does not contain backbone_state or backbone_state_dict')
    
    # 處理 diffusion_state（載入 diffusion_model 時不要管 backbone 的權重）
    if 'diffusion_state' in checkpoint:
        diffusion_state = checkpoint['diffusion_state']
        
        # 過濾掉 backbone 相關的鍵（載入 diffusion_model 時不要管 backbone 的權重）
        diffusion_state_filtered = {
            key: value 
            for key, value in diffusion_state.items() 
            if not key.startswith('backbone.') and not key.startswith('fc.') and not key.startswith('linear.')
        }
        
        print('  Loading diffusion model state...')
        missing_keys, unexpected_keys = diffusion_model.load_state_dict(diffusion_state_filtered, strict=False)
        if missing_keys:
            print(f'  Diffusion model - Missing keys: {missing_keys[:5]}...' if len(missing_keys) > 5 else f'  Diffusion model - Missing keys: {missing_keys}')
        if unexpected_keys:
            # 過濾掉 backbone 相關的 unexpected keys
            unexpected_keys_filtered = [k for k in unexpected_keys if not k.startswith('backbone.') and not k.startswith('fc.') and not k.startswith('linear.')]
            if unexpected_keys_filtered:
                print(f'  Diffusion model - Unexpected keys: {unexpected_keys_filtered[:5]}...' if len(unexpected_keys_filtered) > 5 else f'  Diffusion model - Unexpected keys: {unexpected_keys_filtered}')
        print('  Diffusion model loaded successfully')
    elif 'diffusion_state_dict' in checkpoint:
        diffusion_state_dict = checkpoint['diffusion_state_dict']
        
        # 過濾掉 backbone 相關的鍵
        diffusion_state_dict_filtered = {
            key: value 
            for key, value in diffusion_state_dict.items() 
            if not key.startswith('backbone.') and not key.startswith('fc.') and not key.startswith('linear.')
        }
        
        print('  Loading diffusion model state...')
        missing_keys, unexpected_keys = diffusion_model.load_state_dict(diffusion_state_dict_filtered, strict=False)
        if missing_keys:
            print(f'  Diffusion model - Missing keys: {missing_keys[:5]}...' if len(missing_keys) > 5 else f'  Diffusion model - Missing keys: {missing_keys}')
        if unexpected_keys:
            unexpected_keys_filtered = [k for k in unexpected_keys if not k.startswith('backbone.') and not k.startswith('fc.') and not k.startswith('linear.')]
            if unexpected_keys_filtered:
                print(f'  Diffusion model - Unexpected keys: {unexpected_keys_filtered[:5]}...' if len(unexpected_keys_filtered) > 5 else f'  Diffusion model - Unexpected keys: {unexpected_keys_filtered}')
        print('  Diffusion model loaded successfully')
    else:
        raise ValueError(f'Checkpoint does not contain diffusion_state or diffusion_state_dict')
    
    backbone.eval()
    diffusion_model.eval()
    
    return checkpoint


class NoiseDataset(Dataset):
    """生成随机噪声图像作为OOD数据集"""
    def __init__(self, num_samples, image_size=(224, 224), num_channels=3):
        """
        Args:
            num_samples: 生成的噪声样本数量
            image_size: 图像尺寸 (height, width)，默认 (224, 224) 用于PACS
            num_channels: 图像通道数，默认3（RGB）
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_channels = num_channels
        
        # PACS的normalization参数
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机噪声图像 [0, 1] 范围（ToTensor后的范围）
        # 使用均匀分布生成随机噪声
        noise_image = torch.rand(self.num_channels, self.image_size[0], self.image_size[1])
        
        # 应用normalization（与PACS测试数据一致）
        noise_image = self.normalize(noise_image)
        
        # 返回噪声图像和dummy标签（OOD不需要真实标签）
        return noise_image, 0


def get_noise_loader(num_samples, batch_size, num_workers, image_size=(224, 224)):
    """创建噪声数据集的DataLoader"""
    noise_dataset = NoiseDataset(num_samples=num_samples, image_size=image_size)
    
    loader = DataLoader(
        noise_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


def resolve_svhn_mat_root(dataset_root, svhn_root_explicit):
    """Directory that contains test_32x32.mat for torchvision.datasets.SVHN."""
    if svhn_root_explicit:
        root = os.path.abspath(svhn_root_explicit)
        if not os.path.isfile(os.path.join(root, 'test_32x32.mat')):
            raise FileNotFoundError(
                f'SVHN test_32x32.mat not found under --svhn_root={root}'
            )
        return root
    root = dataset_root
    if root.endswith('/PACS') or root.endswith('\\PACS'):
        root = os.path.dirname(root)
    for c in (os.path.join(root, 'SVHN'), root):
        if os.path.isfile(os.path.join(c, 'test_32x32.mat')):
            return c
    raise FileNotFoundError(
        'SVHN test_32x32.mat not found. Expected e.g. <datasets>/SVHN/test_32x32.mat '
        'or set --svhn_root to the folder containing test_32x32.mat'
    )


def load_svhn_ood_loader(dataset_root, svhn_root_explicit, batch_size, num_workers):
    """SVHN test；與 PACS ID 相同之 224×224 + ImageNet normalize。"""
    mat_root = resolve_svhn_mat_root(dataset_root, svhn_root_explicit)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_mat = os.path.join(mat_root, 'test_32x32.mat')
    svhn_ds = datasets.SVHN(
        root=mat_root,
        split='test',
        download=not os.path.isfile(test_mat),
        transform=transform,
    )
    loader = DataLoader(
        svhn_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(
        f'  Loaded SVHN (test) from {mat_root}: {len(svhn_ds)} samples '
        f'(224×224, ImageNet norm, same as PACS ID)'
    )
    return loader, len(svhn_ds)


def resolve_textures_root(dataset_root, textures_root_explicit):
    """
    Resolve DTD textures path.
    Accepts either:
      - <datasets>/dtd (contains images/, labels/, imdb/)
      - <datasets>/dtd/images (ImageFolder root)
    """
    if textures_root_explicit:
        root = os.path.abspath(textures_root_explicit)
        if os.path.isdir(root) and os.path.isdir(os.path.join(root, 'images')):
            return os.path.join(root, 'images')
        if os.path.isdir(root) and os.path.isdir(os.path.join(root, '..')):  # allow direct images/ root
            # If user passed .../dtd/images, keep as-is.
            if os.path.basename(root) == 'images':
                return root
        raise FileNotFoundError(
            f'DTD textures not found under --textures_root={root}. Expected a folder containing images/ or the images/ folder itself.'
        )

    root = dataset_root
    if root.endswith('/PACS') or root.endswith('\\PACS'):
        root = os.path.dirname(root)

    dtd_dir = os.path.join(root, 'dtd')
    images_dir = os.path.join(dtd_dir, 'images')
    if os.path.isdir(images_dir):
        return images_dir

    raise FileNotFoundError(
        'DTD textures not found. Expected e.g. <datasets>/dtd/images (you already downloaded DTD as dtd-r1.0.1). '
        'Set --textures_root to <datasets>/dtd or <datasets>/dtd/images.'
    )


def load_textures_ood_loader(dataset_root, textures_root_explicit, batch_size, num_workers):
    """DTD(Textures)；用 ImageFolder 讀取 dtd/images/*，並套用與 PACS 相同前處理。"""
    images_root = resolve_textures_root(dataset_root, textures_root_explicit)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dtd_ds = datasets.ImageFolder(root=images_root, transform=transform)
    loader = DataLoader(
        dtd_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f'  Loaded DTD textures from {images_root}: {len(dtd_ds)} samples (224×224, ImageNet norm)')
    return loader, len(dtd_ds)


def load_pacs_test_data(dataset_root, domain_name, batch_size, num_workers):
    """加载PACS指定domain的测试数据"""
    # Test transforms (与util.py中的一致)
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # PACSDataset 會在 root 後面自動加上 'PACS'
    # 如果 dataset_root 已經是 datasets/PACS，需要移除最後的 'PACS'
    if dataset_root.endswith('/PACS') or dataset_root.endswith('\\PACS'):
        # 移除最後的 'PACS'
        dataset_root = os.path.dirname(dataset_root)
        print(f'  Adjusted dataset_root (removed trailing PACS): {dataset_root}')
    
    test_dataset = PACSDataset(
        root=dataset_root,
        dataset_name=domain_name,
        transform=transform_test
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader, len(test_dataset)


def compute_ood_scores(backbone, diffusion_model, test_loader, diffusion_steps, 
                       ood_eval_scores_type, device):
    """计算OOD分数（完全参考evaluate_cifar10_ood.py的方法）"""
    scores_list = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Computing scores ({ood_eval_scores_type})'):
            data, _, _ = util.unpack_batch(batch)
            data = data.to(device)
            # 特征提取
            latents = backbone.intermediate_forward(data)
            
            # 获取diffusion scores（normalize=True表示使用checkpoint中的normalization统计量）
            scores, _ = get_diffusion_scores(
                latents,
                diffusion_model,
                diffusion_steps,
                ood_eval_scores_type,
                normalize=True,  # 关键：使用domain-specific的normalization
                dtype=torch.float32
            )
            
            # scores可能是标量或tensor，确保转换为numpy数组
            if isinstance(scores, torch.Tensor):
                scores_np = scores.cpu().numpy()
            else:
                scores_np = np.array([scores])
            # 如果是多维，展平
            scores_list.append(scores_np.flatten())
    
    all_scores = np.concatenate(scores_list)
    return all_scores


def evaluate_domain_ood_scores(args):
    """评估domain OOD分数"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Domains:
    # - For PACS, infer train domains from checkpoint filenames under checkpoint_dir,
    #   but evaluate test domains over the full PACS set (including leave-out).
    if args.dataset == 'pacs':
        test_domains = list(PACS_ALL_DOMAINS)
        train_domains = _infer_train_domains_from_checkpoints(
            args.checkpoint_dir, args.description, PACS_ALL_DOMAINS
        )
        inferred_leave_out = args.leave_out or _infer_leave_out_from_any_checkpoint(
            args.checkpoint_dir, args.description, PACS_ALL_DOMAINS, device
        )
        if not train_domains:
            if inferred_leave_out in PACS_ALL_DOMAINS:
                train_domains = [d for d in PACS_ALL_DOMAINS if d != inferred_leave_out]
            else:
                train_domains = list(PACS_ALL_DOMAINS)
        if inferred_leave_out:
            print(f'  Inferred leave_out: {inferred_leave_out}')
        print(f'  Train domains (from checkpoints): {train_domains}')
        print(f'  Test domains: {test_domains}')
    else:
        train_domains = ['default']
        test_domains = ['default']
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 准备diffusion steps
    diffusion_steps = list(range(args.num_eval_steps))
    print(f'Using {len(diffusion_steps)} diffusion steps: {diffusion_steps[:5]}...{diffusion_steps[-5:]}')
    
    # 存储结果
    results = []
    
    # 根據資料集自動判斷 ResNet 類型
    # PACS 使用標準 ResNet (standard)，CIFAR-10 使用簡化 ResNet (simplified)
    # 注意：測試時不使用預訓練權重，因為會從 checkpoint 加載權重
    if args.dataset == 'pacs':
        args.resnet_type = 'standard'
        args.pretrained = False  # 測試時不使用預訓練，從 checkpoint 加載
        print(f'  Auto-detected: PACS dataset → using standard ResNet (weights will be loaded from checkpoint)')
    elif args.dataset == 'cifar10':
        args.resnet_type = 'simplified'
        args.pretrained = False  # 測試時不使用預訓練，從 checkpoint 加載
        print(f'  Auto-detected: CIFAR-10 dataset → using simplified ResNet (weights will be loaded from checkpoint)')
    else:
        # 如果沒有明確指定，嘗試從 checkpoint 讀取
        first_checkpoint_path = None
        for train_domain in train_domains:
            checkpoint_filename = f'{args.description}_{train_domain}_final.pth'
            checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_filename)
            if not os.path.isabs(args.checkpoint_dir):
                checkpoint_path = os.path.abspath(checkpoint_path)
            if os.path.exists(checkpoint_path):
                first_checkpoint_path = checkpoint_path
                break
        
        if first_checkpoint_path:
            print(f'  Loading checkpoint to extract training parameters: {first_checkpoint_path}')
            temp_checkpoint = torch.load(first_checkpoint_path, map_location=device, weights_only=False)
            if 'args' in temp_checkpoint:
                checkpoint_args = temp_checkpoint['args']
                if hasattr(checkpoint_args, 'resnet_type'):
                    args.resnet_type = checkpoint_args.resnet_type
                    print(f'  Using resnet_type from checkpoint: {args.resnet_type}')
                if hasattr(checkpoint_args, 'pretrained'):
                    args.pretrained = checkpoint_args.pretrained
                    print(f'  Using pretrained from checkpoint: {args.pretrained}')
    
    # 创建backbone和diffusion模型（使用自動判斷的參數）
    backbone = util.select_model(args.num_classes, args).to(device)
    
    diffusion_model = get_diffusion_model(
        ft_size=512,
        denoiser_type="unet0d",
        diffusion_denoiser_channels=args.diffusion_channels,
        num_diffusion_steps=args.diffusion_steps,
    ).to(device)
    
    # 9次测试循环：每个train_domain × 每个test_domain
    for train_domain in train_domains:
        print(f'\n{"="*80}')
        print(f'Train Domain: {train_domain}')
        print(f'{"="*80}')
        
        # 加载checkpoint
        checkpoint_filename = f'{args.description}_{train_domain}_final.pth'
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_filename)
        
        # 如果checkpoint_dir是相对路径，转换为绝对路径
        if not os.path.isabs(args.checkpoint_dir):
            checkpoint_path = os.path.abspath(checkpoint_path)
        
        print(f'Looking for checkpoint: {checkpoint_path}')
        
        if not os.path.exists(checkpoint_path):
            print(f'Warning: Checkpoint not found: {checkpoint_path}')
            print(f'Current working directory: {os.getcwd()}')
            print(f'Checkpoint dir: {args.checkpoint_dir}')
            print(f'Checkpoint filename: {checkpoint_filename}')
            print(f'Skipping train_domain={train_domain}')
            continue
        
        # 加载模型
        checkpoint = load_checkpoint(checkpoint_path, backbone, diffusion_model, device)
        
        # 获取checkpoint中的domain信息（如果有）
        checkpoint_domain = checkpoint.get('domain', train_domain)
        print(f'Loaded checkpoint for domain: {checkpoint_domain}')
        
        # 額外 OOD（noise / SVHN / textures）vs ID（train_domain 或 target，由 --id_source 決定）
        if args.external_ood in ('noise', 'svhn', 'textures'):
            # 1) OOD 分數每個 (node, external_ood) 只需算一次，兩種 ID 共用
            if args.external_ood == 'noise':
                print(f'  Loading {args.noise_samples} noise samples as OOD...')
                ood_loader = get_noise_loader(
                    args.noise_samples,
                    args.batch_size,
                    args.num_workers,
                    image_size=(224, 224)
                )
                ood_name = 'noise'
                ood_num_samples = args.noise_samples
                ood_label = 'Noise_OOD'
            elif args.external_ood == 'svhn':
                ood_loader, ood_num_samples = load_svhn_ood_loader(
                    args.datasetRoot,
                    args.svhn_root,
                    args.batch_size,
                    args.num_workers,
                )
                ood_name = 'svhn'
                ood_label = 'SVHN_OOD'
            else:
                ood_loader, ood_num_samples = load_textures_ood_loader(
                    args.datasetRoot,
                    args.textures_root,
                    args.batch_size,
                    args.num_workers,
                )
                ood_name = 'textures'
                ood_label = 'Textures_OOD'

            print(f'  Computing {ood_name} OOD scores...')
            ext_ood_scores = compute_ood_scores(
                backbone,
                diffusion_model,
                ood_loader,
                diffusion_steps,
                args.ood_eval_scores_type,
                device
            )
            ext_ood_appended = False  # OOD 統計列只存一次（與 id_source 無關）

            # 2) 決定要評估哪些 ID 來源
            if args.id_source == 'both':
                id_sources = ['train_domain', 'target']
            else:
                id_sources = [args.id_source]

            for id_src in id_sources:
                if id_src == 'train_domain':
                    id_domain = train_domain
                else:  # target = leave-out 域
                    id_domain = inferred_leave_out
                    if not id_domain:
                        print(
                            '  Warning: id_source=target 但無法推得 leave_out 域，跳過此 ID 來源'
                        )
                        continue

                print(
                    f'\n  Evaluating {ood_name.upper()} OOD vs ID '
                    f'(id_source={id_src}, id_domain={id_domain}, node={train_domain})'
                )
                id_loader, id_num_samples = load_pacs_test_data(
                    args.datasetRoot,
                    id_domain,
                    args.batch_size,
                    args.num_workers
                )
                print(f'  Loaded {id_num_samples} ID samples from {id_domain}')

                print('  Computing ID scores...')
                id_scores = compute_ood_scores(
                    backbone,
                    diffusion_model,
                    id_loader,
                    diffusion_steps,
                    args.ood_eval_scores_type,
                    device
                )

                print('  Computing metrics...')
                if np.mean(id_scores) > np.mean(ext_ood_scores):
                    print(
                        f'  Warning: ID scores are higher than {ood_name} OOD scores. Inverting scores.'
                    )
                    id_scores_for_metric = -id_scores
                    ext_ood_for_metric = -ext_ood_scores
                else:
                    id_scores_for_metric = id_scores
                    ext_ood_for_metric = ext_ood_scores

                auroc = compute_auroc(id_scores_for_metric, ext_ood_for_metric)
                fpr95 = compute_fpr_at_tpr(id_scores_for_metric, ext_ood_for_metric, tpr=0.95)

                print(f'\n  {ood_name.upper()} OOD Detection Results (ID={id_src}:{id_domain}):')
                print(
                    f'    ID scores: mean={np.mean(id_scores):.4f}, std={np.std(id_scores):.4f}'
                )
                print(
                    f'    OOD scores: mean={np.mean(ext_ood_scores):.4f}, '
                    f'std={np.std(ext_ood_scores):.4f}'
                )
                print(f'    AUROC: {auroc:.4f}')
                print(f'    FPR@95%TPR: {fpr95:.4f}')

                if not ext_ood_appended:
                    results.append({
                        'train_domain': train_domain,
                        'test_domain': ood_name,
                        'id_source': '-',
                        'id_domain': '-',
                        'scores': ext_ood_scores,
                        'mean_score': np.mean(ext_ood_scores),
                        'std_score': np.std(ext_ood_scores),
                        'min_score': np.min(ext_ood_scores),
                        'max_score': np.max(ext_ood_scores),
                        'num_samples': ood_num_samples,
                        'score_type': args.ood_eval_scores_type,
                        'is_id': False,
                        'label': ood_label,
                        'auroc': None,
                        'fpr95': None
                    })
                    ext_ood_appended = True

                results.append({
                    'train_domain': train_domain,
                    'test_domain': id_domain,
                    'id_source': id_src,
                    'id_domain': id_domain,
                    'scores': id_scores,
                    'mean_score': np.mean(id_scores),
                    'std_score': np.std(id_scores),
                    'min_score': np.min(id_scores),
                    'max_score': np.max(id_scores),
                    'num_samples': id_num_samples,
                    'score_type': args.ood_eval_scores_type,
                    'is_id': True,
                    'label': f'ID_{id_src}_vs_{ood_name}',
                    'auroc': auroc,
                    'fpr95': fpr95
                })

        # 对每个test_domain进行测试
        if args.skip_cross_domain:
            print('\n  [skip_cross_domain] 跳過 PACS 跨域掃描')
            continue
        for test_domain in test_domains:
            print(f'\n  Test Domain: {test_domain}')
            
            # 加载测试数据
            test_loader, num_samples = load_pacs_test_data(
                args.datasetRoot,
                test_domain,
                args.batch_size,
                args.num_workers
            )
            
            print(f'  Loaded {num_samples} test samples')
            
            # 计算OOD分数
            scores = compute_ood_scores(
                backbone,
                diffusion_model,
                test_loader,
                diffusion_steps,
                args.ood_eval_scores_type,
                device
            )
            
            # 保存所有样本级别的分数（用于后续分析）
            # 判断是否为ID数据
            is_id = (train_domain == test_domain)
            label = 'ID' if is_id else 'Neighbor'
            
            # 计算统计量
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            print(f'  Label: {label}')
            print(f'  Mean Score: {mean_score:.6f}, Std: {std_score:.6f}')
            print(f'  Min Score: {min_score:.6f}, Max Score: {max_score:.6f}')
            
            # 保存结果
            result_dict = {
                'train_domain': train_domain,
                'test_domain': test_domain,
                'scores': scores,  # 保存所有样本级别的分数
                'mean_score': mean_score,
                'std_score': std_score,
                'min_score': min_score,
                'max_score': max_score,
                'num_samples': num_samples,
                'score_type': args.ood_eval_scores_type,
                'is_id': is_id,
                'label': label
            }
            # PACS domain 交叉結果本身不帶 AUROC/FPR（僅 external_ood 區塊另存）
            result_dict['auroc'] = None
            result_dict['fpr95'] = None
            results.append(result_dict)
    
    # 保存结果到CSV（不包含scores数组，只保存统计量）
    results_for_csv = []
    for r in results:
        csv_dict = {
            'train_domain': r['train_domain'],
            'test_domain': r['test_domain'],
            'id_source': r.get('id_source', '-'),
            'id_domain': r.get('id_domain', '-'),
            'mean_score': r['mean_score'],
            'std_score': r['std_score'],
            'min_score': r['min_score'],
            'max_score': r['max_score'],
            'num_samples': r['num_samples'],
            'score_type': r['score_type'],
            'is_id': r['is_id'],
            'label': r['label']
        }
        # 添加AUROC和FPR95（如果存在）
        if 'auroc' in r:
            csv_dict['auroc'] = r['auroc']
        else:
            csv_dict['auroc'] = None
        if 'fpr95' in r:
            csv_dict['fpr95'] = r['fpr95']
        else:
            csv_dict['fpr95'] = None
        results_for_csv.append(csv_dict)
    
    df = pd.DataFrame(results_for_csv)
    ext_tag = f'_{args.external_ood}' if args.external_ood != 'none' else ''
    csv_path = os.path.join(args.output_dir, f'domain_ood_scores_{args.ood_eval_scores_type}{ext_tag}.csv')
    df.to_csv(csv_path, index=False)
    print(f'\n{"="*80}')
    print(f'Results saved to: {csv_path}')
    print(f'{"="*80}')
    
    # 打印结果表格
    print('\nResults Summary:')
    if args.external_ood != 'none':
        print('-' * 150)
        print(f'{"Train Domain":<20} {"Test Domain":<20} {"Mean Score":<15} {"Std":<15} {"Min":<15} {"Max":<15} {"Label":<15} {"AUROC":<10} {"FPR95":<10}')
        print('-' * 150)
        
        for result in results:
            auroc_str = f'{result.get("auroc", None):.4f}' if result.get("auroc") is not None else 'N/A'
            fpr95_str = f'{result.get("fpr95", None):.4f}' if result.get("fpr95") is not None else 'N/A'
            print(f'{result["train_domain"]:<20} {result["test_domain"]:<20} '
                  f'{result["mean_score"]:<15.6f} {result["std_score"]:<15.6f} '
                  f'{result["min_score"]:<15.6f} {result["max_score"]:<15.6f} '
                  f'{result["label"]:<15} {auroc_str:<10} {fpr95_str:<10}')
    else:
        print('-' * 120)
        print(f'{"Train Domain":<20} {"Test Domain":<20} {"Mean Score":<15} {"Std":<15} {"Min":<15} {"Max":<15} {"Label":<10}')
        print('-' * 120)
        
        for result in results:
            print(f'{result["train_domain"]:<20} {result["test_domain"]:<20} '
                  f'{result["mean_score"]:<15.6f} {result["std_score"]:<15.6f} '
                  f'{result["min_score"]:<15.6f} {result["max_score"]:<15.6f} {result["label"]:<10}')
    
    print('-' * (150 if args.external_ood != 'none' else 120))
    
    return results


if __name__ == '__main__':
    args = parse_args()
    evaluate_domain_ood_scores(args)

