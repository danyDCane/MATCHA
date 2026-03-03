import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

import util
from pacs_dataset import PACSDataset
from dood.utils.diffusion import get_diffusion_model, get_diffusion_scores


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
    
    # 噪声OOD评估相关
    parser.add_argument('--use_noise_ood', action='store_true', 
                       help='Use noise as OOD dataset for evaluation')
    parser.add_argument('--noise_samples', type=int, default=3000, 
                       help='Number of noise samples to generate for OOD evaluation')
    
    return parser.parse_args()


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
        for data, _ in tqdm(test_loader, desc=f'Computing scores ({ood_eval_scores_type})'):
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
    
    # PACS的4個 domain（含 cartoon 當作 ID-C 看分數）
    domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    
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
        for train_domain in domains:
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
    for train_domain in domains:
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
        
        # =====================================================================
        # 步骤1: 先计算所有test_domain的分数（包括train_domain）
        # =====================================================================
        domain_scores = {}  # 存储每个domain的分数和样本数
        
        for test_domain in domains:
            print(f'\n  Test Domain: {test_domain}')
            
            # 判断是否为ID数据
            is_id = (train_domain == test_domain)
            
            # 加载测试数据
            test_loader, num_samples = load_pacs_test_data(
                args.datasetRoot,
                test_domain,
                args.batch_size,
                args.num_workers
            )
            
            print(f'  Loaded {num_samples} test samples')
            
            # 计算分数
            scores = compute_ood_scores(
                backbone,
                diffusion_model,
                test_loader,
                diffusion_steps,
                args.ood_eval_scores_type,
                device
            )
            
            # 保存分数
            domain_scores[test_domain] = {
                'scores': scores,
                'num_samples': num_samples,
                'is_id': is_id
            }
            
            # 计算统计量
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            label = 'ID' if is_id else 'Neighbor'
            print(f'  Label: {label}')
            print(f'  Mean Score: {mean_score:.6f}, Std: {std_score:.6f}')
            print(f'  Min Score: {min_score:.6f}, Max Score: {max_score:.6f}')
            
            # 保存结果（暂时不包含AUROC和FPR95，后面如果需要会更新）
            result_dict = {
                'train_domain': train_domain,
                'test_domain': test_domain,
                'scores': scores,
                'mean_score': mean_score,
                'std_score': std_score,
                'min_score': min_score,
                'max_score': max_score,
                'num_samples': num_samples,
                'score_type': args.ood_eval_scores_type,
                'is_id': is_id,
                'label': label,
                'auroc': None,
                'fpr95': None
            }
            results.append(result_dict)
        
        # =====================================================================
        # 準備每張圖分數表：每欄一個 test domain + 稍後可加 OOD，依圖片編號順序
        # 各 domain 樣本數不同，以最大長度為準，較短欄位用 NaN 補齊
        # =====================================================================
        max_len = max(len(domain_scores[d]['scores']) for d in domains)
        per_image_df = pd.DataFrame({
            test_domain: np.concatenate([
                domain_scores[test_domain]['scores'],
                np.full(max_len - len(domain_scores[test_domain]['scores']), np.nan)
            ])
            for test_domain in domains
        })
        per_image_df.insert(0, 'image_idx', np.arange(max_len))
        
        # =====================================================================
        # 步骤2: 如果需要噪声OOD评估，计算噪声OOD分数並加入表格
        # =====================================================================
        if args.use_noise_ood:
            print(f'\n  Evaluating Noise OOD Detection for train_domain={train_domain}')
            
            # 从已计算的分数中获取ID分数（train_domain的分数）
            id_scores = domain_scores[train_domain]['scores']
            id_num_samples = domain_scores[train_domain]['num_samples']
            
            print(f'  Using ID scores from {train_domain} (already computed above)')
            print(f'    ID scores: mean={np.mean(id_scores):.4f}, std={np.std(id_scores):.4f}')
            
            # 加载噪声OOD数据
            print(f'  Loading {args.noise_samples} noise samples as OOD...')
            noise_loader = get_noise_loader(
                args.noise_samples,
                args.batch_size,
                args.num_workers,
                image_size=(224, 224)
            )
            
            # 计算噪声OOD分数
            print(f'  Computing noise OOD scores...')
            noise_ood_scores = compute_ood_scores(
                backbone,
                diffusion_model,
                noise_loader,
                diffusion_steps,
                args.ood_eval_scores_type,
                device
            )
            # 確保只使用請求的 noise 數量（DataLoader 可能因 batch 邊界產生不同數量）
            noise_ood_scores = np.asarray(noise_ood_scores)[: args.noise_samples]
            print(f'  Noise OOD scores: {len(noise_ood_scores)} (requested {args.noise_samples})')
            # 將 OOD（noise）分數加入每張圖分數表（noise 較少時後段為 NaN；noise 較多時擴充表格）
            new_len = max(len(per_image_df), len(noise_ood_scores))
            if new_len > len(per_image_df):
                extra = new_len - len(per_image_df)
                extra_df = pd.DataFrame(
                    {c: [np.nan] * extra for c in per_image_df.columns},
                    index=np.arange(len(per_image_df), new_len)
                )
                extra_df['image_idx'] = np.arange(len(per_image_df), new_len)
                per_image_df = pd.concat([per_image_df, extra_df], ignore_index=True)
            noise_col = np.full(new_len, np.nan, dtype=float)
            noise_col[: len(noise_ood_scores)] = np.asarray(noise_ood_scores)
            per_image_df['noise_ood'] = noise_col
            
            # 计算AUROC和FPR95（与噪声OOD分数计算放在一起）
            print(f'  Computing AUROC and FPR95 metrics...')
            
            # 如果ID分数更高，需要反转（确保OOD分数更高）
            if np.mean(id_scores) > np.mean(noise_ood_scores):
                print('  Warning: ID scores are higher than noise OOD scores. Inverting scores.')
                id_scores_for_metric = -id_scores
                noise_ood_scores_for_metric = -noise_ood_scores
            else:
                id_scores_for_metric = id_scores
                noise_ood_scores_for_metric = noise_ood_scores
            
            noise_ood_auroc = compute_auroc(id_scores_for_metric, noise_ood_scores_for_metric)
            noise_ood_fpr95 = compute_fpr_at_tpr(id_scores_for_metric, noise_ood_scores_for_metric, tpr=0.95)
            
            print(f'\n  Noise OOD Detection Results:')
            print(f'    ID scores: mean={np.mean(id_scores):.4f}, std={np.std(id_scores):.4f}')
            print(f'    Noise OOD scores: mean={np.mean(noise_ood_scores):.4f}, std={np.std(noise_ood_scores):.4f}')
            print(f'    AUROC: {noise_ood_auroc:.4f}')
            print(f'    FPR@95%TPR: {noise_ood_fpr95:.4f}')
            
            # 保存噪声OOD评估结果
            results.append({
                'train_domain': train_domain,
                'test_domain': 'noise',
                'scores': noise_ood_scores,
                'mean_score': np.mean(noise_ood_scores),
                'std_score': np.std(noise_ood_scores),
                'min_score': np.min(noise_ood_scores),
                'max_score': np.max(noise_ood_scores),
                'num_samples': args.noise_samples,
                'score_type': args.ood_eval_scores_type,
                'is_id': False,
                'label': 'Noise_OOD',
                'auroc': noise_ood_auroc,
                'fpr95': noise_ood_fpr95
            })
            
            # 更新train_domain的结果，添加AUROC和FPR95（用于噪声OOD评估的ID部分）
            # 找到对应的结果并更新
            for result in results:
                if result['train_domain'] == train_domain and result['test_domain'] == train_domain:
                    result['label'] = 'ID_for_noise_ood'
                    result['auroc'] = noise_ood_auroc
                    result['fpr95'] = noise_ood_fpr95
                    break
        
        # 儲存此 train_domain 的每張圖分數 CSV（含各 test domain；若有跑 noise OOD 則含 noise_ood 欄）
        per_image_path = os.path.join(
            args.output_dir,
            f'per_image_scores_{train_domain}_{args.ood_eval_scores_type}.csv'
        )
        per_image_df.to_csv(per_image_path, index=False)
        cols_note = 'test domains + noise_ood' if args.use_noise_ood else 'test domains'
        print(f'  Per-image scores saved: {per_image_path} (rows=image index, cols={cols_note})')
    
    # 保存结果到CSV（不包含scores数组，只保存统计量）
    results_for_csv = []
    for r in results:
        csv_dict = {
            'train_domain': r['train_domain'],
            'test_domain': r['test_domain'],
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
    csv_path = os.path.join(args.output_dir, f'domain_ood_scores_{args.ood_eval_scores_type}.csv')
    df.to_csv(csv_path, index=False)
    print(f'\n{"="*80}')
    print(f'Results saved to: {csv_path}')
    print(f'{"="*80}')
    
    # 打印结果表格
    print('\nResults Summary:')
    if args.use_noise_ood:
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
    
    print('-' * (150 if args.use_noise_ood else 120))
    
    return results


if __name__ == '__main__':
    args = parse_args()
    evaluate_domain_ood_scores(args)

