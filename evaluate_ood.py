import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import wandb

import util
from dood.utils.diffusion import get_diffusion_model, get_diffusion_scores


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate OOD detection on MATCHA trained models')
    
    # 模型相关
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--model', default="res", type=str, help='model name: res/VGG/wrn')
    parser.add_argument('--resnet_type', default='standard', type=str, 
                        choices=['simplified', 'standard'], 
                        help='ResNet type: simplified or standard (torchvision)')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pretrained ImageNet weights for ResNet')
    
    # 数据相关
    parser.add_argument('--data_root', type=str, default='../datasets', help='Root directory for datasets')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # OOD数据集相关
    parser.add_argument('--ood_dataset', type=str, default='LSUN', 
                       choices=['cifar100', 'LSUN', 'SVHN'],
                       help='OOD dataset to use')
    
    # Diffusion相关
    parser.add_argument('--diffusion_channels', type=int, default=512, help='Diffusion denoiser channels')
    parser.add_argument('--diffusion_steps', type=int, default=1000, help='Number of diffusion steps')
    parser.add_argument('--ood_eval_scores_type', type=str, default='eps_mse',
                       choices=['eps_mse', 'eps_cos', 'recon_mse', 'bpd'],
                       help='Type of OOD scoring function')
    parser.add_argument('--num_eval_steps', type=int, default=25, help='Number of diffusion steps for evaluation')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    # Wandb相关
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='matcha-ood', help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name (default: auto-generated)')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity/team name')
    
    return parser.parse_args()


def compute_auroc(id_scores, ood_scores):
    """计算AUROC"""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    auroc = roc_auc_score(labels, scores)
    return auroc


def compute_fpr_at_tpr(id_scores, ood_scores, tpr=0.95):
    """计算FPR@TPR (False Positive Rate at True Positive Rate)"""
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    y_scores = np.concatenate([id_scores, ood_scores])
    
    # 處理分數方向：確保 OOD 分數比 ID 高
    if np.mean(id_scores) > np.mean(ood_scores):
        y_scores = -y_scores
    
    # 計算 ROC curve
    fpr, tpr_array, thresholds = roc_curve(y_true, y_scores)
    
    # 找到 TPR >= tpr 的第一個點
    idx = np.searchsorted(tpr_array, tpr)
    
    # 處理邊界情況
    if idx == 0:
        return fpr[0]
    elif idx >= len(fpr):
        return fpr[-1]
    
    # 線性插值
    if idx > 0 and tpr_array[idx-1] < tpr < tpr_array[idx]:
        tpr_diff = tpr_array[idx] - tpr_array[idx-1]
        if tpr_diff > 0:
            weight = (tpr - tpr_array[idx-1]) / tpr_diff
            fpr_interpolated = fpr[idx-1] + weight * (fpr[idx] - fpr[idx-1])
            return fpr_interpolated
    
    return fpr[idx]


def get_cifar10_loader(data_root, batch_size, num_workers, train=False):
    """加载CIFAR-10数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    dataset = datasets.CIFAR10(
        root=data_root,
        train=train,
        download=True,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


def get_cifar100_loader(data_root, batch_size, num_workers):
    """加载CIFAR-100数据集作为OOD"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    dataset = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


def get_svhn_loader(data_root, batch_size, num_workers):
    """加载SVHN数据集作为OOD"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    ])
    
    dataset = datasets.SVHN(
        root=data_root,
        split='test',
        download=True,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


def get_lsun_loader(data_root, batch_size, num_workers):
    """加载LSUN数据集作为OOD"""
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomCrop(32, padding=4),
    ])
    
    lsun_path = os.path.join(data_root, 'LSUN')
    if not os.path.exists(lsun_path):
        alternative_paths = [
            os.path.join(data_root, 'lsun'),
            os.path.join(data_root, 'lsun_resize'),
            data_root,
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                lsun_path = alt_path
                break
        else:
            raise FileNotFoundError(f'LSUN dataset not found at {data_root}')
    
    dataset = datasets.ImageFolder(root=lsun_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f'Loaded LSUN dataset from {lsun_path} with {len(dataset)} images')
    return loader


def get_ood_loader(args):
    """根据参数获取OOD数据集loader"""
    if args.ood_dataset == 'cifar100':
        return get_cifar100_loader(args.data_root, args.batch_size, args.num_workers)
    elif args.ood_dataset == 'svhn':
        return get_svhn_loader(args.data_root, args.batch_size, args.num_workers)
    elif args.ood_dataset == 'LSUN':
        return get_lsun_loader(args.data_root, args.batch_size, args.num_workers)
    else:
        raise NotImplementedError(f'OOD dataset {args.ood_dataset} not implemented')


def evaluate_ood_detection(args):
    """评估OOD检测性能"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 初始化 wandb
    if args.use_wandb:
        run_name = args.wandb_name or f'eval_{args.ood_dataset}_{args.ood_eval_scores_type}'
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            entity=args.wandb_entity,
            config=vars(args),
            reinit=True
        )
        print(f'Wandb initialized: project={args.wandb_project}, name={wandb.run.name}')
    
    # 加载模型
    print('Loading models...')
    # Create model using MATCHA's util.select_model
    backbone = util.select_model(args.num_classes, args).to(device)
    
    diffusion_model = get_diffusion_model(
        ft_size=512,
        denoiser_type="unet0d",
        diffusion_denoiser_channels=args.diffusion_channels,
        num_diffusion_steps=args.diffusion_steps,
    ).to(device)
    
    # 加载checkpoint
    print(f'Loading checkpoint from {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load backbone state dict
    if 'backbone_state_dict' in checkpoint:
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
    elif 'model_state_dict' in checkpoint:
        backbone.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Try loading directly (assuming checkpoint contains model state dict)
        backbone.load_state_dict(checkpoint)
    
    # Load diffusion model state dict
    if 'diffusion_state_dict' in checkpoint:
        diffusion_model.load_state_dict(checkpoint['diffusion_state_dict'])
    elif 'diffusion_model' in checkpoint:
        diffusion_model.load_state_dict(checkpoint['diffusion_model'])
    else:
        print('Warning: No diffusion model found in checkpoint. Using randomly initialized model.')
    
    backbone.eval()
    diffusion_model.eval()
    
    # 准备diffusion steps
    diffusion_steps = list(range(args.num_eval_steps))
    
    # 加载数据集
    print('Loading datasets...')
    id_loader = get_cifar10_loader(args.data_root, args.batch_size, args.num_workers, train=False)
    ood_loader = get_ood_loader(args)
    
    print(f'Evaluating on ID dataset (CIFAR-10 test set)...')
    id_scores = []
    with torch.no_grad():
        for data, _ in tqdm(id_loader, desc='ID samples'):
            data = data.to(device)
            latents = backbone.intermediate_forward(data)
            
            # 获取diffusion scores
            scores, _ = get_diffusion_scores(
                latents,
                diffusion_model,
                diffusion_steps,
                args.ood_eval_scores_type,
                normalize=True,
                dtype=torch.float32
            )
            if isinstance(scores, torch.Tensor):
                scores_np = scores.cpu().numpy()
            else:
                scores_np = np.array([scores])
            id_scores.append(scores_np.flatten())
    
    id_scores = np.concatenate(id_scores)
    
    print(f'Evaluating on OOD dataset ({args.ood_dataset})...')
    ood_scores = []
    with torch.no_grad():
        for data, _ in tqdm(ood_loader, desc='OOD samples'):
            data = data.to(device)
            latents = backbone.intermediate_forward(data)
            
            # 获取diffusion scores
            scores, _ = get_diffusion_scores(
                latents,
                diffusion_model,
                diffusion_steps,
                args.ood_eval_scores_type,
                normalize=True,
                dtype=torch.float32
            )
            if isinstance(scores, torch.Tensor):
                scores_np = scores.cpu().numpy()
            else:
                scores_np = np.array([scores])
            ood_scores.append(scores_np.flatten())
    
    ood_scores = np.concatenate(ood_scores)
    
    # 计算指标
    print('\nComputing metrics...')
    
    # 如果ID分数更高，需要反转
    if np.mean(id_scores) > np.mean(ood_scores):
        print('Warning: ID scores are higher than OOD scores. Inverting scores.')
        id_scores = -id_scores
        ood_scores = -ood_scores
    
    auroc = compute_auroc(id_scores, ood_scores)
    fpr95 = compute_fpr_at_tpr(id_scores, ood_scores, tpr=0.95)
    
    print(f'\nResults:')
    print(f'  ID scores: mean={np.mean(id_scores):.4f}, std={np.std(id_scores):.4f}')
    print(f'  OOD scores: mean={np.mean(ood_scores):.4f}, std={np.std(ood_scores):.4f}')
    print(f'  AUROC: {auroc:.4f}')
    print(f'  FPR@95%TPR: {fpr95:.4f}')
    
    # 记录到wandb
    if args.use_wandb:
        wandb.log({
            'ood/id_score_mean': np.mean(id_scores),
            'ood/id_score_std': np.std(id_scores),
            'ood/ood_score_mean': np.mean(ood_scores),
            'ood/ood_score_std': np.std(ood_scores),
            'ood/auroc': auroc,
            'ood/fpr_at_95_tpr': fpr95,
            'ood/dataset': args.ood_dataset,
            'ood/score_type': args.ood_eval_scores_type,
        })
        wandb.finish()


if __name__ == '__main__':
    args = parse_args()
    evaluate_ood_detection(args)

