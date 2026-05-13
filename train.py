import os
import numpy as np
import time
import argparse
import sys
from copy import deepcopy
import random
import re

from math import ceil
from random import Random
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torchvision.models as models
import wandb

from style_stats import (
    compute_multi_layer_style_stats,
    flatten_style_stats,
)
cudnn.benchmark = True

import util
from graph_manager import FixedProcessor, MatchaProcessor
from communicator import SingleProcessCommunicator


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _to_string_list(value, batch_size: int):
    if value is None:
        return [""] * batch_size
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)] * batch_size


def save_spatial_feature_dump(
    save_root,
    epoch,
    step,
    domain,
    data,
    target,
    meta,
    z_clean,
    z_style,
    z_hard,
    style_aug_activated,
    max_samples,
):
    os.makedirs(save_root, exist_ok=True)
    sample_count = min(int(max_samples), int(data.size(0)))
    meta = meta or {}
    paths = _to_string_list(meta.get("path") if isinstance(meta, dict) else None, sample_count)
    domains = _to_string_list(meta.get("domain") if isinstance(meta, dict) else domain, sample_count)

    payload = {
        "epoch": int(epoch),
        "step": int(step),
        "domain": domain,
        "style_aug_activated": bool(style_aug_activated),
        "input": data[:sample_count].detach().cpu(),
        "target": target[:sample_count].detach().cpu(),
        "image_paths": paths[:sample_count],
        "image_domains": domains[:sample_count],
        "z_clean": None if z_clean is None else z_clean[:sample_count].detach().cpu(),
        "z_style": z_style[:sample_count].detach().cpu(),
        "z_hard": None if z_hard is None else z_hard[:sample_count].detach().cpu(),
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
    }
    filename = f"epoch_{int(epoch):04d}_step_{int(step):06d}_{_sanitize_filename(domain)}.pt"
    save_path = os.path.join(save_root, filename)
    torch.save(payload, save_path)
    print(f"[SpatialDump] Saved {sample_count} samples for domain '{domain}' to: {save_path}")


def compute_loader_avg_confidence(model, data_loader):
    was_training = model.training
    model.eval()
    total_conf = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in data_loader:
            data, _, _ = util.unpack_batch(batch)
            data = data.cuda(non_blocking=True)
            logits = model(data)
            probs = F.softmax(logits, dim=1)
            conf = probs.max(dim=1).values
            total_conf += float(conf.sum().item())
            total_count += int(conf.numel())
    if was_training:
        model.train()
    return (total_conf / max(1, total_count))


def compute_loader_per_class_accuracy(model, data_loader, num_classes):
    """
    Compute per-class accuracy on a data loader.

    Returns:
        per_class_acc: list[float] length=num_classes, each in [0, 100]
        class_correct: list[int]
        class_total: list[int]
    """
    was_training = model.training
    model.eval()
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]
    with torch.no_grad():
        for batch in data_loader:
            data, target, _ = util.unpack_batch(batch)
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            logits = model(data)
            pred = logits.argmax(dim=1)
            for c in range(num_classes):
                mask = (target == c)
                cnt = int(mask.sum().item())
                if cnt == 0:
                    continue
                class_total[c] += cnt
                class_correct[c] += int((pred[mask] == target[mask]).sum().item())
    if was_training:
        model.train()
    per_class_acc = []
    for c in range(num_classes):
        if class_total[c] > 0:
            per_class_acc.append(100.0 * float(class_correct[c]) / float(class_total[c]))
        else:
            per_class_acc.append(0.0)
    return per_class_acc, class_correct, class_total


def _collect_test_style_pool_layer3_mu_sigma(model, dataset, indices, batch_size, adv_eps, pin_memory=True):
    """
    Run leave-out test images through the current backbone and stack per-sample layer3
    spatial mean/std (same convention as hard-adv mu_orig/sigma_orig).
    Returns (mu_stack, sigma_stack) each [N,C,1,1] on CUDA, or (None, None) if empty.
    """
    from torch.utils.data import Subset, DataLoader

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    was_training = model.training
    model.eval()
    mu_chunks, sig_chunks = [], []
    try:
        with torch.no_grad():
            for batch in loader:
                inputs, _, _ = util.unpack_batch(batch)
                inputs = inputs.cuda(non_blocking=True)
                feats = model.extract_features_to_layer3(inputs)
                z3 = feats["layer3"]
                mu = z3.mean(dim=(2, 3), keepdim=True)
                var = z3.var(dim=(2, 3), unbiased=False, keepdim=True)
                sigma = torch.sqrt(var + adv_eps)
                mu_chunks.append(mu)
                sig_chunks.append(sigma)
    finally:
        model.train(was_training)

    if len(mu_chunks) == 0:
        return None, None
    return torch.cat(mu_chunks, dim=0), torch.cat(sig_chunks, dim=0)


def run(num_domains):
    """
    Single process training function for decentralized learning.
    Processes multiple domains sequentially within a single process.
    
    Args:
        num_domains: Number of training domains (e.g., 3 for PACS)
    """
    # set random seed
    torch.manual_seed(args.randomSeed)
    np.random.seed(args.randomSeed)
    random.seed(args.randomSeed)

    # initialize wandb (only once for single process)
    wandb.init(
        project=args.wandb_project,
        name=f"{args.name}_single",
        config={
            "num_domains": num_domains,
            "num_nodes": args.num_nodes,
            "model": args.model,
            "lr": args.lr,
            "epoch": args.epoch,
            "batch_size": args.bs,
            "budget": args.budget,
            "graphid": args.graphid,
            "dataset": args.dataset,
            "matcha": args.matcha,
            "randomSeed": args.randomSeed,
            "description": args.description,
        },
        reinit=True
    )

    # ===== Checkpoint settings =====
    # Save model weights periodically during training.
    # Default: every 50 epochs (can be disabled with --save_every_epoch 0).
    save_dir = args.savePath if args.savePath is not None else "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_every_epoch = getattr(args, "save_every_epoch", 50)
    spatial_dump_dir = os.path.join(save_dir, "spatial_feature_dumps")
    spatial_dump_enabled = bool(getattr(args, "save_spatial_debug_tensors", False))
    spatial_dump_domain = str(getattr(args, "spatial_debug_domain", "sketch"))
    spatial_dump_every = max(1, int(getattr(args, "spatial_debug_every", 1)))
    spatial_dump_max_samples = max(1, int(getattr(args, "spatial_debug_max_samples", 8)))
    spatial_dump_saved_epochs = set()

    # load data for all domains
    node_to_domain = {}
    if args.dataset in ['pacs', 'vlcs']:
        loaded = util.load_dataset_single_process(args)
        if isinstance(loaded, tuple) and len(loaded) == 2:
            domain_loaders, node_to_domain = loaded
        else:
            domain_loaders = loaded
            node_to_domain = {k: k for k in domain_loaders.keys()}
        domain_names = list(domain_loaders.keys())
        num_domains = len(domain_names)  # Update num_domains based on actual loaded domains
        print(f"[Single Process] Loaded {num_domains} domains: {domain_names}")
    else:
        # For other datasets, use single train/test loader
        train_loader, test_loader = util.load_dataset_single_process(args)
        domain_loaders = {'default': (train_loader, test_loader)}
        domain_names = ['default']
        num_domains = 1

    # ====== Calculate STEPS_PER_EPOCH ======
    # Get maximum steps across all domains
    max_steps = max(len(train_loader) for train_loader, _ in domain_loaders.values())
    STEPS_PER_EPOCH = max_steps

    # total epochs as in original arguments
    TOTAL_EPOCHS = args.epoch
    # total iterations K (Algorithm 1)
    if args.total_iter is not None:
        K = args.total_iter
    else:
        K = STEPS_PER_EPOCH * TOTAL_EPOCHS

    # build infinite iterators for each domain
    def get_infinite_iterator(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    train_iters = {domain: get_infinite_iterator(train_loader) 
                   for domain, (train_loader, _) in domain_loaders.items()}

    # 載入基礎網路拓撲結構
    # graphid=-1：使用 3 節點全連接圖（用於 PACS）
    # graphid=6：使用隨機幾何圖（RGG），9 個節點
    if args.graphid == -1:
        subGraphs = util.select_graph(-1)
    elif args.graphid == 6:
        # RGG: use virtual node count in single-process mode
        rgg_nodes = args.num_nodes if args.num_nodes is not None else num_domains
        subGraphs = util.select_graph(6, num_nodes=rgg_nodes, radius=0.8, seed=args.randomSeed)
    else:
        subGraphs = util.select_graph(args.graphid)
    
    # Create dummy rank and size for topology (will be mapped to domains)
    dummy_rank = 0
    dummy_size = num_domains
    
    # define graph activation scheme with K iterations (no MPI comm for single process)
    if args.matcha:
        GP = MatchaProcessor(subGraphs, args.budget, dummy_rank, dummy_size, K, True, comm=None)
    else:
        GP = FixedProcessor(subGraphs, args.budget, dummy_rank, dummy_size, K, True, comm=None)

    if getattr(args, "debug_topology", False):
        print("\n====== TOPOLOGY CHECK (single-process) ======")
        print(f"graphid={args.graphid}, num_nodes={num_domains}, subgraphs={len(GP.subGraphs)}")
        print(f"neighbor_weight(alpha)={GP.neighbor_weight}")
        for i, neighbors in enumerate(GP.neighbors_info):
            print(f"subgraph[{i}] neighbors: {neighbors}")
        if args.dataset in ['pacs', 'vlcs'] and node_to_domain:
            print("node -> source domain mapping:")
            for node_name in sorted(node_to_domain.keys(), key=lambda x: int(x.split('_')[-1]) if x.startswith('node_') else x):
                print(f"  {node_name} -> {node_to_domain[node_name]}")
        print("=============================================\n")

    # define single process communicator
    communicator = SingleProcessCommunicator(domain_names, GP)

    # select neural network model for each domain
    if args.dataset == 'pacs':
        num_classes = 7
    elif args.dataset == 'vlcs':
        num_classes = 5
    else:
        num_classes = 10
    
    # Initialize models for each domain
    models_dict = {}
    optimizers_dict = {}
    schedulers_dict = {}
    diffusion_models_dict = {}
    optimizers_diffusion_dict = {}
    
    if getattr(args, 'pretrained', False):
        # Create first model (download pretrained weights if needed)
        first_model = util.select_model(num_classes, args)
        first_model = first_model.cuda()
        models_dict[domain_names[0]] = first_model
        
        # Copy first model's parameters to all other models
        for domain in domain_names[1:]:
            models_dict[domain] = util.select_model(num_classes, args)
            models_dict[domain] = models_dict[domain].cuda()
            # Copy parameters from first model
            models_dict[domain].load_state_dict(first_model.state_dict())
    else:
        # Create first model
        first_model = util.select_model(num_classes, args)
        first_model = first_model.cuda()
        models_dict[domain_names[0]] = first_model
        
        # Copy first model's parameters to all other models
        for domain in domain_names[1:]:
            models_dict[domain] = util.select_model(num_classes, args)
            models_dict[domain] = models_dict[domain].cuda()
            # Copy parameters from first model
            models_dict[domain].load_state_dict(first_model.state_dict())
    
    # Move all models to GPU and create optimizers
    for domain in domain_names:
        criterion = nn.CrossEntropyLoss().cuda()
        optimizers_dict[domain] = optim.SGD(
            models_dict[domain].parameters(), 
            lr=args.lr,
            momentum=args.momentum, 
            weight_decay=5e-4,
            nesterov=args.nesterov
        )
        schedulers_dict[domain] = CosineAnnealingLR(optimizers_dict[domain], T_max=K, eta_min=0.0)
        
        # Initialize diffusion model for OOD detection if enabled
        if getattr(args, 'use_ood', False):
            from dood.utils.diffusion import get_diffusion_model
            diffusion_model = get_diffusion_model(
                ft_size=512,
                denoiser_type="unet0d",
                diffusion_denoiser_channels=getattr(args, 'diffusion_channels', 512),
                num_diffusion_steps=getattr(args, 'diffusion_steps', 1000),
            ).cuda()
            optimizers_diffusion_dict[domain] = optim.Adam(
                diffusion_model.parameters(),
                lr=getattr(args, 'lr_diffusion', 5e-5)
            )
            models_dict[domain].diffusion_model = diffusion_model
            diffusion_models_dict[domain] = diffusion_model
    
    # All models already have identical initial parameters (copied from first model)

    # ====== Initialize BatchNorm state isolation for each domain =======
    # In multi-process mode, each rank has independent BatchNorm statistics.
    # In single-process mode, we need to maintain separate BatchNorm states for each domain
    # to prevent cross-domain contamination of running_mean/running_var.
    bn_states_dict = {}
    for domain in domain_names:
        bn_states_dict[domain] = {}
        for name, module in models_dict[domain].named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                bn_states_dict[domain][name] = {
                    'running_mean': module.running_mean.clone(),
                    'running_var': module.running_var.clone(),
                    'num_batches_tracked': module.num_batches_tracked.clone() if hasattr(module, 'num_batches_tracked') else None
                }
    # init recorders for each domain
    comp_time, comm_time = 0, 0
    # Use domain index for recorder (compatible with existing Recorder interface)
    recorders = {domain: util.Recorder(args, domain_names.index(domain)) for domain in domain_names}
    losses_dict = {domain: util.AverageMeter() for domain in domain_names}
    top1_dict = {domain: util.AverageMeter() for domain in domain_names}
    style_aug_flag_meters = {domain: util.AverageMeter() for domain in domain_names}
    # Meters for z_style / z_hard training (enabled by --use_hard_style_adv)
    task_loss_meters = {domain: util.AverageMeter() for domain in domain_names}
    clean_ce_meters = {domain: util.AverageMeter() for domain in domain_names}
    clean_conf_meters = {domain: util.AverageMeter() for domain in domain_names}
    style_conf_meters = {domain: util.AverageMeter() for domain in domain_names}
    clean_style_ce_cos_meters = {domain: util.AverageMeter() for domain in domain_names}
    clean_style_sym_kl_meters = {domain: util.AverageMeter() for domain in domain_names}
    style_ce_meters = {domain: util.AverageMeter() for domain in domain_names}
    hard_ce_meters = {domain: util.AverageMeter() for domain in domain_names}
    hard_ce_delta_meters = {domain: util.AverageMeter() for domain in domain_names}
    hard_ce_harder_ratio_meters = {domain: util.AverageMeter() for domain in domain_names}
    # Terminal-only: accumulate per-sample hard CE deltas for epoch-level distribution stats.
    hard_ce_delta_samples = {domain: [] for domain in domain_names}
    hard_ood_loss_meters = {domain: util.AverageMeter() for domain in domain_names}
    # Terminal-only hard-adv classification behavior monitors.
    hard_style_conf_meters = {domain: util.AverageMeter() for domain in domain_names}
    hard_hard_conf_meters = {domain: util.AverageMeter() for domain in domain_names}
    hard_style_target_max_gap_meters = {domain: util.AverageMeter() for domain in domain_names}
    hard_hard_target_max_gap_meters = {domain: util.AverageMeter() for domain in domain_names}
    hard_flip_all_ratio_meters = {domain: util.AverageMeter() for domain in domain_names}
    hard_flip_on_style_correct_ratio_meters = {domain: util.AverageMeter() for domain in domain_names}
    hard_mu_hit_lower_ratio_meters = {domain: util.AverageMeter() for domain in domain_names}
    hard_mu_hit_upper_ratio_meters = {domain: util.AverageMeter() for domain in domain_names}
    hard_sigma_hit_lower_ratio_meters = {domain: util.AverageMeter() for domain in domain_names}
    hard_sigma_hit_upper_ratio_meters = {domain: util.AverageMeter() for domain in domain_names}
    adv_ood_loss_meters = {domain: util.AverageMeter() for domain in domain_names}
    adv_cls_inner_meters = {domain: util.AverageMeter() for domain in domain_names}
    # Monitor statistics and gradients for mu/sigma in inner loop
    mu_orig_norm_meters = {domain: util.AverageMeter() for domain in domain_names}
    sigma_orig_norm_meters = {domain: util.AverageMeter() for domain in domain_names}
    grad_mu_norm_meters = {domain: util.AverageMeter() for domain in domain_names}
    grad_sigma_norm_meters = {domain: util.AverageMeter() for domain in domain_names}
    # Monitor relative update size (adv_eta * grad) vs parameter scale.
    # Useful when grad norms shrink due to normalization/reduction in diffusion loss.
    rel_mu_step_meters = {domain: util.AverageMeter() for domain in domain_names}
    rel_sigma_step_meters = {domain: util.AverageMeter() for domain in domain_names}
    # Optional: decompose inner-loop gradients into OOD vs CLS parts.
    grad_ood_norm_meters = {domain: util.AverageMeter() for domain in domain_names}
    grad_cls_norm_meters = {domain: util.AverageMeter() for domain in domain_names}
    grad_ood_cls_cos_meters = {domain: util.AverageMeter() for domain in domain_names}
    # grad_mu_cls sparsity: mean over batch of (sum_c |g_c|) / max_c |g_c|  (in [1, C]; lower => peakier / fewer large channels)
    grad_mu_cls_l1_max_ratio_meters = {domain: util.AverageMeter() for domain in domain_names}
    # Pooled feature shift: vec_hard vs vec_style (same forward path intent as style branch baseline).
    vec_hard_vs_style_delta_meters = {domain: util.AverageMeter() for domain in domain_names}
    # Style CE vs Hard CE gradient monitors (per-parameter-group).
    grad_style_norm_layer3_meters = {domain: util.AverageMeter() for domain in domain_names}
    grad_hard_norm_layer3_meters = {domain: util.AverageMeter() for domain in domain_names}
    grad_style_hard_cos_layer3_meters = {domain: util.AverageMeter() for domain in domain_names}
    grad_style_norm_layer4_meters = {domain: util.AverageMeter() for domain in domain_names}
    grad_hard_norm_layer4_meters = {domain: util.AverageMeter() for domain in domain_names}
    grad_style_hard_cos_layer4_meters = {domain: util.AverageMeter() for domain in domain_names}
    grad_style_norm_head_meters = {domain: util.AverageMeter() for domain in domain_names}
    grad_hard_norm_head_meters = {domain: util.AverageMeter() for domain in domain_names}
    grad_style_hard_cos_head_meters = {domain: util.AverageMeter() for domain in domain_names}
    grad_style_norm_backbone_meters = {domain: util.AverageMeter() for domain in domain_names}
    grad_hard_norm_backbone_meters = {domain: util.AverageMeter() for domain in domain_names}
    grad_style_hard_cos_backbone_meters = {domain: util.AverageMeter() for domain in domain_names}
    # Consensus drift monitors (model aggregation pre/post communicate).
    drift_l2_layer4_meters = {domain: util.AverageMeter() for domain in domain_names}
    drift_rel_layer4_meters = {domain: util.AverageMeter() for domain in domain_names}
    drift_l2_head_meters = {domain: util.AverageMeter() for domain in domain_names}
    drift_rel_head_meters = {domain: util.AverageMeter() for domain in domain_names}
    drift_l2_backbone_meters = {domain: util.AverageMeter() for domain in domain_names}
    drift_rel_backbone_meters = {domain: util.AverageMeter() for domain in domain_names}
    drift_pairwise_head_pre_meters = util.AverageMeter()
    drift_pairwise_head_post_meters = util.AverageMeter()
    drift_pairwise_backbone_pre_meters = util.AverageMeter()
    drift_pairwise_backbone_post_meters = util.AverageMeter()
    # Terminal-only monitoring; enabled by --hard_adv_monitor_sparsity / --hard_adv_monitor_direction.
    z_hard_sparsity_meters = {domain: util.AverageMeter() for domain in domain_names}
    vec_hard_sparsity_meters = {domain: util.AverageMeter() for domain in domain_names}
    vec_clean_l4_sparsity_meters = {domain: util.AverageMeter() for domain in domain_names}
    vec_hard_vs_clean_l4_delta_meters = {domain: util.AverageMeter() for domain in domain_names}
    vec_style_hard_dir_cos_meters = {domain: util.AverageMeter() for domain in domain_names}
    vec_style_from_clean_norm_meters = {domain: util.AverageMeter() for domain in domain_names}
    vec_hard_from_clean_norm_meters = {domain: util.AverageMeter() for domain in domain_names}
    vec_style_hard_dir_cos_low03_meters = {domain: util.AverageMeter() for domain in domain_names}
    # Orthogonal hard-adv debug (only meaningful when --hard_adv_orthogonal_style_clean).
    # removed_ratio = ||proj_v(g)|| / ||g|| in concat(mu,sigma) space, computed per-sample and averaged.
    orth_removed_ratio_meters = {domain: util.AverageMeter() for domain in domain_names}
    # Fraction of samples that fall back to the original gradient because ||g_orth|| is too small.
    orth_fallback_frac_meters = {domain: util.AverageMeter() for domain in domain_names}
    # concat(mu,sigma) distance vs leave-out test style pool (Phase A: --hard_style_test_pool).
    style_triplet_l2_orig_pool_meters = {domain: util.AverageMeter() for domain in domain_names}
    style_triplet_l2_hard_pool_meters = {domain: util.AverageMeter() for domain in domain_names}
    style_triplet_l2_orig_hard_meters = {domain: util.AverageMeter() for domain in domain_names}
    style_triplet_cos_orig_pool_meters = {domain: util.AverageMeter() for domain in domain_names}
    style_triplet_cos_hard_pool_meters = {domain: util.AverageMeter() for domain in domain_names}
    style_triplet_cos_orig_hard_meters = {domain: util.AverageMeter() for domain in domain_names}
    # Terminal-only: per-sample direction cosine distribution for epoch-level quantiles.
    vec_style_hard_dir_cos_samples = {domain: [] for domain in domain_names}
    loss_diff_meters = {domain: util.AverageMeter() for domain in domain_names} if getattr(args, 'use_ood', False) else None
    # Accumulate batch-mean |grad_mu_cls| per channel for optional epoch dump (only when log_grad_parts batches run).
    grad_mu_cls_abs_ch_sum = {domain: None for domain in domain_names}
    grad_mu_cls_abs_ch_count = {domain: 0 for domain in domain_names}
    # Sketch-on-photo class-wise monitor (PACS has 7 classes).
    sketch_num_classes = 7
    sketch_vol_window = 10
    sketch_class_acc_history = [[] for _ in range(sketch_num_classes)]
    tic = time.time()

    hard_style_test_pool = bool(getattr(args, "hard_style_test_pool", False))
    test_style_pool_frac = float(getattr(args, "hard_style_test_pool_frac", 0.1))
    # Per-domain stacked [N,C,1,1]; rebuilt at each pseudo-epoch start when --hard_style_test_pool.
    test_style_pool_mu = {d: None for d in domain_names}
    test_style_pool_sigma = {d: None for d in domain_names}
    # Mean over pool -> [1,C,1,1] anchor for (1-beta)*orig + beta*anchor.
    test_style_anchor_mu = {d: None for d in domain_names}
    test_style_anchor_sigma = {d: None for d in domain_names}

    # ===== start training with fixed total steps K (Algorithm 1) =====
    for k in range(K):
        # Set all models to training mode
        for model in models_dict.values():
            model.train()

        start_time = time.time()

        # Phase A: rebuild leave-out test style pool at epoch boundaries (current backbone, eval extract).
        if (
            hard_style_test_pool
            and getattr(args, "use_hard_style_adv", False)
            and STEPS_PER_EPOCH > 0
            and (k % STEPS_PER_EPOCH == 0)
            and args.model == "res"
        ):
            epoch_for_pool = int(k // STEPS_PER_EPOCH) + 1
            _, test_loader_shared = domain_loaders[domain_names[0]]
            test_ds = test_loader_shared.dataset
            n_tot = len(test_ds)
            pool_n = max(1, int(round(n_tot * test_style_pool_frac)))
            gen = torch.Generator()
            gen.manual_seed(int(args.randomSeed) + epoch_for_pool * 100_003 + 17)
            perm = torch.randperm(n_tot, generator=gen)[:pool_n]
            pool_indices = perm.tolist()
            pool_bs = min(int(args.bs), int(test_loader_shared.batch_size or 64))
            adv_eps_pool = float(getattr(args, "hard_adv_eps", 1e-6))
            pin_mem = bool(getattr(test_loader_shared, "pin_memory", True))
            print(
                f"[test_style_pool] epoch={epoch_for_pool} rebuild leave-out test indices: "
                f"n_test={n_tot} pool_size={pool_n} frac={test_style_pool_frac} "
                f"mode={str(getattr(args, 'hard_style_test_pool_mode', 'random')).lower()} "
                f"beta={max(0.0, min(1.0, float(getattr(args, 'hard_style_test_anchor_beta', 0.1))))}"
            )
            for _dom in domain_names:
                mdl = models_dict[_dom]
                try:
                    pmu, psig = _collect_test_style_pool_layer3_mu_sigma(
                        mdl,
                        test_ds,
                        pool_indices,
                        pool_bs,
                        adv_eps_pool,
                        pin_memory=pin_mem,
                    )
                except AttributeError as exc:
                    print(
                        f"[test_style_pool] ERROR domain={_dom}: need extract_features_to_layer3 on model ({exc}). "
                        "Disabling pool for this run step."
                    )
                    pmu, psig = None, None
                test_style_pool_mu[_dom] = pmu
                test_style_pool_sigma[_dom] = psig
                if pmu is None or psig is None:
                    test_style_anchor_mu[_dom] = None
                    test_style_anchor_sigma[_dom] = None
                    print(f"[test_style_pool] WARNING domain={_dom}: empty pool; z_hard falls back to adversarial mu/sigma.")
                else:
                    _mode = str(getattr(args, "hard_style_test_pool_mode", "random")).lower()
                    if _mode == "mean":
                        test_style_anchor_mu[_dom] = pmu.mean(dim=0, keepdim=True)
                        test_style_anchor_sigma[_dom] = psig.mean(dim=0, keepdim=True)
                    else:
                        test_style_anchor_mu[_dom] = None
                        test_style_anchor_sigma[_dom] = None
                    if _mode == "mean":
                        print(
                            f"[test_style_pool] domain={_dom} stacked_mu_sigma shape={tuple(pmu.shape)} "
                            f"anchor_mu_sigma shape={tuple(test_style_anchor_mu[_dom].shape)} mode=mean"
                        )
                    else:
                        print(
                            f"[test_style_pool] domain={_dom} stacked_mu_sigma shape={tuple(pmu.shape)} mode=random"
                        )

        # ========== 第一阶段：计算所有 domain 的风格统计量（不训练）==========
        style_vecs_dict = {}
        # Cache one batch per domain so style stats and training share identical data
        # (matching train_mpi.py behavior).
        batch_cache = {}
        clean_layer3_cache = {}
        if (getattr(args, "use_style_stats", False) or getattr(args, "use_style_shift", False)) and args.model == "res":
            for domain in domain_names:
                batch = next(train_iters[domain])
                data, target, meta = util.unpack_batch(batch)
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                batch_cache[domain] = (data, target, meta)
                
                with torch.no_grad():  # 不计算梯度，节省内存
                    model = models_dict[domain]
                    # 只提取特征到 layer3，不应用 style shift
                    feats = model.extract_features_to_layer3(data)
                    if spatial_dump_enabled and domain == spatial_dump_domain:
                        clean_layer3_cache[domain] = feats["layer3"].detach()
                    
                    # Compute STYLEDDG-style statistics (batch-level, per channel)
                    style_stats = compute_multi_layer_style_stats(
                        feats,
                        eta=getattr(args, "style_eta", 1e-5),
                    )
                    
                    # Extract channel information from style_stats for unflattening received neighbor stats
                    # Only set once (on first iteration or when not set)
                    if communicator.channels_per_layer is None:
                        channels_per_layer = {}
                        for layer_name in ["layer1", "layer2", "layer3"]:
                            if layer_name in style_stats:
                                # Get channel count from mu_bar shape
                                channels_per_layer[layer_name] = style_stats[layer_name]["mu_bar"].shape[0]
                        communicator.set_style_channels(channels_per_layer)
                    
                    # Flatten to a single style vector per batch/device for communication
                    style_vec = flatten_style_stats(
                        style_stats,
                        layer_order=["layer1", "layer2", "layer3"],
                    )
                    # Keep on GPU for single process (no need to move to CPU)
                    style_vecs_dict[domain] = style_vec.detach()
        
        # ========== 通信交换风格统计量 ==========
        # Exchange style statistics only (no model parameters) with neighbors
        if style_vecs_dict:
            d_comm_time = communicator.communicate(models_dict, style_vecs_dict=style_vecs_dict)
            comm_time += d_comm_time

        # ========== 第二阶段：正式训练所有 domain（使用交换到的风格统计量）==========
        use_style_stats = getattr(args, "use_style_stats", False)
        use_style_shift = getattr(args, "use_style_shift", False)
        debug_style_shift = getattr(args, "debug_style_shift", False)
        
        # Process each domain sequentially
        for domain in domain_names:
            model = models_dict[domain]
            optimizer = optimizers_dict[domain]
            criterion = nn.CrossEntropyLoss().cuda()
            
            # ====== Restore BatchNorm state for this domain ======
            # This ensures each domain maintains independent BatchNorm statistics,
            # matching multi-process behavior where each rank has separate BatchNorm states.
            for name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    if name in bn_states_dict[domain]:
                        module.running_mean.copy_(bn_states_dict[domain][name]['running_mean'])
                        module.running_var.copy_(bn_states_dict[domain][name]['running_var'])
                        if bn_states_dict[domain][name]['num_batches_tracked'] is not None:
                            module.num_batches_tracked.copy_(bn_states_dict[domain][name]['num_batches_tracked'])
            
            # Reuse phase-1 batch when style stats are enabled; otherwise fetch normally.
            if domain in batch_cache:
                data, target, batch_meta = batch_cache[domain]
            else:
                batch = next(train_iters[domain])
                data, target, batch_meta = util.unpack_batch(batch)
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            
            # Forward + loss
            # Note: communicator.neighbor_style_stats uses domain names as keys in single process mode.
            use_hard_style_adv = getattr(args, "use_hard_style_adv", False)
            adv_only_when_style_aug = not bool(getattr(args, "hard_adv_always", False))
            adv_steps = int(getattr(args, "hard_adv_steps", 1))
            adv_eta = float(getattr(args, "hard_adv_eta", 0.01))
            adv_lambda = float(getattr(args, "hard_adv_lambda", 1.0))
            adv_eps = float(getattr(args, "hard_adv_eps", 1e-6))
            adv_ood_timestep = int(getattr(args, "hard_adv_ood_timestep", 250))
            adv_update_mode = str(getattr(args, "hard_adv_update_mode", "gd")).lower()
            adv_max_rel_margin = float(getattr(args, "hard_adv_max_rel_margin", 0.1))
            adv_grad_norm_eps = float(getattr(args, "hard_adv_grad_norm_eps", 1e-8))
            adv_grad_gate_tau = float(getattr(args, "hard_adv_grad_gate_tau", 0.0))
            adv_cls_only = bool(getattr(args, "hard_adv_cls_only", False))
            adv_equal_ce_weight = bool(getattr(args, "hard_adv_equal_ce_weight", False))
            hard_adv_z_base = str(getattr(args, "hard_adv_z_base", "clean")).lower()
            hard_adv_clean_loss_weight = float(getattr(args, "hard_adv_clean_loss_weight", 0.0))
            hard_adv_monitor_sparsity = bool(getattr(args, "hard_adv_monitor_sparsity", False))
            hard_adv_monitor_direction = bool(getattr(args, "hard_adv_monitor_direction", False))
            hard_adv_orthogonal_style_clean = bool(getattr(args, "hard_adv_orthogonal_style_clean", False))
            hard_adv_orth_fallback_ratio = float(getattr(args, "hard_adv_orth_fallback_ratio", 0.05))
            hard_style_test_anchor_beta = max(
                0.0, min(1.0, float(getattr(args, "hard_style_test_anchor_beta", 0.1)))
            )
            hard_style_test_pool_mode = str(getattr(args, "hard_style_test_pool_mode", "random")).lower()
            log_grad_parts_every = int(getattr(args, "hard_adv_log_grad_parts_every", 10))
            # Use step index to estimate the current (integer) epoch.
            # Note: wandb logging uses 1-based epoch = (k+1)//STEPS_PER_EPOCH.
            # Keep inner-loop logging aligned with that convention.
            epoch_i_1based = int(k // STEPS_PER_EPOCH) + 1 if STEPS_PER_EPOCH > 0 else 1

            if use_hard_style_adv:
                # Require diffusion model for OOD guidance.
                if not (getattr(args, 'use_ood', False) and domain in diffusion_models_dict):
                    raise RuntimeError("--use_hard_style_adv requires --use_ood and an initialized diffusion model.")
                diffusion_model = diffusion_models_dict[domain]

                if args.model != "res":
                    raise RuntimeError("--use_hard_style_adv currently supports only --model res.")

                if not hasattr(model, "forward_to_layer3_style") or not hasattr(model, "forward_from_layer3"):
                    raise RuntimeError("Model must implement forward_to_layer3_style/forward_from_layer3 for hard style adv.")
                # This method assumes a ResNet with layer4 producing a 512-d pooled vector (standard ResNet wrapper).
                expected_ft = int(getattr(diffusion_model.denoiser, "in_channels", 512))
                if not hasattr(model, "backbone") or not hasattr(getattr(model, "backbone"), "layer4"):
                    raise RuntimeError(
                        "--use_hard_style_adv requires a backbone with layer4 (set --resnet_type standard)."
                    )

                # Switch communicator view to current domain only (for style shift).
                communicator.set_active_domain(domain)

                should_save_spatial_dump = (
                    spatial_dump_enabled
                    and domain == spatial_dump_domain
                    and STEPS_PER_EPOCH > 0
                    and (epoch_i_1based % spatial_dump_every == 0)
                    and ((k % STEPS_PER_EPOCH) == 0)
                    and (epoch_i_1based not in spatial_dump_saved_epochs)
                )

                z_clean = None
                need_clean_for_adv = (hard_adv_z_base == "clean")
                need_clean_for_dump = should_save_spatial_dump and (clean_layer3_cache.get(domain) is None)
                # For monitor_direction, always compute z_clean reference regardless of hard_adv_z_base.
                # This runs in the existing safe monitor path (temporary eval + restored mode).
                need_clean_for_monitor = hard_adv_monitor_sparsity or hard_adv_monitor_direction
                need_clean_for_orth = bool(getattr(args, "hard_adv_orthogonal_style_clean", False))
                if need_clean_for_adv or need_clean_for_dump or need_clean_for_monitor or need_clean_for_orth:
                    # Get z_clean with gradient graph while preventing an extra BN running-stats update.
                    # We temporarily switch backbone/model to eval() for this forward only.
                    backbone_for_clean = getattr(model, "backbone", None)
                    prev_backbone_training_for_clean = None
                    prev_model_training_for_clean = model.training
                    if backbone_for_clean is not None:
                        prev_backbone_training_for_clean = backbone_for_clean.training
                        backbone_for_clean.eval()
                    else:
                        model.eval()
                    try:
                        z_clean = model.extract_features_to_layer3(data)["layer3"]
                    finally:
                        if backbone_for_clean is not None and prev_backbone_training_for_clean is not None:
                            backbone_for_clean.train(prev_backbone_training_for_clean)
                        else:
                            model.train(prev_model_training_for_clean)

                # 1) Get z_style at layer3 (may or may not be style-augmented).
                z_style = model.forward_to_layer3_style(
                    data,
                    communicator=communicator if (use_style_stats or use_style_shift) else None,
                    debug_style_shift=debug_style_shift,
                    iter_num=k+1,
                    rank=domain_names.index(domain),
                )

                # Conservative batch-level flag.
                style_aug_activated = bool(getattr(model, "last_style_aug_activated", False))
                style_aug_flag_meters[domain].update(1.0 if style_aug_activated else 0.0, data.size(0))

                # 2) Optionally skip inner loop if no style aug happened.
                run_adv = (not adv_only_when_style_aug) or style_aug_activated
                z_hard = None
                z_base = z_clean if hard_adv_z_base == "clean" else z_style

                logits_style, vec_style = model.forward_from_layer3(z_style)
                if vec_style.ndim != 2 or vec_style.size(1) != expected_ft:
                    raise RuntimeError(
                        f"Diffusion expects ft_size={expected_ft}, but got vec_style shape={tuple(vec_style.shape)}. "
                        "Ensure you are using --resnet_type standard and ft_size matches the pooled feature dim."
                    )

                # Inner loop: generate z_hard (one or few steps), only updating mu/sigma.
                if run_adv:
                    with torch.no_grad():
                        mu_base = z_base.detach().mean(dim=(2, 3), keepdim=True)
                        var_base = z_base.detach().var(dim=(2, 3), unbiased=False, keepdim=True)
                        sigma_base = torch.sqrt(var_base + adv_eps)
                        mu_orig = z_style.detach().mean(dim=(2, 3), keepdim=True)
                        var_orig = z_style.detach().var(dim=(2, 3), unbiased=False, keepdim=True)
                        sigma_orig = torch.sqrt(var_orig + adv_eps)
                        # 監控 mu_orig / sigma_orig 的 L2 範數（batch-level）
                        mu_orig_norm = mu_orig.norm().item()
                        sigma_orig_norm = sigma_orig.norm().item()
                        mu_orig_norm_meters[domain].update(mu_orig_norm, data.size(0))
                        sigma_orig_norm_meters[domain].update(sigma_orig_norm, data.size(0))

                    # Unit direction of (style - clean) in (mu, sigma) concat space for optional orthogonal inner grads.
                    v_style_clean_unit = None
                    if hard_adv_orthogonal_style_clean and z_clean is not None:
                        with torch.no_grad():
                            mu_clean_ref = z_clean.detach().mean(dim=(2, 3), keepdim=True)
                            var_clean_ref = z_clean.detach().var(dim=(2, 3), unbiased=False, keepdim=True)
                            sigma_clean_ref = torch.sqrt(var_clean_ref + adv_eps)
                            d_mu_sc = (mu_orig - mu_clean_ref).reshape(z_style.size(0), -1)
                            d_sig_sc = (sigma_orig - sigma_clean_ref).reshape(z_style.size(0), -1)
                            v_cat = torch.cat([d_mu_sc, d_sig_sc], dim=1)
                            v_norm = v_cat.norm(dim=1, keepdim=True).clamp_min(1e-12)
                            v_style_clean_unit = v_cat / v_norm

                    # z_hat uses detached z_base-normalized content (inner loop should not affect backbone).
                    z_norm_detached = (z_base.detach() - mu_base) / (sigma_base + adv_eps)

                    mu_adv = mu_orig.clone().requires_grad_(True)
                    sigma_adv = sigma_orig.clone().requires_grad_(True)

                    # Freeze params & stop BN/normalization updates during inner loop.
                    backbone = getattr(model, "backbone", None)
                    prev_backbone_training = None
                    if backbone is not None:
                        prev_backbone_training = backbone.training
                        backbone.eval()
                    prev_model_training = model.training
                    model.eval()
                    prev_diff_training = diffusion_model.training
                    diffusion_model.eval()

                    # Freeze parameter grads (we only need grads for mu_adv/sigma_adv).
                    for p in model.parameters():
                        p.requires_grad_(False)
                    for p in diffusion_model.parameters():
                        p.requires_grad_(False)

                    # Multi-step (default 1)
                    mu_cur, sigma_cur = mu_adv, sigma_adv
                    L_cls_inner_vec = None
                    for _ in range(max(1, adv_steps)):
                        z_hat = z_norm_detached * sigma_cur + mu_cur
                        logits_hat, vec_hat = model.forward_from_layer3(z_hat)
                        if vec_hat.ndim != 2 or vec_hat.size(1) != expected_ft:
                            raise RuntimeError(
                                f"Expected vec_hat shape (B,{expected_ft}), got {tuple(vec_hat.shape)}."
                            )

                        vec_hat_norm = diffusion_model.normalize(vec_hat)
                        L_ood = diffusion_model.get_loss_at_timestep(vec_hat_norm, adv_ood_timestep)
                        L_cls_inner_vec = F.cross_entropy(logits_hat, target, reduction='none')
                        L_cls_inner = L_cls_inner_vec.mean()

                        if adv_cls_only:
                            L_adv = adv_lambda * L_cls_inner
                        else:
                            L_adv = -L_ood + adv_lambda * L_cls_inner

                        log_grad_parts = bool(getattr(args, "hard_adv_log_grad_parts", False)) and (
                            log_grad_parts_every > 0 and (epoch_i_1based % log_grad_parts_every == 0)
                        )
                        if log_grad_parts:
                            # Compute parts first, then sum to get total grad.
                            # This avoids "backward through graph a second time" errors.
                            if adv_cls_only:
                                grad_mu_ood = torch.zeros_like(mu_cur)
                                grad_sigma_ood = torch.zeros_like(sigma_cur)
                                grad_mu_cls, grad_sigma_cls = torch.autograd.grad(
                                    adv_lambda * L_cls_inner, [mu_cur, sigma_cur], create_graph=False, retain_graph=False
                                )
                                grad_mu = grad_mu_cls
                                grad_sigma = grad_sigma_cls
                            else:
                                grad_mu_ood, grad_sigma_ood = torch.autograd.grad(
                                    -L_ood, [mu_cur, sigma_cur], create_graph=False, retain_graph=True
                                )
                                grad_mu_cls, grad_sigma_cls = torch.autograd.grad(
                                    adv_lambda * L_cls_inner, [mu_cur, sigma_cur], create_graph=False, retain_graph=False
                                )
                                grad_mu = grad_mu_ood + grad_mu_cls
                                grad_sigma = grad_sigma_ood + grad_sigma_cls

                            g_ood = torch.cat([grad_mu_ood.reshape(-1), grad_sigma_ood.reshape(-1)])
                            g_cls = torch.cat([grad_mu_cls.reshape(-1), grad_sigma_cls.reshape(-1)])
                            g_ood_norm = float(g_ood.norm().item())
                            g_cls_norm = float(g_cls.norm().item())
                            denom = (g_ood_norm * g_cls_norm) + 1e-12
                            g_cos = float((g_ood @ g_cls).item() / denom) if denom > 0 else 0.0
                            grad_ood_norm_meters[domain].update(g_ood_norm, data.size(0))
                            grad_cls_norm_meters[domain].update(g_cls_norm, data.size(0))
                            grad_ood_cls_cos_meters[domain].update(g_cos, data.size(0))

                            # grad_mu_cls: L1/max per sample (batch mean) + accumulate epoch-mean |g| per channel.
                            g_flat = grad_mu_cls.abs().squeeze(-1).squeeze(-1)
                            l1_per = g_flat.sum(dim=1)
                            max_per = g_flat.max(dim=1).values.clamp_min(1e-12)
                            ratio = (l1_per / max_per).mean().item()
                            grad_mu_cls_l1_max_ratio_meters[domain].update(ratio, data.size(0))
                            ch_mean = grad_mu_cls.abs().mean(dim=(0, 2, 3)).detach()
                            bs = int(data.size(0))
                            if grad_mu_cls_abs_ch_sum[domain] is None:
                                grad_mu_cls_abs_ch_sum[domain] = ch_mean * bs
                            else:
                                grad_mu_cls_abs_ch_sum[domain] = grad_mu_cls_abs_ch_sum[domain] + ch_mean * bs
                            grad_mu_cls_abs_ch_count[domain] += bs
                        else:
                            grad_mu, grad_sigma = torch.autograd.grad(
                                L_adv, [mu_cur, sigma_cur], create_graph=False
                            )

                        # Optional: remove inner-loop gradient component parallel to (style-clean) in (mu,sigma) space.
                        if v_style_clean_unit is not None:
                            Bsz, Cch = z_style.size(0), z_style.size(1)
                            g_mu_f = grad_mu.reshape(Bsz, -1)
                            g_sig_f = grad_sigma.reshape(Bsz, -1)
                            g_cat = torch.cat([g_mu_f, g_sig_f], dim=1)
                            v = v_style_clean_unit
                            dot = (g_cat * v).sum(dim=1, keepdim=True)
                            g_par = dot * v
                            g_orth = g_cat - g_par
                            g_orth_n = g_orth.norm(dim=1, keepdim=True)
                            g_cat_n = g_cat.norm(dim=1, keepdim=True).clamp_min(1e-12)
                            tiny = g_orth_n < (hard_adv_orth_fallback_ratio * g_cat_n)
                            g_use = torch.where(tiny, g_cat, g_orth)
                            grad_mu = g_use[:, :Cch].reshape_as(grad_mu)
                            grad_sigma = g_use[:, Cch:].reshape_as(grad_sigma)
                            with torch.no_grad():
                                removed_ratio = (g_par.norm(dim=1) / g_cat_n.squeeze(1)).mean().item()
                                fallback_frac = tiny.float().mean().item()
                            orth_removed_ratio_meters[domain].update(float(removed_ratio), data.size(0))
                            orth_fallback_frac_meters[domain].update(float(fallback_frac), data.size(0))

                        # 監控梯度範數（取最後一步為代表）
                        grad_mu_norm = grad_mu.norm().item()
                        grad_sigma_norm = grad_sigma.norm().item()
                        grad_mu_norm_meters[domain].update(grad_mu_norm, data.size(0))
                        grad_sigma_norm_meters[domain].update(grad_sigma_norm, data.size(0))

                        # Update mu/sigma (either plain GD or Projected Normalized GD).
                        if adv_update_mode == "pngd":
                            # 1) Normalize gradient direction per-sample (across channels).
                            # Shapes: (B,C,1,1) -> norms: (B,1,1,1)
                            gmu_norm = grad_mu.norm(dim=1, keepdim=True)
                            gsig_norm = grad_sigma.norm(dim=1, keepdim=True)

                            # 2) Optional gate: if gradient is too small, fall back to plain GD step.
                            # This avoids amplifying numerical noise when grads ~ 0.
                            use_fallback_mu = gmu_norm < adv_grad_gate_tau
                            use_fallback_sigma = gsig_norm < adv_grad_gate_tau

                            normed_grad_mu = grad_mu / (gmu_norm + adv_grad_norm_eps)
                            normed_grad_sigma = grad_sigma / (gsig_norm + adv_grad_norm_eps)

                            mu_step = mu_cur - adv_eta * normed_grad_mu
                            sigma_step = sigma_cur - adv_eta * normed_grad_sigma

                            if adv_grad_gate_tau > 0.0:
                                mu_step = torch.where(use_fallback_mu, mu_cur - adv_eta * grad_mu, mu_step)
                                sigma_step = torch.where(use_fallback_sigma, sigma_cur - adv_eta * grad_sigma, sigma_step)

                            # 3) Project to a relative box around (mu_orig, sigma_orig).
                            margin_mu = adv_max_rel_margin * mu_orig.abs().clamp_min(1e-3)
                            margin_sigma = adv_max_rel_margin * sigma_orig
                            mu_next = torch.max(torch.min(mu_step, mu_orig + margin_mu), mu_orig - margin_mu)
                            sigma_next = torch.max(torch.min(sigma_step, sigma_orig + margin_sigma), sigma_orig - margin_sigma)
                            sigma_next = sigma_next.clamp_min(adv_eps)
                        else:
                            # Plain GD step (original behavior).
                            mu_next = mu_cur - adv_eta * grad_mu
                            sigma_next = (sigma_cur - adv_eta * grad_sigma).clamp_min(adv_eps)

                        # Relative step size based on *actual* update after projection/clamp.
                        step_mu_norm = (mu_next.detach() - mu_cur.detach()).norm().item()
                        step_sigma_norm = (sigma_next.detach() - sigma_cur.detach()).norm().item()
                        mu_scale = mu_cur.detach().norm().item() + 1e-12
                        sigma_scale = sigma_cur.detach().norm().item() + 1e-12
                        rel_mu_step_meters[domain].update(step_mu_norm / mu_scale, data.size(0))
                        rel_sigma_step_meters[domain].update(step_sigma_norm / sigma_scale, data.size(0))

                        mu_cur, sigma_cur = mu_next, sigma_next

                    mu_hard = mu_cur.detach()
                    sigma_hard = sigma_cur.detach()
                    if not torch.isfinite(mu_hard).all() or not torch.isfinite(sigma_hard).all():
                        raise FloatingPointError("Non-finite mu_hard/sigma_hard detected during inner loop.")
                    with torch.no_grad():
                        bound_tol = 1e-7
                        if adv_update_mode == "pngd":
                            mu_lower = mu_orig - margin_mu
                            mu_upper = mu_orig + margin_mu
                            sigma_lower = sigma_orig - margin_sigma
                            sigma_upper = sigma_orig + margin_sigma
                            mu_hit_lower_ratio = float((mu_hard <= (mu_lower + bound_tol)).float().mean().item())
                            mu_hit_upper_ratio = float((mu_hard >= (mu_upper - bound_tol)).float().mean().item())
                            sigma_hit_lower_ratio = float((sigma_hard <= (sigma_lower + bound_tol)).float().mean().item())
                            sigma_hit_upper_ratio = float((sigma_hard >= (sigma_upper - bound_tol)).float().mean().item())
                        else:
                            # No explicit box projection in plain GD; keep zero ratios.
                            mu_hit_lower_ratio = 0.0
                            mu_hit_upper_ratio = 0.0
                            sigma_hit_lower_ratio = 0.0
                            sigma_hit_upper_ratio = 0.0

                    # Restore training modes and requires_grad for outer loop.
                    if prev_backbone_training is not None:
                        backbone.train(prev_backbone_training)
                    model.train(prev_model_training)
                    diffusion_model.train(prev_diff_training)
                    for p in model.parameters():
                        p.requires_grad_(True)
                    for p in diffusion_model.parameters():
                        p.requires_grad_(True)

                    # Outer loop: z_hard re-composed from LIVE z_base-normalized content.
                    # Default: adversarial mu_hard/sigma_hard.
                    # --hard_style_test_pool: leave-out test subset -> stacked [N,C,1,1] pmu/psig per epoch.
                    #   mode=random: per-sample idx~U(0,N-1), mu_t=pmu[idx], mix (1-beta)*orig+beta*drawn.
                    #   mode=mean: pool mean anchor, mix (1-beta)*orig+beta*anchor. beta=1 -> pure ref side.
                    z_norm_live = (z_base - mu_base) / (sigma_base + adv_eps)
                    pmu = test_style_pool_mu.get(domain) if hard_style_test_pool else None
                    psig = test_style_pool_sigma.get(domain) if hard_style_test_pool else None
                    pool_ok = (
                        hard_style_test_pool
                        and pmu is not None
                        and psig is not None
                        and pmu.numel() > 0
                        and pmu.size(0) > 0
                    )
                    if pool_ok:
                        Bsz = z_style.size(0)
                        b = hard_style_test_anchor_beta
                        if hard_style_test_pool_mode == "mean":
                            amu = test_style_anchor_mu.get(domain)
                            asig = test_style_anchor_sigma.get(domain)
                            if amu is None or asig is None:
                                z_hard = z_norm_live * sigma_hard + mu_hard
                            else:
                                mu_mix = (1.0 - b) * mu_orig + b * amu
                                sigma_mix = (1.0 - b) * sigma_orig + b * asig
                                z_hard = z_norm_live * sigma_mix + mu_mix
                                with torch.no_grad():
                                    amu_b = amu.expand(Bsz, -1, -1, -1)
                                    asig_b = asig.expand(Bsz, -1, -1, -1)
                                    co = torch.cat(
                                        [mu_orig.reshape(Bsz, -1), sigma_orig.reshape(Bsz, -1)], dim=1
                                    )
                                    ch = torch.cat(
                                        [mu_hard.reshape(Bsz, -1), sigma_hard.reshape(Bsz, -1)], dim=1
                                    )
                                    ct = torch.cat(
                                        [amu_b.reshape(Bsz, -1), asig_b.reshape(Bsz, -1)], dim=1
                                    )
                                    l2_ot = float((co - ct).norm(dim=1).mean().item())
                                    l2_ht = float((ch - ct).norm(dim=1).mean().item())
                                    l2_oh = float((co - ch).norm(dim=1).mean().item())
                                    cos_ot = float(F.cosine_similarity(co, ct, dim=1, eps=1e-12).mean().item())
                                    cos_ht = float(F.cosine_similarity(ch, ct, dim=1, eps=1e-12).mean().item())
                                    cos_oh = float(F.cosine_similarity(co, ch, dim=1, eps=1e-12).mean().item())
                                style_triplet_l2_orig_pool_meters[domain].update(l2_ot, data.size(0))
                                style_triplet_l2_hard_pool_meters[domain].update(l2_ht, data.size(0))
                                style_triplet_l2_orig_hard_meters[domain].update(l2_oh, data.size(0))
                                style_triplet_cos_orig_pool_meters[domain].update(cos_ot, data.size(0))
                                style_triplet_cos_hard_pool_meters[domain].update(cos_ht, data.size(0))
                                style_triplet_cos_orig_hard_meters[domain].update(cos_oh, data.size(0))
                        else:
                            idx = torch.randint(
                                0,
                                pmu.size(0),
                                (Bsz,),
                                device=z_style.device,
                                dtype=torch.long,
                            )
                            mu_t = pmu[idx]
                            sigma_t = psig[idx]
                            mu_mix = (1.0 - b) * mu_orig + b * mu_t
                            sigma_mix = (1.0 - b) * sigma_orig + b * sigma_t
                            z_hard = z_norm_live * sigma_mix + mu_mix
                            with torch.no_grad():
                                co = torch.cat(
                                    [mu_orig.reshape(Bsz, -1), sigma_orig.reshape(Bsz, -1)], dim=1
                                )
                                ch = torch.cat(
                                    [mu_hard.reshape(Bsz, -1), sigma_hard.reshape(Bsz, -1)], dim=1
                                )
                                ct = torch.cat(
                                    [mu_t.reshape(Bsz, -1), sigma_t.reshape(Bsz, -1)], dim=1
                                )
                                l2_ot = float((co - ct).norm(dim=1).mean().item())
                                l2_ht = float((ch - ct).norm(dim=1).mean().item())
                                l2_oh = float((co - ch).norm(dim=1).mean().item())
                                cos_ot = float(F.cosine_similarity(co, ct, dim=1, eps=1e-12).mean().item())
                                cos_ht = float(F.cosine_similarity(ch, ct, dim=1, eps=1e-12).mean().item())
                                cos_oh = float(F.cosine_similarity(co, ch, dim=1, eps=1e-12).mean().item())
                            style_triplet_l2_orig_pool_meters[domain].update(l2_ot, data.size(0))
                            style_triplet_l2_hard_pool_meters[domain].update(l2_ht, data.size(0))
                            style_triplet_l2_orig_hard_meters[domain].update(l2_oh, data.size(0))
                            style_triplet_cos_orig_pool_meters[domain].update(cos_ot, data.size(0))
                            style_triplet_cos_hard_pool_meters[domain].update(cos_ht, data.size(0))
                            style_triplet_cos_orig_hard_meters[domain].update(cos_oh, data.size(0))
                    else:
                        z_hard = z_norm_live * sigma_hard + mu_hard
                    if not torch.isfinite(z_hard).all():
                        raise FloatingPointError("Non-finite z_hard detected before outer loop.")
                    # z_style分支保留 train() 让 layer4 BN running stats 更新一次；
                    # z_hard 分支暂时把 layer4 设为 eval()，避免 layer4 BN running stats 再更新一次，
                    # 但不影响 autograd 梯度回传。
                    backbone = getattr(model, "backbone", None)
                    prev_layer4_training = None
                    if backbone is not None and hasattr(backbone, "layer4"):
                        prev_layer4_training = backbone.layer4.training
                        backbone.layer4.eval()
                    try:
                        logits_hard, _vec_hard = model.forward_from_layer3(z_hard)
                        # Monitor z_hard OOD loss with diffusion in eval mode (no state update).
                        with torch.no_grad():
                            prev_diff_training_for_monitor = diffusion_model.training
                            diffusion_model.eval()
                            try:
                                vec_hard_norm = diffusion_model.normalize(_vec_hard.detach())
                                hard_ood_loss = diffusion_model.get_loss_at_timestep(vec_hard_norm, adv_ood_timestep)
                            finally:
                                diffusion_model.train(prev_diff_training_for_monitor)
                        # Monitoring-only reference: run z_clean through layer4+pool once when available.
                        if (hard_adv_monitor_sparsity or hard_adv_monitor_direction) and z_clean is not None:
                            with torch.no_grad():
                                _logits_clean_l4, _vec_clean_l4 = model.forward_from_layer3(z_clean.detach())
                    finally:
                        if backbone is not None and hasattr(backbone, "layer4") and prev_layer4_training is not None:
                            backbone.layer4.train(prev_layer4_training)

                    # 監控：style 分支 pooled 向量(vec_style) vs z_hard 的 pooled 向量(_vec_hard)
                    # 代表 hard 相對於原本 z_style 在 avgpool 後表徵上的改動量。
                    with torch.no_grad():
                        vec_delta = (_vec_hard - vec_style.detach()).norm().item()
                    vec_hard_vs_style_delta_meters[domain].update(vec_delta, data.size(0))
                    if hard_adv_monitor_sparsity:
                        with torch.no_grad():
                            z_hard_sparsity = (z_hard == 0).float().mean().item()
                            vec_hard_sparsity = (_vec_hard == 0).float().mean().item()
                            if z_clean is not None:
                                vec_clean_l4_sparsity = (_vec_clean_l4 == 0).float().mean().item()
                                vec_hard_vs_clean_l4_delta = (_vec_hard - _vec_clean_l4).norm().item()
                        z_hard_sparsity_meters[domain].update(z_hard_sparsity, data.size(0))
                        vec_hard_sparsity_meters[domain].update(vec_hard_sparsity, data.size(0))
                        if z_clean is not None:
                            vec_clean_l4_sparsity_meters[domain].update(vec_clean_l4_sparsity, data.size(0))
                            vec_hard_vs_clean_l4_delta_meters[domain].update(vec_hard_vs_clean_l4_delta, data.size(0))
                    if hard_adv_monitor_direction and z_clean is not None:
                        with torch.no_grad():
                            d_style = vec_style.detach() - _vec_clean_l4.detach()
                            d_hard = _vec_hard.detach() - _vec_clean_l4.detach()
                            d_style_norm = d_style.norm(dim=1)
                            d_hard_norm = d_hard.norm(dim=1)
                            dir_cos_vec = F.cosine_similarity(d_style, d_hard, dim=1, eps=1e-12)
                            dir_cos = dir_cos_vec.mean().item()
                            dir_cos_low03 = float((dir_cos_vec < 0.3).float().mean().item())
                            vec_style_from_clean_norm = d_style_norm.mean().item()
                            vec_hard_from_clean_norm = d_hard_norm.mean().item()
                        vec_style_hard_dir_cos_meters[domain].update(float(dir_cos), data.size(0))
                        vec_style_hard_dir_cos_low03_meters[domain].update(float(dir_cos_low03), data.size(0))
                        vec_style_from_clean_norm_meters[domain].update(float(vec_style_from_clean_norm), data.size(0))
                        vec_hard_from_clean_norm_meters[domain].update(float(vec_hard_from_clean_norm), data.size(0))
                        vec_style_hard_dir_cos_samples[domain].append(dir_cos_vec.detach().cpu())

                    loss_style_ce = criterion(logits_style, target)
                    hard_ce_vec = F.cross_entropy(logits_hard, target, reduction='none')
                    loss_hard_ce = hard_ce_vec.mean()
                    with torch.no_grad():
                        hard_ce_delta_vec = hard_ce_vec - L_cls_inner_vec.detach()
                        hard_ce_delta = float(hard_ce_delta_vec.mean().item())
                        hard_ce_harder_ratio = float((hard_ce_delta_vec > 0).float().mean().item())
                        probs_style = F.softmax(logits_style, dim=1)
                        probs_hard = F.softmax(logits_hard, dim=1)
                        style_conf = float(probs_style.max(dim=1).values.mean().item())
                        hard_conf = float(probs_hard.max(dim=1).values.mean().item())

                        style_target_logit = logits_style.gather(1, target.view(-1, 1)).squeeze(1)
                        hard_target_logit = logits_hard.gather(1, target.view(-1, 1)).squeeze(1)
                        style_max_logit = logits_style.max(dim=1).values
                        hard_max_logit = logits_hard.max(dim=1).values
                        style_target_max_gap = float((style_max_logit - style_target_logit).mean().item())
                        hard_target_max_gap = float((hard_max_logit - hard_target_logit).mean().item())

                        style_pred = logits_style.argmax(dim=1)
                        hard_pred = logits_hard.argmax(dim=1)
                        style_correct = (style_pred == target)
                        hard_wrong = (hard_pred != target)
                        flip_mask = style_correct & hard_wrong
                        flip_all_ratio = float(flip_mask.float().mean().item())
                        style_correct_count = int(style_correct.sum().item())
                        if style_correct_count > 0:
                            flip_on_style_correct_ratio = float(
                                flip_mask[style_correct].float().mean().item()
                            )
                        else:
                            flip_on_style_correct_ratio = 0.0

                    hard_ce_delta_samples[domain].append(hard_ce_delta_vec.detach().cpu())

                    # Gradient conflict monitor: compare grads from style/hard CE over selected groups.
                    backbone_mod = getattr(model, "backbone", None)
                    if backbone_mod is not None and hasattr(backbone_mod, "layer3") and hasattr(backbone_mod, "layer4") and hasattr(backbone_mod, "fc"):
                        full_backbone_params = [p for p in backbone_mod.parameters() if p.requires_grad]
                        if len(full_backbone_params) > 0:
                            grad_style_all = torch.autograd.grad(
                                loss_style_ce,
                                full_backbone_params,
                                retain_graph=True,
                                allow_unused=True,
                            )
                            grad_hard_all = torch.autograd.grad(
                                loss_hard_ce,
                                full_backbone_params,
                                retain_graph=True,
                                allow_unused=True,
                            )
                            grad_style_map = {id(p): g for p, g in zip(full_backbone_params, grad_style_all)}
                            grad_hard_map = {id(p): g for p, g in zip(full_backbone_params, grad_hard_all)}

                            def _norm_and_cos(subset_params):
                                style_chunks, hard_chunks = [], []
                                for _p in subset_params:
                                    gs = grad_style_map.get(id(_p))
                                    gh = grad_hard_map.get(id(_p))
                                    if gs is None or gh is None:
                                        continue
                                    style_chunks.append(gs.reshape(-1))
                                    hard_chunks.append(gh.reshape(-1))
                                if len(style_chunks) == 0:
                                    return 0.0, 0.0, 0.0
                                gs_vec = torch.cat(style_chunks, dim=0)
                                gh_vec = torch.cat(hard_chunks, dim=0)
                                gs_norm = float(gs_vec.norm().item())
                                gh_norm = float(gh_vec.norm().item())
                                if gs_norm <= 0.0 or gh_norm <= 0.0:
                                    return gs_norm, gh_norm, 0.0
                                cos = float(
                                    F.cosine_similarity(
                                        gs_vec.unsqueeze(0),
                                        gh_vec.unsqueeze(0),
                                        dim=1,
                                        eps=1e-12,
                                    ).item()
                                )
                                return gs_norm, gh_norm, cos

                            layer3_params = [p for p in backbone_mod.layer3.parameters() if p.requires_grad]
                            layer4_params = [p for p in backbone_mod.layer4.parameters() if p.requires_grad]
                            head_params = [p for p in backbone_mod.fc.parameters() if p.requires_grad]

                            s3, h3, c3 = _norm_and_cos(layer3_params)
                            s4, h4, c4 = _norm_and_cos(layer4_params)
                            sh, hh, ch = _norm_and_cos(head_params)
                            sb, hb, cb = _norm_and_cos(full_backbone_params)

                            grad_style_norm_layer3_meters[domain].update(s3, data.size(0))
                            grad_hard_norm_layer3_meters[domain].update(h3, data.size(0))
                            grad_style_hard_cos_layer3_meters[domain].update(c3, data.size(0))
                            grad_style_norm_layer4_meters[domain].update(s4, data.size(0))
                            grad_hard_norm_layer4_meters[domain].update(h4, data.size(0))
                            grad_style_hard_cos_layer4_meters[domain].update(c4, data.size(0))
                            grad_style_norm_head_meters[domain].update(sh, data.size(0))
                            grad_hard_norm_head_meters[domain].update(hh, data.size(0))
                            grad_style_hard_cos_head_meters[domain].update(ch, data.size(0))
                            grad_style_norm_backbone_meters[domain].update(sb, data.size(0))
                            grad_hard_norm_backbone_meters[domain].update(hb, data.size(0))
                            grad_style_hard_cos_backbone_meters[domain].update(cb, data.size(0))

                    # 線性 warmup：從 epoch 0 到 60，z_hard 權重從 0 → 0.5
                    # 之後維持 0.5；z_style 權重始終為 1 - w_hard。
                    warmup_epochs = 60.0
                    # 以當前 iteration 推出「目前是第幾個 epoch」（浮點數）
                    current_epoch = (k + 1) / float(STEPS_PER_EPOCH) if STEPS_PER_EPOCH > 0 else 0.0
                    hard_weight = 0.2 * min(max(current_epoch / warmup_epochs, 0.0), 1.0)
                    style_weight = 1.0 - hard_weight

                    if adv_equal_ce_weight:
                        loss = 0.5 * (loss_style_ce + loss_hard_ce)
                    else:
                        loss = style_weight * loss_style_ce + hard_weight * loss_hard_ce

                    # Record adv meters (batch-level; treat scalar as per-sample weight).
                    task_loss_meters[domain].update(float(loss.item()), data.size(0))
                    style_ce_meters[domain].update(float(loss_style_ce.item()), data.size(0))
                    hard_ce_meters[domain].update(float(loss_hard_ce.item()), data.size(0))
                    hard_ce_delta_meters[domain].update(hard_ce_delta, data.size(0))
                    hard_ce_harder_ratio_meters[domain].update(hard_ce_harder_ratio, data.size(0))
                    hard_ood_loss_meters[domain].update(float(hard_ood_loss.item()), data.size(0))
                    hard_style_conf_meters[domain].update(style_conf, data.size(0))
                    hard_hard_conf_meters[domain].update(hard_conf, data.size(0))
                    hard_style_target_max_gap_meters[domain].update(style_target_max_gap, data.size(0))
                    hard_hard_target_max_gap_meters[domain].update(hard_target_max_gap, data.size(0))
                    hard_flip_all_ratio_meters[domain].update(flip_all_ratio, data.size(0))
                    hard_flip_on_style_correct_ratio_meters[domain].update(
                        flip_on_style_correct_ratio, data.size(0)
                    )
                    hard_mu_hit_lower_ratio_meters[domain].update(mu_hit_lower_ratio, data.size(0))
                    hard_mu_hit_upper_ratio_meters[domain].update(mu_hit_upper_ratio, data.size(0))
                    hard_sigma_hit_lower_ratio_meters[domain].update(sigma_hit_lower_ratio, data.size(0))
                    hard_sigma_hit_upper_ratio_meters[domain].update(sigma_hit_upper_ratio, data.size(0))
                    adv_ood_loss_meters[domain].update(float(L_ood.item()), data.size(0))
                    adv_cls_inner_meters[domain].update(float(L_cls_inner.item()), data.size(0))
                else:
                    # No adv: just use z_style branch.
                    loss = criterion(logits_style, target)
                    task_loss_meters[domain].update(float(loss.item()), data.size(0))

                if should_save_spatial_dump:
                    z_clean_to_save = clean_layer3_cache.get(domain)
                    if z_clean_to_save is None:
                        if z_clean is None:
                            print(
                                f"[SpatialDump][WARN] Cache miss for z_clean at epoch={epoch_i_1based}, "
                                f"iter={k+1}, domain={domain}; z_clean is unavailable under hard_adv_z_base={hard_adv_z_base}, saving z_clean=None."
                            )
                        else:
                            print(
                                f"[SpatialDump][WARN] Cache miss for z_clean at epoch={epoch_i_1based}, "
                                f"iter={k+1}, domain={domain}; using current-graph z_clean.detach() for dump."
                            )
                            z_clean_to_save = z_clean.detach()

                    save_spatial_feature_dump(
                        save_root=spatial_dump_dir,
                        epoch=epoch_i_1based,
                        step=k + 1,
                        domain=domain,
                        data=data,
                        target=target,
                        meta=batch_meta,
                        z_clean=z_clean_to_save,
                        z_style=z_style,
                        z_hard=z_hard,
                        style_aug_activated=style_aug_activated,
                        max_samples=spatial_dump_max_samples,
                    )
                    spatial_dump_saved_epochs.add(epoch_i_1based)

                output = logits_style
            else:
                # Original training path (logits only)
                # Isolate BN updates for clean branch:
                # use eval() so BN running stats are not updated, while keeping autograd enabled.
                backbone_for_clean = getattr(model, "backbone", None)
                prev_backbone_training_for_clean = None
                prev_model_training_for_clean = model.training
                if backbone_for_clean is not None:
                    prev_backbone_training_for_clean = backbone_for_clean.training
                    backbone_for_clean.eval()
                else:
                    model.eval()
                try:
                    logits_clean = model(
                        data,
                        return_blocks=False,
                        communicator=None,
                        debug_style_shift=False,
                        iter_num=k+1,
                        rank=domain_names.index(domain),
                    )
                finally:
                    if backbone_for_clean is not None and prev_backbone_training_for_clean is not None:
                        backbone_for_clean.train(prev_backbone_training_for_clean)
                    else:
                        model.train(prev_model_training_for_clean)
                loss_clean_ce = criterion(logits_clean, target)
                clean_ce_meters[domain].update(float(loss_clean_ce.item()), data.size(0))

                if (use_style_stats or use_style_shift) and args.model == "res":
                    communicator.set_active_domain(domain)
                    output = model(
                        data,
                        return_blocks=False,
                        communicator=communicator,
                        debug_style_shift=debug_style_shift,
                        iter_num=k+1,
                        rank=domain_names.index(domain),
                    )
                else:
                    output = model(data)

                style_aug_activated = bool(getattr(model, "last_style_aug_activated", False))
                style_aug_flag_meters[domain].update(1.0 if style_aug_activated else 0.0, data.size(0))

                loss = criterion(output, target)
                if hard_adv_clean_loss_weight > 0.0:
                    loss = loss + hard_adv_clean_loss_weight * loss_clean_ce

                with torch.no_grad():
                    probs_clean = F.softmax(logits_clean, dim=1)
                    probs_style = F.softmax(output, dim=1)
                    clean_conf = probs_clean.max(dim=1).values.mean().item()
                    style_conf = probs_style.max(dim=1).values.mean().item()
                    clean_conf_meters[domain].update(clean_conf, data.size(0))
                    style_conf_meters[domain].update(style_conf, data.size(0))

                    # Cosine between per-sample CE vectors to detect opposite trend.
                    loss_clean_vec = F.cross_entropy(logits_clean, target, reduction='none')
                    loss_style_vec = F.cross_entropy(output, target, reduction='none')
                    ce_cos = F.cosine_similarity(
                        loss_clean_vec.unsqueeze(0),
                        loss_style_vec.unsqueeze(0),
                        dim=1,
                        eps=1e-8,
                    ).item()
                    clean_style_ce_cos_meters[domain].update(float(ce_cos), data.size(0))

                    # Symmetric KL between clean/style prediction distributions.
                    eps_kl = 1e-8
                    kl_clean_to_style = F.kl_div(
                        torch.log(probs_clean.clamp_min(eps_kl)),
                        probs_style,
                        reduction='batchmean',
                    )
                    kl_style_to_clean = F.kl_div(
                        torch.log(probs_style.clamp_min(eps_kl)),
                        probs_clean,
                        reduction='batchmean',
                    )
                    sym_kl = 0.5 * (kl_clean_to_style + kl_style_to_clean)
                    clean_style_sym_kl_meters[domain].update(float(sym_kl.item()), data.size(0))
            
            # Compute diffusion loss if OOD detection is enabled
            if getattr(args, 'use_ood', False) and domain in diffusion_models_dict:
                diffusion_model = diffusion_models_dict[domain]
                optimizer_diffusion = optimizers_diffusion_dict[domain]
                
                # Extract intermediate features for diffusion model.
                # Run this extra forward in eval + no_grad so BatchNorm running stats are NOT
                # updated a second time (the classification forward above already updated them).
                was_training = model.training
                # IMPORTANT: only switch the backbone to eval() for feature extraction,
                # so BN uses running stats (matching test-time backbone.eval()), while keeping
                # diffusion_model in train() to keep FeatureNormalization buffers updating.
                backbone = getattr(model, "backbone", None)
                if backbone is not None:
                    backbone.eval()
                else:
                    model.eval()
                with torch.no_grad():
                    latents = model.intermediate_forward(data)
                if was_training:
                    if backbone is not None:
                        backbone.train()
                    else:
                        model.train()
                
                # Normalize features and compute diffusion loss
                # Detach latents to avoid affecting backbone gradients
                latents_for_diff = latents.detach().requires_grad_(True)
                diffusion_model.train()
                latents_normalized = diffusion_model.normalize(latents_for_diff)
                
                loss_diff = diffusion_model.get_loss_iter(latents_normalized)
                
                # Backward pass for diffusion model
                optimizer_diffusion.zero_grad()
                loss_diff.backward()
                optimizer_diffusion.step()
                
                # Record diffusion loss for averaging
                if loss_diff_meters is not None:
                    loss_diff_meters[domain].update(loss_diff.item(), data.size(0))

            # record training loss and accuracy
            record_start = time.time()
            acc1 = util.comp_accuracy(output, target)
            losses_dict[domain].update(loss.item(), data.size(0))
            top1_dict[domain].update(acc1[0], data.size(0))
            record_end = time.time()

            # backward pass for classification
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update cosine annealing scheduler (purely step-based)
            schedulers_dict[domain].step()
            
            # ====== Save updated BatchNorm state for this domain ======
            # After training, save the updated BatchNorm statistics to maintain domain isolation.
            # This prevents BatchNorm statistics from one domain from contaminating others.
            for name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    if name not in bn_states_dict[domain]:
                        bn_states_dict[domain][name] = {}
                    bn_states_dict[domain][name]['running_mean'] = module.running_mean.clone()
                    bn_states_dict[domain][name]['running_var'] = module.running_var.clone()
                    if hasattr(module, 'num_batches_tracked'):
                        bn_states_dict[domain][name]['num_batches_tracked'] = module.num_batches_tracked.clone()
        
        # ========== 第三阶段：交换训练后的模型参数 ==========
        # Snapshot model parameters before consensus aggregation for drift monitoring.
        pre_comm_snapshots = {}
        for domain in domain_names:
            backbone = getattr(models_dict[domain], "backbone", None)
            if backbone is None:
                continue
            pre_comm_snapshots[domain] = {
                "layer4": [p.detach().clone() for p in backbone.layer4.parameters() if p.requires_grad],
                "head": [p.detach().clone() for p in backbone.fc.parameters() if p.requires_grad],
                "backbone": [p.detach().clone() for p in backbone.parameters() if p.requires_grad],
            }

        def _flatten_param_list(params):
            if len(params) == 0:
                return None
            return torch.cat([p.reshape(-1) for p in params], dim=0)

        pre_head_vecs = {}
        pre_backbone_vecs = {}
        for domain in domain_names:
            if domain not in pre_comm_snapshots:
                continue
            v_head = _flatten_param_list(pre_comm_snapshots[domain]["head"])
            v_backbone = _flatten_param_list(pre_comm_snapshots[domain]["backbone"])
            if v_head is not None:
                pre_head_vecs[domain] = v_head
            if v_backbone is not None:
                pre_backbone_vecs[domain] = v_backbone

        def _pairwise_mean(vec_dict):
            keys = list(vec_dict.keys())
            if len(keys) < 2:
                return 0.0
            vals = []
            for i in range(len(keys)):
                vi = vec_dict[keys[i]]
                for j in range(i + 1, len(keys)):
                    vj = vec_dict[keys[j]]
                    vals.append(float((vi - vj).norm().item()))
            if len(vals) == 0:
                return 0.0
            return float(sum(vals) / len(vals))

        pairwise_head_pre = _pairwise_mean(pre_head_vecs)
        pairwise_backbone_pre = _pairwise_mean(pre_backbone_vecs)

        # Exchange updated model parameters after training step
        # Note: style_vecs_dict is None here since we only exchange model parameters (not style stats)
        d_comm_time_after = communicator.communicate(models_dict, style_vecs_dict=None)
        comm_time += d_comm_time_after

        # Compute per-domain self drift and pairwise post-aggregation drift.
        post_head_vecs = {}
        post_backbone_vecs = {}
        for domain in domain_names:
            if domain not in pre_comm_snapshots:
                continue
            backbone = getattr(models_dict[domain], "backbone", None)
            if backbone is None:
                continue
            post_layer4 = [p.detach() for p in backbone.layer4.parameters() if p.requires_grad]
            post_head = [p.detach() for p in backbone.fc.parameters() if p.requires_grad]
            post_backbone = [p.detach() for p in backbone.parameters() if p.requires_grad]

            def _drift_stats(pre_list, post_list):
                if len(pre_list) == 0 or len(post_list) == 0:
                    return 0.0, 0.0
                delta_sq = 0.0
                base_sq = 0.0
                for p0, p1 in zip(pre_list, post_list):
                    d = (p1 - p0).reshape(-1)
                    b = p0.reshape(-1)
                    delta_sq += float(torch.dot(d, d).item())
                    base_sq += float(torch.dot(b, b).item())
                l2 = (delta_sq + 1e-24) ** 0.5
                rel = l2 / ((base_sq + 1e-24) ** 0.5 + 1e-12)
                return l2, rel

            l2_l4, rel_l4 = _drift_stats(pre_comm_snapshots[domain]["layer4"], post_layer4)
            l2_head, rel_head = _drift_stats(pre_comm_snapshots[domain]["head"], post_head)
            l2_backbone, rel_backbone = _drift_stats(pre_comm_snapshots[domain]["backbone"], post_backbone)
            drift_l2_layer4_meters[domain].update(l2_l4, 1)
            drift_rel_layer4_meters[domain].update(rel_l4, 1)
            drift_l2_head_meters[domain].update(l2_head, 1)
            drift_rel_head_meters[domain].update(rel_head, 1)
            drift_l2_backbone_meters[domain].update(l2_backbone, 1)
            drift_rel_backbone_meters[domain].update(rel_backbone, 1)

            v_head_post = _flatten_param_list(post_head)
            v_backbone_post = _flatten_param_list(post_backbone)
            if v_head_post is not None:
                post_head_vecs[domain] = v_head_post
            if v_backbone_post is not None:
                post_backbone_vecs[domain] = v_backbone_post

        pairwise_head_post = _pairwise_mean(post_head_vecs)
        pairwise_backbone_post = _pairwise_mean(post_backbone_vecs)
        drift_pairwise_head_pre_meters.update(pairwise_head_pre, 1)
        drift_pairwise_head_post_meters.update(pairwise_head_post, 1)
        drift_pairwise_backbone_pre_meters.update(pairwise_backbone_pre, 1)
        drift_pairwise_backbone_post_meters.update(pairwise_backbone_post, 1)
        
        # 检查通信后的参数一致性（每epoch检查一次，只在不一致时打印）
        if (k + 1) % STEPS_PER_EPOCH == 0:
            first_params = {}
            for domain in domain_names:
                first_param = list(models_dict[domain].parameters())[0].view(-1)[0].item()
                first_params[domain] = first_param
            
            # 只在参数不一致时打印警告
            param_set = set([round(first_params[d], 4) for d in domain_names])
            if len(param_set) > 1:
                print(f"⚠ WARNING: Parameters INCONSISTENT after comm iter {k+1}: {param_set}")
        
        end_time = time.time()
        d_comp_time = (end_time - start_time - (record_end - record_start))
        comp_time += d_comp_time

        # 每隔一定次數清理 GPU 緩存，避免記憶體累積
        if (k + 1) % 20 == 0:
            torch.cuda.empty_cache()

        # Print progress (average across domains)
        avg_loss = sum(losses_dict[d].avg for d in domain_names) / len(domain_names)
        avg_acc = sum(top1_dict[d].avg for d in domain_names) / len(domain_names)
        avg_loss_val = avg_loss.item() if hasattr(avg_loss, 'item') else float(avg_loss)
        avg_acc_val = avg_acc.item() if hasattr(avg_acc, 'item') else float(avg_acc)
        print("iter: %d/%d, comp_time: %.3f, comm_time: %.3f, total time: %.3f, avg_loss: %.3f, avg_acc: %.3f"
              % (k+1, K, d_comp_time, d_comm_time_after, comp_time + comm_time, avg_loss_val, avg_acc_val), end='\r')

        # measure and log after each pseudo-epoch
        if (k + 1) % STEPS_PER_EPOCH == 0:
            epoch = (k + 1) // STEPS_PER_EPOCH
            toc = time.time()
            record_time = toc - tic  # includes everything
            epoch_time = comp_time + comm_time  # only important parts

            # 在測試前清理 GPU 緩存，釋放未使用的記憶體
            torch.cuda.empty_cache()
            
            # evaluate test accuracy for each domain
            test_accs = {}
            test_confidences = {}
            for domain in domain_names:
                _, test_loader = domain_loaders[domain]
                test_acc = util.test(models_dict[domain], test_loader)
                test_accs[domain] = test_acc
                test_confidences[domain] = compute_loader_avg_confidence(models_dict[domain], test_loader)
                if args.dataset == "pacs" and domain == "sketch":
                    per_cls_acc, _cls_correct, _cls_total = compute_loader_per_class_accuracy(
                        models_dict[domain], test_loader, sketch_num_classes
                    )
                    for _c in range(sketch_num_classes):
                        sketch_class_acc_history[_c].append(float(per_cls_acc[_c]))
                    # Terminal summary: print per-class acc and top-3 volatile classes.
                    per_cls_str = ", ".join(
                        f"c{_c}={per_cls_acc[_c]:.2f}" for _c in range(sketch_num_classes)
                    )
                    print(
                        f"[sketch_photo_per_class] epoch={epoch} acc: {per_cls_str}"
                    )
                    cls_stats = []
                    for _c in range(sketch_num_classes):
                        hist = sketch_class_acc_history[_c]
                        cur = hist[-1]
                        prev = hist[-2] if len(hist) >= 2 else cur
                        delta = abs(cur - prev)
                        recent = hist[-min(len(hist), sketch_vol_window):]
                        recent_t = torch.tensor(recent, dtype=torch.float32)
                        vol_std = float(recent_t.std(unbiased=False).item()) if len(recent) > 1 else 0.0
                        cls_stats.append((_c, cur, delta, vol_std))
                    top_delta = sorted(cls_stats, key=lambda x: x[2], reverse=True)[:3]
                    top_std = sorted(cls_stats, key=lambda x: x[3], reverse=True)[:3]
                    top_delta_str = ", ".join(
                        f"c{_c}(d={_d:.2f})" for _c, _cur, _d, _s in top_delta
                    )
                    top_std_str = ", ".join(
                        f"c{_c}(std={_s:.2f})" for _c, _cur, _d, _s in top_std
                    )
                    print(
                        f"[sketch_photo_class_volatility] epoch={epoch} "
                        f"top_delta: {top_delta_str} | top_std(w={sketch_vol_window}): {top_std_str}"
                    )
            
            # 測試後再次清理緩存，確保記憶體被釋放
            torch.cuda.empty_cache()

            # Log results for each domain
            # Convert test_accs to float values before averaging
            test_accs_float = {d: (acc.item() if hasattr(acc, 'item') else float(acc)) for d, acc in test_accs.items()}
            avg_test_acc = sum(test_accs_float.values()) / len(test_accs_float)
            avg_train_loss = sum(losses_dict[d].avg for d in domain_names) / len(domain_names)
            avg_train_acc = sum(top1_dict[d].avg for d in domain_names) / len(domain_names)
            
            for domain in domain_names:
                recorder = recorders[domain]
                recorder.add_new(record_time, comp_time, comm_time, epoch_time,
                               top1_dict[domain].avg, losses_dict[domain].avg, test_accs[domain])
                if getattr(args, "debug_per_node_metrics", False) and args.dataset in ['pacs', 'vlcs'] and domain in node_to_domain:
                    print(f"  {domain} (source_domain={node_to_domain[domain]}): test_acc={float(test_accs[domain]):.2f}")
            
            # 简化的epoch总结打印（先换行清除\r的效果）
            print()  # 换行，清除之前使用\r的进度条
            avg_train_loss_val = avg_train_loss.item() if hasattr(avg_train_loss, 'item') else float(avg_train_loss)
            avg_train_acc_val = avg_train_acc.item() if hasattr(avg_train_acc, 'item') else float(avg_train_acc)
            avg_test_acc_val = avg_test_acc.item() if hasattr(avg_test_acc, 'item') else float(avg_test_acc)
            print(f"Epoch {epoch}: avg_loss={avg_train_loss_val:.3f}, avg_train_acc={avg_train_acc_val:.2f}%, avg_test_acc={avg_test_acc_val:.2f}%")
            if not getattr(args, "use_hard_style_adv", False):
                for _d in domain_names:
                    print(
                        f"[confidence] epoch={epoch} domain={_d} "
                        f"clean_train={float(clean_conf_meters[_d].avg):.4f} "
                        f"style_train={float(style_conf_meters[_d].avg):.4f} "
                        f"test={float(test_confidences[_d]):.4f} "
                        f"ce_cos={float(clean_style_ce_cos_meters[_d].avg):.4f} "
                        f"sym_kl={float(clean_style_sym_kl_meters[_d].avg):.6f}"
                    )
            if getattr(args, "use_hard_style_adv", False):
                for _d in domain_names:
                    print(
                        f"[hard_cls] epoch={epoch} domain={_d} "
                        f"style_conf={float(hard_style_conf_meters[_d].avg):.4f} "
                        f"hard_conf={float(hard_hard_conf_meters[_d].avg):.4f} "
                        f"style_gap(max-target)={float(hard_style_target_max_gap_meters[_d].avg):.6f} "
                        f"hard_gap(max-target)={float(hard_hard_target_max_gap_meters[_d].avg):.6f} "
                        f"flip_all={float(hard_flip_all_ratio_meters[_d].avg):.4f} "
                        f"flip_on_style_correct={float(hard_flip_on_style_correct_ratio_meters[_d].avg):.4f}"
                    )
                    mean_delta = float(hard_ce_delta_meters[_d].avg)
                    if len(hard_ce_delta_samples[_d]) == 0:
                        print(
                            f"[hard_ce_delta_dist] epoch={epoch} domain={_d} "
                            f"mean={mean_delta:.6f} no_samples"
                        )
                    else:
                        delta_all = torch.cat(hard_ce_delta_samples[_d], dim=0)
                        qs = torch.quantile(
                            delta_all,
                            torch.tensor([0.25, 0.50, 0.75, 0.90], dtype=delta_all.dtype),
                        )
                        p25, p50, p75, p90 = float(qs[0].item()), float(qs[1].item()), float(qs[2].item()), float(qs[3].item())
                        std = float(delta_all.std(unbiased=False).item())
                        print(
                            f"[hard_ce_delta_dist] epoch={epoch} domain={_d} "
                            f"mean={mean_delta:.6f} std={std:.6f} p25={p25:.6f} p50={p50:.6f} p75={p75:.6f} p90={p90:.6f}"
                        )
            if getattr(args, "use_hard_style_adv", False) and bool(getattr(args, "hard_adv_monitor_sparsity", False)):
                for _d in domain_names:
                    print(
                        f"[sparsity] epoch={epoch} domain={_d} "
                        f"z_hard={float(z_hard_sparsity_meters[_d].avg):.6f} "
                        f"vec_hard={float(vec_hard_sparsity_meters[_d].avg):.6f} "
                        f"vec_clean_l4={float(vec_clean_l4_sparsity_meters[_d].avg):.6f} "
                        f"vec_hard_vs_clean_l4_l2={float(vec_hard_vs_clean_l4_delta_meters[_d].avg):.6f}"
                    )
            if getattr(args, "use_hard_style_adv", False) and bool(getattr(args, "hard_adv_monitor_direction", False)):
                for _d in domain_names:
                    if len(vec_style_hard_dir_cos_samples[_d]) > 0:
                        cos_all = torch.cat(vec_style_hard_dir_cos_samples[_d], dim=0)
                        qs = torch.quantile(
                            cos_all,
                            torch.tensor([0.10, 0.50, 0.90], dtype=cos_all.dtype),
                        )
                        p10, p50, p90 = float(qs[0].item()), float(qs[1].item()), float(qs[2].item())
                    else:
                        p10, p50, p90 = 0.0, 0.0, 0.0
                    print(
                        f"[style_hard_direction] epoch={epoch} domain={_d} "
                        f"cos={float(vec_style_hard_dir_cos_meters[_d].avg):.6f} "
                        f"p10={p10:.6f} p50={p50:.6f} p90={p90:.6f} "
                        f"frac_lt_0.3={float(vec_style_hard_dir_cos_low03_meters[_d].avg):.6f} "
                        f"|style-clean|={float(vec_style_from_clean_norm_meters[_d].avg):.6f} "
                        f"|hard-clean|={float(vec_hard_from_clean_norm_meters[_d].avg):.6f} "
                        f"|hard-style|={float(vec_hard_vs_style_delta_meters[_d].avg):.6f} "
                        f"orth_removed={float(orth_removed_ratio_meters[_d].avg):.6f} "
                        f"orth_fallback={float(orth_fallback_frac_meters[_d].avg):.6f}"
                    )
            if getattr(args, "hard_style_test_pool", False) and getattr(args, "use_hard_style_adv", False):
                _bprint = max(0.0, min(1.0, float(getattr(args, "hard_style_test_anchor_beta", 0.1))))
                _mprint = str(getattr(args, "hard_style_test_pool_mode", "random")).lower()
                _ref = "anchor" if _mprint == "mean" else "draw"
                for _d in domain_names:
                    print(
                        f"[style_mu_sigma_triplet] epoch={epoch} domain={_d} mode={_mprint} beta={_bprint:.4f} "
                        f"L2(orig,{_ref})={float(style_triplet_l2_orig_pool_meters[_d].avg):.6f} "
                        f"L2(hard,{_ref})={float(style_triplet_l2_hard_pool_meters[_d].avg):.6f} "
                        f"L2(orig,hard)={float(style_triplet_l2_orig_hard_meters[_d].avg):.6f} "
                        f"cos(orig,{_ref})={float(style_triplet_cos_orig_pool_meters[_d].avg):.6f} "
                        f"cos(hard,{_ref})={float(style_triplet_cos_hard_pool_meters[_d].avg):.6f} "
                        f"cos(orig,hard)={float(style_triplet_cos_orig_hard_meters[_d].avg):.6f}"
                    )

            # Epoch-mean |grad_mu_cls| per channel (only from batches where log_grad_parts ran this epoch).
            grad_mu_cls_ch_avg_epoch = {}
            for _d in domain_names:
                if grad_mu_cls_abs_ch_count[_d] > 0 and grad_mu_cls_abs_ch_sum[_d] is not None:
                    grad_mu_cls_ch_avg_epoch[_d] = (
                        grad_mu_cls_abs_ch_sum[_d] / float(grad_mu_cls_abs_ch_count[_d])
                    ).detach().cpu()
                grad_mu_cls_abs_ch_sum[_d] = None
                grad_mu_cls_abs_ch_count[_d] = 0
            
            # log to wandb (average across domains)
            log_dict = {
                "epoch": epoch,
                "iter": k,
                "loss": avg_train_loss.item() if hasattr(avg_train_loss, 'item') else float(avg_train_loss),
                "train_acc": avg_train_acc.item() if hasattr(avg_train_acc, 'item') else float(avg_train_acc),
                "test_acc": avg_test_acc.item() if hasattr(avg_test_acc, 'item') else float(avg_test_acc),
                "lr": optimizers_dict[domain_names[0]].param_groups[0]['lr'],
                "comp_time": comp_time,
                "comm_time": comm_time,
                "epoch_time": epoch_time,
            }
            
            # Add per-domain metrics
            for domain in domain_names:
                log_dict[f"{domain}/loss"] = losses_dict[domain].avg.item() if hasattr(losses_dict[domain].avg, 'item') else float(losses_dict[domain].avg)
                log_dict[f"{domain}/train_acc"] = top1_dict[domain].avg.item() if hasattr(top1_dict[domain].avg, 'item') else float(top1_dict[domain].avg)
                log_dict[f"{domain}/test_acc"] = test_accs[domain].item() if hasattr(test_accs[domain], 'item') else float(test_accs[domain])
                # Hard-style-adv metrics (only meaningful when --use_hard_style_adv)
                log_dict[f"{domain}/task_loss"] = float(task_loss_meters[domain].avg)
                log_dict[f"{domain}/clean_ce"] = float(clean_ce_meters[domain].avg)
                log_dict[f"{domain}/style_ce"] = float(style_ce_meters[domain].avg)
                log_dict[f"{domain}/hard_ce"] = float(hard_ce_meters[domain].avg)
                log_dict[f"{domain}/hard_ce_delta"] = float(hard_ce_delta_meters[domain].avg)
                log_dict[f"{domain}/hard_ce_harder_ratio"] = float(hard_ce_harder_ratio_meters[domain].avg)
                log_dict[f"{domain}/adv_cls_inner"] = float(adv_cls_inner_meters[domain].avg)
                log_dict[f"{domain}/mu_orig_norm"] = float(mu_orig_norm_meters[domain].avg)
                log_dict[f"{domain}/sigma_orig_norm"] = float(sigma_orig_norm_meters[domain].avg)
                log_dict[f"{domain}/grad_mu_norm"] = float(grad_mu_norm_meters[domain].avg)
                log_dict[f"{domain}/grad_sigma_norm"] = float(grad_sigma_norm_meters[domain].avg)
                log_dict[f"{domain}/rel_mu_step"] = float(rel_mu_step_meters[domain].avg)
                log_dict[f"{domain}/rel_sigma_step"] = float(rel_sigma_step_meters[domain].avg)
                log_dict[f"{domain}/grad_style_norm/layer3"] = float(grad_style_norm_layer3_meters[domain].avg)
                log_dict[f"{domain}/grad_hard_norm/layer3"] = float(grad_hard_norm_layer3_meters[domain].avg)
                log_dict[f"{domain}/grad_style_hard_cos/layer3"] = float(grad_style_hard_cos_layer3_meters[domain].avg)
                log_dict[f"{domain}/grad_style_norm/layer4"] = float(grad_style_norm_layer4_meters[domain].avg)
                log_dict[f"{domain}/grad_hard_norm/layer4"] = float(grad_hard_norm_layer4_meters[domain].avg)
                log_dict[f"{domain}/grad_style_hard_cos/layer4"] = float(grad_style_hard_cos_layer4_meters[domain].avg)
                log_dict[f"{domain}/grad_style_norm/head"] = float(grad_style_norm_head_meters[domain].avg)
                log_dict[f"{domain}/grad_hard_norm/head"] = float(grad_hard_norm_head_meters[domain].avg)
                log_dict[f"{domain}/grad_style_hard_cos/head"] = float(grad_style_hard_cos_head_meters[domain].avg)
                log_dict[f"{domain}/grad_style_norm/full_backbone"] = float(grad_style_norm_backbone_meters[domain].avg)
                log_dict[f"{domain}/grad_hard_norm/full_backbone"] = float(grad_hard_norm_backbone_meters[domain].avg)
                log_dict[f"{domain}/grad_style_hard_cos/full_backbone"] = float(grad_style_hard_cos_backbone_meters[domain].avg)
                log_dict[f"{domain}/consensus_drift/layer4_l2"] = float(drift_l2_layer4_meters[domain].avg)
                log_dict[f"{domain}/consensus_drift/layer4_rel"] = float(drift_rel_layer4_meters[domain].avg)
                log_dict[f"{domain}/consensus_drift/head_l2"] = float(drift_l2_head_meters[domain].avg)
                log_dict[f"{domain}/consensus_drift/head_rel"] = float(drift_rel_head_meters[domain].avg)
                log_dict[f"{domain}/consensus_drift/full_backbone_l2"] = float(drift_l2_backbone_meters[domain].avg)
                log_dict[f"{domain}/consensus_drift/full_backbone_rel"] = float(drift_rel_backbone_meters[domain].avg)
                log_dict[f"{domain}/style_hard_direction/cos"] = float(vec_style_hard_dir_cos_meters[domain].avg)
                log_dict[f"{domain}/style_hard_direction/style_clean_norm"] = float(vec_style_from_clean_norm_meters[domain].avg)
                log_dict[f"{domain}/style_hard_direction/hard_clean_norm"] = float(vec_hard_from_clean_norm_meters[domain].avg)
                log_grad_parts_every = int(getattr(args, "hard_adv_log_grad_parts_every", 10))
                do_log_grad_parts = bool(getattr(args, "hard_adv_log_grad_parts", False)) and (
                    log_grad_parts_every > 0 and (epoch % log_grad_parts_every == 0)
                )
                if do_log_grad_parts:
                    log_dict[f"{domain}/grad_ood_norm"] = float(grad_ood_norm_meters[domain].avg)
                    log_dict[f"{domain}/grad_cls_norm"] = float(grad_cls_norm_meters[domain].avg)
                    log_dict[f"{domain}/grad_ood_cls_cos"] = float(grad_ood_cls_cos_meters[domain].avg)
                    log_dict[f"{domain}/grad_mu_cls_l1_max_ratio"] = float(
                        grad_mu_cls_l1_max_ratio_meters[domain].avg
                    )
                    if domain in grad_mu_cls_ch_avg_epoch:
                        _v = grad_mu_cls_ch_avg_epoch[domain].numpy()
                        log_dict[f"{domain}/grad_mu_cls_abs_per_channel_hist"] = wandb.Histogram(_v)
                log_dict[f"{domain}/vec_hard_vs_style_delta"] = float(vec_hard_vs_style_delta_meters[domain].avg)
                if args.dataset == "pacs" and domain == "sketch":
                    cls_stats = []
                    for _c in range(sketch_num_classes):
                        hist = sketch_class_acc_history[_c]
                        if len(hist) == 0:
                            continue
                        cur = hist[-1]
                        prev = hist[-2] if len(hist) >= 2 else cur
                        delta = abs(cur - prev)
                        recent = hist[-min(len(hist), sketch_vol_window):]
                        recent_t = torch.tensor(recent, dtype=torch.float32)
                        vol_std = float(recent_t.std(unbiased=False).item()) if len(recent) > 1 else 0.0
                        cls_stats.append((_c, cur, delta, vol_std))
                    if len(cls_stats) > 0:
                        worst_cur = min(cls_stats, key=lambda x: x[1])
                        top_delta_one = max(cls_stats, key=lambda x: x[2])
                        top_std_one = max(cls_stats, key=lambda x: x[3])
                        log_dict[f"{domain}/class_monitor/worst_class_id"] = float(worst_cur[0])
                        log_dict[f"{domain}/class_monitor/worst_class_acc"] = float(worst_cur[1])
                        log_dict[f"{domain}/class_monitor/max_delta_class_id"] = float(top_delta_one[0])
                        log_dict[f"{domain}/class_monitor/max_delta"] = float(top_delta_one[2])
                        log_dict[f"{domain}/class_monitor/max_std_class_id"] = float(top_std_one[0])
                        log_dict[f"{domain}/class_monitor/max_std"] = float(top_std_one[3])
            log_dict["consensus_drift/pairwise_head_pre"] = float(drift_pairwise_head_pre_meters.avg)
            log_dict["consensus_drift/pairwise_head_post"] = float(drift_pairwise_head_post_meters.avg)
            log_dict["consensus_drift/pairwise_full_backbone_pre"] = float(drift_pairwise_backbone_pre_meters.avg)
            log_dict["consensus_drift/pairwise_full_backbone_post"] = float(drift_pairwise_backbone_post_meters.avg)

            log_grad_parts_every_ep = int(getattr(args, "hard_adv_log_grad_parts_every", 10))
            do_log_grad_parts_ep = bool(getattr(args, "hard_adv_log_grad_parts", False)) and (
                log_grad_parts_every_ep > 0 and (epoch % log_grad_parts_every_ep == 0)
            )
            if do_log_grad_parts_ep:
                _topk = 15
                for _d in domain_names:
                    if _d not in grad_mu_cls_ch_avg_epoch:
                        continue
                    v = grad_mu_cls_ch_avg_epoch[_d]
                    tk = min(_topk, int(v.numel()))
                    vals, idx = torch.topk(v, k=tk)
                    pairs = ", ".join(
                        f"ch{int(i)}={float(x):.5f}" for i, x in zip(idx.tolist(), vals.tolist())
                    )
                    print(
                        f"[grad_mu_cls |.|] epoch={epoch} domain={_d} "
                        f"L1/max={float(grad_mu_cls_l1_max_ratio_meters[_d].avg):.4f}  "
                        f"top{tk}: {pairs}"
                    )
            
            # Add diffusion loss if OOD detection is enabled
            if getattr(args, 'use_ood', False) and loss_diff_meters is not None:
                # ===== 为每个domain记录各自的diffusion监控指标 =====
                # 虽然denoiser参数会被聚合，但normalization buffers是独立的，所以每个domain的指标不同
                for domain in domain_names:
                    if domain in diffusion_models_dict:
                        diffusion_model = diffusion_models_dict[domain]
                        
                        # 为每个domain记录diffusion loss
                        log_dict[f"{domain}/diffusion_loss"] = loss_diff_meters[domain].avg.item() if hasattr(loss_diff_meters[domain].avg, 'item') else float(loss_diff_meters[domain].avg)
                        
                        # 预测噪声余弦相似度指标（每个domain独立）
                        if hasattr(diffusion_model.diffusion_process, '_last_snr_info'):
                            monitor_info = diffusion_model.diffusion_process._last_snr_info
                            if 'noise_pred_cosine' in monitor_info:
                                log_dict[f"{domain}/noise_pred_cosine/mean"] = monitor_info['noise_pred_cosine']
                                if 'noise_pred_cosine_std' in monitor_info:
                                    log_dict[f"{domain}/noise_pred_cosine/std"] = monitor_info['noise_pred_cosine_std']
                        
                        # ID vs Pseudo-OOD Score Gap（使用固定时间步 t=15）
                        try:
                            with torch.no_grad():
                                # 获取真实特征（ID）
                                sample_data, _, _ = util.unpack_batch(next(iter(domain_loaders[domain][1])))  # test_loader
                                sample_data = sample_data.cuda()
                                sample_latents = models_dict[domain].intermediate_forward(sample_data)
                                sample_latents_normalized = diffusion_model.normalize(sample_latents.detach())
                                
                                # 生成纯噪声（Pseudo-OOD）
                                noise_latents = torch.randn_like(sample_latents_normalized)
                                
                                # 使用固定时间步 t=15 计算 score
                                t_fixed = torch.full((sample_latents_normalized.size(0),), 15, 
                                                    device=sample_latents_normalized.device, dtype=torch.long)
                                
                                # 计算 ID 的 score norm
                                id_pred_noise = diffusion_model.denoiser(sample_latents_normalized, t_fixed)
                                # 计算 std_t for t=15
                                sqrt_one_minus_alpha_cumprod_t = diffusion_model.diffusion_process._extract(
                                    diffusion_model.diffusion_process.sqrt_one_minus_alphas_cumprod.to(sample_latents_normalized.device),
                                    t_fixed, sample_latents_normalized.shape
                                )
                                std_t = sqrt_one_minus_alpha_cumprod_t
                                id_score = -id_pred_noise / (std_t + 1e-8)
                                id_score_norm = id_score.norm(p=2, dim=1).mean().item()
                                
                                # 计算 Noise (Pseudo-OOD) 的 score norm
                                noise_pred_noise = diffusion_model.denoiser(noise_latents, t_fixed)
                                noise_score = -noise_pred_noise / (std_t + 1e-8)
                                noise_score_norm = noise_score.norm(p=2, dim=1).mean().item()
                                
                                # 计算 Gap
                                score_gap = noise_score_norm - id_score_norm
                                
                                log_dict[f"{domain}/score_gap/id_score_norm"] = id_score_norm
                                log_dict[f"{domain}/score_gap/noise_score_norm"] = noise_score_norm
                                log_dict[f"{domain}/score_gap/gap"] = score_gap
                        except Exception as e:
                            # 如果计算失败，跳过（不影响训练）
                            pass
                        
                        # Diffusion学习率（所有domain应该相同，只记录一次）
                        if domain == domain_names[0]:
                            log_dict["train/lr_diffusion"] = optimizers_diffusion_dict[domain].param_groups[0]['lr']
            
            wandb.log(log_dict)

            # ===== Periodic checkpoint saving =====
            if save_every_epoch is not None and save_every_epoch > 0 and (epoch % save_every_epoch == 0):
                for domain in domain_names:
                    model = models_dict[domain]
                    checkpoint = {
                        "args": args,
                        "domain": domain,
                        "all_domains": domain_names,
                        "dataset": args.dataset,
                        "leave_out": getattr(args, "leave_out", None),
                        # Full backbone model (feature extractor + classifier head)
                        "backbone_state": model.state_dict(),
                    }

                    if hasattr(model, "diffusion_model") and model.diffusion_model is not None:
                        checkpoint["diffusion_state"] = model.diffusion_model.state_dict()

                    save_name = f"{args.description}_{domain}_epoch_{epoch}.pth"
                    save_path = os.path.join(save_dir, save_name)
                    torch.save(checkpoint, save_path)
                    print(f"[Checkpoint] Saved checkpoint at epoch {epoch} for domain '{domain}' to: {save_path}")

            # reset recorders for next epoch
            comp_time, comm_time = 0, 0
            for domain in domain_names:
                losses_dict[domain].reset()
                top1_dict[domain].reset()
                style_aug_flag_meters[domain].reset()
                task_loss_meters[domain].reset()
                clean_ce_meters[domain].reset()
                clean_conf_meters[domain].reset()
                style_conf_meters[domain].reset()
                clean_style_ce_cos_meters[domain].reset()
                clean_style_sym_kl_meters[domain].reset()
                style_ce_meters[domain].reset()
                hard_ce_meters[domain].reset()
                hard_ce_delta_meters[domain].reset()
                hard_ce_harder_ratio_meters[domain].reset()
                hard_ce_delta_samples[domain].clear()
                hard_ood_loss_meters[domain].reset()
                hard_style_conf_meters[domain].reset()
                hard_hard_conf_meters[domain].reset()
                hard_style_target_max_gap_meters[domain].reset()
                hard_hard_target_max_gap_meters[domain].reset()
                hard_flip_all_ratio_meters[domain].reset()
                hard_flip_on_style_correct_ratio_meters[domain].reset()
                hard_mu_hit_lower_ratio_meters[domain].reset()
                hard_mu_hit_upper_ratio_meters[domain].reset()
                hard_sigma_hit_lower_ratio_meters[domain].reset()
                hard_sigma_hit_upper_ratio_meters[domain].reset()
                adv_ood_loss_meters[domain].reset()
                adv_cls_inner_meters[domain].reset()
                mu_orig_norm_meters[domain].reset()
                sigma_orig_norm_meters[domain].reset()
                grad_mu_norm_meters[domain].reset()
                grad_sigma_norm_meters[domain].reset()
                rel_mu_step_meters[domain].reset()
                rel_sigma_step_meters[domain].reset()
                grad_ood_norm_meters[domain].reset()
                grad_cls_norm_meters[domain].reset()
                grad_ood_cls_cos_meters[domain].reset()
                grad_mu_cls_l1_max_ratio_meters[domain].reset()
                vec_hard_vs_style_delta_meters[domain].reset()
                grad_style_norm_layer3_meters[domain].reset()
                grad_hard_norm_layer3_meters[domain].reset()
                grad_style_hard_cos_layer3_meters[domain].reset()
                grad_style_norm_layer4_meters[domain].reset()
                grad_hard_norm_layer4_meters[domain].reset()
                grad_style_hard_cos_layer4_meters[domain].reset()
                grad_style_norm_head_meters[domain].reset()
                grad_hard_norm_head_meters[domain].reset()
                grad_style_hard_cos_head_meters[domain].reset()
                grad_style_norm_backbone_meters[domain].reset()
                grad_hard_norm_backbone_meters[domain].reset()
                grad_style_hard_cos_backbone_meters[domain].reset()
                drift_l2_layer4_meters[domain].reset()
                drift_rel_layer4_meters[domain].reset()
                drift_l2_head_meters[domain].reset()
                drift_rel_head_meters[domain].reset()
                drift_l2_backbone_meters[domain].reset()
                drift_rel_backbone_meters[domain].reset()
                vec_style_hard_dir_cos_meters[domain].reset()
                vec_style_from_clean_norm_meters[domain].reset()
                vec_hard_from_clean_norm_meters[domain].reset()
                vec_style_hard_dir_cos_low03_meters[domain].reset()
                orth_removed_ratio_meters[domain].reset()
                orth_fallback_frac_meters[domain].reset()
                style_triplet_l2_orig_pool_meters[domain].reset()
                style_triplet_l2_hard_pool_meters[domain].reset()
                style_triplet_l2_orig_hard_meters[domain].reset()
                style_triplet_cos_orig_pool_meters[domain].reset()
                style_triplet_cos_hard_pool_meters[domain].reset()
                style_triplet_cos_orig_hard_meters[domain].reset()
                vec_style_hard_dir_cos_samples[domain].clear()
                if bool(getattr(args, "hard_adv_monitor_sparsity", False)):
                    z_hard_sparsity_meters[domain].reset()
                    vec_hard_sparsity_meters[domain].reset()
                    vec_clean_l4_sparsity_meters[domain].reset()
                    vec_hard_vs_clean_l4_delta_meters[domain].reset()
                if loss_diff_meters is not None:
                    loss_diff_meters[domain].reset()
            drift_pairwise_head_pre_meters.reset()
            drift_pairwise_head_post_meters.reset()
            drift_pairwise_backbone_pre_meters.reset()
            drift_pairwise_backbone_post_meters.reset()
            tic = time.time()

    # ===== Save final models (backbone + diffusion + normalization stats) =====
    # We save one checkpoint per domain so that:
    # - Backbone (feature extractor + classifier head) weights are preserved per domain
    # - Diffusion model (denoiser + diffusion process + FeatureNormalization buffers) are preserved per domain
    # - Domain-specific FeatureNormalization statistics (mean/std/shift/scale) are not lost
    for domain in domain_names:
        model = models_dict[domain]
        checkpoint = {
            "args": args,
            "domain": domain,
            "all_domains": domain_names,
            "dataset": args.dataset,
            "leave_out": getattr(args, "leave_out", None),
            # Full backbone model (feature extractor + classifier head)
            "backbone_state": model.state_dict(),
        }

        # Save full diffusion model state (includes denoiser + diffusion process + FeatureNormalization buffers)
        if hasattr(model, "diffusion_model") and model.diffusion_model is not None:
            checkpoint["diffusion_state"] = model.diffusion_model.state_dict()

        save_name = f"{args.description}_{domain}_final.pth"
        save_path = os.path.join(save_dir, save_name)
        torch.save(checkpoint, save_path)
        print(f"[Checkpoint] Saved final backbone + diffusion model for domain '{domain}' to: {save_path}")

    # Save recorders
    for domain, recorder in recorders.items():
        recorder.save_to_file()
    wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Single Process Training for Decentralized Learning')
    parser.add_argument('--name','-n', default="default", type=str, help='experiment name')
    parser.add_argument('--description', type=str, help='experiment description')

    parser.add_argument('--model', default="res", type=str, help='model name: res/VGG/wrn')
    parser.add_argument('--resnet_type', default='simplified', type=str, 
                        choices=['simplified', 'standard'], 
                        help='ResNet type: simplified (current) or standard (torchvision)')
    parser.add_argument('--lr', default=0.8, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.0, type=float, help='momentum')
    parser.add_argument('--epoch', '-e', default=10, type=int, help='total epoch')
    parser.add_argument('--bs', default=64, type=int, help='batch size on each worker')
    parser.add_argument('--num_nodes', type=int, default=None,
                        help='number of virtual nodes in single-process mode (default: train domains count)')
    parser.add_argument('--node_split_mode', type=str, default=None,
                        choices=['class_balanced', 'random_contiguous'],
                        help='how to split one domain among multiple virtual nodes '
                             '(auto: class_balanced for graphid=6, otherwise random_contiguous)')
    parser.add_argument('--debug_topology', action='store_true',
                        help='print topology neighbors and node-domain mapping at startup')
    parser.add_argument('--debug_per_node_metrics', action='store_true',
                        help='print per-domain/per-node test_acc each epoch (default: only avg_test_acc)')
    parser.add_argument('--sampler_type', type=str, default='random',
                        choices=['random', 'random_class'],
                        help='train sampler type: random (shuffle) or random_class (balanced classes per batch)')
    parser.add_argument('--n_ins', type=int, default=16,
                        help='for random_class sampler: number of instances per class in each batch')
    parser.add_argument('--warmup', action='store_true', help='use lr warmup or not')
    parser.add_argument('--nesterov', action='store_true', help='use nesterov momentum or not')

    parser.add_argument('--matcha', action='store_true', help='use MATCHA or not')
    parser.add_argument('--budget', type=float, help='comm budget')
    parser.add_argument('--graphid', default=0, type=int, help='the idx of base graph')
    
    parser.add_argument('--dataset', default='cifar10', type=str, help='the dataset')
    parser.add_argument('--datasetRoot', type=str, help='the path of dataset')
    parser.add_argument('--leave_out', type=str, default=None, help='leave out domain for PACS dataset (art_painting, cartoon, photo, sketch)')
    parser.add_argument('--p', '-p', action='store_true', help='partition the dataset or not')
    parser.add_argument('--savePath' ,type=str, help='save path')
    parser.add_argument('--save_every_epoch', type=int, default=50,
                        help='save checkpoints every N epochs during training (0 to disable)')
    
    parser.add_argument('--compress', action='store_true', help='use chocoSGD or not')    
    parser.add_argument('--consensus_lr', default=0.1, type=float, help='consensus_lr')
    parser.add_argument('--randomSeed', type=int, help='random seed')
    parser.add_argument('--total_iter', type=int, help='total training iterations (if not set, uses epoch * max_steps_per_epoch)')
    parser.add_argument('--wandb_project', default='MATCHA', type=str, help='wandb project name')

    # ===== Style statistics / StyleDDG-related options =====
    parser.add_argument('--use_style_stats', action='store_true',
                        help='compute style statistics from first three conv blocks in a single forward pass')
    parser.add_argument('--style_eta', type=float, default=1e-5,
                        help='numerical stability constant for style statistics')
    
    # ===== Style Shift options =====
    parser.add_argument('--use_style_shift', action='store_true',
                        help='enable style shift: transform part of batch to neighbor style using AdaIN')
    parser.add_argument('--style_shift_prob', type=float, default=0.5,
                        help='probability of activating style shift module (default: 0.5)')
    parser.add_argument('--style_shift_ratio', type=float, default=0.5,
                        help='ratio of samples in batch to be transformed (default: 0.5)')
    parser.add_argument('--style_explore_alpha', type=float, default=3.0,
                        help='extrapolation coefficient for style explore module (default: 3.0)')
    parser.add_argument('--style_explore_ratio', type=float, default=0.5,
                        help='ratio of samples in batch to be explored (default: 0.5)')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pretrained ImageNet weights for ResNet (default: False). Only works with resnet_type=standard')

    # ===== Hard-style adversarial feature augmentation (z_style -> z_hard) =====
    parser.add_argument('--use_hard_style_adv', action='store_true',
                        help='enable inner-loop mu/sigma adversarial update at layer3 to generate z_hard (requires --use_ood and resnet_type=standard)')
    parser.add_argument('--hard_adv_always', action='store_true',
                        help='run inner loop for every batch (default: off; when off, run only if style modules activated)')
    parser.add_argument('--hard_adv_steps', type=int, default=1,
                        help='number of inner-loop gradient steps (default: 1)')
    parser.add_argument('--hard_adv_eta', type=float, default=0.01,
                        help='inner-loop step size eta for mu/sigma update (default: 0.01)')
    parser.add_argument('--hard_adv_lambda', type=float, default=1.0,
                        help='weight for inner-loop classification term lambda (default: 1.0)')
    parser.add_argument('--hard_adv_update_mode', type=str, default='gd', choices=['gd', 'pngd'],
                        help='inner-loop update mode for mu/sigma: gd (mu-=eta*grad) or pngd (normalized step + projection) (default: gd)')
    parser.add_argument('--hard_adv_max_rel_margin', type=float, default=0.1,
                        help='pngd: relative projection margin ratio around mu_orig/sigma_orig (default: 0.1)')
    parser.add_argument('--hard_adv_grad_norm_eps', type=float, default=1e-8,
                        help='pngd: epsilon added to grad norm during normalization (default: 1e-8)')
    parser.add_argument('--hard_adv_grad_gate_tau', type=float, default=0.0,
                        help='pngd: if per-sample grad norm < tau, fall back to plain GD step (0 disables) (default: 0.0)')
    parser.add_argument('--hard_adv_cls_only', action='store_true',
                        help='inner loop: use only classification term for L_adv (L_adv = lambda * L_cls_inner)')
    parser.add_argument('--hard_adv_equal_ce_weight', action='store_true',
                        help='outer loop: use equal CE weighting, loss = 0.5 * (loss_style_ce + loss_hard_ce)')
    parser.add_argument('--hard_adv_eps', type=float, default=1e-6,
                        help='epsilon for numerical stability and sigma clamp_min (default: 1e-6)')
    parser.add_argument('--hard_adv_ood_timestep', type=int, default=250,
                        help='fixed diffusion timestep t for L_ood in inner loop (0 .. num_timesteps-1; default: 250)')
    parser.add_argument('--hard_adv_pool_delta_probe', action='store_true',
                        help='inner loop: measure ||z512_after-z512_before|| after one FGSM-style sign step on mu/sigma (extra 2 forwards)')
    parser.add_argument('--hard_adv_pool_delta_print', action='store_true',
                        help='print pool_delta_probe on first iter of each pseudo-epoch (domain 0 only)')
    parser.add_argument('--hard_adv_log_grad_parts', action='store_true',
                        help='log inner-loop grad parts: OOD/CLS norms+cos; grad_mu_cls L1/max ratio; '
                             'wandb histogram of epoch-mean |grad_mu_cls| per channel; print top-15 channels (extra autograd.grad)')
    parser.add_argument('--hard_adv_log_grad_parts_every', type=int, default=10,
                        help='only compute/log hard_adv_log_grad_parts every N epochs (0 disables). Default: 10')
    parser.add_argument('--hard_adv_monitor_sparsity', action='store_true',
                        help='extra hard-adv monitor: print epoch-level sparsity for z_hard/_vec_hard/vec_clean_l4 in terminal (adds one extra forward_from_layer3 per batch)')
    parser.add_argument('--hard_adv_monitor_direction', action='store_true',
                        help='extra hard-adv monitor: compare direction (z_style-z_clean) vs (z_hard-z_clean) in pooled feature space')
    parser.add_argument('--hard_adv_orthogonal_style_clean', action='store_true',
                        help='inner loop: project (grad_mu,grad_sigma) orthogonal to per-sample '
                             '(mu_style-mu_clean, sigma_style-sigma_clean) in concat space (requires z_clean; '
                             'adds one extract_features_to_layer3 when not already computed)')
    parser.add_argument('--hard_adv_orth_fallback_ratio', type=float, default=0.05,
                        help='after orthogonal projection, if ||g_orth|| < ratio * ||g|| for a sample, use original g '
                             '(default: 0.05)')
    parser.add_argument('--hard_adv_z_base', type=str, default='clean', choices=['clean', 'style'],
                        help='base feature map used to compose z_hard content: clean (default) or style')
    parser.add_argument('--hard_adv_clean_loss_weight', type=float, default=0.0,
                        help='extra CE weight on clean logits from model(data, communicator=None); 0 disables (default: 0.0)')
    parser.add_argument('--hard_style_test_pool', action='store_true',
                        help='Phase A oracle: each pseudo-epoch rebuild layer3 mu/sigma pool from a random subset of '
                             'leave-out test set (current backbone). Compose z_hard with '
                             '(mu,sigma)=(1-beta)(mu_orig,sigma_orig)+beta*ref where ref is chosen by '
                             '--hard_style_test_pool_mode (random per-sample pool row, or mean anchor). '
                             'Inner loop still runs for metrics.')
    parser.add_argument('--hard_style_test_pool_frac', type=float, default=0.1,
                        help='fraction of leave-out test images in the pool (default: 0.1)')
    parser.add_argument('--hard_style_test_pool_mode', type=str, default='random', choices=['random', 'mean'],
                        help='random: per-sample draw from pool then mix with orig (matches prior full-replace when '
                             'beta=1); mean: mix with pool-mean anchor (default: random)')
    parser.add_argument('--hard_style_test_anchor_beta', type=float, default=0.1,
                        help='mixing weight beta in [0,1]: (1-beta)*(mu_orig,sigma_orig)+beta*ref; beta=1 uses only '
                             'ref (drawn row in random mode, pool mean in mean mode) (default: 0.1)')
    parser.add_argument('--save_spatial_debug_tensors', action='store_true',
                        help='save layer3 spatial debug tensors (input, z_clean, z_style, z_hard) for one batch per selected epoch')
    parser.add_argument('--spatial_debug_domain', type=str, default='sketch',
                        help='which training domain to dump spatial debug tensors for (default: sketch)')
    parser.add_argument('--spatial_debug_every', type=int, default=10,
                        help='save one spatial debug dump every N epochs (default: 10)')
    parser.add_argument('--spatial_debug_max_samples', type=int, default=8,
                        help='maximum number of samples to save in each spatial debug dump (default: 8)')

    # ===== OOD Detection (Diffusion) options =====
    parser.add_argument('--use_ood', action='store_true',
                        help='enable OOD detection with diffusion model')
    parser.add_argument('--diffusion_channels', type=int, default=512,
                        help='Diffusion denoiser channels (default: 512)')
    parser.add_argument('--diffusion_steps', type=int, default=1000,
                        help='Number of diffusion steps (default: 1000)')
    parser.add_argument('--lr_diffusion', type=float, default=5e-5,
                        help='Learning rate for diffusion model (default: 5e-5)')
    parser.add_argument('--lambda_diff', type=float, default=1.0,
                        help='Weight for diffusion loss (default: 1.0)')

    args = parser.parse_args()

    # Auto split mode policy:
    # - If user did not set --node_split_mode, use class_balanced for RGG (graphid=6)
    # - Otherwise use random_contiguous as default fallback
    if args.node_split_mode is None:
        args.node_split_mode = 'class_balanced' if args.graphid == 6 else 'random_contiguous'

    if not args.description:
        print('No experiment description, exit!')
        exit()

    # Validate dataset requirements
    if args.dataset == 'pacs':
        if not args.leave_out:
            print('Error: --leave_out must be specified when using PACS dataset.')
            print('Valid options: art_painting, cartoon, photo, sketch')
            exit(1)
        valid_domains = ['art_painting', 'cartoon', 'photo', 'sketch']
        if args.leave_out not in valid_domains:
            print(f'Error: Invalid leave_out domain: {args.leave_out}')
            print(f'Valid options: {valid_domains}')
            exit(1)
        # For PACS, we have 3 training domains
        num_domains = 3
    elif args.dataset == 'vlcs':
        if not args.leave_out:
            print('Error: --leave_out must be specified when using VLCS dataset.')
            print('Valid options: caltech, labelme, pascal, sun')
            exit(1)
        valid_domains = ['caltech', 'labelme', 'pascal', 'sun']
        leave_out = str(args.leave_out).lower()
        if leave_out not in valid_domains:
            print(f'Error: Invalid leave_out domain: {args.leave_out}')
            print(f'Valid options: {valid_domains}')
            exit(1)
        args.leave_out = leave_out
        # For VLCS, we have 3 training domains
        num_domains = 3
    else:
        # For other datasets, single domain
        num_domains = 1

    run(num_domains)  # 開始訓練

