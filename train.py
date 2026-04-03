import os
import numpy as np
import time
import argparse
import sys
from copy import deepcopy
import random

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
    style_ce_meters = {domain: util.AverageMeter() for domain in domain_names}
    hard_ce_meters = {domain: util.AverageMeter() for domain in domain_names}
    adv_ood_loss_meters = {domain: util.AverageMeter() for domain in domain_names}
    adv_cls_inner_meters = {domain: util.AverageMeter() for domain in domain_names}
    # Monitor statistics and gradients for mu/sigma in inner loop
    mu_orig_norm_meters = {domain: util.AverageMeter() for domain in domain_names}
    sigma_orig_norm_meters = {domain: util.AverageMeter() for domain in domain_names}
    grad_mu_norm_meters = {domain: util.AverageMeter() for domain in domain_names}
    grad_sigma_norm_meters = {domain: util.AverageMeter() for domain in domain_names}
    loss_diff_meters = {domain: util.AverageMeter() for domain in domain_names} if getattr(args, 'use_ood', False) else None
    tic = time.time()

    # ===== start training with fixed total steps K (Algorithm 1) =====
    for k in range(K):
        # Set all models to training mode
        for model in models_dict.values():
            model.train()

        start_time = time.time()

        # ========== 第一阶段：计算所有 domain 的风格统计量（不训练）==========
        style_vecs_dict = {}
        # Cache one batch per domain so style stats and training share identical data
        # (matching train_mpi.py behavior).
        batch_cache = {}
        if (getattr(args, "use_style_stats", False) or getattr(args, "use_style_shift", False)) and args.model == "res":
            for domain in domain_names:
                data, target = next(train_iters[domain])
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                batch_cache[domain] = (data, target)
                
                with torch.no_grad():  # 不计算梯度，节省内存
                    model = models_dict[domain]
                    # 只提取特征到 layer3，不应用 style shift
                    feats = model.extract_features_to_layer3(data)
                    
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
                data, target = batch_cache[domain]
            else:
                data, target = next(train_iters[domain])
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

                logits_style, vec_style = model.forward_from_layer3(z_style)
                if vec_style.ndim != 2 or vec_style.size(1) != expected_ft:
                    raise RuntimeError(
                        f"Diffusion expects ft_size={expected_ft}, but got vec_style shape={tuple(vec_style.shape)}. "
                        "Ensure you are using --resnet_type standard and ft_size matches the pooled feature dim."
                    )

                # Inner loop: generate z_hard (one or few steps), only updating mu/sigma.
                if run_adv:
                    with torch.no_grad():
                        mu_orig = z_style.detach().mean(dim=(2, 3), keepdim=True)
                        var_orig = z_style.detach().var(dim=(2, 3), unbiased=False, keepdim=True)
                        sigma_orig = torch.sqrt(var_orig + adv_eps)
                        # 監控 mu_orig / sigma_orig 的 L2 範數（batch-level）
                        mu_orig_norm = mu_orig.norm().item()
                        sigma_orig_norm = sigma_orig.norm().item()
                        mu_orig_norm_meters[domain].update(mu_orig_norm, data.size(0))
                        sigma_orig_norm_meters[domain].update(sigma_orig_norm, data.size(0))

                    # z_hat uses detached z_style (inner loop should not affect backbone).
                    z_norm_detached = (z_style.detach() - mu_orig) / (sigma_orig + adv_eps)

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
                    for _ in range(max(1, adv_steps)):
                        z_hat = z_norm_detached * sigma_cur + mu_cur
                        logits_hat, vec_hat = model.forward_from_layer3(z_hat)
                        if vec_hat.ndim != 2 or vec_hat.size(1) != expected_ft:
                            raise RuntimeError(
                                f"Expected vec_hat shape (B,{expected_ft}), got {tuple(vec_hat.shape)}."
                            )

                        vec_hat_norm = diffusion_model.normalize(vec_hat)
                        L_ood = diffusion_model.get_loss_at_timestep(vec_hat_norm, adv_ood_timestep)
                        L_cls_inner = criterion(logits_hat, target)
                        L_adv = -L_ood + adv_lambda * L_cls_inner

                        grad_mu, grad_sigma = torch.autograd.grad(L_adv, [mu_cur, sigma_cur], create_graph=False)
                        # 監控梯度範數（取最後一步為代表）
                        grad_mu_norm = grad_mu.norm().item()
                        grad_sigma_norm = grad_sigma.norm().item()
                        grad_mu_norm_meters[domain].update(grad_mu_norm, data.size(0))
                        grad_sigma_norm_meters[domain].update(grad_sigma_norm, data.size(0))
                        mu_next = mu_cur - adv_eta * grad_mu
                        sigma_next = (sigma_cur - adv_eta * grad_sigma).clamp_min(adv_eps)
                        mu_cur, sigma_cur = mu_next, sigma_next

                    mu_hard = mu_cur.detach()
                    sigma_hard = sigma_cur.detach()
                    if not torch.isfinite(mu_hard).all() or not torch.isfinite(sigma_hard).all():
                        raise FloatingPointError("Non-finite mu_hard/sigma_hard detected during inner loop.")

                    # Restore training modes and requires_grad for outer loop.
                    if prev_backbone_training is not None:
                        backbone.train(prev_backbone_training)
                    model.train(prev_model_training)
                    diffusion_model.train(prev_diff_training)
                    for p in model.parameters():
                        p.requires_grad_(True)
                    for p in diffusion_model.parameters():
                        p.requires_grad_(True)

                    # Outer loop: z_hard re-composed from LIVE z_style (gradient flows to layer1/2/3).
                    z_norm_live = (z_style - mu_orig) / (sigma_orig + adv_eps)
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
                    finally:
                        if backbone is not None and hasattr(backbone, "layer4") and prev_layer4_training is not None:
                            backbone.layer4.train(prev_layer4_training)

                    loss_style_ce = criterion(logits_style, target)
                    loss_hard_ce = criterion(logits_hard, target)

                    # 線性 warmup：從 epoch 0 到 60，z_hard 權重從 0 → 0.5
                    # 之後維持 0.5；z_style 權重始終為 1 - w_hard。
                    warmup_epochs = 60.0
                    # 以當前 iteration 推出「目前是第幾個 epoch」（浮點數）
                    current_epoch = (k + 1) / float(STEPS_PER_EPOCH) if STEPS_PER_EPOCH > 0 else 0.0
                    hard_weight = 0.5 * min(max(current_epoch / warmup_epochs, 0.0), 1.0)
                    style_weight = 1.0 - hard_weight

                    loss = style_weight * loss_style_ce + hard_weight * loss_hard_ce

                    # Record adv meters (batch-level; treat scalar as per-sample weight).
                    task_loss_meters[domain].update(float(loss.item()), data.size(0))
                    style_ce_meters[domain].update(float(loss_style_ce.item()), data.size(0))
                    hard_ce_meters[domain].update(float(loss_hard_ce.item()), data.size(0))
                    adv_ood_loss_meters[domain].update(float(L_ood.item()), data.size(0))
                    adv_cls_inner_meters[domain].update(float(L_cls_inner.item()), data.size(0))
                else:
                    # No adv: just use z_style branch.
                    loss = criterion(logits_style, target)
                    task_loss_meters[domain].update(float(loss.item()), data.size(0))

                output = logits_style
            else:
                # Original training path (logits only)
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
        # Exchange updated model parameters after training step
        # Note: style_vecs_dict is None here since we only exchange model parameters (not style stats)
        d_comm_time_after = communicator.communicate(models_dict, style_vecs_dict=None)
        comm_time += d_comm_time_after
        
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
            for domain in domain_names:
                _, test_loader = domain_loaders[domain]
                test_acc = util.test(models_dict[domain], test_loader)
                test_accs[domain] = test_acc
            
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
                log_dict[f"{domain}/style_aug_activated_rate"] = float(style_aug_flag_meters[domain].avg)
                # Hard-style-adv metrics (only meaningful when --use_hard_style_adv)
                log_dict[f"{domain}/task_loss"] = float(task_loss_meters[domain].avg)
                log_dict[f"{domain}/style_ce"] = float(style_ce_meters[domain].avg)
                log_dict[f"{domain}/hard_ce"] = float(hard_ce_meters[domain].avg)
                log_dict[f"{domain}/adv_ood_loss"] = float(adv_ood_loss_meters[domain].avg)
                log_dict[f"{domain}/adv_cls_inner"] = float(adv_cls_inner_meters[domain].avg)
                log_dict[f"{domain}/mu_orig_norm"] = float(mu_orig_norm_meters[domain].avg)
                log_dict[f"{domain}/sigma_orig_norm"] = float(sigma_orig_norm_meters[domain].avg)
                log_dict[f"{domain}/grad_mu_norm"] = float(grad_mu_norm_meters[domain].avg)
                log_dict[f"{domain}/grad_sigma_norm"] = float(grad_sigma_norm_meters[domain].avg)
            
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
                                sample_data, _ = next(iter(domain_loaders[domain][1]))  # test_loader
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
                style_ce_meters[domain].reset()
                hard_ce_meters[domain].reset()
                adv_ood_loss_meters[domain].reset()
                adv_cls_inner_meters[domain].reset()
                mu_orig_norm_meters[domain].reset()
                sigma_orig_norm_meters[domain].reset()
                grad_mu_norm_meters[domain].reset()
                grad_sigma_norm_meters[domain].reset()
                if loss_diff_meters is not None:
                    loss_diff_meters[domain].reset()
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
    parser.add_argument('--hard_adv_eps', type=float, default=1e-6,
                        help='epsilon for numerical stability and sigma clamp_min (default: 1e-6)')
    parser.add_argument('--hard_adv_ood_timestep', type=int, default=250,
                        help='fixed diffusion timestep t for L_ood in inner loop (0 .. num_timesteps-1; default: 250)')

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

