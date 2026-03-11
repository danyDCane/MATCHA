import os
import numpy as np
import time
import argparse
import sys
from copy import deepcopy

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

    # initialize wandb (only once for single process)
    wandb.init(
        project=args.wandb_project,
        name=f"{args.name}_single",
        config={
            "num_domains": num_domains,
            "model": args.model,
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

    # load data for all domains
    if args.dataset == 'pacs':
        domain_loaders = util.load_dataset_single_process(args)
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
        # RGG：9 個節點，半徑=0.8，使用 randomSeed 確保可重現性
        subGraphs = util.select_graph(6, num_nodes=9, radius=0.8, seed=args.randomSeed)
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

    # define single process communicator
    communicator = SingleProcessCommunicator(domain_names, GP)

    # select neural network model for each domain
    num_classes = 7 if args.dataset == 'pacs' else 10
    
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
    loss_diff_meters = {domain: util.AverageMeter() for domain in domain_names} if getattr(args, 'use_ood', False) else None
    loss_pair_meters = {domain: util.AverageMeter() for domain in domain_names} if (getattr(args, 'use_ood', False) and getattr(args, 'use_style_shift', False)) else None
    loss_supcon_meters = {domain: util.AverageMeter() for domain in domain_names} if (getattr(args, 'use_style_shift', False) and getattr(args, 'use_supcon', False)) else None
    # Style Removal Power (指標二) + SVD Effective Rank，只在三個 domain 時計算
    style_removal_mse_z_zs = None
    style_removal_mse_z_zs0hat = None
    svd_effective_rank_95 = None
    svd_top10_energy_ratio = None
    if num_domains == 3 and loss_pair_meters is not None:
        style_removal_mse_z_zs = {domain: util.AverageMeter() for domain in domain_names}
        style_removal_mse_z_zs0hat = {domain: util.AverageMeter() for domain in domain_names}
        svd_effective_rank_95 = {domain: util.AverageMeter() for domain in domain_names}
        svd_top10_energy_ratio = {domain: util.AverageMeter() for domain in domain_names}
    tic = time.time()

    # ===== start training with fixed total steps K (Algorithm 1) =====
    for k in range(K):
        # Set all models to training mode
        for model in models_dict.values():
            model.train()

        start_time = time.time()

        # ========== 第一阶段：计算所有 domain 的风格统计量（不训练）==========
        style_vecs_dict = {}
        if (getattr(args, "use_style_stats", False) or getattr(args, "use_style_shift", False)) and args.model == "res":
            for domain in domain_names:
                data, target = next(train_iters[domain])
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                
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
        use_ood = getattr(args, 'use_ood', False)
        warmup_epochs = getattr(args, 'warmup_epochs', 50)
        pair_warmup_end_epoch = getattr(args, 'pair_warmup_end_epoch', 100)
        lambda_pair = getattr(args, 'lambda_pair', 0.2)
        lambda_supcon = getattr(args, 'lambda_supcon', 0.1)
        supcon_temperature = getattr(args, 'supcon_temperature', 0.1)
        pair_t_min = getattr(args, 'pair_t_min', 200)
        pair_t_max = getattr(args, 'pair_t_max', 400)
        
        # Epoch for Pair Loss schedule (1-based)
        epoch = (k + 1) // STEPS_PER_EPOCH
        if epoch < warmup_epochs:
            current_lambda_pair = 0.0
        else:
            progress = (epoch - warmup_epochs) / max(1, pair_warmup_end_epoch - warmup_epochs)
            current_lambda_pair = lambda_pair * min(1.0, progress)
        
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
            
            # Get batch for this domain
            data, target = next(train_iters[domain])
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            
            # Dual forward when use_style_shift (for Pair Loss); otherwise single forward
            z_s = None
            if args.model == "res" and use_style_shift:
                # Clean forward (no style shift): communicator=None
                out_clean, z = model(data, communicator=None, return_feature=True)
                # Style-shifted forward (force style shift)
                out_shift, z_s = model(
                    data,
                    communicator=communicator,
                    force_style_shift=True,
                    debug_style_shift=debug_style_shift,
                    iter_num=k + 1,
                    rank=domain_names.index(domain),
                    return_feature=True,
                )
                loss_cls = criterion(out_clean, target) + criterion(out_shift, target)
                output = out_clean  # for accuracy logging (use clean as primary)
            elif args.model == "res":
                if use_style_stats or use_style_shift:
                    output, feat = model(
                        data,
                        communicator=communicator,
                        debug_style_shift=debug_style_shift,
                        iter_num=k + 1,
                        rank=domain_names.index(domain),
                        return_feature=True,
                    )
                else:
                    output, feat = model(data, return_feature=True)
                loss_cls = criterion(output, target)
                z = feat
            else:
                output = model(data)
                loss_cls = criterion(output, target)
                z = None
            
            # Pair Loss (only when epoch >= warmup_epochs, use_ood, use_style_shift)
            loss_pair = None
            if (epoch >= warmup_epochs and use_ood and use_style_shift and 
                domain in diffusion_models_dict and args.model == "res" and current_lambda_pair > 0):
                diffusion_model = diffusion_models_dict[domain]
                dp = diffusion_model.diffusion_process
                
                # Use eval() to avoid in-place buffer updates in FeatureNormalization
                # (normalize() updates shift/scale in-place when training, which breaks grad for z/z_s)
                was_diffusion_training = diffusion_model.training
                diffusion_model.eval()
                z_norm = diffusion_model.normalize(z)
                z_s_norm = diffusion_model.normalize(z_s)
                
                # 檢測哪些樣本有被風格偏移（因 style_shift_prob/ratio 可能使部分未偏移）
                with torch.no_grad():
                    diff_per_sample = (z_norm - z_s_norm).pow(2).sum(dim=1)
                    shifted_mask = diff_per_sample > 1e-8
                    n_shifted = shifted_mask.sum().item()
                
                N = z.size(0)
                t = torch.randint(pair_t_min, pair_t_max + 1, (N,), device=z.device, dtype=torch.long)
                epsilon = torch.randn_like(z, device=z.device)
                # 將z加噪聲得到z_t
                z_t = dp.q_sample(z_norm, t, noise=epsilon)
                # 將z_s加噪聲得到z_s_t
                z_s_t = dp.q_sample(z_s_norm, t, noise=epsilon)
                
                with torch.no_grad():
                    # 用denoiser預測z_t和z_s_t的x_0
                    eps_hat = diffusion_model.denoiser(z_t, t)
                    eps_hat_s = diffusion_model.denoiser(z_s_t, t)
                
                # 用denoiser預測z_t和z_s_t的噪聲後，去噪得到z_0_hat和z_s_0_hat
                z_0_hat = dp._predict_xstart_from_eps(z_t, t=t, eps=eps_hat)
                z_s_0_hat = dp._predict_xstart_from_eps(z_s_t, t=t, eps=eps_hat_s)
                
                # Pair Loss：只對有風格偏移的樣本計算，否則無意義（z≈z_s 時 loss 本就接近 0）
                if n_shifted > 0:
                    z_0_hat_shifted = z_0_hat[shifted_mask]
                    z_s_0_hat_shifted = z_s_0_hat[shifted_mask]
                    # Stop-Gradient 於 z_0_hat：z 作為固定錨點，z_s_0_hat 單向逼近
                    loss_pair = F.mse_loss(z_s_0_hat_shifted, z_0_hat_shifted.detach())
                    if loss_pair_meters is not None:
                        loss_pair_meters[domain].update(loss_pair.item(), n_shifted)
                else:
                    loss_pair = 0.0
                
                # 指標二：風格濾除力 / 語義還原度 (Style Removal Power)，僅在三個 domain 時計算
                if style_removal_mse_z_zs is not None and style_removal_mse_z_zs0hat is not None:
                    with torch.no_grad():
                        if n_shifted > 0:
                            z_norm_shifted = z_norm[shifted_mask]
                            z_s_norm_shifted = z_s_norm[shifted_mask]
                            z_s_0_hat_shifted = z_s_0_hat[shifted_mask]
                            mse_z_zs_val = F.mse_loss(z_norm_shifted, z_s_norm_shifted).item()
                            mse_z_zs0hat_val = F.mse_loss(z_norm_shifted, z_s_0_hat_shifted).item()
                            style_removal_mse_z_zs[domain].update(mse_z_zs_val, n_shifted)
                            style_removal_mse_z_zs0hat[domain].update(mse_z_zs0hat_val, n_shifted)
                
                if was_diffusion_training:
                    diffusion_model.train()

            # Supervised Contrastive Loss: 需手動開 --use_supcon，且 use_style_shift、有 z/z_s、且至少一筆有風格偏移
            loss_supcon = None
            use_supcon = getattr(args, 'use_supcon', False)
            if use_supcon and use_style_shift and z is not None and z_s is not None:
                with torch.no_grad():
                    diff_per_sample = (z - z_s).pow(2).sum(dim=1)
                    supcon_shifted_mask = diff_per_sample > 1e-8
                    n_supcon_shifted = supcon_shifted_mask.sum().item()
                if n_supcon_shifted > 0:
                    features_combined = torch.cat([z, z_s], dim=0)
                    targets_combined = torch.cat([target, target], dim=0)
                    loss_supcon = util.supervised_contrastive_loss(features_combined, targets_combined, temperature=supcon_temperature)
                    if loss_supcon_meters is not None:
                        loss_supcon_meters[domain].update(loss_supcon.item(), data.size(0))
            
            # SVD - 有效秩 (Effective Rank)：維度坍縮指標，能量集中於少數奇異值
            # 從一開始就監測 backbone 特徵的有效維度，而非僅在 Pair Loss 啟用後
            if svd_effective_rank_95 is not None and svd_top10_energy_ratio is not None and z is not None:
                with torch.no_grad():
                    z_centered = z - z.mean(dim=0, keepdim=True)
                    S = torch.linalg.svdvals(z_centered)
                    total = S.sum().clamp(min=1e-8)
                    cumsum = S.cumsum(0)
                    idx_95 = (cumsum / total >= 0.95).nonzero(as_tuple=True)[0]
                    eff_rank_95 = (idx_95[0].item() + 1) if len(idx_95) > 0 else len(S)
                    top10 = min(10, len(S))
                    top10_ratio = S[:top10].sum() / total
                    svd_effective_rank_95[domain].update(float(eff_rank_95), data.size(0))
                    svd_top10_energy_ratio[domain].update(top10_ratio.item(), data.size(0))
            
            # Total loss for backbone
            loss_total = loss_cls
            if loss_pair is not None:
                loss_total = loss_total + current_lambda_pair * loss_pair
            if loss_supcon is not None:
                loss_total = loss_total + lambda_supcon * loss_supcon

            # Record training loss and accuracy
            record_start = time.time()
            acc1 = util.comp_accuracy(output, target)
            losses_dict[domain].update(loss_total.item(), data.size(0))
            top1_dict[domain].update(acc1[0], data.size(0))
            record_end = time.time()

            # Backbone update (must be BEFORE Diffusion update to avoid in-place buffer overwrite)
            # Pair Loss uses normalize(z) which references shift/scale; Diffusion update's
            # normalize() modifies them in-place. backward() needs unmodified buffers.
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # Diffusion update (teacher self-learning, use clean features z)
            # Done AFTER backbone backward so FeatureNormalization buffers are not overwritten
            # before loss_total.backward() runs
            if use_ood and domain in diffusion_models_dict and z is not None:
                diffusion_model = diffusion_models_dict[domain]
                optimizer_diffusion = optimizers_diffusion_dict[domain]
                
                latents_for_diff = z.detach().requires_grad_(True)
                latents_normalized = diffusion_model.normalize(latents_for_diff)
                loss_diff = diffusion_model.get_loss_iter(latents_normalized)
                
                optimizer_diffusion.zero_grad()
                loss_diff.backward()
                optimizer_diffusion.step()
                
                if loss_diff_meters is not None:
                    loss_diff_meters[domain].update(loss_diff.item(), data.size(0))

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
            
            # 每輪模型交換聚合結束後存 checkpoint（若啟用）
            if getattr(args, 'save_every_epoch', False):
                save_dir = args.savePath if args.savePath is not None else "./checkpoints"
                os.makedirs(save_dir, exist_ok=True)
                for domain in domain_names:
                    model = models_dict[domain]
                    ckpt = {
                        "args": args,
                        "domain": domain,
                        "all_domains": domain_names,
                        "dataset": args.dataset,
                        "leave_out": getattr(args, "leave_out", None),
                        "epoch": epoch,
                        "backbone_state": model.state_dict(),
                    }
                    if hasattr(model, "diffusion_model") and model.diffusion_model is not None:
                        ckpt["diffusion_state"] = model.diffusion_model.state_dict()
                    path = os.path.join(save_dir, f"{args.description}_{domain}_latest.pth")
                    torch.save(ckpt, path)
                print(f"[Checkpoint] Saved latest (epoch {epoch}, post-comm) to {save_dir}")
            record_time = toc - tic  # includes everything
            epoch_time = comp_time + comm_time  # only important parts

            # 在測試前清理 GPU 緩存，釋放未使用的記憶體
            torch.cuda.empty_cache()
            
            # evaluate test accuracy and class separation (test + train) per domain
            test_accs = {}
            class_sep_intra = {}
            class_sep_inter = {}
            class_sep_intra_train = {}
            class_sep_inter_train = {}
            tsne_features_by_domain = {}
            tsne_train_features_by_domain = {}
            for domain in domain_names:
                train_loader, test_loader = domain_loaders[domain]
                model_eval = models_dict[domain]
                test_acc = util.test(model_eval, test_loader)
                test_accs[domain] = test_acc

                if not hasattr(model_eval, "intermediate_forward"):
                    continue

                # Test: 一次掃取得 centroid + 特徵（供 t-SNE），再一次掃算 D_intra；D_inter 由 centroid 算
                centroids, features_np, labels_np = util.get_centroids_and_features(model_eval, test_loader)
                tsne_features_by_domain[domain] = (features_np, labels_np)
                if centroids is not None:
                    class_sep_intra[domain] = util.compute_intra_distance(model_eval, test_loader, centroids)
                    class_sep_inter[domain] = util.compute_inter_distance(centroids)

                # Train: 同上，並存下 train 特徵供疊加 t-SNE 重用（不再多掃 train）
                centroids_tr, feat_train_np, lab_train_np = util.get_centroids_and_features(model_eval, train_loader)
                tsne_train_features_by_domain[domain] = (feat_train_np, lab_train_np)
                if centroids_tr is not None:
                    class_sep_intra_train[domain] = util.compute_intra_distance(model_eval, train_loader, centroids_tr)
                    class_sep_inter_train[domain] = util.compute_inter_distance(centroids_tr)

            torch.cuda.empty_cache()

            # t-SNE：使用上面已收集的 tsne_features_by_domain，不再多掃 test_loader
            tsne_every_epoch = getattr(args, 'tsne_every_epoch', 0)
            tsne_overlay_every_epoch = getattr(args, 'tsne_overlay_every_epoch', 50)
            if getattr(util, '_TSNE_AVAILABLE', False):
                save_dir = args.savePath if args.savePath is not None else "./checkpoints"
                tsne_dir = os.path.join(save_dir, "tsne")
                tsne_n_samples = getattr(args, 'tsne_n_samples', 500)
                # (1) 僅測試集的 t-SNE（原有）：依 tsne_every_epoch
                if tsne_every_epoch > 0 and epoch % tsne_every_epoch == 0:
                    for domain in domain_names:
                        if domain not in tsne_features_by_domain:
                            continue
                        features_np, labels_np = tsne_features_by_domain[domain]
                        save_path = os.path.join(tsne_dir, f"{args.description}_{domain}_epoch{epoch}.png")
                        util.draw_tsne_feature_distribution(
                            features_np, labels_np, save_path,
                            title=f"{domain} epoch {epoch}",
                            random_state=args.randomSeed,
                        )
                    print(f"[t-SNE] Saved feature distribution plots to {tsne_dir}")
                # (2) 疊加圖：每 tsne_overlay_every_epoch（預設 50）畫一次
                # 測試集雖是同一份（leave_out，如 cartoon），但特徵必須用「各自 domain 的模型」抽取，不可跨模型通用。
                # 算 centroid 時已存：tsne_train_features_by_domain[d]=d 模型看 d 訓練；tsne_features_by_domain[d]=d 模型看 leave_out 測試。
                if tsne_overlay_every_epoch > 0 and epoch % tsne_overlay_every_epoch == 0:
                    leave_out = getattr(args, 'leave_out', None)  # 測試集 domain 名稱（如 cartoon）
                    for source_domain in domain_names:
                        if source_domain not in tsne_train_features_by_domain or source_domain not in tsne_features_by_domain:
                            continue
                        # 只用「該 domain 模型」的特徵：train 與 test 皆來自 source_domain 的模型，不混用他域模型
                        feat_train_full, lab_train_full = tsne_train_features_by_domain[source_domain]
                        feat_test_full, lab_test_full = tsne_features_by_domain[source_domain]
                        feat_train, lab_train = util.subsample_stratified(
                            feat_train_full, lab_train_full,
                            tsne_n_samples, num_classes, random_state=args.randomSeed,
                        )
                        feat_test, lab_test = util.subsample_stratified(
                            feat_test_full, lab_test_full,
                            tsne_n_samples, num_classes, random_state=args.randomSeed,
                        )
                        if len(lab_train) == 0 or len(lab_test) == 0:
                            continue
                        test_domain_name = leave_out if leave_out else "test"
                        overlay_path = os.path.join(tsne_dir, f"{args.description}_overlay_{source_domain}_train_vs_{test_domain_name}_test_epoch{epoch}.png")
                        util.draw_tsne_train_test_overlay(
                            feat_train, lab_train, feat_test, lab_test,
                            overlay_path,
                            title=f"Train ({source_domain}) vs Test ({test_domain_name}) epoch {epoch}",
                            source_domain_name=source_domain,
                            target_domain_name=test_domain_name,
                            random_state=args.randomSeed,
                        )
                    print(f"[t-SNE] Saved overlay plots (every {tsne_overlay_every_epoch} epochs, stratified by class) to {tsne_dir}")
            elif tsne_every_epoch > 0 and epoch % tsne_every_epoch == 0:
                print("[t-SNE] Skipped (install matplotlib and scikit-learn to enable)")

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
                "comp_time": comp_time,
                "comm_time": comm_time,
                "epoch_time": epoch_time,
            }
            
            # Add per-domain metrics
            for domain in domain_names:
                log_dict[f"{domain}/loss"] = losses_dict[domain].avg.item() if hasattr(losses_dict[domain].avg, 'item') else float(losses_dict[domain].avg)
                log_dict[f"{domain}/train_acc"] = top1_dict[domain].avg.item() if hasattr(top1_dict[domain].avg, 'item') else float(top1_dict[domain].avg)
                log_dict[f"{domain}/test_acc"] = test_accs[domain].item() if hasattr(test_accs[domain], 'item') else float(test_accs[domain])
                # 特徵類別分離度：test / train 的類內、類間距離
                if domain in class_sep_intra:
                    log_dict[f"{domain}/class_sep/D_intra"] = float(class_sep_intra[domain])
                    log_dict[f"{domain}/class_sep/D_inter"] = float(class_sep_inter[domain])
                if domain in class_sep_intra_train:
                    log_dict[f"{domain}/class_sep/D_intra_train"] = float(class_sep_intra_train[domain])
                    log_dict[f"{domain}/class_sep/D_inter_train"] = float(class_sep_inter_train[domain])
            
            # Add Pair Loss if Diffusion-guided Pair Loss is enabled
            if loss_pair_meters is not None:
                for domain in domain_names:
                    log_dict[f"{domain}/loss_pair"] = loss_pair_meters[domain].avg.item() if hasattr(loss_pair_meters[domain].avg, 'item') else float(loss_pair_meters[domain].avg)
                avg_pair = sum(loss_pair_meters[d].avg for d in domain_names) / len(domain_names)
                log_dict["loss_pair"] = avg_pair.item() if hasattr(avg_pair, 'item') else float(avg_pair)

            # SupCon (Supervised Contrastive Loss) when use_style_shift
            if loss_supcon_meters is not None:
                for domain in domain_names:
                    log_dict[f"{domain}/loss_supcon"] = loss_supcon_meters[domain].avg.item() if hasattr(loss_supcon_meters[domain].avg, 'item') else float(loss_supcon_meters[domain].avg)
                avg_supcon = sum(loss_supcon_meters[d].avg for d in domain_names) / len(domain_names)
                log_dict["loss_supcon"] = avg_supcon.item() if hasattr(avg_supcon, 'item') else float(avg_supcon)
            
            # 指標二：風格濾除力 / 語義還原度 (Style Removal Power)，僅在三個 domain 時計算
            # 判讀：MSE(z,z_s_0_hat) < MSE(z,z_s) 代表 Diffusion 老師成功還原語義（卸妝能力）
            if style_removal_mse_z_zs is not None and style_removal_mse_z_zs0hat is not None:
                for domain in domain_names:
                    mse_z_zs = style_removal_mse_z_zs[domain].avg.item() if hasattr(style_removal_mse_z_zs[domain].avg, 'item') else float(style_removal_mse_z_zs[domain].avg)
                    mse_z_zs0hat = style_removal_mse_z_zs0hat[domain].avg.item() if hasattr(style_removal_mse_z_zs0hat[domain].avg, 'item') else float(style_removal_mse_z_zs0hat[domain].avg)
                    log_dict[f"{domain}/style_removal/MSE_z_zs"] = mse_z_zs
                    log_dict[f"{domain}/style_removal/MSE_z_zs0hat"] = mse_z_zs0hat
                    log_dict[f"{domain}/style_removal/removal_power"] = mse_z_zs - mse_z_zs0hat  # >0 代表還原成功
                avg_mse_z_zs = sum(style_removal_mse_z_zs[d].avg for d in domain_names) / len(domain_names)
                avg_mse_z_zs0hat = sum(style_removal_mse_z_zs0hat[d].avg for d in domain_names) / len(domain_names)
                log_dict["style_removal/MSE_z_zs"] = avg_mse_z_zs.item() if hasattr(avg_mse_z_zs, 'item') else float(avg_mse_z_zs)
                log_dict["style_removal/MSE_z_zs0hat"] = avg_mse_z_zs0hat.item() if hasattr(avg_mse_z_zs0hat, 'item') else float(avg_mse_z_zs0hat)
                log_dict["style_removal/removal_power"] = (avg_mse_z_zs - avg_mse_z_zs0hat).item() if hasattr(avg_mse_z_zs, 'item') else float(avg_mse_z_zs - avg_mse_z_zs0hat)
            
            # SVD - 有效秩 (Effective Rank)：維度坍縮，前10奇異值佔95%+代表特徵本質約10維
            if svd_effective_rank_95 is not None and svd_top10_energy_ratio is not None:
                for domain in domain_names:
                    log_dict[f"{domain}/svd/effective_rank_95"] = svd_effective_rank_95[domain].avg.item() if hasattr(svd_effective_rank_95[domain].avg, 'item') else float(svd_effective_rank_95[domain].avg)
                    log_dict[f"{domain}/svd/top10_energy_ratio"] = svd_top10_energy_ratio[domain].avg.item() if hasattr(svd_top10_energy_ratio[domain].avg, 'item') else float(svd_top10_energy_ratio[domain].avg)
                avg_eff_rank = sum(svd_effective_rank_95[d].avg for d in domain_names) / len(domain_names)
                avg_top10_ratio = sum(svd_top10_energy_ratio[d].avg for d in domain_names) / len(domain_names)
                log_dict["svd/effective_rank_95"] = avg_eff_rank.item() if hasattr(avg_eff_rank, 'item') else float(avg_eff_rank)
                log_dict["svd/top10_energy_ratio"] = avg_top10_ratio.item() if hasattr(avg_top10_ratio, 'item') else float(avg_top10_ratio)
            
            # Add diffusion loss if OOD detection is enabled
            if getattr(args, 'use_ood', False) and loss_diff_meters is not None:
                # ===== 为每个domain记录各自的diffusion监控指标 =====
                # 虽然denoiser参数会被聚合，但normalization buffers是独立的，所以每个domain的指标不同
                for domain in domain_names:
                    if domain in diffusion_models_dict:
                        diffusion_model = diffusion_models_dict[domain]
                        
                        # 为每个domain记录diffusion loss
                        log_dict[f"{domain}/diffusion_loss"] = loss_diff_meters[domain].avg.item() if hasattr(loss_diff_meters[domain].avg, 'item') else float(loss_diff_meters[domain].avg)
                        
                        # 预测噪声余弦相似度指标（每个domain独立）- 已關閉以減少雜亂
                        # if hasattr(diffusion_model.diffusion_process, '_last_snr_info'):
                        #     monitor_info = diffusion_model.diffusion_process._last_snr_info
                        #     if 'noise_pred_cosine' in monitor_info:
                        #         log_dict[f"{domain}/noise_pred_cosine/mean"] = monitor_info['noise_pred_cosine']
                        #         if 'noise_pred_cosine_std' in monitor_info:
                        #             log_dict[f"{domain}/noise_pred_cosine/std"] = monitor_info['noise_pred_cosine_std']
                        
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

            # reset recorders for next epoch
            comp_time, comm_time = 0, 0
            for domain in domain_names:
                losses_dict[domain].reset()
                top1_dict[domain].reset()
                if loss_diff_meters is not None:
                    loss_diff_meters[domain].reset()
                if loss_pair_meters is not None:
                    loss_pair_meters[domain].reset()
                if loss_supcon_meters is not None:
                    loss_supcon_meters[domain].reset()
                if style_removal_mse_z_zs is not None and style_removal_mse_z_zs0hat is not None:
                    style_removal_mse_z_zs[domain].reset()
                    style_removal_mse_z_zs0hat[domain].reset()
                if svd_effective_rank_95 is not None and svd_top10_energy_ratio is not None:
                    svd_effective_rank_95[domain].reset()
                    svd_top10_energy_ratio[domain].reset()
            tic = time.time()

    # ===== Save final models (backbone + diffusion + normalization stats) =====
    # We save one checkpoint per domain so that:
    # - Backbone (feature extractor + classifier head) weights are preserved per domain
    # - Diffusion model (denoiser + diffusion process + FeatureNormalization buffers) are preserved per domain
    # - Domain-specific FeatureNormalization statistics (mean/std/shift/scale) are not lost
    save_dir = args.savePath if args.savePath is not None else "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

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
    parser.add_argument('--save_every_epoch', action='store_true',
                        help='save checkpoint every epoch after model exchange (default: only save at end)')
    
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

    # ===== Cosine classifier head options =====
    parser.add_argument('--use_cosine_classifier', action='store_true',
                        help='use cosine classifier head (L2-normalized features and weights with scale s) instead of linear fc')
    parser.add_argument('--cosine_scale', type=float, default=30.0,
                        help='scale s for cosine classifier (default: 30.0)')
    parser.add_argument('--cosine_learn_scale', action='store_true',
                        help='make cosine classifier scale s a learnable parameter instead of fixed constant')

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

    # Pair Loss (Diffusion-guided) options
    parser.add_argument('--lambda_pair', type=float, default=0.2,
                        help='Final target weight for Pair Loss (default: 0.2)')
    parser.add_argument('--warmup_epochs', type=int, default=50,
                        help='Epoch when Pair Loss starts (default: 50)')
    parser.add_argument('--pair_warmup_end_epoch', type=int, default=100,
                        help='Epoch when Pair Loss weight reaches lambda_pair (linear warmup, default: 100)')
    parser.add_argument('--pair_t_min', type=int, default=15,
                        help='Min timestep for Pair Loss sampling (default: 15)')
    parser.add_argument('--pair_t_max', type=int, default=100,
                        help='Max timestep for Pair Loss sampling (default: 100)')

    # SupCon (Supervised Contrastive Loss) for representation alignment
    parser.add_argument('--use_supcon', action='store_true',
                        help='enable Supervised Contrastive Loss (requires --use_style_shift)')
    parser.add_argument('--lambda_supcon', type=float, default=0.1,
                        help='Weight for Supervised Contrastive Loss (default: 0.1)')
    parser.add_argument('--supcon_temperature', type=float, default=0.1,
                        help='Temperature tau for SupCon (default: 0.1)')

    # t-SNE feature distribution visualization
    parser.add_argument('--tsne_every_epoch', type=int, default=0,
                        help='Draw t-SNE feature plot every N epochs (0=disabled, default: 0)')
    parser.add_argument('--tsne_n_samples', type=int, default=500,
                        help='Max samples per domain for t-SNE overlay (train vs test, default: 500)')
    parser.add_argument('--tsne_overlay_every_epoch', type=int, default=50,
                        help='Draw t-SNE overlay (train+test) every N epochs (default: 50)')

    args = parser.parse_args()

    if not args.description:
        print('No experiment description, exit!')
        exit()

    # Validate PACS requirements
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
    else:
        # For other datasets, single domain
        num_domains = 1

    run(num_domains)  # 開始訓練

