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
cudnn.benchmark = False
cudnn.deterministic = True

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
        _topo_seed = args.topo_seed if args.topo_seed is not None else args.randomSeed
        subGraphs = util.select_graph(6, num_nodes=rgg_nodes, radius=args.rgg_radius, seed=_topo_seed, topology=args.topology)
        try:
            _n_edges = sum(len(m) for m in subGraphs)  # subGraphs = list of matchings(邊列表)
        except Exception:
            _n_edges = 'n/a'
        print(f"[TOPO] graphid6 RGG seed={_topo_seed} (topo_seed={args.topo_seed}, randomSeed={args.randomSeed}) "
              f"edges={_n_edges}", flush=True)
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
        # Open-Set DG: when --exclude_class is set, one PACS class is held out as the
        # unknown (never trained), so the classifier becomes (7-1)-way. person is the
        # last ImageFolder index (alphabetical) -> remaining labels 0..5 stay contiguous.
        num_classes = 7 - (1 if getattr(args, 'exclude_class', None) else 0)
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
    # ===== 聚合健康度診斷（D1–D4）初始化（純記錄、對訓練零副作用）=====
    diag = None
    if getattr(args, 'enable_agg_diag', False):
        if not getattr(args, 'use_ood', False):
            print("\033[91m[AGG-DIAG] --enable_agg_diag 需搭配 --use_ood（diffusion_model 才存在），診斷停用\033[0m")
        else:
            from dood.agg_diagnostics import AggDiagnostics
            t_probe = args.diag_t_probe if args.diag_t_probe >= 0 else getattr(args, 'diffusion_steps', 1000) // 2
            diag_dir = os.path.join(args.savePath, 'agg_diag')
            diag = AggDiagnostics(
                save_dir=diag_dir,
                node_to_domain=node_to_domain if node_to_domain else {d: d for d in domain_names},
                t_probe=t_probe,
                d4_every=args.diag_d4_every,
            )
            # 對每節點抽一批固定 ID probe 影像（存著重複使用，不隨訓練改變）。
            # 用獨立 iterator 取，不動 train_iters。
            for domain in domain_names:
                _tl, _ = domain_loaders[domain]
                _batch = next(iter(_tl))
                if getattr(args, 'use_fourier_aug', False):
                    # Fourier wrapper yields (raw_norm, aug_norm, target, meta); probe uses raw.
                    _data, _target = _batch[0], _batch[2]
                else:
                    _data, _target, _ = util.unpack_batch(_batch)
                _data = _data.cuda(non_blocking=True)[:args.diag_probe_bs]
                _target = _target.cuda(non_blocking=True)[:args.diag_probe_bs]
                diag.set_probe_images(domain, _data, _target)
            communicator.diag = diag
            print(f"[AGG-DIAG] enabled. t_probe={t_probe}, d4_every={args.diag_d4_every}, "
                  f"N={args.diag_every_n_epoch}, M={args.diag_every_m_epoch}, "
                  f"nodes={len(domain_names)}, dir={diag_dir}", flush=True)

    # init recorders for each domain
    comp_time, comm_time = 0, 0
    # Use domain index for recorder (compatible with existing Recorder interface)
    recorders = {domain: util.Recorder(args, domain_names.index(domain)) for domain in domain_names}
    losses_dict = {domain: util.AverageMeter() for domain in domain_names}
    top1_dict = {domain: util.AverageMeter() for domain in domain_names}
    # Fourier aug (Option B) Path F/S separation meters (R1b); unused/zero when fourier off
    fourier_acc_f_dict = {domain: util.AverageMeter() for domain in domain_names}
    fourier_loss_s_meters = {domain: util.AverageMeter() for domain in domain_names}
    # Gradient-conflict between Path S and Path F (cos<0 => fighting on backbone)
    fourier_g_cos_meters = {domain: util.AverageMeter() for domain in domain_names}
    fourier_g_s_norm_meters = {domain: util.AverageMeter() for domain in domain_names}
    fourier_g_f_norm_meters = {domain: util.AverageMeter() for domain in domain_names}
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
    hard_ood_loss_meters = {domain: util.AverageMeter() for domain in domain_names}
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
    # Terminal-only monitoring (not logged to wandb); enabled by --hard_adv_monitor_sparsity.
    z_hard_sparsity_meters = {domain: util.AverageMeter() for domain in domain_names}
    vec_hard_sparsity_meters = {domain: util.AverageMeter() for domain in domain_names}
    vec_clean_l4_sparsity_meters = {domain: util.AverageMeter() for domain in domain_names}
    vec_hard_vs_clean_l4_delta_meters = {domain: util.AverageMeter() for domain in domain_names}
    loss_diff_meters = {domain: util.AverageMeter() for domain in domain_names} if getattr(args, 'use_ood', False) else None
    # Accumulate batch-mean |grad_mu_cls| per channel for optional epoch dump (only when log_grad_parts batches run).
    grad_mu_cls_abs_ch_sum = {domain: None for domain in domain_names}
    grad_mu_cls_abs_ch_count = {domain: 0 for domain in domain_names}

    # ===== KSD generalization coupling meters (only meaningful when --use_ksd_reg) =====
    # Phase 1 (replaces V2-B-1 score-norm reg): pull the style-shifted (augmented) penultimate
    # toward the clean-source density using KSD with the FROZEN diffusion as score witness, plus
    # GRADIENT BALANCING that sets lambda so ||lam*g_ksd|| = ksd_grad_ratio * ||g_cls|| (breaks the
    # V2-B-1 ~4e-4 gradient damping). ksd_grad_ratio=0 => KSD-off control (same two-step forward,
    # no KSD loss) == the deterministic V1 73.29 setup.
    use_ksd_reg = bool(getattr(args, 'use_ksd_reg', False))
    ksd_grad_ratio = float(getattr(args, 'ksd_grad_ratio', 0.1))
    ksd_score_t = int(getattr(args, 'ksd_score_timestep', 25))
    ksd_balance_every = max(1, int(getattr(args, 'ksd_balance_every', 20)))
    # Lambda held between balance measurements (gradient balancing runs every ksd_balance_every steps).
    ksd_lambda_held = {domain: 0.0 for domain in domain_names}
    # Monitoring meters (KSD做動健康監控).
    loss_ksd_meters = {domain: util.AverageMeter() for domain in domain_names}
    ksd_lambda_meters = {domain: util.AverageMeter() for domain in domain_names}
    ksd_bw_meters = {domain: util.AverageMeter() for domain in domain_names}
    ksd_g_cls_norm_meters = {domain: util.AverageMeter() for domain in domain_names}
    ksd_g_ksd_norm_meters = {domain: util.AverageMeter() for domain in domain_names}
    ksd_g_cos_meters = {domain: util.AverageMeter() for domain in domain_names}
    ksd_achieved_ratio_meters = {domain: util.AverageMeter() for domain in domain_names}

    if use_ksd_reg:
        if not getattr(args, 'use_ood', False):
            raise RuntimeError("--use_ksd_reg requires --use_ood (needs diffusion score witness).")
        if args.model != 'res':
            raise RuntimeError("--use_ksd_reg currently supports only --model res.")
        if getattr(args, 'resnet_type', 'simplified') != 'standard':
            raise RuntimeError("--use_ksd_reg requires --resnet_type standard (needs forward_to_layer3_style/forward_from_layer3).")
        if getattr(args, 'use_hard_style_adv', False):
            raise RuntimeError("--use_ksd_reg is mutually exclusive with --use_hard_style_adv (clean A/B baseline).")
        print(
            f"[KSD] generalization coupling enabled. "
            f"grad_ratio={ksd_grad_ratio}, score_t={ksd_score_t} (low-sigma witness), "
            f"balance_every={ksd_balance_every} steps. grad_ratio=0 => KSD-off control (= V1 forward).",
            flush=True,
        )

    # ===== Stage 1: prototype detection readout (0803 §2.2/§2.6; 0810 implementation plan) =====
    use_proto_reg = bool(getattr(args, 'use_proto_reg', False))
    proto_m = float(getattr(args, 'proto_m', 0.95))
    proto_temp = float(getattr(args, 'proto_temp', 0.1))
    proto_warmup_epochs = int(getattr(args, 'proto_warmup_epochs', 10))
    proto_balance_every = max(1, int(getattr(args, 'proto_balance_every', 20)))
    proto_rel_margin = float(getattr(args, 'proto_rel_margin', 0.0))
    # target ratios: ||lam*g_term|| = ratio * ||g_cls||. All-zero => lambda=0 BASELINE arm.
    proto_ratios = {
        'comp':  float(getattr(args, 'proto_comp_ratio', 0.1)),
        'style': float(getattr(args, 'proto_style_ratio', 0.0)),
        'disp':  float(getattr(args, 'proto_disp_ratio', 0.1)),
        'rel':   float(getattr(args, 'proto_rel_ratio', 0.0)),
    }
    # lambda held between balance measurements (same pattern as KSD).
    proto_lambda_held = {d: {k: 0.0 for k in proto_ratios} for d in domain_names}

    proto_loss_meters = {k: {d: util.AverageMeter() for d in domain_names} for k in proto_ratios}
    proto_lambda_meters = {k: {d: util.AverageMeter() for d in domain_names} for k in proto_ratios}
    proto_gnorm_meters = {k: {d: util.AverageMeter() for d in domain_names} for k in proto_ratios}
    proto_gcos_meters = {k: {d: util.AverageMeter() for d in domain_names} for k in proto_ratios}
    proto_gcls_norm_meters = {d: util.AverageMeter() for d in domain_names}
    # ★ 觀察量①（0803 §4.1）：**同一樣本**乾淨 vs 風格擾動後的特徵夾角 —— 擾動把特徵推多遠，
    #   同時是 L_rel 的 margin 取值依據。concat 不需要，兩次前向本來就同時有兩者。
    proto_zangle_meters = {d: util.AverageMeter() for d in domain_names}
    # ★ 健康檢查：原型 vs fc 權重向量的夾角 —— 高度對齊 ⇒ 投影層沒學到與分類器不同的東西。
    proto_vs_fc_meters = {d: util.AverageMeter() for d in domain_names}

    if use_proto_reg:
        if args.model != 'res':
            raise RuntimeError("--use_proto_reg currently supports only --model res.")
        if getattr(args, 'resnet_type', 'simplified') != 'standard':
            raise RuntimeError(
                "--use_proto_reg requires --resnet_type standard "
                "(needs forward_to_layer3_style/forward_from_layer3/project).")
        if use_ksd_reg:
            raise RuntimeError("--use_proto_reg is mutually exclusive with --use_ksd_reg "
                               "(both claim the two-step style forward; keep a clean single-variable arm).")
        if proto_ratios['style'] != 0.0:
            # 每節點單域 ⇒ 類別中心 ≡ 該畫風原型 ⇒ L_style 與 L_comp 是同一個量（0803 §2.6.5）
            print("[proto][WARN] proto_style_ratio != 0 but stage 1 has one domain per node: "
                  "L_style is the SAME quantity as L_comp and will double-count. Set it to 0 "
                  "unless prototypes are already aggregated (stage 2b).", flush=True)
        from dood.prototype import init_prototype_buffers
        _pdim = (512 if bool(getattr(args, 'proto_no_projection', False))
                 else int(getattr(args, 'proj_dim', 128)))
        # ⚠️ 原型的第二軸是「來源畫風」不是「節點」——`domain_names` 在 9 節點設定下是
        #    ['node_0'..'node_8']（節點），真正的來源畫風要查 node_to_domain（3 個）。
        #    初版誤用 len(domain_names) ⇒ buffer 變成 [7,9,128]、cells_filled 7/63，
        #    由 smoke test 抓到。
        proto_domain_list = sorted(set(node_to_domain.values())) if node_to_domain \
            else sorted(domain_names)
        proto_dom_index = {d: i for i, d in enumerate(proto_domain_list)}
        for _d in domain_names:
            init_prototype_buffers(models_dict[_d], num_classes, len(proto_domain_list), _pdim,
                                   device=next(models_dict[_d].parameters()).device)
        _armtag = ("lambda=0 BASELINE" if all(v == 0.0 for v in proto_ratios.values())
                   else ("1c (no projection)" if getattr(args, 'proto_no_projection', False)
                         else ("1b (+L_rel)" if proto_ratios['rel'] > 0 else "1a")))
        print(f"[proto] stage-1 prototype readout enabled — arm {_armtag}. "
              f"buffers=[{num_classes},{len(proto_domain_list)},{_pdim}] "
              f"(第二軸＝來源畫風 {proto_domain_list}，不是節點) "
              f"proto_m={proto_m} tau={proto_temp} "
              f"warmup={proto_warmup_epochs}ep ratios={proto_ratios} rel_margin={proto_rel_margin}. "
              f"Forward count = 2 (perturbed + clean, BOTH grad-enabled since 0812: "
              f"L_disp acts on the stored prototypes via the clean-feature EMA chain).", flush=True)

    tic = time.time()

    use_fourier_aug = getattr(args, 'use_fourier_aug', False)

    # ===== Async event-triggered setup (Stage 1; replaces synchronous Phase-3) =====
    async_enabled = getattr(args, 'async_trigger', False)
    async_diag_f = None
    async_diag_writer = None
    async_recv = {}   # per-sweep: domain -> list of staleness ages (from R step)
    async_gamma_mode = getattr(args, 'trigger_gamma_mode', 'lr')
    async_style_enabled = getattr(args, 'async_style', False)
    if async_style_enabled and not async_enabled:
        raise RuntimeError("--async_style requires --async_trigger (style rides on the model-broadcast event).")
    _init_lr = args.lr
    if async_enabled:
        import os as _os, csv as _csv
        communicator.async_configure(threshold=args.trigger_threshold,
                                     max_interval=args.trigger_max_interval,
                                     buffer_max=getattr(args, 'async_buffer_max', None),
                                     async_style=async_style_enabled,
                                     aggregate_bn=getattr(args, 'aggregate_bn', False))
        communicator.async_init_snapshot(models_dict)
        _diag_dir = _os.path.join(args.savePath, 'async_diag')
        _os.makedirs(_diag_dir, exist_ok=True)
        async_diag_f = open(_os.path.join(_diag_dir, 'broadcast_log.csv'), 'w', newline='')
        async_diag_writer = _csv.writer(async_diag_f)
        async_diag_writer.writerow(['step', 'epoch', 'domain', 'delta', 'thresh', 'gamma', 'fired', 'forced',
                                    'n_pushed', 'n_received', 'mean_staleness', 'n_style_nb', 'mean_style_age',
                                    'mean_style_dist'])
        _stage = "Stage2: model+style async" if async_style_enabled else "Stage1: model-only, style stays sync"
        print(f"[ASYNC] event-trigger enabled: tau={args.trigger_threshold}, gamma_mode={async_gamma_mode}, "
              f"max_interval={args.trigger_max_interval} ({_stage})", flush=True)
        # Consensus-deviation trajectory (取代 async 下被 skip 的 sync D1)：每 epoch 一列
        async_consensus_f = open(_os.path.join(_diag_dir, 'consensus.csv'), 'w', newline='')
        async_consensus_writer = _csv.writer(async_consensus_f)
        async_consensus_writer.writerow(['epoch', 'consensus_rms', 'max_node_dev'])
    else:
        async_consensus_f = None
        async_consensus_writer = None

    # ===== start training with fixed total steps K (Algorithm 1) =====
    for k in range(K):
        # Set all models to training mode
        for model in models_dict.values():
            model.train()

        start_time = time.time()

        # Async delay-1：每個 k 開始把『上一輪』push 的風格 swap 進 active inbox（消同輪偷看；
        # 本輪稍後 async C push 的風格進 pending、下一輪才可見＝staleness≥1）。只在 async_style 分支。
        if async_enabled:
            communicator.swap_style_buffers()

        # ========== 第一阶段：计算所有 domain 的风格统计量（不训练）==========
        style_vecs_dict = {}
        # Cache one batch per domain so style stats and training share identical data
        # (matching train_mpi.py behavior).
        batch_cache = {}
        clean_layer3_cache = {}
        if (getattr(args, "use_style_stats", False) or getattr(args, "use_style_shift", False)) and args.model == "res":
            for domain in domain_names:
                batch = next(train_iters[domain])
                if use_fourier_aug:
                    # 4-tuple: (raw_norm, fourier_aug_norm, target, meta)
                    data, data_aug, target, meta = batch[0], batch[1], batch[2], batch[3]
                    data = data.cuda(non_blocking=True)
                    data_aug = data_aug.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                    batch_cache[domain] = (data, data_aug, target, meta)
                else:
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
        if style_vecs_dict and not getattr(args, 'isolated_nodes', False):
            if async_style_enabled:
                # Stage 2：風格不同步交換（跳過同步 barrier）。自風格改由 C 步 push 時直接夾帶
                # style_vecs_dict[domain]（不另存 communicator 狀態）；鄰居風格由持久 style_inbox 提供
                # （set_active_domain 即時建視圖）；空 buffer → StyleShift skip(R1)。
                pass
            else:
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

            # ===== Async R (RECEIVE) 已移到「訓練 + push 之後」（切斷同輪訓練接力；見 0717 報告 §4）=====
            # 本輪訓練改用「上一輪融合後」的起點，不站前面節點本輪剛訓練的肩膀上 → 消除收斂假快。

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

            # ⚠️ 原型「不」走 bn_states_dict 那套 save/restore（0810 plan §2.2e 原本規劃要加，
            #    實作時查證後撤回）：`models_dict[domain]` 是**每個域各自的 model 物件**
            #    （:262-283 逐一 select_model），原型 buffer 天生就是 per-node、沒有東西會覆蓋它。
            #    反而**加了才危險**——BN 那套的 restore 會用 dict 快照覆蓋模型現值，正是 0730
            #    「聚合後不寫回就靜默失效」的來源。階段 2 的原型聚合直接寫進 models_dict[domain]，
            #    不經過 dict 中轉 ⇒ 沒有這個坑。

            # Reuse phase-1 batch when style stats are enabled; otherwise fetch normally.
            if domain in batch_cache:
                if use_fourier_aug:
                    data, data_aug, target, batch_meta = batch_cache[domain]
                else:
                    data, target, batch_meta = batch_cache[domain]
            else:
                batch = next(train_iters[domain])
                if use_fourier_aug:
                    data, data_aug, target, batch_meta = batch[0], batch[1], batch[2], batch[3]
                    data = data.cuda(non_blocking=True)
                    data_aug = data_aug.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                else:
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
                need_clean_for_monitor = hard_adv_monitor_sparsity and (hard_adv_z_base == "clean")
                if need_clean_for_adv or need_clean_for_dump or need_clean_for_monitor:
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

                    # Restore training modes and requires_grad for outer loop.
                    if prev_backbone_training is not None:
                        backbone.train(prev_backbone_training)
                    model.train(prev_model_training)
                    diffusion_model.train(prev_diff_training)
                    for p in model.parameters():
                        p.requires_grad_(True)
                    for p in diffusion_model.parameters():
                        p.requires_grad_(True)

                    # Outer loop: z_hard re-composed from LIVE z_base-normalized content
                    # while using style-derived (mu/sigma) adversarial updates.
                    z_norm_live = (z_base - mu_base) / (sigma_base + adv_eps)
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
                        if hard_adv_monitor_sparsity and z_clean is not None:
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

                    loss_style_ce = criterion(logits_style, target)
                    hard_ce_vec = F.cross_entropy(logits_hard, target, reduction='none')
                    loss_hard_ce = hard_ce_vec.mean()
                    with torch.no_grad():
                        hard_ce_delta_vec = hard_ce_vec - L_cls_inner_vec.detach()
                        hard_ce_delta = float(hard_ce_delta_vec.mean().item())
                        hard_ce_harder_ratio = float((hard_ce_delta_vec > 0).float().mean().item())

                    # 線性 warmup：從 epoch 0 到 60，z_hard 權重從 0 → 0.5
                    # 之後維持 0.5；z_style 權重始終為 1 - w_hard。
                    warmup_epochs = 60.0
                    # 以當前 iteration 推出「目前是第幾個 epoch」（浮點數）
                    current_epoch = (k + 1) / float(STEPS_PER_EPOCH) if STEPS_PER_EPOCH > 0 else 0.0
                    hard_weight = 0.5 * min(max(current_epoch / warmup_epochs, 0.0), 1.0)
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
            elif use_fourier_aug and getattr(args, 'fourier_concat_forward', False):
                # arm5c (FOOGD-exact): ONE forward over cat([clean, fourier]) + ONE CE, so BN
                # sees the mixed clean+fourier batch statistics (matches FOOGD client_fedavg
                # data=cat([x_ori,x_s2o]) + CE). Contrast arm5b's TWO separate forwards
                # (0.5*CE_clean + 0.5*CE_fourier) where BN normalizes clean / fourier separately.
                # Loss math is equivalent; the only difference is forward/BN coupling.
                # No style / score-reg here (same flags-off config as arm5b). train-BN is implicit
                # (model is in train()). The two-forward fourier_grad probe does not apply.
                _n_clean = data.size(0)
                _combined = torch.cat([data, data_aug], dim=0)
                _logits_cat = model(
                    _combined,
                    return_blocks=False,
                    communicator=None,
                    debug_style_shift=False,
                    iter_num=k+1,
                    rank=domain_names.index(domain),
                )
                _target_cat = torch.cat([target, target], dim=0)
                loss = criterion(_logits_cat, _target_cat)
                # Split halves only for meters (clean = Path S analogue, fourier = Path F analogue).
                output = _logits_cat[:_n_clean]
                logits_clean = _logits_cat[_n_clean:]
                loss_clean_ce = criterion(logits_clean, target)
                clean_ce_meters[domain].update(float(loss_clean_ce.item()), data.size(0))
                with torch.no_grad():
                    acc_f = util.comp_accuracy(logits_clean, target)
                fourier_acc_f_dict[domain].update(acc_f[0], data.size(0))
                fourier_loss_s_meters[domain].update(float(criterion(output, target).item()), data.size(0))
                style_aug_activated = False
                style_aug_flag_meters[domain].update(0.0, data.size(0))
            else:
                # Original training path (logits only)
                # Isolate BN updates for clean branch: use eval() so BN running stats are
                # not updated, while keeping autograd enabled.
                # arm1 (--fourier_train_bn): for the Fourier Path F, do NOT freeze BN, so the
                # fourier features update BN running stats (matches FOOGD's train-BN fourier view).
                # This tests whether the eval-BN choice was handicapping Path F.
                freeze_clean_bn = not (use_fourier_aug and getattr(args, 'fourier_train_bn', False))
                backbone_for_clean = getattr(model, "backbone", None)
                prev_backbone_training_for_clean = None
                prev_model_training_for_clean = model.training
                if freeze_clean_bn:
                    if backbone_for_clean is not None:
                        prev_backbone_training_for_clean = backbone_for_clean.training
                        backbone_for_clean.eval()
                    else:
                        model.eval()
                try:
                    # Path F (Option B): Fourier-augmented input through the clean forward
                    # (communicator=None). BN frozen by default (eval-BN); train-BN under arm1.
                    logits_clean = model(
                        data_aug if use_fourier_aug else data,
                        return_blocks=False,
                        communicator=None,
                        debug_style_shift=False,
                        iter_num=k+1,
                        rank=domain_names.index(domain),
                    )
                finally:
                    if freeze_clean_bn:
                        if backbone_for_clean is not None and prev_backbone_training_for_clean is not None:
                            backbone_for_clean.train(prev_backbone_training_for_clean)
                        else:
                            model.train(prev_model_training_for_clean)
                loss_clean_ce = criterion(logits_clean, target)
                clean_ce_meters[domain].update(float(loss_clean_ce.item()), data.size(0))
                if use_fourier_aug:
                    # Path F train acc (R1b): does the model learn amplitude-invariance?
                    with torch.no_grad():
                        acc_f = util.comp_accuracy(logits_clean, target)
                    fourier_acc_f_dict[domain].update(acc_f[0], data.size(0))

                # vec_style_for_reg (= grad-enabled style-shifted penultimate = KSD's z_aug) is only
                # produced by the two-step forward; otherwise None. Triggered by KSD coupling OR the
                # passive --score_diag probe (both consume the same z_aug).
                vec_style_for_reg = None
                if use_ksd_reg or use_proto_reg or bool(getattr(args, 'score_diag', False)):
                    # Two-step forward so logits and vec_style_for_reg share the SAME z_style graph.
                    # Avoids re-running style aug (which is stochastic) for the KSD / probe input.
                    communicator.set_active_domain(domain)
                    z_style_pooled = model.forward_to_layer3_style(
                        data,
                        communicator=communicator if (use_style_stats or use_style_shift) else None,
                        debug_style_shift=debug_style_shift,
                        iter_num=k+1,
                        rank=domain_names.index(domain),
                    )
                    output, vec_style_for_reg = model.forward_from_layer3(z_style_pooled)
                elif (use_style_stats or use_style_shift) and args.model == "res":
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

                # ===== Phase 0 (--score_diag): passive diffusion-score DIRECTION diagnostic =====
                # Reuses vec_style_for_reg (the REAL style-shifted penultimate from the two-step
                # forward above) as z_style; z_clean from a passive clean forward. Checks whether the
                # diffusion score (∇log p of clean-source density) points the style-shifted feature
                # BACK toward the clean manifold (=> usable KSD witness for Phase 1). PASSIVE: logs
                # only, applies NO loss / no backward; diffusion frozen (no queue / BN / RNG change).
                if (bool(getattr(args, 'score_diag', False)) and vec_style_for_reg is not None
                        and domain in diffusion_models_dict
                        and int(getattr(args, 'score_diag_every', 50)) > 0
                        and (k % int(getattr(args, 'score_diag_every', 50)) == 0)):
                    try:
                        _dm = diffusion_models_dict[domain]
                        _dp = _dm.diffusion_process
                        _sigmas = _dp.sqrt_one_minus_alphas_cumprod.to(data.device)
                        _t_list = [int(s) for s in str(getattr(args, 'score_diag_t', '10,25,50')).split(',') if s.strip()]
                        _bb = getattr(model, 'backbone', None)
                        _prev_bb = _bb.training if _bb is not None else model.training
                        _prev_dm = _dm.training
                        (_bb.eval() if _bb is not None else model.eval())  # clean forward: no BN update / no styleshift
                        _dm.eval()                                          # don't touch normalize queue / denoiser BN
                        with torch.no_grad():
                            z_clean = model.intermediate_forward(data)      # clean penultimate (no style shift)
                            z_style = vec_style_for_reg.detach()            # REAL style-shifted penultimate (reused)
                            zc_n = _dm.normalize(z_clean)
                            zs_n = _dm.normalize(z_style)
                            _disp = zs_n - zc_n
                            _dispn = _disp.norm(dim=1).mean().item()
                            _act = 1.0 if style_aug_activated else 0.0
                            for _t in _t_list:
                                _t = max(1, min(_t, int(_dp.num_timesteps) - 1))
                                _sig = float(_sigmas[_t])
                                _tt = torch.full((z_clean.size(0),), _t, device=data.device, dtype=torch.long)
                                _sc = -_dm.denoiser(zc_n, _tt) / (_sig + 1e-8)
                                _ss = -_dm.denoiser(zs_n, _tt) / (_sig + 1e-8)
                                _nc = _sc.norm(dim=1).mean().item()
                                _ns = _ss.norm(dim=1).mean().item()
                                _cv = F.cosine_similarity(_ss, _disp, dim=1, eps=1e-8)
                                _cos = _cv.mean().item()
                                _fn = (_cv < 0).float().mean().item()
                                # console-only (NOT wandb): extra wandb.log calls would bump the
                                # global step and shift the V1 metrics' x-axis out of alignment with
                                # the baseline run. Parse [score_diag] from the log for Phase 0 analysis.
                                print(f"[score_diag] epoch={epoch_i_1based} domain={domain} t={_t} "
                                      f"sigma={_sig:.4f} sty_act={_act:.1f} disp={_dispn:.4f} "
                                      f"|s_clean|={_nc:.4f} |s_style|={_ns:.4f} ratio={_ns/(_nc+1e-8):.3f} "
                                      f"cos={_cos:+.4f} frac_cos_neg={_fn:.3f}", flush=True)
                        # ---- ratio_aug: z_aug->clean DIRECT-OBJECTIVE measure (Phase 1.5) ----
                        # ratio_aug = KSD(z_aug;q) / KSD(z_clean;q) under the frozen witness at the
                        # SAME t the coupling optimizes (ksd_score_timestep). ->1 = z_aug as in-dist as
                        # clean (objective met); >>1 = not pulled. The ratio cancels the diffusion-maturity
                        # confound within a run; decisive read = KSD-on vs ratio=0 control trajectory.
                        # compute_KSD uses autograd.grad (Term2/3) so this MUST run OUTSIDE the no_grad
                        # block above with grad-enabled inputs (flag_retain/create=False => value-only).
                        if int(getattr(args, 'score_diag_ksd', 1)) == 1:
                            try:
                                from dood.ksd import (
                                    compute_KSD, SE_kernel_multi, trace_SE_kernel_multi, median_heruistic,
                                )
                                _tk = max(1, min(int(getattr(args, 'ksd_score_timestep', 25)),
                                                 int(_dp.num_timesteps) - 1))
                                _sigk = float(_sigmas[_tk])
                                for _pp in _dm.parameters():
                                    _pp.requires_grad_(False)  # frozen witness: grad to z only, not params

                                def _score_fn_diag(z):
                                    z_n = _dm.normalize(z)
                                    _tt = torch.full((z.size(0),), _tk, device=z.device, dtype=torch.long)
                                    return -_dm.denoiser(z_n, _tt) / (_sigk + 1e-8)

                                _zc_g = z_clean.detach().clone().requires_grad_(True)
                                _za_g = z_style.detach().clone().requires_grad_(True)
                                _bwk = median_heruistic(_zc_g.detach(), _zc_g.detach())
                                _ksd_c = compute_KSD(_zc_g, _zc_g, _score_fn_diag, SE_kernel_multi,
                                                     trace_SE_kernel_multi, _bwk,
                                                     flag_U=True, flag_retain=False, flag_create=False)
                                _ksd_a = compute_KSD(_za_g, _za_g, _score_fn_diag, SE_kernel_multi,
                                                     trace_SE_kernel_multi, _bwk,
                                                     flag_U=True, flag_retain=False, flag_create=False)
                                _kc = float(_ksd_c.item()); _ka = float(_ksd_a.item())
                                print(f"[score_diag_ksd] epoch={epoch_i_1based} domain={domain} t={_tk} "
                                      f"ratio_aug={_ka/(_kc+1e-12):.4f} ksd_clean={_kc:.4f} "
                                      f"ksd_aug={_ka:.4f} bw={float(_bwk):.3f}", flush=True)
                            except Exception as _e2:
                                print(f"[score_diag_ksd][WARN] failed: {_e2}", flush=True)
                            finally:
                                for _pp in _dm.parameters():
                                    _pp.requires_grad_(True)
                        _dm.train(_prev_dm)
                        (_bb.train(_prev_bb) if _bb is not None else model.train(_prev_bb))
                    except Exception as _e:
                        print(f"[score_diag][WARN] failed: {_e}", flush=True)

                loss_cls = criterion(output, target)   # Path S (channel-stat aug) loss
                loss = loss_cls
                if use_fourier_aug:
                    # Option B: equal-weight Path S + Path F, single backward.
                    fourier_loss_s_meters[domain].update(float(loss_cls.item()), data.size(0))
                    # Gradient-conflict probe (do BEFORE finalizing/backward so we can
                    # autograd.grad each weighted path loss on the shared backbone).
                    # cos(g_S, g_F) < 0 => the two paths fight on the backbone (the −acc could
                    # be gradient cancellation, not "fourier useless"). reuses compute_grad_conflict.
                    if bool(getattr(args, 'fourier_grad_diag', False)) and (
                        int(getattr(args, 'fourier_grad_diag_every', 50)) > 0
                        and (k % int(getattr(args, 'fourier_grad_diag_every', 50)) == 0)
                    ):
                        from dood.score_reg_diagnostics import compute_grad_conflict
                        _bb_params = [p for p in model.parameters() if p.requires_grad]
                        try:
                            _gc = compute_grad_conflict(
                                loss_cls=0.5 * loss_cls,          # Path S contribution
                                loss_reg_weighted=0.5 * loss_clean_ce,  # Path F contribution
                                params=_bb_params,
                            )
                            fourier_g_s_norm_meters[domain].update(_gc['g_cls_norm'], 1)
                            fourier_g_f_norm_meters[domain].update(_gc['g_reg_norm'], 1)
                            fourier_g_cos_meters[domain].update(_gc['g_cos'], 1)
                        except Exception as _e:
                            print(f"[fourier][diag][WARN] grad_conflict failed: {_e}", flush=True)
                    loss = 0.5 * loss_cls + 0.5 * loss_clean_ce
                elif hard_adv_clean_loss_weight > 0.0:
                    loss = loss + hard_adv_clean_loss_weight * loss_clean_ce

                # ===== KSD generalization coupling (replaces V2-B-1 score-norm reg) =====
                # KSD pulls z_aug (= grad-enabled style-shifted penultimate) toward the clean-source
                # density; diffusion is a FROZEN witness (grad flows THROUGH the denoiser to z_aug, not
                # to its params). Gradient balancing sets lambda so ||lam*g_ksd|| = ratio*||g_cls||.
                if (use_ksd_reg and ksd_grad_ratio > 0.0
                        and vec_style_for_reg is not None and domain in diffusion_models_dict):
                    from dood.ksd import (
                        compute_KSD, SE_kernel_multi, trace_SE_kernel_multi, median_heruistic,
                    )
                    from dood.score_reg_diagnostics import compute_grad_conflict
                    diffusion_model = diffusion_models_dict[domain]
                    z_aug = vec_style_for_reg  # do NOT detach: KSD gradient must reach the backbone
                    prev_diff_training = diffusion_model.training
                    diffusion_model.eval()  # freeze normalize queue + denoiser BN (witness only)
                    for _p in diffusion_model.parameters():
                        _p.requires_grad_(False)  # frozen witness: grad flows through to z_aug, not params
                    try:
                        _dp = diffusion_model.diffusion_process
                        _t = max(1, min(int(ksd_score_t), int(_dp.num_timesteps) - 1))
                        _sig_t = float(_dp.sqrt_one_minus_alphas_cumprod[_t])

                        def _score_fn(z):
                            # grad-enabled witness direction field: score = -eps_pred / sigma at low t
                            z_n = diffusion_model.normalize(z)
                            _tt = torch.full((z.size(0),), _t, device=z.device, dtype=torch.long)
                            return -diffusion_model.denoiser(z_n, _tt) / (_sig_t + 1e-8)

                        bw = median_heruistic(z_aug.detach(), z_aug.detach())
                        loss_ksd = compute_KSD(
                            z_aug, z_aug, _score_fn, SE_kernel_multi, trace_SE_kernel_multi, bw,
                            flag_U=True, flag_retain=True, flag_create=True,  # MUST be True (grad to backbone)
                        )

                        # ---- Gradient balancing (key lever, every ksd_balance_every steps) ----
                        if (k % ksd_balance_every == 0) and ksd_grad_ratio > 0.0:
                            backbone_params = [p for p in model.parameters() if p.requires_grad]
                            gc = compute_grad_conflict(
                                loss_cls=loss_cls,
                                loss_reg_weighted=loss_ksd,  # raw g_ksd at lambda=1
                                params=backbone_params,
                            )
                            g_cls_n, g_ksd_n = gc['g_cls_norm'], gc['g_reg_norm']
                            ksd_lambda_held[domain] = float(ksd_grad_ratio * g_cls_n / (g_ksd_n + 1e-12))
                            ksd_g_cls_norm_meters[domain].update(g_cls_n, 1)
                            ksd_g_ksd_norm_meters[domain].update(g_ksd_n, 1)
                            ksd_g_cos_meters[domain].update(gc['g_cos'], 1)
                            ksd_achieved_ratio_meters[domain].update(
                                (ksd_lambda_held[domain] * g_ksd_n) / (g_cls_n + 1e-12), 1)

                        lam_eff = ksd_lambda_held[domain]
                        if lam_eff > 0.0:
                            loss = loss + lam_eff * loss_ksd

                        loss_ksd_meters[domain].update(float(loss_ksd.item()), data.size(0))
                        ksd_lambda_meters[domain].update(lam_eff, data.size(0))
                        ksd_bw_meters[domain].update(
                            float(bw.item()) if torch.is_tensor(bw) else float(bw), data.size(0))
                    except Exception as _e:
                        print(f"[KSD][WARN] coupling failed: {_e}", flush=True)
                    finally:
                        for _p in diffusion_model.parameters():
                            _p.requires_grad_(True)
                        diffusion_model.train(prev_diff_training)

                # ===== Stage 1: prototype detection readout (0803 §2.2/§2.6; 0810 plan §2.2) =====
                # 參照物要乾淨、被拉的要是擾動的：原型的 EMA 只吃「未擾動」特徵（否則 (類別,畫風)
                # 語意會被鄰居畫風污染、兩層結構與跨節點聚合全都失效）；三個損失拉的是「擾動後」
                # 特徵（否則從未訓練過「畫風變了要回到原位」，而 target 域正是一種沒見過的擾動）。
                if use_proto_reg and vec_style_for_reg is not None:
                    from dood.prototype import (
                        class_centers, comp_loss, style_loss,
                        disp_loss, rel_loss, detection_score, ema_prototypes_live,
                    )
                    # ⚠️ 這是「來源畫風」的索引、不是節點索引（domain_names 是 node_0..node_8）
                    _dom_idx = proto_dom_index[node_to_domain.get(domain, domain)]
                    z_aug = model.project(vec_style_for_reg)          # 擾動後、grad-enabled

                    # --- 乾淨前向（★ 2026-08-12 起帶梯度）---------------------------------
                    # ⚠️ 走「同一個」forward_to_layer3_style 但傳 communicator=None：同一段程式碼、
                    #    同一組 BN 模組、同一個 train 模式，只差風格模組沒觸發。
                    # ⚠️⚠️ 不可改用 model.eval()（見 :672-683 的舊做法）——eval 會讓 BN 改用 running
                    #    統計量，乾淨與擾動特徵就落在兩個不同的正規化座標系，等於在兩個座標系之間
                    #    拉扯。這與 0805 探針抓到的病同類（denoiser 已共識、正規化座標系未共識，
                    #    分歧差 5e6 倍）。正解是「維持 train()、暫時凍結 BN 的 momentum」。
                    # ⚠️ 因守衛 `use_style_shift and communicator is not None` 不成立，這次前向
                    #    完全不呼叫 random.random()、不跑任何風格模組 ⇒ 零亂數消耗
                    #    ⇒ λ=0 控制組與其他臂的亂數流逐位相同（0810 plan §1）。
                    # ★ 為什麼拿掉 no_grad（0812）：L_disp 必須直接作用在「存檔原型」上。原本的
                    #   影子原型只讓梯度從 z_aug 批平均流 ⇒ 推的不是存檔原型、通道還窄 8 倍。
                    #   代價＝乾淨前向要建計算圖（記憶體增加；bs=64 的 ResNet18 可承受）。
                    _bn_mom = {}
                    for _n, _m_ in model.named_modules():
                        if isinstance(_m_, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                            _bn_mom[_n] = _m_.momentum
                            _m_.momentum = 0.0
                    try:
                        _z3_clean = model.forward_to_layer3_style(
                            data, communicator=None, debug_style_shift=False,
                            iter_num=k + 1, rank=domain_names.index(domain),
                        )
                        _, _vec_clean = model.forward_from_layer3(_z3_clean)
                        z_clean_p = model.project(_vec_clean)          # ★ 帶計算圖
                    finally:
                        for _n, _m_ in model.named_modules():
                            if _n in _bn_mom:
                                _m_.momentum = _bn_mom[_n]

                    # --- 原型 EMA（逐樣本，對齊 CIDER losses.py:261-264）--------------------
                    # ★ 回傳的 _proto_live 與存檔原型**數值完全相同**，只差有沒有帶梯度：
                    #   存檔用 .detach()（給檢測分數、comp_loss、階段 2 聚合），
                    #   _proto_live 給 L_disp（梯度沿 EMA 鏈流回 z_clean_p → backbone）。
                    # ★ λ=0 baseline 也照常累積，確保程式路徑與亂數流與其他臂完全一致。
                    _proto_live, _count_live = ema_prototypes_live(
                        model.prototypes, model.proto_count, z_clean_p, target, _dom_idx, proto_m)
                    with torch.no_grad():
                        model.prototypes.copy_(_proto_live.detach())
                        model.proto_count.copy_(_count_live)

                    with torch.no_grad():
                        # 觀察量①：**同一個樣本**在「乾淨」與「風格擾動後」的特徵夾角
                        # （不是「同類不同樣本」之間的夾角——量的是擾動把特徵推多遠，
                        #   這才是 L_rel margin 的取值依據，也才能對照 §3.5 的張力）
                        _cos = (z_aug.detach() * z_clean_p).sum(dim=1).clamp(-1 + 1e-7, 1 - 1e-7)
                        proto_zangle_meters[domain].update(
                            float(torch.arccos(_cos).mean().item()), data.size(0))

                    # --- 三個原型損失 + 相對約束 -------------------------------------------
                    # warmup：訓練初期特徵近乎隨機 ⇒ 原型也近乎隨機 ⇒ 拉向隨機目標＝注入噪聲。
                    # 原型在 warmup 期間照常累積，只是不產生損失。
                    _warm_done = (epoch_i_1based > proto_warmup_epochs)
                    _centers = class_centers(model.prototypes, model.proto_count)
                    _terms = {}
                    if _warm_done:
                        if proto_ratios['comp'] > 0:
                            _terms['comp'] = comp_loss(z_aug, target, model.prototypes,
                                                       model.proto_count, proto_temp)
                        if proto_ratios['style'] > 0:
                            _terms['style'] = style_loss(z_aug, target, _centers)
                        if proto_ratios['disp'] > 0:
                            # ★ 直接作用在「存檔原型的帶梯度版本」上（_proto_live 與 model.prototypes
                            #   數值相同、只差有沒有梯度）⇒ 推的就是部署時量到的那組中心。
                            # ⚠️ 不可改用 _centers（來自 buffer、對參數是常數）⇒ 梯度恆為 0、該項純裝飾
                            #    ＝V2B1 失效模式（0810 smoke test 實測 ||g_disp||=0、lam=0）。
                            _terms['disp'] = disp_loss(
                                class_centers(_proto_live, _count_live), _count_live, proto_temp)
                        if proto_ratios['rel'] > 0:
                            _s_aug = detection_score(z_aug, _centers)
                            # ⚠️⚠️ 必須 detach：z_clean_p 自 0812 起帶梯度，若不切斷，
                            #    relu(s_aug - s_clean - margin) 可以靠「**把 s_clean 拉高**」降到 0
                            #    （讓原圖看起來更像 OOD），s_aug 完全不用動 ⇒ 退化解，
                            #    而且損失曲線照樣在降，是最難察覺的那種失效。
                            _s_cln = detection_score(z_clean_p.detach(), _centers)
                            _terms['rel'] = rel_loss(_s_aug, _s_cln, proto_rel_margin)

                    # --- 梯度範數平衡 -------------------------------------------------------
                    # ⚠️ g_cls 只算一次、各輔助項各算一次（不要對每項都呼叫 compute_grad_conflict
                    #    那種成對介面、會重複算 g_cls）。
                    # ⚠️ 每 N 步才量一次，lambda 在量測之間保持不變（階梯式，與 KSD 同樣式）。
                    # λ=0 臂也量 ||g_cls||（不需平衡，但跨臂要能比梯度尺度）。
                    # autograd.grad 不消耗亂數 ⇒ 不影響「λ=0 與其他臂亂數流逐位相同」。
                    #
                    # ★ 0812 修正：範數只在**兩股梯度共享的參數**上算（backbone，排除 fc 與 proj_head）。
                    #   CE 走不到 proj_head、原型損失走不到 fc，allow_unused 會把走不到的補零 ⇒ 舊寫法
                    #   等於「‖g_cls‖ 含 fc」對上「‖g_term‖ 含 proj_head」，兩個範數在不同的參數集合上
                    #   算，比值被各自獨有的 head 稀釋 ⇒ ratio=0.1 不等於「在共享 backbone 上佔 10%」。
                    _shared_params = [p for _pn, p in model.named_parameters()
                                      if p.requires_grad
                                      and not _pn.startswith('proj_head')
                                      and not _pn.startswith('backbone.fc')
                                      and not _pn.startswith('diffusion_model.')]

                    def _gnorm_flat(_loss, _params):
                        _g = torch.autograd.grad(_loss, _params, retain_graph=True,
                                                 create_graph=False, allow_unused=True)
                        return torch.cat([(g.detach().flatten() if g is not None
                                           else p.new_zeros(p.numel()))
                                          for g, p in zip(_g, _params)])

                    if (not _terms) and use_proto_reg and (k % proto_balance_every == 0):
                        try:
                            proto_gcls_norm_meters[domain].update(
                                float(_gnorm_flat(loss_cls, _shared_params).norm().item()), 1)
                        except Exception:
                            pass
                    if _terms and (k % proto_balance_every == 0):
                        try:
                            _f_cls = _gnorm_flat(loss_cls, _shared_params)
                            _gcls_n = float(_f_cls.norm().item())
                            proto_gcls_norm_meters[domain].update(_gcls_n, 1)
                            for _name, _t in _terms.items():
                                _f = _gnorm_flat(_t, _shared_params)
                                _gn = float(_f.norm().item())
                                proto_gnorm_meters[_name][domain].update(_gn, 1)
                                proto_lambda_held[domain][_name] = (
                                    (proto_ratios[_name] * _gcls_n / _gn) if _gn > 1e-12 else 0.0)
                                # ★ 0812 新增：梯度方向衝突。範數對了但方向相反的話，輔助損失
                                #   是在抵銷分類（0730 V2B1 只看範數沒看方向的教訓）。
                                _den = _gcls_n * _gn
                                proto_gcos_meters[_name][domain].update(
                                    float((_f_cls @ _f).item() / _den) if _den > 1e-12 else 0.0, 1)
                        except Exception as _e:
                            print(f"[proto][WARN] gradient balancing failed: {_e}", flush=True)

                    for _name, _t in _terms.items():
                        _lam = proto_lambda_held[domain][_name]
                        loss = loss + _lam * _t
                        proto_loss_meters[_name][domain].update(float(_t.item()), data.size(0))
                        proto_lambda_meters[_name][domain].update(_lam, data.size(0))

                    # --- 健康檢查：原型 vs fc 權重向量的夾角 --------------------------------
                    # 高度對齊 ⇒ 投影層沒學到與分類器不同的東西、檢測模組沒提供額外資訊。
                    # ⚠️ 僅在無投影層時維度才對得上（1c）；有投影層時 128 vs 512 不可直接比。
                    if (k % proto_balance_every == 0) and bool(getattr(args, 'proto_no_projection', False)):
                        with torch.no_grad():
                            _w = F.normalize(model.backbone.fc.weight, dim=1)      # [C, 512]
                            _alive = (model.proto_count > 0).any(dim=1)
                            if _alive.any():
                                _ang = torch.arccos(
                                    (_centers[_alive] * _w[_alive]).sum(dim=1).clamp(-1 + 1e-7, 1 - 1e-7))
                                proto_vs_fc_meters[domain].update(float(_ang.mean().item()), 1)

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
            
            # record training loss and accuracy
            record_start = time.time()
            acc1 = util.comp_accuracy(output, target)
            losses_dict[domain].update(loss.item(), data.size(0))
            top1_dict[domain].update(acc1[0], data.size(0))
            record_end = time.time()

            # backward pass for classification
            # IMPORTANT: this MUST run BEFORE the diffusion training block below.
            # V2-B-1 puts diffusion_model parameters inside the main loss graph
            # (via loss_reg = diffusion_model.get_loss_at_timestep(vec_norm, t_probe));
            # if diffusion training (optimizer_diffusion.step) updates those params
            # in-place first, loss.backward() detects the version mismatch and raises
            # "variable modified by inplace operation". Pre-V2-B-1 the order was reversed
            # because main loss did not touch diffusion params.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute diffusion loss if OOD detection is enabled
            # (Moved AFTER main optimizer.step() — see comment above.)
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

            # ===== Async C (CHECK): event-trigger broadcast after this node finishes training =====
            if async_enabled:
                if async_gamma_mode == 'lr':
                    _cur_lr = optimizers_dict[domain].param_groups[0]['lr']
                    _gamma = (_cur_lr / _init_lr) if _init_lr > 0 else 1.0
                else:
                    _gamma = 1.0
                fired, delta, forced, _thresh = communicator.should_broadcast(domain, model, k, gamma=_gamma)
                n_pushed = communicator.push_to_neighbors(
                    domain, model, k,
                    style_vec=(style_vecs_dict.get(domain) if async_style_enabled else None)
                ) if fired else 0
                # ===== Async R (RECEIVE): 融合移到「訓練 + push 之後」（切斷訓練接力；0717 報告 §4）=====
                # push 推的是「本輪訓練後、未融合」的自身成果；receive 後模型才含鄰居（下一輪訓練才用 → staleness≥1）。
                async_recv[domain] = communicator.receive_and_aggregate(domain, model, k)
                # ⚠️ aggregate_bn：把 receive 融合後的 BN 寫回 bn_states_dict。不做的話，下一輪
                # Phase-2 的 restore（本檔上方「Restore BatchNorm state for this domain」）會用
                # 「聚合前」的舊值覆蓋掉，聚合等於沒發生——且不會報錯（靜默失敗）。
                if getattr(args, 'aggregate_bn', False):
                    for _bn_name, _bn_mod in model.named_modules():
                        if isinstance(_bn_mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) \
                                and _bn_name in bn_states_dict[domain]:
                            bn_states_dict[domain][_bn_name]['running_mean'] = _bn_mod.running_mean.clone()
                            bn_states_dict[domain][_bn_name]['running_var'] = _bn_mod.running_var.clone()
                _ages = async_recv.get(domain, [])
                _mean_stale = (sum(_ages) / len(_ages)) if _ages else 0.0
                if async_style_enabled:
                    _n_style_nb, _mean_style_age = communicator.style_buffer_age(domain, k)
                    _mean_style_dist = communicator.style_buffer_dist(domain)
                else:
                    _n_style_nb, _mean_style_age, _mean_style_dist = 0, 0.0, 0.0
                async_diag_writer.writerow([k, (k // STEPS_PER_EPOCH) + 1, domain,
                                            f"{delta:.6e}", f"{_thresh:.6e}", f"{_gamma:.4f}",
                                            int(bool(fired)), int(bool(forced)),
                                            n_pushed, len(_ages), f"{_mean_stale:.2f}",
                                            _n_style_nb, f"{_mean_style_age:.2f}", f"{_mean_style_dist:.4f}"])
                if (k + 1) % STEPS_PER_EPOCH == 0:
                    async_diag_f.flush()

        # ========== 第三阶段：交换训练后的模型参数 ==========
        # Exchange updated model parameters after training step
        # Note: style_vecs_dict is None here since we only exchange model parameters (not style stats)
        if async_enabled:
            # Async mode: model aggregation已由 per-node R/C(inbox)處理；跳過同步 Phase-3。
            # 仍推進 communicator.iter 以維持風格交換(Phase1)的 active_flags 排程。
            communicator.iter += 1
            d_comm_time_after = 0.0  # 佔位（下方 progress print 會引用）；async 無同步聚合通訊時間
        elif getattr(args, 'isolated_nodes', False):
            # Isolated ablation: 跳過 Phase-3 模型聚合，各節點獨立訓練（風格交換也已在上方 skip）。
            # 仍推進 iter 保持 communicator 內部排程一致（風格已停用故不影響行為）。
            communicator.iter += 1
            d_comm_time_after = 0.0
        else:
            # 更新診斷 epoch 標記（聚合 hook 會在 communicate 內讀取）
            communicator.diag_epoch = (k // STEPS_PER_EPOCH) + 1
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

            # BN 跨節點分歧診斷（0628 定義 mean‖v−μ‖/‖μ‖；keep-local 於 ep200 為 0.341）。
            # aggregate_bn 開啟時此值應明顯低於 keep-local；若持平 ⇒ 聚合被 Phase-2 restore
            # 覆蓋掉（靜默失敗），必須在 dry-run 就抓出來。
            if getattr(args, 'aggregate_bn', False) or getattr(args, 'bn_div_diag', False):
                with torch.no_grad():
                    _vecs = []
                    for _d in domain_names:
                        _bd = dict(models_dict[_d].named_buffers())
                        _vecs.append(torch.cat([_bd[_n].flatten().float() for _n in sorted(_bd)
                                                if _n.endswith(("running_mean", "running_var"))]))
                    _V = torch.stack(_vecs, 0)
                    _mu = _V.mean(0)
                    _div = (torch.norm(_V - _mu, dim=1).mean() / (torch.norm(_mu) + 1e-12)).item()
                    print(f"[BN-DIV] epoch={(k+1)//STEPS_PER_EPOCH} cross-node BN divergence={_div:.6f}", flush=True)

            # Async consensus-deviation 診斷（每 epoch）：看 const/高τ 末期是否 drift
            if async_enabled and async_consensus_writer is not None:
                _cons_rms, _cons_max = communicator.consensus_deviation(models_dict)
                async_consensus_writer.writerow([(k // STEPS_PER_EPOCH) + 1,
                                                 f"{_cons_rms:.6e}", f"{_cons_max:.6e}"])
                async_consensus_f.flush()
        
        end_time = time.time()
        d_comp_time = (end_time - start_time - (record_end - record_start))
        comp_time += d_comp_time

        # 每隔一定次數清理 GPU 緩存，避免記憶體累積
        if (k + 1) % 20 == 0:
            torch.cuda.empty_cache()

        # Print progress (average across domains).
        # 只在 TTY(互動終端)顯示自我覆蓋(\r)的 per-step 進度條；重導到 log 檔時(isatty=False)不印，
        # 避免 K 步進度行灌爆 log(每 epoch 的 `Epoch N:` 摘要才是分析主軸)。不動 RNG、不影響決定論。
        if sys.stdout.isatty():
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

            # ===== KSD coupling: epoch summary (做動健康監控) =====
            if use_ksd_reg:
                _avg = lambda m: sum(float(m[d].avg) for d in domain_names) / max(1, len(domain_names))
                _style_rate_avg = sum(float(style_aug_flag_meters[d].avg) for d in domain_names) / max(1, len(domain_names))
                _ksd_msg = (
                    f"  [KSD] loss_ksd={_avg(loss_ksd_meters):.4f}  lambda={_avg(ksd_lambda_meters):.4g}  "
                    f"bw={_avg(ksd_bw_meters):.4g}  achieved_ratio={_avg(ksd_achieved_ratio_meters):.3f}  "
                    f"style_aug_rate={_style_rate_avg*100:.1f}%"
                    f"  | grad_cos(cls,ksd)={_avg(ksd_g_cos_meters):+.3f}  "
                    f"||g_cls||={_avg(ksd_g_cls_norm_meters):.3f}  ||g_ksd_raw||={_avg(ksd_g_ksd_norm_meters):.4g}"
                )
                print(_ksd_msg, flush=True)

            # ===== 階段 1 原型讀出：epoch 摘要（0810 plan §4.1/§4.3 的觀察量）=====
            if use_proto_reg:
                _pavg = lambda m: sum(float(m[d].avg) for d in domain_names) / max(1, len(domain_names))
                _srate = sum(float(style_aug_flag_meters[d].avg) for d in domain_names) / max(1, len(domain_names))
                _gcls = _pavg(proto_gcls_norm_meters)
                # ★ 觀察量②：三項梯度範數「分開」記 —— 不要只記合併值，否則泛化掉時無法歸因
                #    （只有拉緊項跟 MixStyle 對打；推開項作用在類別中心、與畫風無關）。
                #    ⚠️ V2B1 教訓：當時 ||g_reg|| 僅 ||g_cls|| 的 0.1%、grad_cos≈0 ⇒ 根本沒動 backbone。
                _gterms = "  ".join(
                    f"{n}:L={_pavg(proto_loss_meters[n]):.4f}/lam={_pavg(proto_lambda_meters[n]):.3g}"
                    f"/||g||={_pavg(proto_gnorm_meters[n]):.4g}"
                    f"/ratio={(_pavg(proto_lambda_meters[n]) * _pavg(proto_gnorm_meters[n]) / _gcls if _gcls > 1e-12 else 0.0):.3f}"
                    f"/cos={_pavg(proto_gcos_meters[n]):+.3f}"
                    for n in ('comp', 'style', 'disp', 'rel') if proto_ratios[n] > 0
                )
                _warm = "WARMUP(losses off)" if epoch <= proto_warmup_epochs else "active"
                _filled = int((models_dict[domain_names[0]].proto_count > 0).sum().item())
                print(
                    f"  [proto] {_warm}  ||g_cls||={_gcls:.3f}  style_aug_rate={_srate*100:.1f}%  "
                    f"cells_filled={_filled}/{models_dict[domain_names[0]].proto_count.numel()}  "
                    # ★ 觀察量①：同類樣本 z 與 z~ 的夾角 —— 被壓小 ⇒ 拉緊項在抵銷風格增強；
                    #   同時是 L_rel margin 的取值依據（0803 §4.1-R）。
                    f"angle(z,z~)={_pavg(proto_zangle_meters):.4f}rad"
                    + (f"  angle(proto,fc_w)={_pavg(proto_vs_fc_meters):.4f}rad"
                       if getattr(args, 'proto_no_projection', False) else "")
                    + (f"\n         {_gterms}" if _gterms else "  [lambda=0 BASELINE: 原型照常累積、無損失]"),
                    flush=True,
                )

            # ===== 聚合診斷 D2/D3（epoch 級）=====
            if diag is not None:
                if args.diag_every_n_epoch > 0 and epoch % args.diag_every_n_epoch == 0:
                    diag.log_d2(models_dict, epoch)
                if args.diag_every_m_epoch > 0 and epoch % args.diag_every_m_epoch == 0:
                    diag.log_d3(models_dict, epoch)
            for _d in domain_names:
                print(
                    f"[confidence] epoch={epoch} domain={_d} "
                    f"clean_train={float(clean_conf_meters[_d].avg):.4f} "
                    f"style_train={float(style_conf_meters[_d].avg):.4f} "
                    f"test={float(test_confidences[_d]):.4f} "
                    f"ce_cos={float(clean_style_ce_cos_meters[_d].avg):.4f} "
                    f"sym_kl={float(clean_style_sym_kl_meters[_d].avg):.6f}"
                )
            if getattr(args, 'fourier_grad_diag', False):
                for _d in domain_names:
                    print(
                        f"[fourier_grad] epoch={epoch} domain={_d} "
                        f"grad_cos_SF={float(fourier_g_cos_meters[_d].avg):+.4f} "
                        f"g_S_norm={float(fourier_g_s_norm_meters[_d].avg):.4f} "
                        f"g_F_norm={float(fourier_g_f_norm_meters[_d].avg):.4f} "
                        f"acc_F={float(fourier_acc_f_dict[_d].avg):.2f}"
                    )
            if getattr(args, "use_hard_style_adv", False):
                for _d in domain_names:
                    print(
                        f"[hard_ood] epoch={epoch} domain={_d} "
                        f"hard_ood_loss={float(hard_ood_loss_meters[_d].avg):.6f} "
                        f"inner_ood_loss={float(adv_ood_loss_meters[_d].avg):.6f}"
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
                if use_fourier_aug:
                    # Path F / Path S separation (R1b). clean_ce == loss_F; train_acc == Path S acc.
                    _acc_f = fourier_acc_f_dict[domain].avg
                    log_dict[f"{domain}/fourier_acc_F"] = _acc_f.item() if hasattr(_acc_f, 'item') else float(_acc_f)
                    log_dict[f"{domain}/fourier_loss_S"] = float(fourier_loss_s_meters[domain].avg)
                    log_dict[f"{domain}/fourier_loss_F"] = float(clean_ce_meters[domain].avg)
                    if getattr(args, 'fourier_grad_diag', False):
                        log_dict[f"{domain}/fourier_grad_cos_SF"] = float(fourier_g_cos_meters[domain].avg)
                        log_dict[f"{domain}/fourier_g_S_norm"] = float(fourier_g_s_norm_meters[domain].avg)
                        log_dict[f"{domain}/fourier_g_F_norm"] = float(fourier_g_f_norm_meters[domain].avg)
                log_dict[f"{domain}/style_ce"] = float(style_ce_meters[domain].avg)
                log_dict[f"{domain}/hard_ce"] = float(hard_ce_meters[domain].avg)
                log_dict[f"{domain}/hard_ce_delta"] = float(hard_ce_delta_meters[domain].avg)
                log_dict[f"{domain}/hard_ce_harder_ratio"] = float(hard_ce_harder_ratio_meters[domain].avg)
                log_dict[f"{domain}/hard_ood_loss"] = float(hard_ood_loss_meters[domain].avg)
                log_dict[f"{domain}/adv_ood_loss"] = float(adv_ood_loss_meters[domain].avg)
                log_dict[f"{domain}/adv_cls_inner"] = float(adv_cls_inner_meters[domain].avg)
                log_dict[f"{domain}/mu_orig_norm"] = float(mu_orig_norm_meters[domain].avg)
                log_dict[f"{domain}/sigma_orig_norm"] = float(sigma_orig_norm_meters[domain].avg)
                log_dict[f"{domain}/grad_mu_norm"] = float(grad_mu_norm_meters[domain].avg)
                log_dict[f"{domain}/grad_sigma_norm"] = float(grad_sigma_norm_meters[domain].avg)
                log_dict[f"{domain}/rel_mu_step"] = float(rel_mu_step_meters[domain].avg)
                log_dict[f"{domain}/rel_sigma_step"] = float(rel_sigma_step_meters[domain].avg)
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

                # KSD coupling metrics (做動健康監控; only meaningful when --use_ksd_reg)
                if use_ksd_reg:
                    log_dict[f"{domain}/loss_ksd"] = float(loss_ksd_meters[domain].avg)
                    log_dict[f"{domain}/ksd_lambda"] = float(ksd_lambda_meters[domain].avg)
                    log_dict[f"{domain}/ksd_bw"] = float(ksd_bw_meters[domain].avg)
                    log_dict[f"{domain}/ksd_achieved_ratio"] = float(ksd_achieved_ratio_meters[domain].avg)
                    log_dict[f"{domain}/style_aug_activated_rate"] = float(style_aug_flag_meters[domain].avg)
                    log_dict[f"{domain}/ksd_g_cls_norm"] = float(ksd_g_cls_norm_meters[domain].avg)
                    log_dict[f"{domain}/ksd_g_ksd_norm"] = float(ksd_g_ksd_norm_meters[domain].avg)
                    log_dict[f"{domain}/ksd_g_cos"] = float(ksd_g_cos_meters[domain].avg)
                if use_proto_reg:
                    # ★ 三項梯度範數分開記（觀察量②）＋ 夾角軌跡（觀察量①）
                    for _k in proto_ratios:
                        log_dict[f"{domain}/proto_loss_{_k}"] = float(proto_loss_meters[_k][domain].avg)
                        log_dict[f"{domain}/proto_lambda_{_k}"] = float(proto_lambda_meters[_k][domain].avg)
                        log_dict[f"{domain}/proto_gnorm_{_k}"] = float(proto_gnorm_meters[_k][domain].avg)
                        log_dict[f"{domain}/proto_gcos_{_k}"] = float(proto_gcos_meters[_k][domain].avg)
                    log_dict[f"{domain}/proto_g_cls_norm"] = float(proto_gcls_norm_meters[domain].avg)
                    log_dict[f"{domain}/proto_angle_z_zaug"] = float(proto_zangle_meters[domain].avg)
                    log_dict[f"{domain}/proto_angle_vs_fc"] = float(proto_vs_fc_meters[domain].avg)
                    log_dict[f"{domain}/proto_cells_filled"] = int(
                        (models_dict[domain].proto_count > 0).sum().item())

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
                fourier_acc_f_dict[domain].reset()
                fourier_loss_s_meters[domain].reset()
                fourier_g_cos_meters[domain].reset()
                fourier_g_s_norm_meters[domain].reset()
                fourier_g_f_norm_meters[domain].reset()
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
                hard_ood_loss_meters[domain].reset()
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
                if bool(getattr(args, "hard_adv_monitor_sparsity", False)):
                    z_hard_sparsity_meters[domain].reset()
                    vec_hard_sparsity_meters[domain].reset()
                    vec_clean_l4_sparsity_meters[domain].reset()
                    vec_hard_vs_clean_l4_delta_meters[domain].reset()
                if loss_diff_meters is not None:
                    loss_diff_meters[domain].reset()
                # KSD coupling meters
                loss_ksd_meters[domain].reset()
                ksd_lambda_meters[domain].reset()
                ksd_bw_meters[domain].reset()
                ksd_g_cls_norm_meters[domain].reset()
                ksd_g_ksd_norm_meters[domain].reset()
                ksd_g_cos_meters[domain].reset()
                ksd_achieved_ratio_meters[domain].reset()
                # Stage-1 prototype meters
                if use_proto_reg:
                    for _k in proto_ratios:
                        proto_loss_meters[_k][domain].reset()
                        proto_lambda_meters[_k][domain].reset()
                        proto_gnorm_meters[_k][domain].reset()
                        proto_gcos_meters[_k][domain].reset()
                    proto_gcls_norm_meters[domain].reset()
                    proto_zangle_meters[domain].reset()
                    proto_vs_fc_meters[domain].reset()
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

    # 關閉診斷 CSV
    if diag is not None:
        diag.close()


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
    parser.add_argument('--rgg_radius', default=0.8, type=float, help='RGG connection radius for graphid=6 (default 0.8=dense; lower=sparser, 1-hop may miss domains)')
    parser.add_argument('--topology', default='rgg', choices=['rgg', 'ring'], help='graphid=6 topology: rgg (random geometric, default) or ring (9-node cycle, degree 2, most-sparse connected)')
    parser.add_argument('--isolated_nodes', action='store_true', help='ablation: each node trains fully independently. Skips BOTH Phase-1 style exchange and Phase-3 model aggregation, so no neighbor style is received (StyleShift skips on empty buffer) and no model averaging happens. Functionally equals a disconnected topology; topology object stays valid (dense) but its aggregation is never invoked. sync-only.')

    parser.add_argument('--dataset', default='cifar10', type=str, help='the dataset')
    parser.add_argument('--datasetRoot', type=str, help='the path of dataset')
    parser.add_argument('--leave_out', type=str, default=None, help='leave out domain for PACS dataset (art_painting, cartoon, photo, sketch)')
    parser.add_argument('--exclude_class', type=str, default=None, help='Open-Set DG: PACS class name held out as the unknown (never trained), e.g. "person". Filtered from TRAIN sets only; test sets keep all classes for OSCR/H-score eval. Only the last-index class (person) is supported without relabel.')
    parser.add_argument('--p', '-p', action='store_true', help='partition the dataset or not')
    parser.add_argument('--savePath' ,type=str, help='save path')
    parser.add_argument('--save_every_epoch', type=int, default=50,
                        help='save checkpoints every N epochs during training (0 to disable)')
    
    parser.add_argument('--compress', action='store_true', help='use chocoSGD or not')    
    parser.add_argument('--consensus_lr', default=0.1, type=float, help='consensus_lr')
    parser.add_argument('--randomSeed', type=int, help='random seed')
    parser.add_argument('--topo_seed', type=int, default=None,
                        help='拓樸生成專用 seed（RGG）；不設則退回用 randomSeed（向後相容）。'
                             '設此可固定拓樸、只讓 randomSeed 變訓練 RNG（拆 topo/train seed 混淆）。')
    parser.add_argument('--total_iter', type=int, help='total training iterations (if not set, uses epoch * max_steps_per_epoch)')
    parser.add_argument('--wandb_project', default='MATCHA', type=str, help='wandb project name')

    # ===== 聚合健康度診斷（D1–D4），純記錄、零 sys.exit；需搭配 --use_ood =====
    parser.add_argument('--enable_agg_diag', action='store_true',
                        help='enable diffusion aggregation health diagnostics (D1-D4 logger)')
    parser.add_argument('--diag_t_probe', type=int, default=-1,
                        help='fixed diffusion timestep for D2/D4 DSM loss (-1 => diffusion_steps//2)')
    parser.add_argument('--diag_every_n_epoch', type=int, default=1,
                        help='D2 (aggregated-model ID NLL) every N epochs')
    parser.add_argument('--diag_every_m_epoch', type=int, default=20,
                        help='D3 (cross-node heterogeneity matrix) every M epochs (M>>N)')
    parser.add_argument('--diag_d4_every', type=int, default=1,
                        help='D4 (pre/post-agg dL) every K communication rounds (D1 every round)')
    parser.add_argument('--diag_probe_bs', type=int, default=64,
                        help='batch size of the fixed ID probe images cached for D2/D3/D4')

    # ===== Fourier amplitude augmentation (Option B, Path F) =====
    parser.add_argument('--use_fourier_aug', action='store_true',
                        help='enable image-level Fourier amplitude augmentation (within-node partner). '
                             'Adds Path F (fourier aug, channel-stat OFF) alongside Path S; loss = 0.5*S + 0.5*F')
    parser.add_argument('--fourier_alpha', type=float, default=1.0,
                        help='lambda ~ U(0, alpha) amplitude mixing range (FOOGD default 1.0)')
    parser.add_argument('--fourier_ratio', type=float, default=1.0,
                        help='fraction of centered low-freq band to mix (1.0 = full spectrum)')
    parser.add_argument('--fourier_workers', type=int, default=6,
                        help='DataLoader workers for the Fourier-aug train loader (CPU FFT parallelism). '
                             '0 = main process (slow). Deterministic via fourier_worker_init_fn.')
    parser.add_argument('--train_workers', type=int, default=0,
                        help='DataLoader workers for the NON-fourier train loader. Default 0 = main '
                             'process (baseline behavior). arm5a (pure-ERM control) sets this = '
                             'fourier_workers so it nw-matches arm5b and arm5b-arm5a isolates fourier '
                             '(not the nw=6-vs-0 dataloader confound).')
    parser.add_argument('--fourier_train_bn', action='store_true',
                        help='arm1: let Path F (fourier) forward update BN running stats (train-BN) '
                             'instead of the default eval-BN. Tests if eval-BN handicapped fourier '
                             '(FOOGD fourier view uses train-BN). Only effective with --use_fourier_aug.')
    parser.add_argument('--fourier_concat_forward', action='store_true',
                        help='arm5c (FOOGD-exact): single forward over cat([clean, fourier]) + single '
                             'CE, so BN sees the mixed clean+fourier batch statistics (matches FOOGD '
                             'client_fedavg data=cat([x_ori,x_s2o])+CE). Replaces arm5b two separate '
                             'forwards (0.5*CE_clean+0.5*CE_fourier, separate BN). The only difference '
                             'vs arm5b is the forward/BN structure (loss math is equivalent). '
                             'Only with --use_fourier_aug; ignores --fourier_train_bn (implicit train-BN).')
    parser.add_argument('--fourier_grad_diag', action='store_true',
                        help='probe backbone gradient conflict cos(grad_PathS, grad_PathF) + norms '
                             'every --fourier_grad_diag_every steps (cos<0 => paths fight).')
    parser.add_argument('--fourier_grad_diag_every', type=int, default=50,
                        help='step interval for the Fourier gradient-conflict probe (default 50).')

    # ===== KSD Phase 0: passive diffusion-score direction diagnostic (no loss applied) =====
    parser.add_argument('--score_diag', action='store_true',
                        help='Phase 0: passively log the diffusion score direction on clean vs '
                             'style-shifted features — ||score(z_clean)|| vs ||score(z_style)|| and '
                             'cos(score(z_style), z_style - z_clean). Tests if the score is a usable '
                             'KSD witness (points style-shifted feats back to clean density). '
                             'Computes & logs ONLY, applies NO loss / no backward. Requires --use_ood.')
    parser.add_argument('--score_diag_every', type=int, default=50,
                        help='step interval for the --score_diag probe (default 50).')
    parser.add_argument('--score_diag_t', type=str, default='10,25,50',
                        help='comma-separated low diffusion timesteps to evaluate score at '
                             '(sweep to find sigma~0.1; sigma printed per t).')
    parser.add_argument('--score_diag_ksd', type=int, default=1,
                        help='Phase 1.5: also log ratio_aug = KSD(z_aug;q)/KSD(z_clean;q) at the KSD '
                             'witness t (ksd_score_timestep) inside the --score_diag probe — the DIRECT '
                             'objective (did style-shifted z_aug get pulled to clean density). 1=on '
                             '(default, follows --score_diag), 0=off (skip the extra compute_KSD cost).')

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
    parser.add_argument('--hard_adv_z_base', type=str, default='clean', choices=['clean', 'style'],
                        help='base feature map used to compose z_hard content: clean (default) or style')
    parser.add_argument('--hard_adv_clean_loss_weight', type=float, default=0.0,
                        help='extra CE weight on clean logits from model(data, communicator=None); 0 disables (default: 0.0)')
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

    # ===== KSD generalization coupling (Phase 1; replaces V2-B-1 score-norm reg) =====
    parser.add_argument('--use_ksd_reg', action='store_true',
                        help='KSD coupling: pull z_aug (style-shifted penultimate) toward clean-source density '
                             'via KSD with the FROZEN diffusion as score witness. Forces the '
                             'forward_to_layer3_style + forward_from_layer3 two-step forward (grad-enabled z_aug). '
                             'Requires --use_ood and resnet_type=standard.')
    parser.add_argument('--ksd_grad_ratio', type=float, default=0.1,
                        help='KSD gradient balancing target: set lambda so ||lam*g_ksd|| = ratio*||g_cls|| '
                             '(default 0.1). ratio=0 => KSD-off control (same two-step forward, no KSD loss = V1).')
    parser.add_argument('--ksd_score_timestep', type=int, default=25,
                        help='Low diffusion timestep for the KSD score witness (default 25, sigma~0.095).')
    parser.add_argument('--ksd_balance_every', type=int, default=20,
                        help='Run gradient balancing (measure ||g_ksd||/||g_cls|| to set lambda) every N steps '
                             '(default 20); lambda is held between measurements.')

    # ===== Stage 1: prototype detection readout (0803 §2.2/§2.6, 0810 plan) =====
    # Projection head (512->512->128, parallel to fc) + (class, domain) prototype buffers +
    # angular detection score to the 6 CLASS CENTERS. Forces the same two-step style forward as
    # KSD (grad-enabled perturbed feature) PLUS a second no-grad CLEAN forward for the prototype
    # EMA. proto_comp_ratio=0 => prototype-off control (same two forwards, no prototype loss).
    parser.add_argument('--use_proto_reg', action='store_true',
                        help='Stage 1: enable projection head + prototype losses + angular readout. '
                             'Requires resnet_type=standard.')
    parser.add_argument('--proj_dim', type=int, default=128,
                        help='Projection head output dim (default 128 = CIDER/PALM source-verified).')
    parser.add_argument('--proto_no_projection', action='store_true',
                        help='Arm 1c: skip the projection head, apply prototype losses directly on the '
                             '512-d penultimate (control for inter-set interference, 0803 §2.3.1).')
    parser.add_argument('--proto_m', type=float, default=0.95,
                        help='Prototype EMA memory (CIDER proto_m). NOTE there is no single "CIDER value": '
                             'CIFAR-100 uses 0.5, CIFAR-10/ImageNet-100 use 0.95. Update is PER-SAMPLE, so the '
                             'effective window is ~1/(1-m) SAMPLES (~2 batches at 0.95), not batches.')
    parser.add_argument('--proto_temp', type=float, default=0.1,
                        help='Temperature for comp/disp losses (CIDER tau=0.1).')
    parser.add_argument('--proto_comp_ratio', type=float, default=0.1,
                        help='Gradient balancing target for L_comp: set lambda so ||lam*g_comp||=ratio*||g_cls||. '
                             'ratio=0 => that term off. Set comp=disp=rel=0 for the lambda=0 BASELINE arm '
                             '(same two forwards, prototypes still accumulate => identical RNG stream to 1a).')
    parser.add_argument('--proto_disp_ratio', type=float, default=0.1,
                        help='Gradient balancing target for L_disp (class centers pushed apart).')
    parser.add_argument('--proto_style_ratio', type=float, default=0.0,
                        help='Gradient balancing target for L_style (styles pulled to class center). '
                             'MUST stay 0 in stage 1: with one domain per node the class center IS that '
                             'domain prototype, so L_style is the same quantity as L_comp (0803 §2.6.5).')
    parser.add_argument('--proto_rel_ratio', type=float, default=0.0,
                        help='Gradient balancing target for L_rel (arm 1b). 0 => arm 1a.')
    parser.add_argument('--proto_rel_margin', type=float, default=0.0,
                        help='Margin for L_rel. Start at 0 (pure one-sided suppression). GorD appendix values '
                             '([0.7,0.5,0.2] / [10,5,5]) are energy-scale and differ 25x => NOT transferable; '
                             'our score is arccos with range [0, pi].')
    parser.add_argument('--proto_warmup_epochs', type=int, default=10,
                        help='Epochs during which prototype losses are disabled (prototypes still accumulate). '
                             'Early features are near-random, so pulling toward a near-random prototype injects '
                             'noise; the EMA window is only ~20 samples so this matters more than it looks.')
    parser.add_argument('--proto_balance_every', type=int, default=20,
                        help='Run prototype gradient balancing every N steps (default 20); lambdas held between.')

    # ===== Async event-triggered communication (Stage 1 baseline) =====
    parser.add_argument('--async_trigger', action='store_true',
                        help='Enable event-triggered ASYNC decentralized training: each node broadcasts its '
                             'model only when (1/sqrt(n))||w_i - w_hat_i|| >= threshold (PersonalizedET Eq.3); '
                             'neighbors receive via a dedup buffer and aggregate on their own schedule. '
                             'Removes the synchronous per-round Phase-3 model aggregation.')
    parser.add_argument('--trigger_threshold', type=float, default=0.0,
                        help='Broadcast trigger threshold tau (r*rho*gamma, rho=gamma=1 in Stage 1). '
                             '0 => always broadcast (~synchronous, sanity control).')
    parser.add_argument('--trigger_max_interval', type=int, default=50,
                        help='Bounded-staleness safety valve: force a broadcast if a node has not broadcast for '
                             'B consecutive sweeps (PersonalizedET B2-connectivity). Guarantees consensus.')
    parser.add_argument('--async_buffer_max', type=int, default=None,
                        help='Optional cap on inbox size per node (naturally bounded by #neighbors).')
    parser.add_argument('--trigger_gamma_mode', type=str, default='lr', choices=['lr', 'const'],
                        help="Time-varying threshold decay gamma^(k) (PersonalizedET-faithful). "
                             "'lr' (default): threshold = tau * lr(k)/lr(0), decays with the cosine LR "
                             "so it stays sensitive as the model converges. 'const': gamma=1 (constant tau).")
    parser.add_argument('--aggregate_bn', action='store_true',
                        help='aggregate BN running_mean/var together with model parameters (same MH weights). '
                             'Default OFF = current SiloBN-style keep-local. Payload +0.086%%. '
                             'Rationale: research/bn_fusion/0729_bn_consensus_first_principles.md')
    parser.add_argument('--bn_div_diag', action='store_true',
                        help='print cross-node BN divergence each epoch (auto-on when --aggregate_bn)')
    parser.add_argument('--async_style', action='store_true',
                        help='Stage 2: also event-trigger the style-statistics exchange (Phase 1). Style rides on '
                             'the SAME model-broadcast event (bundled); neighbors keep a persistent style buffer '
                             '(keep-latest per sender, NOT consumed) and StyleShift reuses the last-received style '
                             'until a newer one arrives. No cold-start seed: empty buffer => StyleShift skips (R1). '
                             'Requires --async_trigger. Removes the synchronous per-sweep style barrier entirely.')

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

