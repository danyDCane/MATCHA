import os
import numpy as np
import time
import argparse
import sys

from mpi4py import MPI
from math import ceil
from random import Random
import networkx as nx

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.multiprocessing import Process
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

# import resnet
# import vggnet
# import wrn 
import util
from graph_manager import FixedProcessor, MatchaProcessor
from communicator import decenCommunicator, ChocoCommunicator, centralizedCommunicator

def sync_allreduce(model, rank, size):
    senddata = {}
    recvdata = {}
    for param in model.parameters():
        tmp = param.data.cpu()
        senddata[param] = tmp.numpy()
        recvdata[param] = np.empty(senddata[param].shape, dtype = senddata[param].dtype)
    torch.cuda.synchronize()
    comm.barrier()

    comm_start = time.time()
    for param in model.parameters():
        comm.Allreduce(senddata[param], recvdata[param], op=MPI.SUM)
    torch.cuda.synchronize()    
    comm.barrier()
    
    comm_end = time.time()
    comm_t = (comm_end - comm_start)
        
    for param in model.parameters():
        param.data = torch.Tensor(recvdata[param]).cuda()
        param.data = param.data/float(size)
    return comm_t

def run(rank, size):

    # set random seed
    torch.manual_seed(args.randomSeed+rank)
    np.random.seed(args.randomSeed)

    # initialize wandb for each rank
    wandb.init(
        project=args.wandb_project,
        name=f"{args.name}_rank{rank}",
        config={
            "rank": rank,
            "size": size,
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

    # load data
    train_loader, test_loader = util.partition_dataset(rank, size, args)

    # ====== Infinite iterator + shared STEPS_PER_EPOCH ======
    # local number of batches on this rank
    local_steps = len(train_loader)
    # use MPI allreduce to get the maximum steps across all ranks
    global comm
    STEPS_PER_EPOCH = comm.allreduce(local_steps, op=MPI.MAX)

    # total epochs as in original arguments
    TOTAL_EPOCHS = args.epoch
    # total iterations K (Algorithm 1)
    if args.total_iter is not None:
        K = args.total_iter
    else:
        K = STEPS_PER_EPOCH * TOTAL_EPOCHS

    # build infinite iterator over local dataloader
    def get_infinite_iterator(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    train_iter = get_infinite_iterator(train_loader)

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
    
    # define graph activation scheme with K iterations
    if args.matcha:
        GP = MatchaProcessor(subGraphs, args.budget, rank, size, K, True)
    else:
        GP = FixedProcessor(subGraphs, args.budget, rank, size, K, True)

    # define communicator
    if args.compress:
        communicator = ChocoCommunicator(rank, size, GP, 0.9, args.consensus_lr)
    else:
        communicator = decenCommunicator(rank, size, GP)

    # [DEBUG START] 請加入這段來檢查拓撲結構
    # if rank == 0:
    #     print("\n====== TOPOLOGY CHECK ======")
    #     print(f"Total Subgraphs: {len(GP.subGraphs)}")
    #     print(f"Neighbor Weight (Alpha): {GP.neighbor_weight}")
    #     # 檢查每個子圖的鄰居定義
    #     for i, neighbors in enumerate(GP.neighbors_info):
    #         print(f"Subgraph {i} Neighbors: {neighbors}")
    #     print("============================\n")
    
    # # 確保所有 Rank 都印完再繼續
    # comm.barrier()
    # [DEBUG END]

    # select neural network model
    num_classes = 7 if args.dataset == 'pacs' else 10
    # For pretrained models: let rank 0 download first to avoid concurrent download conflicts
    if getattr(args, 'pretrained', False):
        if rank == 0:
            print(f"[Rank {rank}] Initializing model with pretrained weights (downloading if needed)...")
            model = util.select_model(num_classes, args)
            comm.barrier()  # Signal that rank 0 is done downloading
        else:
            comm.barrier()  # Other ranks wait for rank 0 to finish downloading
            model = util.select_model(num_classes, args)  # Now use cached weights
    else:
        # No pretrained weights, all ranks can create model simultaneously
        model = util.select_model(num_classes, args)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), 
                          lr=args.lr,
                          momentum=args.momentum, 
                          weight_decay=5e-4,
                          nesterov=args.nesterov)
    
    # Initialize diffusion model for OOD detection if enabled
    diffusion_model = None
    optimizer_diffusion = None
    if getattr(args, 'use_ood', False):
        from dood.utils.diffusion import get_diffusion_model
        # ResNet18 intermediate features are 512-dim
        diffusion_model = get_diffusion_model(
            ft_size=512,
            denoiser_type="unet0d",
            diffusion_denoiser_channels=getattr(args, 'diffusion_channels', 512),
            num_diffusion_steps=getattr(args, 'diffusion_steps', 1000),
        ).cuda()
        optimizer_diffusion = optim.Adam(
            diffusion_model.parameters(),
            lr=getattr(args, 'lr_diffusion', 5e-5)
        )
        # Attach diffusion model to model for easier access during communication
        model.diffusion_model = diffusion_model
        if rank == 0:
            print(f"[Rank {rank}] Diffusion model initialized for OOD detection")
    
    # CosineAnnealingLR: T_max = K (total iterations), purely step-based
    scheduler = CosineAnnealingLR(optimizer, T_max=K, eta_min=0.0)
    
    # guarantee all local models start from the same point
    # can be removed    
    sync_allreduce(model, rank, size)

    # init recorder
    comp_time, comm_time = 0, 0
    recorder = util.Recorder(args,rank)
    losses = util.AverageMeter()
    top1 = util.AverageMeter()
    loss_diff_meter = util.AverageMeter() if getattr(args, 'use_ood', False) else None
    tic = time.time()

    # ===== start training with fixed total steps K (Algorithm 1) =====
    for k in range(K):
        model.train()

        start_time = time.time()

        # sample mini-batch (infinite iterator, cycles through local data)
        data, target = next(train_iter)
        data, target = data.cuda(non_blocking = True), target.cuda(non_blocking = True)
        
        # ========== 第一阶段：计算风格统计量（不训练）==========
        # If enabled, compute style statistics first without gradients
        # This allows us to exchange style statistics before training
        style_vec = None
        if (getattr(args, "use_style_stats", False) or getattr(args, "use_style_shift", False)) and args.model == "res":
            with torch.no_grad():  # 不计算梯度，节省内存
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
                # Detach from computation graph and move to CPU for communication
                style_vec = style_vec.detach().cpu()
        
        # ========== 通信交换风格统计量 ==========
        # Exchange style statistics only (no model parameters) with neighbors
        # If style_vec is provided, only exchange style statistics; if None, only exchange model parameters
        d_comm_time = communicator.communicate(model, style_vec=style_vec)
        comm_time += d_comm_time

        # # Verify style statistics exchange (print every epoch)
        # if getattr(args, "use_style_stats", False) and args.model == "res":
        #     if k == 0 or (k + 1) % STEPS_PER_EPOCH == 0:
        #         # Use MPI barrier to synchronize output
        #         comm.barrier()
        #         # Print in rank order to avoid output mixing
        #         for r in range(size):
        #             if rank == r:
        #                 # Collect all output into a single string first
        #                 output_lines = []
        #                 output_lines.append("\n[StyleStats Exchange] iter {}, rank {}: ".format(k + 1, rank))
        #                 # Print local style vector info
        #                 if communicator.local_style_vec is not None:
        #                     local_vec = communicator.local_style_vec
        #                     output_lines.append("  Local style_vec: shape={}, sample={}".format(
        #                         tuple(local_vec.shape), 
        #                         local_vec[:3].numpy().tolist() if len(local_vec) >= 3 else local_vec.numpy().tolist()))
        #                 else:
        #                     output_lines.append("  Local style_vec: None")
        #                 # Print neighbor style vectors info
        #                 if communicator.neighbor_style_vecs:
        #                     neighbor_info = "  Received {} neighbor(s): ".format(len(communicator.neighbor_style_vecs))
        #                     for neighbor_rank, neighbor_vec in communicator.neighbor_style_vecs.items():
        #                         neighbor_info += "rank{}[shape={}] ".format(neighbor_rank, tuple(neighbor_vec.shape))
        #                     output_lines.append(neighbor_info)
        #                     # Print unflattened stats info if available
        #                     if communicator.neighbor_style_stats:
        #                         output_lines.append("  Unflattened neighbor style stats:")
        #                         for neighbor_rank, neighbor_stats in communicator.neighbor_style_stats.items():
        #                             output_lines.append("    Rank {}:".format(neighbor_rank))
        #                             for layer_name, layer_stats in neighbor_stats.items():
        #                                 output_lines.append("      {}: mu_bar{}, sigma_bar{}, Sigma_mu_sq{}, Sigma_sigma_sq{}".format(
        #                                     layer_name,
        #                                     tuple(layer_stats["mu_bar"].shape),
        #                                     tuple(layer_stats["sigma_bar"].shape),
        #                                     tuple(layer_stats["Sigma_mu_sq"].shape),
        #                                     tuple(layer_stats["Sigma_sigma_sq"].shape)
        #                                 ))
        #                 else:
        #                     output_lines.append("  No neighbor style_vecs received")
        #                 # Print all at once to avoid interleaving
        #                 print("\n".join(output_lines))
        #                 sys.stdout.flush()
        #             comm.barrier()  # Wait for each rank to finish printing
        
        # ========== 第二阶段：正式训练（使用交换到的风格统计量）==========
        # Now forward with style shift using the exchanged neighbor style statistics
        use_style_stats = getattr(args, "use_style_stats", False)
        use_style_shift = getattr(args, "use_style_shift", False)
        # debug_style_shift = ((k % STEPS_PER_EPOCH) == 0) # Enable debug output for style shift
        debug_style_shift = getattr(args, "debug_style_shift", False)
        
        if (use_style_stats or use_style_shift) and args.model == "res":
            # This forward will use communicator.neighbor_style_stats (just exchanged)
            # If use_style_stats is True, we need return_blocks for potential future use
            # If use_style_shift is True, we need communicator for style shift
            if use_style_stats:
                output, feats = model(data, return_blocks=True, communicator=communicator,
                                     debug_style_shift=debug_style_shift, iter_num=k+1, rank=rank)
            else:
                # Only use_style_shift is True, don't need return_blocks
                output = model(data, return_blocks=False, communicator=communicator,
                              debug_style_shift=debug_style_shift, iter_num=k+1, rank=rank)
        else:
            output = model(data)
        
        loss = criterion(output, target)
        
        # Compute diffusion loss if OOD detection is enabled
        loss_diff = None
        if getattr(args, 'use_ood', False) and diffusion_model is not None:
            # Extract intermediate features for diffusion model
            latents = model.intermediate_forward(data)
            
            # Normalize features and compute diffusion loss
            # Detach latents to avoid affecting backbone gradients
            latents_for_diff = latents.detach().requires_grad_(True)
            latents_normalized = diffusion_model.normalize(latents_for_diff)
            loss_diff = diffusion_model.get_loss_iter(latents_normalized)
            
            # Backward pass for diffusion model
            optimizer_diffusion.zero_grad()
            loss_diff.backward()
            optimizer_diffusion.step()
            
            # Record diffusion loss for averaging
            if loss_diff_meter is not None:
                loss_diff_meter.update(loss_diff.item(), data.size(0))

        # record training loss and accuracy
        record_start = time.time()
        acc1 = util.comp_accuracy(output, target)
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))
        record_end = time.time()

        # backward pass for classification
        loss.backward()

        # gradient step
        optimizer.step()
        optimizer.zero_grad()

        # update cosine annealing scheduler (purely step-based)
        scheduler.step()
        
        # ========== 第三阶段：交换训练后的模型参数 ==========
        # Exchange updated model parameters after training step
        # Note: style_vec is None here since we only exchange model parameters (not style stats)
        d_comm_time_after = communicator.communicate(model, style_vec=None)
        comm_time += d_comm_time_after
        
        # 立即检查通信后的参数（在测试之前）
        if (k + 1) % STEPS_PER_EPOCH == 0:
            comm.barrier()
            first_param_immediate = list(model.parameters())[0].view(-1)[0].item()
            if rank == 0:
                params_immediate = [first_param_immediate]
                for r in range(1, size):
                    params_immediate.append(comm.recv(source=r, tag=400))
                print(f"*** IMMEDIATE after comm iter {k+1}: {[f'{p:.6f}' for p in params_immediate]} ***")
                # 检查是否一致
                if len(set([round(p, 4) for p in params_immediate])) == 1:
                    print("✓ Parameters are CONSISTENT after communication!")
                else:
                    print("✗ Parameters are INCONSISTENT after communication!")
            else:
                comm.send(first_param_immediate, dest=0, tag=400)
            comm.barrier()
        
        end_time = time.time()

        d_comp_time = (end_time - start_time - (record_end - record_start))
        comp_time += d_comp_time
        
        # Access neighbor style vectors if needed for loss calculation
        # neighbor_style_vecs = communicator.neighbor_style_vecs  # Dict: {neighbor_rank: style_vec_tensor}

        # 每隔一定次數清理 GPU 緩存，避免記憶體累積（可選，避免頻繁清理影響性能）
        if (k + 1) % 20 == 0:  # 每 20 個 iteration 清理一次
            torch.cuda.empty_cache()

        print("iter: %d/%d, rank: %d, comp_time: %.3f, comm_time: %.3f, total time: %.3f "
              % (k+1, K, rank, d_comp_time, d_comm_time, comp_time + comm_time), end='\r')

        # measure and log after each pseudo-epoch
        if (k + 1) % STEPS_PER_EPOCH == 0:
            epoch = (k + 1) // STEPS_PER_EPOCH
            toc = time.time()
            record_time = toc - tic  # includes everything
            epoch_time = comp_time + comm_time  # only important parts

            # 在測試前清理 GPU 緩存，釋放未使用的記憶體
            torch.cuda.empty_cache()
            
            # evaluate test accuracy
            test_acc = util.test(model, test_loader)
            
            # 測試後再次清理緩存，確保記憶體被釋放
            torch.cuda.empty_cache()

            # 檢查模型共識度，取模型第一層的第一個權重數值
            first_param = list(model.parameters())[0].view(-1)[0].item()
            print(f"*** Iter {k+1} Rank {rank} Param[0]: {first_param:.6f} ***")

            recorder.add_new(record_time, comp_time, comm_time, epoch_time,
                             top1.avg, losses.avg, test_acc)
            print("rank: %d, epoch: %d, loss: %.3f, train_acc: %.3f, test_acc: %.3f epoch time: %.3f"
                  % (rank, epoch, losses.avg, top1.avg, test_acc, epoch_time))
            
            # log to wandb for each rank
            log_dict = {
                "epoch": epoch,
                "iter": k,
                "loss": losses.avg.item() if hasattr(losses.avg, 'item') else float(losses.avg),
                "train_acc": top1.avg.item() if hasattr(top1.avg, 'item') else float(top1.avg),
                "test_acc": test_acc.item() if hasattr(test_acc, 'item') else float(test_acc),
                "lr": optimizer.param_groups[0]['lr'],
                "comp_time": comp_time,
                "comm_time": comm_time,
                "epoch_time": epoch_time,
            }
            # Add diffusion loss if OOD detection is enabled
            if getattr(args, 'use_ood', False) and diffusion_model is not None:
                if loss_diff_meter is not None:
                    log_dict["diffusion_loss"] = loss_diff_meter.avg.item() if hasattr(loss_diff_meter.avg, 'item') else float(loss_diff_meter.avg)
                # Add diffusion model monitoring info if available
                if hasattr(diffusion_model.diffusion_process, '_last_snr_info'):
                    snr_info = diffusion_model.diffusion_process._last_snr_info
                    log_dict["snr"] = snr_info.get('snr', 0.0)
                    if 'identity_correlation' in snr_info:
                        log_dict["identity_correlation"] = snr_info['identity_correlation']
            wandb.log(log_dict)
            
            if rank == 0:
                print("comp_time: %.3f, comm_time: %.3f, comp_time_budget: %.3f, comm_time_budget: %.3f"
                      % (comp_time, comm_time, comp_time/epoch_time, comm_time/epoch_time))

            # reset recorders for next epoch
            comp_time, comm_time = 0, 0
            losses.reset()
            top1.reset()
            if loss_diff_meter is not None:
                loss_diff_meter.reset()
            tic = time.time()

    recorder.save_to_file()
    wandb.finish()


def update_learning_rate(optimizer, epoch, itr=None, itr_per_epoch=None,
                         scale=1):
    """
    1) Linearly warmup to reference learning rate (5 epochs)
    2) Decay learning rate exponentially (epochs 30, 60, 80)
    ** note: args.lr is the reference learning rate from which to scale up
    ** note: minimum global batch-size is 256
    """
    base_lr = 0.1
    target_lr = args.lr
    lr_schedule = [100, 150]

    lr = None
    if args.warmup and epoch < 5:  # warmup to scaled lr
        if target_lr <= base_lr:
            lr = target_lr
        else:
            assert itr is not None and itr_per_epoch is not None
            count = epoch * itr_per_epoch + itr + 1
            incr = (target_lr - base_lr) * (count / (5 * itr_per_epoch))
            lr = base_lr + incr
    else:
        lr = target_lr
        for e in lr_schedule:
            if epoch >= e:
                lr *= 0.1

    if lr is not None:
        # print('Updating learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
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

    if not args.description:
        print('No experiment description, exit!')
        exit()

    comm = MPI.COMM_WORLD  # comm 是 MPI 的 communicator，用於在 process 之間進行通信
    rank = comm.Get_rank() # rank 是當前 process 的 id
    size = comm.Get_size() # size 是總共有多少個 process

    # Validate PACS requirements
    if args.dataset == 'pacs':
        if not args.leave_out:
            if rank == 0:
                print('Error: --leave_out must be specified when using PACS dataset.')
                print('Valid options: art_painting, cartoon, photo, sketch')
            exit(1)
        valid_domains = ['art_painting', 'cartoon', 'photo', 'sketch']
        if args.leave_out not in valid_domains:
            if rank == 0:
                print(f'Error: Invalid leave_out domain: {args.leave_out}')
                print(f'Valid options: {valid_domains}')
            exit(1)

    run(rank, size) # 開始訓練

