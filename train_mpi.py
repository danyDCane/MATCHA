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

    # load base network topology
    # For PACS, use fully connected graph for 3 nodes (graphid = -1)
    if args.dataset == 'pacs':
        subGraphs = util.select_graph(-1)
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

    # select neural network model
    num_classes = 7 if args.dataset == 'pacs' else 10
    model = util.select_model(num_classes, args)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), 
                          lr=args.lr,
                          momentum=args.momentum, 
                          weight_decay=5e-4,
                          nesterov=args.nesterov)
    
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
    tic = time.time()

    # ===== start training with fixed total steps K (Algorithm 1) =====
    for k in range(K):
        model.train()

        start_time = time.time()

        # sample mini-batch (infinite iterator, cycles through local data)
        data, target = next(train_iter)
        data, target = data.cuda(non_blocking = True), target.cuda(non_blocking = True)
        
        # forward pass
        # If enabled and using ResNet backbone, compute style statistics from
        # the first three convolutional blocks in the SAME forward pass to
        # avoid duplicated computation and extra memory.
        if getattr(args, "use_style_stats", False) and args.model == "res":
            output, feats = model(data, return_blocks=True)

            # Compute STYLEDDG-style statistics (batch-level, per channel)
            style_stats = compute_multi_layer_style_stats(
                feats,
                eta=getattr(args, "style_eta", 1e-5),
            )

            # Optional: flatten to a single style vector per batch/device.
            # This is ready for later style sharing (StyleDDG) if needed.
            style_vec = flatten_style_stats(
                style_stats,
                layer_order=["layer1", "layer2", "layer3"],
            )
            # NOTE: Currently we do not modify the loss using style_vec.
            #       This keeps the training identical unless you plug
            #       style-based regularization / sharing on top.
            #       Here we only print a small summary occasionally for sanity check.
            if rank == 0 and (k == 0 or (k + 1) % STEPS_PER_EPOCH == 0):
                l1 = style_stats["layer1"]
                print("\n[StyleStats] iter {}: ".format(k + 1))
                print("  layer1 mu_bar       shape:", tuple(l1["mu_bar"].shape),
                      " sample:", l1["mu_bar"][:5].detach().cpu().numpy())
                print("  layer1 sigma_bar    shape:", tuple(l1["sigma_bar"].shape),
                      " sample:", l1["sigma_bar"][:5].detach().cpu().numpy())
                print("  layer1 Sigma_mu_sq  shape:", tuple(l1["Sigma_mu_sq"].shape),
                      " sample:", l1["Sigma_mu_sq"][:5].detach().cpu().numpy())
                print("  layer1 Sigma_sigma_sq shape:", tuple(l1["Sigma_sigma_sq"].shape),
                      " sample:", l1["Sigma_sigma_sq"][:5].detach().cpu().numpy())
                print("  style_vec length:", int(style_vec.numel()))
        else:
            output = model(data)
        loss = criterion(output, target)

        # record training loss and accuracy
        record_start = time.time()
        acc1 = util.comp_accuracy(output, target)
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))
        record_end = time.time()

        # backward pass
        loss.backward()

        # gradient step
        optimizer.step()
        optimizer.zero_grad()

        # update cosine annealing scheduler (purely step-based)
        scheduler.step()
        end_time = time.time()

        d_comp_time = (end_time - start_time - (record_end - record_start))
        comp_time += d_comp_time

        # communication happens here (all ranks use the same k)
        d_comm_time = communicator.communicate(model)
        comm_time += d_comm_time

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

            recorder.add_new(record_time, comp_time, comm_time, epoch_time,
                             top1.avg, losses.avg, test_acc)
            print("rank: %d, epoch: %d, loss: %.3f, train_acc: %.3f, test_acc: %.3f epoch time: %.3f"
                  % (rank, epoch, losses.avg, top1.avg, test_acc, epoch_time))
            
            # log to wandb for each rank
            wandb.log({
                "epoch": epoch,
                "iter": k,
                "loss": losses.avg.item() if hasattr(losses.avg, 'item') else float(losses.avg),
                "train_acc": top1.avg.item() if hasattr(top1.avg, 'item') else float(top1.avg),
                "test_acc": test_acc.item() if hasattr(test_acc, 'item') else float(test_acc),
                "lr": optimizer.param_groups[0]['lr'],
                "comp_time": comp_time,
                "comm_time": comm_time,
                "epoch_time": epoch_time,
            })
            
            if rank == 0:
                print("comp_time: %.3f, comm_time: %.3f, comp_time_budget: %.3f, comm_time_budget: %.3f"
                      % (comp_time, comm_time, comp_time/epoch_time, comm_time/epoch_time))

            # reset recorders for next epoch
            comp_time, comm_time = 0, 0
            losses.reset()
            top1.reset()
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

    args = parser.parse_args()

    if not args.description:
        print('No experiment description, exit!')
        exit()

    comm = MPI.COMM_WORLD  # comm 是 MPI 的 communicator，用於在 process 之間進行通信
    rank = comm.Get_rank() # rank 是當前 process 的 id
    size = comm.Get_size() # size 是總共有多少個 process

    # Validate PACS requirements
    if args.dataset == 'pacs':
        if size != 3:
            if rank == 0:
                print(f'Error: PACS dataset requires exactly 3 nodes, but {size} nodes were provided.')
                print('Please use: mpirun -np 3 python train_mpi.py ...')
            exit(1)
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

