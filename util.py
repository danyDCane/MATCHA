import os
import numpy as np
import time
import argparse
import random
from collections import defaultdict

# MPI import is optional (only needed for partition_dataset, which is deprecated in single-process mode)
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    MPI = None
from math import ceil
from random import Random
import networkx as nx

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Process
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.utils.data.sampler import Sampler
from torch.utils.data import Subset

from models import *
from pacs_dataset import PACSDataset
from vlcs_dataset import VLCSFullTestDataset

# import GraphPreprocess 

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234, isNonIID=False):
        self.data = data 
        self.partitions = [] 
        rng = Random() 
        rng.seed(seed) 
        data_len = len(data) 
        indexes = [x for x in range(0, data_len)] 
        rng.shuffle(indexes) 
         
 
        for frac in sizes: 
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

        if isNonIID:
            self.partitions = self.__getNonIIDdata__(data, sizes, seed)

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

    def __getNonIIDdata__(self, data, sizes, seed):
        labelList = data.train_labels
        rng = Random()
        rng.seed(seed)
        a = [(label, idx) for idx, label in enumerate(labelList)]
        # Same Part
        labelIdxDict = dict()
        for label, idx in a:
                labelIdxDict.setdefault(label,[])
                labelIdxDict[label].append(idx)
        labelNum = len(labelIdxDict)
        labelNameList = [key for key in labelIdxDict]
        labelIdxPointer = [0] * labelNum
        partitions = [list() for i  in range(len(sizes))]
        eachPartitionLen= int(len(labelList)/len(sizes))
        majorLabelNumPerPartition = ceil(labelNum/len(partitions))
        basicLabelRatio = 0.4

        interval = 1
        labelPointer = 0

        #basic part
        for partPointer in range(len(partitions)):
            requiredLabelList = list()
            for _ in range(majorLabelNumPerPartition):
                requiredLabelList.append(labelPointer)
                labelPointer += interval
                if labelPointer > labelNum - 1:
                    labelPointer = interval
                    interval += 1
            for labelIdx in requiredLabelList:
                start = labelIdxPointer[labelIdx]
                idxIncrement = int(basicLabelRatio*len(labelIdxDict[labelNameList[labelIdx]]))
                partitions[partPointer].extend(labelIdxDict[labelNameList[labelIdx]][start:start+ idxIncrement])
                labelIdxPointer[labelIdx] += idxIncrement

        #random part
        remainLabels = list()
        for labelIdx in range(labelNum):
            remainLabels.extend(labelIdxDict[labelNameList[labelIdx]][labelIdxPointer[labelIdx]:])
        rng.shuffle(remainLabels)
        for partPointer in range(len(partitions)):
            idxIncrement = eachPartitionLen - len(partitions[partPointer])
            partitions[partPointer].extend(remainLabels[:idxIncrement])
            rng.shuffle(partitions[partPointer])
            remainLabels = remainLabels[idxIncrement:]
        return partitions


class RandomClassSampler(Sampler):
    """
    Randomly samples N classes each with K instances to form a minibatch.

    This is a PyTorch-native equivalent of the idea behind Dassl's RandomClassSampler,
    but it works with datasets that expose `targets` (list[int]) as ImageFolder/CIFAR do.
    """

    def __init__(self, dataset, batch_size: int, n_ins: int):
        if batch_size % n_ins != 0:
            raise ValueError(f"batch_size={batch_size} must be divisible by n_ins={n_ins}")

        # `targets` is the common convention for torchvision-style datasets.
        targets = getattr(dataset, "targets", None)
        # Support torch.utils.data.Subset by extracting labels from base dataset.
        if targets is None and hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
            base_targets = getattr(dataset.dataset, "targets", None)
            if base_targets is not None:
                targets = [base_targets[i] for i in dataset.indices]
        if targets is None:
            raise ValueError("RandomClassSampler requires dataset to have attribute `targets`")

        self.targets = targets
        self.batch_size = batch_size
        self.n_ins = n_ins
        self.ncls_per_batch = batch_size // n_ins

        self.index_dic = defaultdict(list)
        for index, label in enumerate(self.targets):
            self.index_dic[int(label)].append(index)

        self.labels = list(self.index_dic.keys())
        if len(self.labels) < self.ncls_per_batch:
            raise ValueError(
                f"Not enough classes for random_class sampler: "
                f"num_classes={len(self.labels)} < ncls_per_batch={self.ncls_per_batch}"
            )

        # Deterministic estimate for number of batches per epoch:
        # - For each class, we can form floor(len(idxs)/n_ins) batches.
        # - If a class has fewer than n_ins samples, we still allow 1 batch by sampling with replacement.
        n_batches_per_label = []
        for label in self.labels:
            cnt = len(self.index_dic[label])
            if cnt <= 0:
                continue
            n_batches_per_label.append(max(1, cnt // self.n_ins))
        total_batches_possible = int(sum(n_batches_per_label))
        self.epoch_batches = max(1, total_batches_possible // self.ncls_per_batch)

        # DataLoader expects sampler.__len__ to be number of indices.
        self.length = self.epoch_batches * self.batch_size

    def __iter__(self):
        # Build per-label batch index lists (each list contains arrays of length n_ins)
        batch_idxs_dict = defaultdict(list)
        for label in self.labels:
            idxs = list(self.index_dic[label])
            if len(idxs) < self.n_ins:
                # Sample with replacement to make one full (n_ins-sized) batch possible.
                idxs = random.choices(idxs, k=self.n_ins)
            random.shuffle(idxs)
            cur = []
            for idx in idxs:
                cur.append(idx)
                if len(cur) == self.n_ins:
                    batch_idxs_dict[label].append(cur)
                    cur = []

        avai_labels = [l for l in self.labels if len(batch_idxs_dict[l]) > 0]
        final_idxs = []

        # Generate up to `epoch_batches` balanced batches
        for _ in range(self.epoch_batches):
            if len(avai_labels) < self.ncls_per_batch:
                break
            selected_labels = random.sample(avai_labels, self.ncls_per_batch)
            for label in selected_labels:
                batch_idxs = batch_idxs_dict[label].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[label]) == 0:
                    avai_labels.remove(label)

        return iter(final_idxs)

    def __len__(self):
        return self.length


def assign_nodes_to_domains(available_domains, num_nodes):
    """
    Assign virtual nodes to domains as evenly as possible.

    Returns:
        node_to_domain: dict like {"node_0": "art_painting", ...}
        domain_to_nodes: dict like {"art_painting": ["node_0", ...], ...}
    """
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive, got {num_nodes}")

    num_domains = len(available_domains)
    if num_domains == 0:
        raise ValueError("No available domains to assign nodes")

    base = num_nodes // num_domains
    extra = num_nodes % num_domains

    node_to_domain = {}
    domain_to_nodes = {d: [] for d in available_domains}

    node_idx = 0
    for i, domain in enumerate(available_domains):
        n_for_domain = base + (1 if i < extra else 0)
        for _ in range(n_for_domain):
            node_name = f"node_{node_idx}"
            node_to_domain[node_name] = domain
            domain_to_nodes[domain].append(node_name)
            node_idx += 1

    return node_to_domain, domain_to_nodes


def partition_domain_dataset_for_nodes(train_dataset, node_names, split_mode, seed):
    """
    Split one domain dataset into multiple subsets for assigned virtual nodes.

    split_mode:
      - random_contiguous: keep original order and slice contiguous chunks
      - class_balanced: split each class bucket evenly across nodes
    """
    n_nodes = len(node_names)
    if n_nodes <= 0:
        raise ValueError("node_names must be non-empty")

    total = len(train_dataset)
    all_indices = list(range(total))

    if n_nodes == 1:
        return {node_names[0]: all_indices}

    if split_mode == "random_contiguous":
        # Contiguous slicing on original index order.
        splits = np.array_split(np.array(all_indices), n_nodes)
        return {node_names[i]: splits[i].tolist() for i in range(n_nodes)}

    if split_mode != "class_balanced":
        raise ValueError(f"Unknown node split mode: {split_mode}")

    targets = getattr(train_dataset, "targets", None)
    if targets is None:
        # Fallback: no class labels available, use contiguous split.
        splits = np.array_split(np.array(all_indices), n_nodes)
        return {node_names[i]: splits[i].tolist() for i in range(n_nodes)}

    # Group indices by class, then split each class bucket across nodes.
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_to_indices[int(label)].append(idx)

    rng = np.random.default_rng(seed)
    node_to_indices = {node: [] for node in node_names}

    for label in sorted(class_to_indices.keys()):
        cls_indices = class_to_indices[label]
        rng.shuffle(cls_indices)
        cls_splits = np.array_split(np.array(cls_indices), n_nodes)
        for i, node in enumerate(node_names):
            node_to_indices[node].extend(cls_splits[i].tolist())

    # Shuffle within each node for training randomness.
    for i, node in enumerate(node_names):
        local_rng = np.random.default_rng(seed + i + 1)
        local_idxs = node_to_indices[node]
        local_rng.shuffle(local_idxs)
        node_to_indices[node] = list(local_idxs)

    return node_to_indices


def _compute_label_histogram_for_indices(dataset, indices):
    """Return label histogram dict for a subset of indices."""
    targets = getattr(dataset, "targets", None)
    if targets is None:
        return {}
    hist = defaultdict(int)
    for idx in indices:
        hist[int(targets[idx])] += 1
    return dict(sorted(hist.items(), key=lambda kv: kv[0]))

def partition_dataset(rank, size, args):
    """
    DEPRECATED: This function is for MPI multi-process training.
    For single-process training, use load_dataset_single_process() instead.
    """
    if not MPI_AVAILABLE:
        raise ImportError("MPI is required for partition_dataset. Use load_dataset_single_process() for single-process training.")
    print('==> load train data')
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root=args.datasetRoot, 
                                                train=True, 
                                                download=(rank == 0),  # 只有 rank 0 下载, 
                                                transform=transform_train)
        # 等待下载完成
        if rank == 0:
            import time
            time.sleep(2)  # 给下载一些时间
        MPI.COMM_WORLD.Barrier()  # 所有进程等待 

        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(trainset, partition_sizes, isNonIID=False)
        partition = partition.use(rank)
        train_loader = torch.utils.data.DataLoader(partition, 
                                                batch_size=args.bs, 
                                                shuffle=True, 
                                                pin_memory=True)
 
        print('==> load test data')
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root=args.datasetRoot, 
                                            train=False, 
                                            download=True, 
                                            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, 
                                                batch_size=64, 
                                                shuffle=False, 
                                                num_workers=size)

    if args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        trainset = torchvision.datasets.CIFAR100(root=args.datasetRoot, 
                                                train=True, 
                                                download=True, 
                                                transform=transform_train)
 
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(trainset, partition_sizes, isNonIID=False)
        partition = partition.use(rank)
        train_loader = torch.utils.data.DataLoader(partition, 
                                                batch_size=args.bs, 
                                                shuffle=True, 
                                                pin_memory=True)
 
        print('==> load test data')
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        testset = torchvision.datasets.CIFAR100(root=args.datasetRoot, 
                                            train=False, 
                                            download=True, 
                                            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, 
                                                batch_size=64, 
                                                shuffle=False, 
                                                num_workers=size)
 

    elif args.dataset == 'imagenet':
        datadir = args.datasetRoot
        traindir = os.path.join(datadir, 'CLS-LOC/train/')
        #valdir = os.path.join(datadir, 'CLS-LOC/')
        #testdir = os.path.join(datadir, 'CLS-LOC/')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
 
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(train_dataset, partition_sizes, isNonIID=False)
        partition = partition.use(rank)
 
        train_loader = torch.utils.data.DataLoader(
            partition, batch_size=args.bs, shuffle=True,
             pin_memory=True)
        '''
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.bs, shuffle=False,
            pin_memory=True)
        val_loader = None
        '''
        test_loader = None

    if args.dataset == 'emnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.EMNIST(root=args.datasetRoot,
                                              split = 'balanced',
                                              train=True,
                                              download=True,
                                              transform=transform_train) 
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(train_dataset, partition_sizes, isNonIID=False)
        partition = partition.use(rank)
 
        train_loader = torch.utils.data.DataLoader(
            partition, batch_size=args.bs, shuffle=True,
             pin_memory=True)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        testset = torchvision.datasets.EMNIST(root=args.datasetRoot,
                                             split = 'balanced',
                                             train=False,
                                             download=True,
                                             transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, 
                                                batch_size=64, 
                                                shuffle=False, 
                                                num_workers=size)
 
    elif args.dataset == 'pacs':
        print('=' * 60)
        print(f'[PACS Dataset] Initializing PACS dataset loading...')
        print(f'[PACS Dataset] Rank: {rank}, Total nodes: {size}')
        print('=' * 60)
        
        # PACS domains
        all_domains = ['art_painting', 'cartoon', 'photo', 'sketch']
        if not args.leave_out:
            raise ValueError('--leave_out must be specified when using PACS dataset.')
        if args.leave_out not in all_domains:
            raise ValueError(f'Invalid leave_out domain: {args.leave_out}. Valid options: {all_domains}')
        
        print(f'[PACS Dataset] Leave-out domain: {args.leave_out}')
        print(f'[PACS Dataset] All domains: {all_domains}')
        
        # Get available domains (exclude leave_out)
        available_domains = [d for d in all_domains if d != args.leave_out]
        print(f'[PACS Dataset] Available training domains: {available_domains}')
        
        # 為每個 rank 分配 domain（如果節點數 > 可用 domain 數，則循環分配）
        domain_for_rank = available_domains[rank % len(available_domains)]
        print(f'[PACS Dataset] Rank {rank} assigned to domain: {domain_for_rank}')
        
        # 計算有多少節點與此 rank 共享相同的 domain
        # 統計具有相同 domain 分配的節點
        nodes_with_same_domain = []
        for r in range(size):
            if available_domains[r % len(available_domains)] == domain_for_rank:
                nodes_with_same_domain.append(r)
        num_nodes_sharing_domain = len(nodes_with_same_domain)
        local_index_in_domain = nodes_with_same_domain.index(rank)  # 此 rank 在共享相同 domain 的節點中的索引
        
        if num_nodes_sharing_domain > 1:
            print(f'[PACS Dataset] Note: {num_nodes_sharing_domain} nodes share domain "{domain_for_rank}" (rank {rank} is index {local_index_in_domain} of {num_nodes_sharing_domain})')
        print('-' * 60)
        
        # 訓練用的資料轉換（符合 Dassl 的 PACS 設定）
        print(f'[PACS Dataset] Rank {rank}: Setting up training transforms...')
        print(f'[PACS Dataset] Rank {rank}: Using Dassl-compatible transforms (random_flip, random_translation, normalize)')
        # 等同於 Dassl 的 Random2DTranslation：
        # 50% 機率：縮放到 1.125x (252x252) 然後隨機裁剪到 224x224
        # 50% 機率：直接縮放到 224x224
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),  # 基礎縮放到 224x224
            # RandomApply，p=0.5：50% 機率應用放大 + 隨機裁剪
            transforms.RandomApply([
                transforms.Compose([
                    transforms.Resize((252, 252)),  # 224 * 1.125 = 252 (放大)
                    transforms.RandomCrop(224)      # 隨機裁剪回 224x224
                ])
            ], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 載入此 rank 的 domain 的訓練資料
        print(f'[PACS Dataset] Rank {rank}: Loading training data from domain "{domain_for_rank}"...')
        print(f'[PACS Dataset] Rank {rank}: Dataset root: {args.datasetRoot}')
        full_domain_dataset = PACSDataset(root=args.datasetRoot, 
                                   dataset_name=domain_for_rank, 
                                   transform=transform_train)
        
        # 如果多個節點共享相同的 domain，則在它們之間分割該 domain 的資料
        if num_nodes_sharing_domain > 1:
            # 在共享此 domain 的節點之間平均分割 domain 的資料
            partition_sizes = [1.0 / num_nodes_sharing_domain for _ in range(num_nodes_sharing_domain)]
            partitioner = DataPartitioner(full_domain_dataset, partition_sizes, seed=args.randomSeed, isNonIID=False)
            train_dataset = partitioner.use(local_index_in_domain)
            print(f'[PACS Dataset] Rank {rank}: Partitioned domain "{domain_for_rank}" data: using partition {local_index_in_domain} of {num_nodes_sharing_domain}')
        else:
            # 只有一個節點使用此 domain，使用完整資料集
            train_dataset = full_domain_dataset
            print(f'[PACS Dataset] Rank {rank}: Using full domain "{domain_for_rank}" data (no partitioning needed)')
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=args.bs, 
            shuffle=True, 
            pin_memory=True
        )
        
        print(f'[PACS Dataset] Rank {rank}: Successfully created training DataLoader')
        print(f'[PACS Dataset] Rank {rank}: Training samples: {len(train_dataset)}')
        print(f'[PACS Dataset] Rank {rank}: Batch size: {args.bs}')
        print(f'[PACS Dataset] Rank {rank}: Number of batches per epoch: {len(train_loader)}')
        print('-' * 60)
        
        # Load test data from leave_out domain
        print(f'[PACS Dataset] Rank {rank}: Loading test data from leave-out domain "{args.leave_out}"...')
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_dataset = PACSDataset(root=args.datasetRoot, 
                                  dataset_name=args.leave_out, 
                                  transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=64, 
            shuffle=False, 
            pin_memory=True
        )
        
        print(f'[PACS Dataset] Rank {rank}: Successfully created test DataLoader')
        print(f'[PACS Dataset] Rank {rank}: Test samples: {len(test_dataset)}')
        print(f'[PACS Dataset] Rank {rank}: Test batch size: 64')
        print(f'[PACS Dataset] Rank {rank}: Number of test batches: {len(test_loader)}')
        print('=' * 60)
        print(f'[PACS Dataset] Rank {rank}: PACS dataset loading completed successfully!')
        print('=' * 60)

    return train_loader, test_loader

def load_dataset_single_process(args):
    """
    Load dataset for single process training.
    This is the main data loading function for single-process decentralized learning.
    
    Returns:
        For PACS: dictionary of {domain_name: (train_loader, test_loader)}
        For other datasets: (train_loader, test_loader) tuple
    """
    print('==> load train data (single process mode)')
    
    if args.dataset == 'pacs':
        print('=' * 60)
        print(f'[PACS Dataset] Initializing PACS dataset loading (single process)...')
        print('=' * 60)
        
        # PACS domains
        all_domains = ['art_painting', 'cartoon', 'photo', 'sketch']
        if not args.leave_out:
            raise ValueError('--leave_out must be specified when using PACS dataset.')
        if args.leave_out not in all_domains:
            raise ValueError(f'Invalid leave_out domain: {args.leave_out}. Valid options: {all_domains}')
        
        print(f'[PACS Dataset] Leave-out domain: {args.leave_out}')
        print(f'[PACS Dataset] All domains: {all_domains}')
        
        # Get available domains (exclude leave_out)
        available_domains = [d for d in all_domains if d != args.leave_out]
        print(f'[PACS Dataset] Available training domains: {available_domains}')
        print('-' * 60)
        
        # Training transforms (same as MPI version)
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomApply([
                transforms.Compose([
                    transforms.Resize((252, 252)),
                    transforms.RandomCrop(224)
                ])
            ], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Test transforms
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Load test data once (shared by all train domains/nodes)
        print(f'[PACS Dataset] Loading test data from leave-out domain "{args.leave_out}"...')
        test_dataset = PACSDataset(
            root=args.datasetRoot,
            dataset_name=args.leave_out,
            transform=transform_test
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            pin_memory=True
        )
        print(f'[PACS Dataset] Test samples: {len(test_dataset)}, Test batches: {len(test_loader)}')

        # number of virtual nodes for single-process
        num_nodes = getattr(args, "num_nodes", len(available_domains)) or len(available_domains)
        split_mode = getattr(args, "node_split_mode", "class_balanced")
        enable_virtual_node_split = (getattr(args, "graphid", None) == 6 and num_nodes > len(available_domains))

        # Case A: original behavior (one loader per train domain)
        # Keep this as default, and ONLY enable virtual-node split for graphid=6.
        if not enable_virtual_node_split:
            domain_loaders = {}
            for domain in available_domains:
                print(f'[PACS Dataset] Loading training data from domain "{domain}"...')
                print(f'[PACS Dataset] Dataset root: {args.datasetRoot}')
                
                train_dataset = PACSDataset(
                    root=args.datasetRoot, 
                    dataset_name=domain, 
                    transform=transform_train
                )
                
                if getattr(args, "sampler_type", "random") == "random_class":
                    train_sampler = RandomClassSampler(train_dataset, batch_size=args.bs, n_ins=args.n_ins)
                    train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=args.bs,
                        sampler=train_sampler,
                        shuffle=False,
                        pin_memory=True
                    )
                else:
                    train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=args.bs,
                        shuffle=True,
                        pin_memory=True
                    )
                
                print(f'[PACS Dataset] Domain "{domain}": {len(train_dataset)} training samples, {len(train_loader)} batches per epoch')
                domain_loaders[domain] = (train_loader, test_loader)

            node_to_domain = {domain: domain for domain in available_domains}
            print('=' * 60)
            print(f'[PACS Dataset] Single process dataset loading completed!')
            print(f'[PACS Dataset] Loaded {len(available_domains)} training domains')
            print('=' * 60)

            if num_nodes > len(available_domains) and getattr(args, "graphid", None) != 6:
                print(
                    f'[PACS Dataset] Note: num_nodes={num_nodes} is ignored for graphid={getattr(args, "graphid", None)}. '
                    f'Virtual-node split is only enabled when graphid=6.'
                )
            return domain_loaders, node_to_domain

        # Case B: virtual-node mode (num_nodes > num_domains)
        print(f'[PACS Dataset] Virtual-node mode: num_nodes={num_nodes}, split_mode={split_mode}')
        node_to_domain, domain_to_nodes = assign_nodes_to_domains(available_domains, num_nodes)
        print(f'[PACS Dataset] Node assignment: {node_to_domain}')

        # Load full datasets once per domain
        full_train_datasets = {}
        for domain in available_domains:
            print(f'[PACS Dataset] Loading training data from domain "{domain}"...')
            print(f'[PACS Dataset] Dataset root: {args.datasetRoot}')
            
            full_train_datasets[domain] = PACSDataset(
                root=args.datasetRoot, 
                dataset_name=domain, 
                transform=transform_train
            )

        # Build node-level loaders
        node_loaders = {}
        for domain in available_domains:
            node_names = domain_to_nodes[domain]
            split_seed = int(args.randomSeed) if args.randomSeed is not None else 1234
            node_indices = partition_domain_dataset_for_nodes(
                full_train_datasets[domain], node_names, split_mode=split_mode, seed=split_seed
            )

            if getattr(args, "debug_topology", False):
                print(f'[PACS Dataset] Class histogram per node in domain "{domain}":')
                # domain-level class counts for rough balance check target
                domain_targets = getattr(full_train_datasets[domain], "targets", [])
                domain_hist = defaultdict(int)
                for y in domain_targets:
                    domain_hist[int(y)] += 1
                print(f'  domain_total_hist: {dict(sorted(domain_hist.items()))}')

            for node_name in node_names:
                subset = Subset(full_train_datasets[domain], node_indices[node_name])
                if len(subset) == 0:
                    raise ValueError(f"Empty subset for {node_name} in domain {domain}")

                if getattr(args, "debug_topology", False):
                    node_hist = _compute_label_histogram_for_indices(full_train_datasets[domain], node_indices[node_name])
                    has_all_classes = (len(node_hist.keys()) == len(domain_hist.keys()))
                    print(
                        f'  {node_name}: n={len(subset)}, classes={len(node_hist)} '
                        f'({"OK" if has_all_classes else "MISSING"}) hist={node_hist}'
                    )

                if getattr(args, "sampler_type", "random") == "random_class":
                    train_sampler = RandomClassSampler(subset, batch_size=args.bs, n_ins=args.n_ins)
                    train_loader = torch.utils.data.DataLoader(
                        subset,
                        batch_size=args.bs,
                        sampler=train_sampler,
                        shuffle=False,
                        pin_memory=True
                    )
                else:
                    train_loader = torch.utils.data.DataLoader(
                        subset,
                        batch_size=args.bs,
                        shuffle=True,
                        pin_memory=True
                    )

                node_loaders[node_name] = (train_loader, test_loader)
                print(
                    f'[PACS Dataset] {node_name} <- {domain}: '
                    f'{len(subset)} samples, {len(train_loader)} batches/epoch'
                )
        
        print('=' * 60)
        print(f'[PACS Dataset] Single process dataset loading completed!')
        print(f'[PACS Dataset] Loaded {len(node_loaders)} training nodes')
        print('=' * 60)
        
        return node_loaders, node_to_domain
    
    elif args.dataset == 'vlcs':
        print('=' * 60)
        print(f'[VLCS Dataset] Initializing VLCS dataset loading...')
        print('=' * 60)

        # VLCS domains
        all_domains = ['caltech', 'labelme', 'pascal', 'sun']
        if not args.leave_out:
            raise ValueError('--leave_out must be specified when using VLCS dataset.')

        leave_out = str(args.leave_out).lower()
        if leave_out not in all_domains:
            raise ValueError(
                f'Invalid leave_out domain: {args.leave_out}. Valid options: {all_domains}'
            )

        available_domains = [d for d in all_domains if d != leave_out]
        print(f'[VLCS Dataset] Leave-out domain: {leave_out}')
        print(f'[VLCS Dataset] Available training domains: {available_domains}')
        print('-' * 60)

        # Training transforms (use the same style as PACS)
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomApply([
                transforms.Compose([
                    transforms.Resize((252, 252)),
                    transforms.RandomCrop(224),
                ])
            ], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Test transforms
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Target test set: leave-out domain full + test
        test_dataset = VLCSFullTestDataset(
            root=args.datasetRoot,
            dataset_name=leave_out,
            transform=transform_test,
            full_dir_name='full',
            test_dir_name='test',
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            pin_memory=True,
        )
        print(f'[VLCS Dataset] Test samples: {len(test_dataset)}, Test batches: {len(test_loader)}')

        # number of virtual nodes for single-process
        num_nodes = getattr(args, "num_nodes", len(available_domains)) or len(available_domains)
        split_mode = getattr(args, "node_split_mode", "class_balanced")
        enable_virtual_node_split = (getattr(args, "graphid", None) == 6 and num_nodes > len(available_domains))

        # Case A: original behavior (one loader per train domain)
        if not enable_virtual_node_split:
            domain_loaders = {}
            for domain in available_domains:
                print(f'[VLCS Dataset] Loading training data from domain "{domain}" (full + test)...')
                print(f'[VLCS Dataset] Dataset root: {args.datasetRoot}')
                train_dataset = VLCSFullTestDataset(
                    root=args.datasetRoot,
                    dataset_name=domain,
                    transform=transform_train,
                    full_dir_name='full',
                    test_dir_name='test',
                )

                if getattr(args, "sampler_type", "random") == "random_class":
                    train_sampler = RandomClassSampler(train_dataset, batch_size=args.bs, n_ins=args.n_ins)
                    train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=args.bs,
                        sampler=train_sampler,
                        shuffle=False,
                        pin_memory=True
                    )
                else:
                    train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=args.bs,
                        shuffle=True,
                        pin_memory=True
                    )

                print(f'[VLCS Dataset] Domain "{domain}": {len(train_dataset)} samples, {len(train_loader)} batches/epoch')
                domain_loaders[domain] = (train_loader, test_loader)

            node_to_domain = {domain: domain for domain in available_domains}
            print('=' * 60)
            print(f'[VLCS Dataset] Single process dataset loading completed!')
            print(f'[VLCS Dataset] Loaded {len(available_domains)} training domains')
            print('=' * 60)

            if num_nodes > len(available_domains) and getattr(args, "graphid", None) != 6:
                print(
                    f'[VLCS Dataset] Note: num_nodes={num_nodes} is ignored for graphid={getattr(args, "graphid", None)}. '
                    f'Virtual-node split is only enabled when graphid=6.'
                )

            return domain_loaders, node_to_domain

        # Case B: virtual-node mode (num_nodes > num_domains)
        print(f'[VLCS Dataset] Virtual-node mode: num_nodes={num_nodes}, split_mode={split_mode}')
        node_to_domain, domain_to_nodes = assign_nodes_to_domains(available_domains, num_nodes)
        print(f'[VLCS Dataset] Node assignment: {node_to_domain}')

        # Load full+test datasets once per domain
        full_train_datasets = {}
        for domain in available_domains:
            print(f'[VLCS Dataset] Loading training data from domain "{domain}"...')
            print(f'[VLCS Dataset] Dataset root: {args.datasetRoot}')
            full_train_datasets[domain] = VLCSFullTestDataset(
                root=args.datasetRoot,
                dataset_name=domain,
                transform=transform_train,
                full_dir_name='full',
                test_dir_name='test',
            )

        node_loaders = {}
        for domain in available_domains:
            node_names = domain_to_nodes[domain]
            split_seed = int(args.randomSeed) if args.randomSeed is not None else 1234
            node_indices = partition_domain_dataset_for_nodes(
                full_train_datasets[domain],
                node_names,
                split_mode=split_mode,
                seed=split_seed
            )

            if getattr(args, "debug_topology", False):
                print(f'[VLCS Dataset] Class histogram per node in domain "{domain}":')
                domain_hist = defaultdict(int)
                for y in getattr(full_train_datasets[domain], "targets", []):
                    domain_hist[int(y)] += 1
                print(f'  domain_total_hist: {dict(sorted(domain_hist.items()))}')

            for node_name in node_names:
                subset = Subset(full_train_datasets[domain], node_indices[node_name])
                if len(subset) == 0:
                    raise ValueError(f"Empty subset for {node_name} in domain {domain}")

                if getattr(args, "debug_topology", False):
                    node_hist = _compute_label_histogram_for_indices(full_train_datasets[domain], node_indices[node_name])
                    has_all_classes = (len(node_hist.keys()) == len(domain_hist.keys()))
                    print(
                        f'  {node_name}: n={len(subset)}, classes={len(node_hist)} '
                        f'({"OK" if has_all_classes else "MISSING"}) hist={node_hist}'
                    )

                if getattr(args, "sampler_type", "random") == "random_class":
                    train_sampler = RandomClassSampler(subset, batch_size=args.bs, n_ins=args.n_ins)
                    train_loader = torch.utils.data.DataLoader(
                        subset,
                        batch_size=args.bs,
                        sampler=train_sampler,
                        shuffle=False,
                        pin_memory=True
                    )
                else:
                    train_loader = torch.utils.data.DataLoader(
                        subset,
                        batch_size=args.bs,
                        shuffle=True,
                        pin_memory=True
                    )

                node_loaders[node_name] = (train_loader, test_loader)
                print(
                    f'[VLCS Dataset] {node_name} <- {domain}: '
                    f'{len(subset)} samples, {len(train_loader)} batches/epoch'
                )

        print('=' * 60)
        print(f'[VLCS Dataset] Single process dataset loading completed!')
        print(f'[VLCS Dataset] Loaded {len(node_loaders)} training nodes')
        print('=' * 60)

        return node_loaders, node_to_domain
    
    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root=args.datasetRoot, 
            train=True, 
            download=True, 
            transform=transform_train
        )
        if getattr(args, "sampler_type", "random") == "random_class":
            train_sampler = RandomClassSampler(trainset, batch_size=args.bs, n_ins=args.n_ins)
            train_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=args.bs,
                sampler=train_sampler,
                shuffle=False,
                pin_memory=True
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=args.bs,
                shuffle=True,
                pin_memory=True
            )
        
        print('==> load test data')
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(
            root=args.datasetRoot, 
            train=False, 
            download=True, 
            transform=transform_test
        )
        test_loader = torch.utils.data.DataLoader(
            testset, 
            batch_size=64, 
            shuffle=False, 
            pin_memory=True
        )
        
        return train_loader, test_loader
    
    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        trainset = torchvision.datasets.CIFAR100(
            root=args.datasetRoot, 
            train=True, 
            download=True, 
            transform=transform_train
        )
        if getattr(args, "sampler_type", "random") == "random_class":
            train_sampler = RandomClassSampler(trainset, batch_size=args.bs, n_ins=args.n_ins)
            train_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=args.bs,
                sampler=train_sampler,
                shuffle=False,
                pin_memory=True
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=args.bs,
                shuffle=True,
                pin_memory=True
            )
        
        print('==> load test data')
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        testset = torchvision.datasets.CIFAR100(
            root=args.datasetRoot, 
            train=False, 
            download=True, 
            transform=transform_test
        )
        test_loader = torch.utils.data.DataLoader(
            testset, 
            batch_size=64, 
            shuffle=False, 
            pin_memory=True
        )
        
        return train_loader, test_loader
    
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported in single process mode yet")

def select_model(num_class, args):
    # Get style shift parameters
    use_style_shift = getattr(args, 'use_style_shift', False)
    style_shift_prob = getattr(args, 'style_shift_prob', 0.5)
    style_shift_ratio = getattr(args, 'style_shift_ratio', 0.5)
    # Get style explore parameters
    style_explore_alpha = getattr(args, 'style_explore_alpha', 3.0)
    style_explore_ratio = getattr(args, 'style_explore_ratio', 0.5)
    # Get mixstyle parameter
    mixstyle_alpha = getattr(args, 'mixstyle_alpha', 0.1)
    # Get pretrained parameter
    pretrained = getattr(args, 'pretrained', False)
    
    if args.model == 'VGG':
        model = vggnet.VGG(16, num_class)
    elif args.model == 'res':
        # 獲取 resnet_type 參數，默認為 'simplified'
        resnet_type = getattr(args, 'resnet_type', 'simplified')
        
        if args.dataset == 'cifar10':
            if resnet_type == 'standard':
                from models.resnet import StandardResNetWrapper
                model = StandardResNetWrapper(18, num_class, 
                                             use_style_shift=use_style_shift,
                                             style_shift_prob=style_shift_prob,
                                             style_shift_ratio=style_shift_ratio,
                                             style_explore_alpha=style_explore_alpha,
                                             style_explore_ratio=style_explore_ratio,
                                             mixstyle_alpha=mixstyle_alpha,
                                             pretrained=pretrained)
            else:
                # model = large_resnet.ResNet18()
                model = resnet.ResNet(18, num_class,
                                     use_style_shift=use_style_shift,
                                     style_shift_prob=style_shift_prob,
                                     style_shift_ratio=style_shift_ratio,
                                     style_explore_alpha=style_explore_alpha,
                                     style_explore_ratio=style_explore_ratio)
        elif args.dataset == 'pacs' or args.dataset == 'vlcs':
            if resnet_type == 'standard':
                from models.resnet import StandardResNetWrapper
                model = StandardResNetWrapper(18, num_class,
                                             use_style_shift=use_style_shift,
                                             style_shift_prob=style_shift_prob,
                                             style_shift_ratio=style_shift_ratio,
                                             style_explore_alpha=style_explore_alpha,
                                             style_explore_ratio=style_explore_ratio,
                                             mixstyle_alpha=mixstyle_alpha,
                                             pretrained=pretrained)
            else:
                model = resnet.ResNet(18, num_class,
                                     use_style_shift=use_style_shift,
                                     style_shift_prob=style_shift_prob,
                                     style_shift_ratio=style_shift_ratio,
                                     style_explore_alpha=style_explore_alpha,
                                     style_explore_ratio=style_explore_ratio)
        elif args.dataset == 'imagenet':
            # ImageNet 默認使用標準 ResNet
            model = models.resnet18()
    elif args.model == 'wrn':
        model = wrn.Wide_ResNet(28,10,0,num_class)
    elif args.model == 'mlp':
        if args.dataset == 'emnist':
            model = MLP.MNIST_MLP(47)
    return model

def select_graph(graphid, num_nodes=None, radius=None, seed=None):
    """
    根據 graphid 選擇圖形拓撲結構。
    
    特殊 graphid：
    - -1: 3 節點全連接圖（用於 PACS）
    - 6: 隨機幾何圖（Random Geometric Graph, RGG），9 個節點，半徑=0.8
    
    參數：
        graphid: 圖形 ID
        num_nodes: 節點數量（RGG 使用，預設=9）
        radius: 連接半徑（RGG 使用，預設=0.8）
        seed: 隨機種子（RGG 使用，如果為 None 則使用預設值）
    """
    # 特殊情況：graphid == -1 為 3 節點全連接圖（用於 PACS）
    if graphid == -1:
        # 3 節點全連接圖
        # 分解為 3 個 matchings：
        # - [(0,1)] (節點 2 孤立)
        # - [(0,2)] (節點 1 孤立)
        # - [(1,2)] (節點 0 孤立)
        return [
            [(0, 1)],  # matching 1: 節點 0 和 1 之間的邊
            [(0, 2)],  # matching 2: 節點 0 和 2 之間的邊
            [(1, 2)]   # matching 3: 節點 1 和 2 之間的邊
        ]
    
    # 特殊情況：graphid == 6 為隨機幾何圖（Random Geometric Graph, RGG）
    if graphid == 6:
        num_nodes = num_nodes or 9
        radius = radius or 0.8
        seed = seed or 42  # 預設種子以確保可重現性
        
        # 生成 RGG，確保連通性
        max_attempts = 100
        G = None
        for attempt in range(max_attempts):
            try:
                G = nx.random_geometric_graph(num_nodes, radius, seed=seed + attempt)
                if nx.is_connected(G):
                    break
            except:
                continue
        else:
            raise RuntimeError(f"在 {max_attempts} 次嘗試後仍無法生成連通的 RGG，參數：num_nodes={num_nodes}, radius={radius}")
        
        # 將 NetworkX 圖轉換為 matchings 格式
        # matching 是一組不相交的邊（沒有共享節點）
        matchings = []
        G_remaining = G.copy()
        
        # 使用貪心演算法將圖分解為 matchings
        while G_remaining.number_of_edges() > 0:
            # 尋找最大匹配
            matching = nx.max_weight_matching(G_remaining)
            if len(matching) > 0:
                # 將 matching 集合轉換為元組列表
                matching_list = list(matching)
                matchings.append(matching_list)
                # 從剩餘圖中移除已匹配的邊
                G_remaining.remove_edges_from(matching_list)
            else:
                # 如果找不到匹配，將剩餘的邊單獨添加
                # 這種情況很少發生，但用於處理邊緣情況
                remaining_edges = list(G_remaining.edges())
                for edge in remaining_edges:
                    matchings.append([edge])
                break
        
        # 驗證所有邊都被覆蓋
        all_edges_in_matchings = set()
        for matching in matchings:
            for edge in matching:
                # 標準化邊的方向（確保較小的節點在前）
                normalized_edge = tuple(sorted(edge))
                all_edges_in_matchings.add(normalized_edge)
        
        original_edges = set(tuple(sorted(edge)) for edge in G.edges())
        if all_edges_in_matchings != original_edges:
            raise RuntimeError("Matchings 分解失敗：並非所有邊都被覆蓋")
        
        return matchings
    
    # pre-defined base network topologies
    # you can add more by extending the list
    Graphs =[ 
             # graph 0: 
             # 8-node erdos-renyi graph as shown in Fig. 1(a) in the main paper
             [[(1, 5), (6, 7), (0, 4), (2, 3)], 
              [(1, 7), (3, 6)], 
              [(1, 0), (3, 7), (5, 6)], 
              [(1, 2), (7, 0)], 
              [(3, 1)]],

             # graph 1:
             # 16-node gemetric graph as shown in Fig. A.3(a) in Appendix
             [[(4, 8), (6, 11), (7, 13), (0, 12), (5, 14), (10, 15), (2, 3), (1, 9)], 
              [(11, 13), (14, 2), (5, 6), (15, 3), (10, 9)], 
              [(11, 8), (2, 5), (13, 4), (14, 3), (0, 10)], 
              [(11, 5), (15, 14), (13, 8)], 
              [(2, 11)]],

             # graph 2:
             # 16-node gemetric graph as shown in Fig. A.3(b) in Appendix
             [[(2, 7), (12, 15), (3, 13), (5, 6), (8, 0), (9, 4), (11, 14), (1, 10)], 
              [(8, 6), (0, 11), (3, 2), (5, 4), (15, 14), (1, 9)], 
              [(8, 3), (0, 6), (11, 2), (4, 1), (12, 14)], 
              [(8, 11), (6, 3), (0, 5)], 
              [(8, 2), (0, 3), (6, 7), (11, 12)], 
              [(8, 5), (6, 4), (0, 2), (11, 7)], 
              [(8, 15), (3, 7), (0, 4), (6, 2)], 
              [(8, 14), (5, 3), (11, 6), (0, 9)], 
              [(8, 7), (15, 11), (2, 5), (4, 3), (1, 0), (13, 6)], 
              [(12, 8)]],

             # graph 3:
             # 16-node gemetric graph as shown in Fig. A.3(c) in Appendix
             [[(3, 12), (4, 8), (1, 13), (5, 7), (9, 10), (11, 14), (6, 15), (0, 2)], 
              [(7, 14), (2, 6), (5, 13), (8, 10), (1, 15), (0, 11), (3, 9), (4, 12)], 
              [(2, 7), (3, 15), (9, 13), (6, 11), (4, 14), (10, 12), (1, 8), (0, 5)], 
              [(5, 14), (1, 12), (13, 8), (9, 4), (2, 11), (7, 0)], 
              [(5, 1), (14, 8), (13, 12), (10, 4), (6, 7)], 
              [(5, 9), (14, 1), (13, 3), (8, 2), (11, 7)], 
              [(5, 12), (14, 13), (1, 9), (8, 0)], 
              [(5, 2), (14, 10), (1, 3), (9, 8), (13, 15)], 
              [(5, 8), (14, 12), (1, 4), (13, 10)], 
              [(5, 3), (14, 2), (9, 12), (1, 10), (13, 4)], 
              [(5, 6), (14, 0), (8, 12), (1, 2)], 
              [(5, 15), (9, 14)], 
              [(11, 5)]],

             # graph 4:
             # 16-node erdos-renyi graph as shown in Fig 3.(b) in the main paper
             [[(2, 7), (3, 15), (13, 14), (8, 9), (1, 5), (0, 10), (6, 12), (4, 11)], 
             [(12, 11), (5, 6), (14, 1), (9, 10), (15, 2), (8, 13)], 
             [(12, 5), (11, 6), (1, 8), (9, 3), (2, 10)], 
             [(12, 14), (11, 9), (5, 15), (0, 6), (1, 7)], 
             [(12, 8), (5, 2), (11, 14), (1, 6)], 
             [(12, 15), (13, 11), (10, 5), (3, 14)], 
             [(12, 9)], 
             [(0, 12)]], 

             # graph 5, 8-node ring
             [[(0, 1), (2, 3), (4, 5), (6, 7)], 
              [(0, 7), (2, 1), (4, 3), (6, 5)]]

            ]
    
    # 驗證預定義圖形的 graphid
    if graphid < 0 or graphid >= len(Graphs):
        raise ValueError(f"無效的 graphid: {graphid}。有效範圍：0-{len(Graphs)-1}，或特殊 ID：-1 (3 節點全連接), 6 (RGG)")
            
    return Graphs[graphid] 

def comp_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0 
        self.avg = 0 
        self.sum = 0 
        self.count = 0 

    def update(self, val, n=1):
        self.val = val 
        self.sum += val * n 
        self.count += n
        self.avg = self.sum / self.count

class Recorder(object):
    def __init__(self, args, rank):
        self.record_accuracy = list()
        self.record_timing = list()
        self.record_comp_timing = list()
        self.record_comm_timing = list()
        self.record_losses = list()
        self.record_trainacc = list()
        self.total_record_timing = list()
        self.args = args
        self.rank = rank
        self.saveFolderName = args.savePath + args.name + '_' + args.model
        # Create directory if it doesn't exist (for single process, rank can be any value)
        # For MPI, only rank 0 creates; for single process, we always create
        if os.path.isdir(self.saveFolderName)==False and getattr(self.args, 'save', self.args.savePath is not None):
            os.makedirs(self.saveFolderName, exist_ok=True)
    
    def add_new(self, record_time, comp_time, comm_time, epoch_time, top1, losses, test_acc):
        self.total_record_timing.append(record_time)
        self.record_timing.append(epoch_time)
        self.record_comp_timing.append(comp_time)
        self.record_comm_timing.append(comm_time)
        # 使用 float() 將 Tensor 轉為純數值，並移出 GPU
        self.record_trainacc.append(float(top1))
        self.record_losses.append(float(losses))
        self.record_accuracy.append(float(test_acc))

    def save_to_file(self):
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-recordtime.log', self.total_record_timing, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-time.log',  self.record_timing, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-comptime.log',  self.record_comp_timing, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-commtime.log',  self.record_comm_timing, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-acc.log',  self.record_accuracy, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-losses.log',  self.record_losses, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-tacc.log',  self.record_trainacc, delimiter=',')
        with open(self.saveFolderName+'/ExpDescription', 'w') as f:
            f.write(str(self.args)+ '\n')
            f.write(self.args.description + '\n')


def test(model, test_loader):
    model.eval()
    top1 = AverageMeter()
    # correct = 0
    # total = 0
    with torch.no_grad():  # 確保不建立計算圖，節省顯存
        for batch_idx, batch in enumerate(test_loader):
            inputs, targets, _ = unpack_batch(batch)
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            acc1 = comp_accuracy(outputs, targets)
            top1.update(acc1[0], inputs.size(0))
            # 每個 batch 後清理臨時變量，避免記憶體累積
            del inputs, targets, outputs
    return top1.avg


def unpack_batch(batch):
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Expected batch to be list/tuple, got {type(batch)}")

    if len(batch) == 2:
        inputs, targets = batch
        meta = None
    elif len(batch) >= 3:
        inputs, targets, meta = batch[0], batch[1], batch[2]
    else:
        raise ValueError("Batch must have at least 2 elements: inputs and targets")

    return inputs, targets, meta
