import numpy as np
import time
import math
import torch
from mpi4py import MPI
from compressors import get_top_k
from typing import Dict

from comm_helpers import flatten_tensors, unflatten_tensors
from style_stats import unflatten_style_stats
        
class Communicator(object):
    """ Classs designed for communicating local models at workers """
    def __init__(self, rank, size):
        self.comm = MPI.COMM_WORLD
        self.rank = rank
        self.size = size
        # Style statistics storage
        self.local_style_vec = None  # Current rank's style statistics (1D tensor or None)
        self.neighbor_style_vecs = {}  # Dict storing received neighbor style statistics {neighbor_rank: style_vec_tensor}
        self.neighbor_style_stats = {}  # Dict storing unflattened neighbor style statistics {neighbor_rank: {layer_name: {stat_name: tensor}}}
        self.channels_per_layer = None  # Dict mapping layer_name -> channel_count, e.g. {"layer1": 64, "layer2": 128, "layer3": 256}
    
    def set_style_channels(self, channels_per_layer: Dict[str, int]):
        """Set channel information for each layer to enable style statistics unflattening.
        
        Args:
            channels_per_layer: Dict mapping layer_name -> channel_count,
                e.g. {"layer1": 64, "layer2": 128, "layer3": 256}
        """
        self.channels_per_layer = channels_per_layer

    def communicate(self, model, style_vec=None):
        """
        Communicate model parameters and/or style statistics with neighbors.
        
        Logic:
        - If style_vec is provided: only exchange style statistics (no model parameters)
        - If style_vec is None: only exchange model parameters (no style statistics)
        
        Args:
            model: The model to exchange parameters for (only used if style_vec is None)
            style_vec: Style statistics vector to exchange (optional)
        """
        # Store local style vector
        self.local_style_vec = style_vec
        
        if style_vec is not None:
            # Only exchange style statistics, skip model parameter exchange
            self.prepare_style_buffer()
            comm_time = self.averaging()
        else:
            # Only exchange model parameters, skip style statistics exchange
            # stack all model parameters into one tensor list
            self.tensor_list = list()
            # Add backbone model parameters
            for param in model.parameters():
                self.tensor_list.append(param.data)
            # Add diffusion model parameters if exists
            if hasattr(model, 'diffusion_model') and model.diffusion_model is not None:
                for param in model.diffusion_model.parameters():
                    self.tensor_list.append(param.data)

            # necessary preprocessing
            self.prepare_comm_buffer()

            # communication happens here
            # record the communication time
            comm_time = self.averaging()

            # Update local models
            self.reset_model()

        return comm_time

    def prepare_comm_buffer(self):
        raise NotImplemented

    def averaging(self):
        raise NotImplemented

    def reset_model(self):
        raise NotImplemented
    
    def prepare_style_buffer(self):
        """Prepare style statistics communication buffer"""
        raise NotImplemented

    
        

class centralizedCommunicator(Communicator):
    """ Perform AllReduce at each iteration """
    def __init__(self, rank, size):
        super(centralizedCommunicator, self).__init__(rank, size)

    
    def prepare_comm_buffer(self):
        # faltten tensors
        self.send_buffer = flatten_tensors(self.tensor_list).cpu()

    def averaging(self):
        self.comm.barrier()
        tic = time.time()

        # 根據是否有 style_vec 決定交換內容
        if self.local_style_vec is not None:
            # 只交換風格統計量，不交換模型參數
            self.neighbor_style_vecs.clear()
            self.neighbor_style_stats.clear()
            
            # AllReduce for style statistics (centralized communication)
            self.style_recv_buffer = self.comm.allreduce(self.style_send_buffer, op=MPI.SUM)
            self.style_recv_buffer.div_(self.size)
            
            # Store received style statistics (for centralized, we store the averaged result)
            # Note: In centralized setting, all workers get the same averaged style stats
            self.neighbor_style_vecs[0] = self.style_recv_buffer.clone()
            
            # Unflatten style statistics if channel info is available
            if self.channels_per_layer is not None:
                try:
                    layer_order = ["layer1", "layer2", "layer3"]
                    unflattened_stats = unflatten_style_stats(
                        self.style_recv_buffer, 
                        layer_order=layer_order,
                        channels_per_layer=self.channels_per_layer
                    )
                    self.neighbor_style_stats[0] = unflattened_stats
                except Exception as e:
                    pass
        else:
            # 只交換模型參數，不交換風格統計量
            # AllReduce
            self.recv_buffer = self.comm.allreduce(self.send_buffer, op=MPI.SUM)
            self.recv_buffer.div_(self.size)
        
        self.comm.barrier()
        toc = time.time()

        return toc - tic

    def reset_model(self):
        # Reset local models to be the averaged model
        unflattened = unflatten_tensors(
            self.recv_buffer.cuda(), self.tensor_list)
        for f, t in zip(unflattened, self.tensor_list):
            t.set_(f)
        # 明確釋放臨時變量，避免記憶體累積
        del unflattened
    
    def prepare_style_buffer(self):
        """Prepare style statistics communication buffer"""
        if self.local_style_vec is None:
            return
        
        # Convert to CPU tensor if needed and detach from computation graph
        if self.local_style_vec.is_cuda:
            self.style_send_buffer = self.local_style_vec.detach().cpu().clone()
        else:
            self.style_send_buffer = self.local_style_vec.detach().clone()
    
class decenCommunicator(Communicator):
    """ decentralized averaging according to a topology sequence """
    def __init__(self, rank, size, topology):
        super(decenCommunicator, self).__init__(rank, size)
        self.topology = topology
        self.neighbor_weight = topology.neighbor_weight
        self.iter = 0


    def prepare_comm_buffer(self):
        # faltten tensors 壓扁成一條超級長的 1D 向量
        self.send_buffer = flatten_tensors(self.tensor_list).cpu()
        self.recv_buffer = torch.zeros_like(self.send_buffer)

    def _compute_mh_degrees(self, active_flags):
        """
        Compute node degrees on the active graph (union of active subgraphs).
        Since each subgraph is a matching, degree increments by one per active incident edge.
        """
        degrees = [0 for _ in range(self.size)]
        for graph_id, flag in enumerate(active_flags):
            if flag == 0:
                continue
            neighbors = self.topology.neighbors_info[graph_id]
            for node in range(self.size):
                if neighbors[node] != -1:
                    degrees[node] += 1
        return degrees


    def averaging(self, active_flags):
        # 等待所有 worker 都準備好
        self.comm.barrier()
        tic = time.time()

        # 根據是否有 style_vec 決定交換內容
        if self.local_style_vec is not None:
            # 只交換風格統計量，不交換模型參數
            self.neighbor_style_vecs.clear()
            self.neighbor_style_stats.clear()
            
            for graph_id, flag in enumerate(active_flags):
                if flag == 0:
                    continue
                else:
                    # 如果我在這個子圖有鄰居
                    if self.topology.neighbors_info[graph_id][self.rank] != -1:
                        neighbor_rank = self.topology.neighbors_info[graph_id][self.rank]
                        
                        # 交換風格統計量
                        style_recv_tmp = self.comm.sendrecv(self.style_send_buffer, source=neighbor_rank, dest=neighbor_rank)
                        # Store received style statistics (flattened vector)
                        self.neighbor_style_vecs[neighbor_rank] = style_recv_tmp.clone()
                        # Unflatten style statistics if channel info is available
                        if self.channels_per_layer is not None:
                            try:
                                layer_order = ["layer1", "layer2", "layer3"]
                                unflattened_stats = unflatten_style_stats(
                                    style_recv_tmp, 
                                    layer_order=layer_order,
                                    channels_per_layer=self.channels_per_layer
                                )
                                self.neighbor_style_stats[neighbor_rank] = unflattened_stats
                            except Exception as e:
                                # If unflattening fails, just store the vector (backward compatibility)
                                pass
        else:
            # 只交換模型參數，不交換風格統計量
            degrees = self._compute_mh_degrees(active_flags)
            degree = degrees[self.rank]
            neighbor_weight_sum = 0.0
            for graph_id, flag in enumerate(active_flags):
                if flag == 0:
                    continue
                else:
                    # 如果我在這個子圖有鄰居
                    if self.topology.neighbors_info[graph_id][self.rank] != -1:
                        # 查表：我在這個子圖的鄰居是誰？
                        neighbor_rank = self.topology.neighbors_info[graph_id][self.rank]
                        # Metropolis-Hastings edge weight
                        w_ij = 1.0 / (1.0 + max(degree, degrees[neighbor_rank]))
                        
                        # 發送我的 (send_buffer) 給他，並接收他的存入 (recv_tmp)
                        self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest = neighbor_rank)
                        # 融合：把他的參數乘上權重，加到我的接收區
                        self.recv_buffer.add_(self.recv_tmp, alpha=w_ij)
                        neighbor_weight_sum += w_ij

                        if self.iter % 100 == 0:  # 每100次打印一次
                            print(
                                f"Rank {self.rank} communicated with {neighbor_rank}, "
                                f"deg_i={degree}, deg_j={degrees[neighbor_rank]}, w_ij={w_ij:.6f}"
                            )
            
            # MH self weight: w_ii = 1 - sum_j w_ij
            selfweight = 1.0 - neighbor_weight_sum
            if self.rank == 0 and self.iter % 62 == 0:
                print(f"DEBUG: rank={self.rank}, send_buffer[0]={self.send_buffer[0]:.6f}, "
                    f"recv_buffer[0]={self.recv_buffer[0]:.6f}, "
                    f"degree={degree}, sum_w_ij={neighbor_weight_sum:.6f}, "
                    f"selfweight={selfweight:.6f}")
            # MH weighted average: w_ii x_i + sum_j w_ij x_j
            self.recv_buffer.add_(self.send_buffer, alpha=selfweight)
            # 在 communicator.py 第238行后添加（在加上自身权重之后）
            if self.iter % 62 == 0:
                # 收集所有 rank 的 send_buffer 值
                self.comm.barrier()
                if self.rank == 0:
                    # rank0 收集所有值
                    send_buffers = [self.send_buffer[0].item()]
                    recv_buffers_final = [self.recv_buffer[0].item()]
                    for r in range(1, self.size):
                        send_buffers.append(self.comm.recv(source=r, tag=500))
                        recv_buffers_final.append(self.comm.recv(source=r, tag=501))
                    
                    # 计算期望值：(1/3) * (x0 + x1 + x2)
                    expected = (1.0 / self.size) * sum(send_buffers)
                    
                    print(f"\n=== Verification at iter {self.iter} ===")
                    print(f"Send buffers: {[f'{x:.6f}' for x in send_buffers]}")
                    print(f"Recv buffers (final): {[f'{x:.6f}' for x in recv_buffers_final]}")
                    print(f"Expected (average): {expected:.6f}")
                    
                    # 检查是否一致（允许小的浮点误差）
                    all_same = all(abs(r - expected) < 1e-5 for r in recv_buffers_final)
                    if all_same:
                        print("✓ Calculation is CORRECT! All ranks have the same averaged value.")
                    else:
                        print("✗ Calculation is WRONG! Values differ.")
                        for i, r in enumerate(recv_buffers_final):
                            diff = abs(r - expected)
                            print(f"  Rank {i}: diff = {diff:.8f}")
                    print("=" * 50)
                else:
                    # 其他 rank 发送值
                    self.comm.send(self.send_buffer[0].item(), dest=0, tag=500)
                    self.comm.send(self.recv_buffer[0].item(), dest=0, tag=501)
                self.comm.barrier()

        self.comm.barrier()
        toc = time.time()

        return toc - tic


    def reset_model(self):
        # Reset local models to be the averaged model
        unflattened = unflatten_tensors(
            self.recv_buffer.cuda(), self.tensor_list)
        for f, t in zip(unflattened, self.tensor_list):
            t.copy_(f)
        # 明確釋放臨時變量，避免記憶體累積
        del unflattened

    def prepare_style_buffer(self):
        """Prepare style statistics communication buffer"""
        if self.local_style_vec is None:
            return
        
        # Convert to CPU tensor if needed and detach from computation graph
        # Style vec should already be on CPU and detached from train.py, but we ensure it here
        if self.local_style_vec.is_cuda:
            self.style_send_buffer = self.local_style_vec.detach().cpu().clone()
        else:
            # Ensure detached (safe to call even if already detached)
            self.style_send_buffer = self.local_style_vec.detach().clone()
    
    def communicate(self, model, style_vec=None):
        # Store local style vector
        self.local_style_vec = style_vec
        
        # get activated topology at current iteration
        active_flags = self.topology.active_flags[self.iter]

        # if no subgraphs are activated,
        # then directly start next iteration
        if np.sum(active_flags) == 0:
            if style_vec is None:
                self.iter += 1
            return 0

        if style_vec is not None:
            # Only exchange style statistics, skip model parameter exchange
            self.prepare_style_buffer()
            comm_time = self.averaging(active_flags)
        else:
            # Only exchange model parameters, skip style statistics exchange
            # stack all model parameters into one tensor list
            self.iter += 1
            self.tensor_list = list()
            # Add backbone model parameters
            for param in model.parameters():
                self.tensor_list.append(param.data)
            # Add diffusion model parameters if exists
            if hasattr(model, 'diffusion_model') and model.diffusion_model is not None:
                for param in model.diffusion_model.parameters():
                    self.tensor_list.append(param.data)

            # necessary preprocess
            self.prepare_comm_buffer()

            # decentralized averaging according to activated topology
            # record the communication time
            comm_time = self.averaging(active_flags)

            # update local models
            self.reset_model()


        return comm_time


class ChocoCommunicator(Communicator):
    """ decentralized averaging using compressed gradients (top-k) """
    
    def __init__(self, rank, size, topology, ratio, consensus_lr):
        super(ChocoCommunicator, self).__init__(rank, size)
        self.topology = topology
        self.neighbor_weight = topology.neighbor_weight
        self.iter = 0

        self.initialized = False
        self.consensus_lr = consensus_lr
        self.ratio = ratio


    def prepare_comm_buffer(self):
        # flatten tensors
        # If not initialized, then initialize x_hat and s
        self.x = flatten_tensors(self.tensor_list).cpu()
        if not self.initialized:
            self.x_hat = torch.zeros_like(self.x)
            self.s = torch.zeros_like(self.x)
            self.initialized = True

        tic = time.time()
        # get compressed message
        # here, we use top_k compressor on GPU
        # one can define more in compressors.py
        self.send_buffer = self.x - self.x_hat
        values, indices = get_top_k(self.send_buffer.cuda(), self.ratio)
        toc = time.time()

        values, indices = values.cpu(), indices.cpu()
        self.compressed = {"values":values, "indices":indices}

        return toc - tic



    def averaging(self, active_flags):
        self.comm.barrier()
        tic = time.time()

        # 根據是否有 style_vec 決定交換內容
        if self.local_style_vec is not None:
            # 只交換風格統計量，不交換模型參數
            self.neighbor_style_vecs.clear()
            self.neighbor_style_stats.clear()
            
            for graph_id, flag in enumerate(active_flags):
                if flag == 0:
                    continue
                else:
                    if self.topology.neighbors_info[graph_id][self.rank] != -1:
                        neighbor_rank = self.topology.neighbors_info[graph_id][self.rank]
                        
                        # 交換風格統計量（使用非壓縮通信）
                        style_recv_tmp = self.comm.sendrecv(self.style_send_buffer, source=neighbor_rank, dest=neighbor_rank)
                        # Store received style statistics (flattened vector)
                        self.neighbor_style_vecs[neighbor_rank] = style_recv_tmp.clone()
                        # Unflatten style statistics if channel info is available
                        if self.channels_per_layer is not None:
                            try:
                                layer_order = ["layer1", "layer2", "layer3"]
                                unflattened_stats = unflatten_style_stats(
                                    style_recv_tmp, 
                                    layer_order=layer_order,
                                    channels_per_layer=self.channels_per_layer
                                )
                                self.neighbor_style_stats[neighbor_rank] = unflattened_stats
                            except Exception as e:
                                # If unflattening fails, just store the vector (backward compatibility)
                                pass
        else:
            # 只交換模型參數，不交換風格統計量
            degree = 0
            for graph_id, flag in enumerate(active_flags):
                if flag == 0:
                    continue
                else:
                    if self.topology.neighbors_info[graph_id][self.rank] != -1:
                        degree += 1
                        neighbor_rank = self.topology.neighbors_info[graph_id][self.rank]
                        # Receive neighbor's message q_j
                        self.recv_tmp = self.comm.sendrecv(self.compressed, source=neighbor_rank, dest = neighbor_rank)
                        # Update aggregated model s += sum w_ij q_j
                        self.s[self.recv_tmp["indices"]] += self.neighbor_weight * self.recv_tmp["values"]

            # Compute self weight
            selfweight = 1 - degree * self.neighbor_weight
            # Update aggregated model s += w_ii q_i
            self.s[self.compressed["indices"]] += selfweight * self.compressed["values"]
            # Update x_hat = x_hat + q_i
            self.x_hat[self.compressed["indices"]] += self.compressed["values"]
            # Update local model parameters: x = x + consensus_lr*(s-x_hat)
            self.x.add_(self.consensus_lr, self.s).sub_(self.consensus_lr, self.x_hat)
        
        self.comm.barrier()
        toc = time.time()

        return toc - tic


    def reset_model(self):
        # Reset local models to be the averaged model
        unflattened = unflatten_tensors(
            self.x.cuda(), self.tensor_list)
        for f, t in zip(unflattened, self.tensor_list):
            t.set_(f)
        # 明確釋放臨時變量，避免記憶體累積
        del unflattened

    def prepare_style_buffer(self):
        """Prepare style statistics communication buffer (non-compressed)"""
        if self.local_style_vec is None:
            return
        
        # Convert to CPU tensor if needed and detach from computation graph
        # Style vec should already be on CPU and detached from train.py, but we ensure it here
        if self.local_style_vec.is_cuda:
            self.style_send_buffer = self.local_style_vec.detach().cpu().clone()
        else:
            # Ensure detached (safe to call even if already detached)
            self.style_send_buffer = self.local_style_vec.detach().clone()
    
    def communicate(self, model, style_vec=None):
        # Store local style vector
        self.local_style_vec = style_vec
        
        # get activated topology at current iteration
        active_flags = self.topology.active_flags[self.iter]
        self.iter += 1

        # if no subgraphs are activated,
        # then directly start next iteration
        if np.sum(active_flags) == 0:
            return 0

        if style_vec is not None:
            # Only exchange style statistics, skip model parameter exchange
            self.prepare_style_buffer()
            comm_time = self.averaging(active_flags)
        else:
            # Only exchange model parameters, skip style statistics exchange
            # stack all model parameters into one tensor list
            self.tensor_list = list()
            # Add backbone model parameters
            for param in model.parameters():
                self.tensor_list.append(param.data)
            # Add diffusion model parameters if exists
            if hasattr(model, 'diffusion_model') and model.diffusion_model is not None:
                for param in model.diffusion_model.parameters():
                    self.tensor_list.append(param.data)

            # necessary preprocess
            # there is an additional encoding time
            encode_time = self.prepare_comm_buffer()

            # decentralized averaging
            # record the communication time
            comm_time = self.averaging(active_flags)

            # update local models
            self.reset_model()
            
            return encode_time + comm_time

        return comm_time


class SingleProcessCommunicator(object):
    """
    Single process communicator for decentralized training without MPI.
    Performs model and style statistics aggregation directly in GPU memory.
    """
    def __init__(self, domain_names, topology):
        """
        Args:
            domain_names: List of domain names (e.g., ['art_painting', 'photo', 'sketch'])
            topology: GraphProcessor instance (can be MatchaProcessor or FixedProcessor)
        """
        # Map domain names to indices (0, 1, 2, ...)
        self.domain_to_idx = {name: idx for idx, name in enumerate(domain_names)}
        self.idx_to_domain = {idx: name for name, idx in self.domain_to_idx.items()}
        self.num_domains = len(domain_names)
        
        # Create a mapping from domain names to rank-like indices for topology
        # The topology uses rank indices, so we map domains to 0, 1, 2, ...
        self.topology = topology
        self.neighbor_weight = topology.neighbor_weight
        self.iter = 0
        # 聚合健康度診斷（D1–D4）；由 train.py 注入 AggDiagnostics 實例，預設關閉。
        # diag_epoch 供診斷 CSV 標記當前 epoch（train.py 每輪更新）。
        self.diag = None
        self.diag_epoch = 0

        # Style statistics storage (same as base class)
        self.local_style_vec = None
        self.neighbor_style_vecs = {}  # {domain_name: style_vec_tensor}
        self.neighbor_style_stats = {}  # {domain_name: {layer_name: {stat_name: tensor}}}
        # Domain-scoped style caches to avoid cross-domain leakage in single-process mode
        self.neighbor_style_vecs_by_domain = {}   # {domain_name: {neighbor_domain: style_vec_tensor}}
        self.neighbor_style_stats_by_domain = {}  # {domain_name: {neighbor_domain: unflattened_stats}}
        self.active_domain = None
        self.channels_per_layer = None
        
        # Build adjacency list for domains based on topology
        self._build_domain_adjacency()

        # ===== Async event-triggered state (Stage 1；預設關、由 train.py 開) =====
        self.async_enabled = False
        self.async_threshold = 0.0
        self.async_max_interval = 50
        self.async_buffer_max = None
        self.last_broadcast_params = {}   # ŵ_i：{domain: {name: tensor}}
        self.last_broadcast_step = {}     # {domain: int}
        self.inbox = {}                   # {domain: {sender: {'backbone':{},'diffusion':{}|None,'push_step':int}}}
        # ===== Stage 2：風格也 event-trigger（預設關；只在 --async_style 開）=====
        # 風格 buffer 與模型 inbox 分離：持久、keep-latest per sender、不 consume（StyleShift 每 sweep 重複讀）。
        self.async_style_enabled = False
        self.style_inbox = {}             # {domain: {sender: {'style_vec':tensor,'push_step':int}}} 持久、覆蓋式
        # 註：節點自風格不另存 communicator 狀態；C 步 push 時由 train.py 直接傳入 style_vecs_dict[domain]。

    def _build_domain_adjacency(self):
        """Build adjacency list for domains based on topology structure."""
        self.domain_adj = {domain: [] for domain in self.domain_to_idx.keys()}
        
        # For each subgraph in the topology, extract edges
        for subgraph in self.topology.subGraphs:
            for edge in subgraph:
                if len(edge) == 2:
                    idx1, idx2 = edge
                    # Map topology indices to domain names
                    if idx1 < self.num_domains and idx2 < self.num_domains:
                        domain1 = self.idx_to_domain[idx1]
                        domain2 = self.idx_to_domain[idx2]
                        if domain2 not in self.domain_adj[domain1]:
                            self.domain_adj[domain1].append(domain2)
                        if domain1 not in self.domain_adj[domain2]:
                            self.domain_adj[domain2].append(domain1)
    
    def set_style_channels(self, channels_per_layer: Dict[str, int]):
        """Set channel information for each layer to enable style statistics unflattening."""
        self.channels_per_layer = channels_per_layer
    
    def prepare_comm_buffer(self):
        """Not needed for single process, but kept for interface compatibility."""
        pass
    
    def prepare_style_buffer(self):
        """Prepare style statistics buffer (keep on GPU for single process)."""
        if self.local_style_vec is None:
            return
        # Keep on GPU, no need to move to CPU
        if self.local_style_vec.is_cuda:
            self.style_send_buffer = self.local_style_vec.detach().clone()
        else:
            self.style_send_buffer = self.local_style_vec.detach().cuda()
    
    def _aggregate_style_stats(self, style_vecs_dict, active_flags):
        """
        Aggregate style statistics directly in GPU memory.
        
        Args:
            style_vecs_dict: Dict mapping domain_name -> style_vec (GPU tensor)
            active_flags: Active topology flags for current iteration
        """
        self.neighbor_style_vecs.clear()
        self.neighbor_style_stats.clear()
        self.neighbor_style_vecs_by_domain = {domain: {} for domain in self.domain_to_idx.keys()}
        self.neighbor_style_stats_by_domain = {domain: {} for domain in self.domain_to_idx.keys()}
        
        # For each active subgraph, exchange style statistics
        for graph_id, flag in enumerate(active_flags):
            if flag == 0:
                continue
            
            # Get neighbors for this subgraph
            if graph_id < len(self.topology.neighbors_info):
                neighbors_info = self.topology.neighbors_info[graph_id]
                
                # For each domain, find its neighbor in this subgraph
                for domain_name, domain_idx in self.domain_to_idx.items():
                    if domain_idx < len(neighbors_info):
                        neighbor_idx = neighbors_info[domain_idx]
                        if neighbor_idx != -1 and neighbor_idx < self.num_domains:
                            neighbor_domain = self.idx_to_domain[neighbor_idx]
                            
                            # Directly access neighbor's style vector (already in GPU)
                            if neighbor_domain in style_vecs_dict:
                                self.neighbor_style_vecs_by_domain[domain_name][neighbor_domain] = (
                                    style_vecs_dict[neighbor_domain].clone()
                                )
                                
                                # Unflatten if channel info is available
                                if self.channels_per_layer is not None:
                                    try:
                                        layer_order = ["layer1", "layer2", "layer3"]
                                        unflattened_stats = unflatten_style_stats(
                                            style_vecs_dict[neighbor_domain],
                                            layer_order=layer_order,
                                            channels_per_layer=self.channels_per_layer
                                        )
                                        self.neighbor_style_stats_by_domain[domain_name][neighbor_domain] = unflattened_stats
                                    except Exception as e:
                                        pass

    def set_active_domain(self, domain_name):
        """
        Expose domain-specific neighbor style stats for the next forward pass.
        This mimics MPI semantics where each rank only sees its own neighbors.
        """
        self.active_domain = domain_name
        if self.async_style_enabled:
            # Stage 2：鄰居風格從持久 style_inbox 建（delay-1：只含『上輪已 swap』的 push；本輪 push 在 pending、尚未 swap＝消同輪偷看、staleness≥1）。
            # 空 buffer → 空 dict → StyleShift 自然 skip（R1、不墊檔）。sorted key 保決定論。
            box = self.style_inbox.get(domain_name, {})
            vecs, stats = {}, {}
            if box and self.channels_per_layer is not None:
                for sender in sorted(box.keys()):
                    sv = box[sender]['style_vec']
                    vecs[sender] = sv
                    try:
                        stats[sender] = unflatten_style_stats(
                            sv, layer_order=["layer1", "layer2", "layer3"],
                            channels_per_layer=self.channels_per_layer)
                    except Exception:
                        pass
            self.neighbor_style_vecs = vecs
            self.neighbor_style_stats = stats
        else:
            self.neighbor_style_vecs = dict(self.neighbor_style_vecs_by_domain.get(domain_name, {}))
            self.neighbor_style_stats = dict(self.neighbor_style_stats_by_domain.get(domain_name, {}))
    
    def _aggregate_models(self, models_dict, active_flags):
        """
        Aggregate model parameters directly in GPU memory.
        Formula: x_i^{t+1} = (1 - d*alpha) * x_i^t + alpha * sum_{j in neighbors} x_j^t
        
        Only aggregates trainable parameters (weight, bias), NOT buffers (running_mean, running_var).
        This matches multi-process behavior where only model.parameters() are aggregated.
        
        IMPORTANT: All domains must use the SAME snapshot of parameters for aggregation
        to ensure consistency (matching multi-process behavior where all ranks exchange simultaneously).
        
        Now also aggregates diffusion_model parameters (denoiser only, not normalization buffers).
        
        Args:
            models_dict: Dict mapping domain_name -> model
            active_flags: Active topology flags for current iteration
        """
        with torch.no_grad():  # Aggregation doesn't need gradients
            # CRITICAL: Save parameter snapshots for ALL domains BEFORE aggregation
            # This ensures all domains aggregate based on the same parameter values
            # (matching multi-process behavior where all ranks exchange simultaneously)
            param_snapshots = {}
            diffusion_param_snapshots = {}
            for domain_name, model in models_dict.items():
                # Save backbone model parameters
                param_snapshots[domain_name] = {
                    name: param.data.clone() 
                    for name, param in model.named_parameters()
                }
                
                # Save diffusion model parameters if exists
                if hasattr(model, 'diffusion_model') and model.diffusion_model is not None:
                    # Only aggregate denoiser parameters, NOT normalization buffers
                    # normalization buffers (mins, maxs, means, stds, etc.) are domain-specific statistics
                    # and should remain independent per domain
                    diffusion_param_snapshots[domain_name] = {
                        name: param.data.clone()
                        for name, param in model.diffusion_model.named_parameters()
                        # Filter out normalization buffers - they are registered as buffers, not parameters
                        # so named_parameters() should only return denoiser parameters
                    }
            
            # ===== 診斷 hook：聚合前（snapshot 已完成，模型參數尚未被 copy_ 覆蓋）=====
            # 用聚合前的 denoiser/backbone 抓 D4 的 L_pre。內部 eval 隔離 + RNG 還原，對訓練零副作用。
            if self.diag is not None:
                self.diag.before_agg(models_dict, self.diag_epoch)

            # Precompute MH degrees on the active graph (node-level)
            degrees = [0 for _ in range(self.num_domains)]
            for graph_id, flag in enumerate(active_flags):
                if flag == 0:
                    continue
                if graph_id < len(self.topology.neighbors_info):
                    neighbors_info = self.topology.neighbors_info[graph_id]
                    for node in range(min(self.num_domains, len(neighbors_info))):
                        if neighbors_info[node] != -1 and neighbors_info[node] < self.num_domains:
                            degrees[node] += 1

            # Now aggregate based on snapshots (all domains use the same snapshot)
            for domain_name in models_dict:
                domain_idx = self.domain_to_idx[domain_name]
                model = models_dict[domain_name]
                param_dict = dict(model.named_parameters())
                
                # Use snapshot for this domain's original parameters
                original_params = param_snapshots[domain_name]
                
                # Count active neighbors and accumulate their parameters from snapshots
                # Note: degree counts the number of active subgraphs with neighbors
                # This matches decenCommunicator logic where degree increments for each active subgraph
                # Backbone and diffusion share the same topology, so we compute degree once and accumulate both
                degree = degrees[domain_idx]
                neighbor_sum = {}  # Accumulate sum of neighbor backbone parameters
                diffusion_neighbor_sum = {}  # Accumulate sum of neighbor diffusion parameters (if exists)
                neighbor_weight_sum = 0.0
                
                # Check if this domain has diffusion model
                has_diffusion = (hasattr(model, 'diffusion_model') and model.diffusion_model is not None and 
                               domain_name in diffusion_param_snapshots)
                if has_diffusion:
                    diffusion_param_dict = dict(model.diffusion_model.named_parameters())
                    original_diffusion_params = diffusion_param_snapshots[domain_name]
                
                for graph_id, flag in enumerate(active_flags):
                    if flag == 0:
                        continue
                    if graph_id < len(self.topology.neighbors_info):
                        neighbors_info = self.topology.neighbors_info[graph_id]
                        if domain_idx < len(neighbors_info):
                            neighbor_idx = neighbors_info[domain_idx]
                            if neighbor_idx != -1 and neighbor_idx < self.num_domains:
                                neighbor_domain = self.idx_to_domain[neighbor_idx]
                                
                                if neighbor_domain in models_dict:
                                    # MH edge weight for this active edge
                                    w_ij = 1.0 / (1.0 + max(degree, degrees[neighbor_idx]))
                                    # Use snapshot instead of current model parameters
                                    neighbor_params = param_snapshots[neighbor_domain]
                                    
                                    # Accumulate neighbor backbone parameters from snapshot
                                    for param_name in param_dict.keys():
                                        if param_name not in neighbor_sum:
                                            neighbor_sum[param_name] = torch.zeros_like(original_params[param_name])
                                        neighbor_sum[param_name] += w_ij * neighbor_params[param_name]
                                    
                                    # Accumulate neighbor diffusion parameters if both domains have diffusion models
                                    if has_diffusion and neighbor_domain in diffusion_param_snapshots:
                                        neighbor_diffusion_params = diffusion_param_snapshots[neighbor_domain]
                                        for param_name in diffusion_param_dict.keys():
                                            if param_name not in diffusion_neighbor_sum:
                                                diffusion_neighbor_sum[param_name] = torch.zeros_like(original_diffusion_params[param_name])
                                            diffusion_neighbor_sum[param_name] += w_ij * neighbor_diffusion_params[param_name]

                                    neighbor_weight_sum += w_ij
                
                # MH self weight
                selfweight = 1.0 - neighbor_weight_sum
                
                # Apply aggregation to backbone parameters only (matching multi-process behavior)
                # Only update parameters, buffers (running_mean, running_var) remain unchanged
                # Use original_params from snapshot, not current param.data
                for param_name, param in param_dict.items():
                    if param_name in neighbor_sum:
                        param.data.copy_(
                            selfweight * original_params[param_name] +
                            neighbor_sum[param_name]
                        )
                    # If no active neighbors, param remains unchanged (selfweight = 1, so no change)
                
                # ===== Aggregate diffusion model parameters =====
                # Reuse the same degree and selfweight computed above (same topology)
                if has_diffusion:
                    # Apply aggregation to diffusion model parameters
                    for param_name, param in diffusion_param_dict.items():
                        if param_name in diffusion_neighbor_sum:
                            param.data.copy_(
                                selfweight * original_diffusion_params[param_name] +
                                diffusion_neighbor_sum[param_name]
                            )

            # ===== 診斷 hook：聚合後（所有節點 backbone+diffusion 已更新）=====
            # 算 D1（denoiser 參數發散度）與 D4（ΔL，用聚合前快取的 feat 隔離 denoiser 效應）。
            if self.diag is not None:
                self.diag.after_agg(models_dict, self.diag_epoch)

    def averaging(self, active_flags=None):
        """
        Perform averaging operation (no-op for single process, but kept for compatibility).
        Actual aggregation is done in communicate() method.
        """
        return 0.0  # No communication time
    
    def reset_model(self):
        """Not needed for single process, but kept for interface compatibility."""
        pass
    
    def communicate(self, models_dict, style_vecs_dict=None):
        """
        Communicate model parameters and/or style statistics in single process.
        
        Args:
            models_dict: Dict mapping domain_name -> model (required if style_vecs_dict is None)
            style_vecs_dict: Dict mapping domain_name -> style_vec (optional)
        
        Returns:
            comm_time: Always 0.0 for single process (no actual communication)
        """
        # Get active topology flags for current iteration
        if hasattr(self.topology, 'active_flags') and self.iter < len(self.topology.active_flags):
            active_flags = self.topology.active_flags[self.iter]
        else:
            # If no active flags, assume all subgraphs are active
            active_flags = [1] * len(self.topology.subGraphs)
        
        # If no subgraphs are activated, skip communication
        if sum(active_flags) == 0:
            if style_vecs_dict is None:
                self.iter += 1
            return 0.0
        
        if style_vecs_dict is not None:
            # Exchange style statistics only
            self._aggregate_style_stats(style_vecs_dict, active_flags)
        else:
            # Exchange model parameters only
            self.iter += 1
            self._aggregate_models(models_dict, active_flags)

        return 0.0  # No communication time for single process

    # ==================================================================
    # Async event-triggered communication (Stage 1)
    # PersonalizedET 式(3) 觸發 + PushCen dedup buffer + SPARQ x̂ 語意。
    # 取代 Phase3 同步 _aggregate_models：train.py 逐節點 R(receive)→T(train)→C(check) 呼叫。
    # ==================================================================
    def async_configure(self, threshold=0.0, max_interval=50, buffer_max=None, async_style=False,
                        aggregate_bn=False):
        """由 train.py 從 args 注入 async 超參並啟用。
        async_style=True → Stage 2：風格也綁模型觸發、持久 buffer（見 push_to_neighbors/set_active_domain）。
        aggregate_bn=True → BN running 統計隨模型一起 push/融合（走同一組 MH 權重，故自動繼承
        event-trigger 語意）。payload 增量 +0.086%（9,600 個數字 vs backbone 11.18M）。"""
        self.async_enabled = True
        self.aggregate_bn = bool(aggregate_bn)
        self.async_threshold = float(threshold)
        self.async_max_interval = int(max_interval)
        self.async_buffer_max = buffer_max
        self.last_broadcast_step = {d: 0 for d in self.domain_to_idx}
        self.inbox = {d: {} for d in self.domain_to_idx}
        self.async_style_enabled = bool(async_style)
        self.style_inbox = {d: {} for d in self.domain_to_idx}
        self.style_no_delay = __import__('os').environ.get('STYLE_NO_DELAY', '0') == '1'  # [PROBE對照] 1=退回無delay(postfuse偷看行為)
        self.style_delay = int(__import__('os').environ.get('STYLE_DELAY', '1'))  # 風格 staleness 深度(step為單位；1 epoch=STEPS_PER_EPOCH steps)；1=原 delay-1
        from collections import deque as _deque
        _kd = max(self.style_delay, 1)
        # 深度 k 管線：push 進 pipeline[0]，每個 k 開始把 pipeline[-1](k 輪前 push)搬進 inbox 後右推一格
        self.style_pipeline = _deque([{d: {} for d in self.domain_to_idx} for _ in range(_kd)], maxlen=_kd)
        self.style_latest = {}  # sender -> 最新已 push 的自風格向量（供 style_buffer_dist 量 staleness 造成的風格值偏離）

    def async_init_snapshot(self, models_dict):
        """ŵ_i = 目前 backbone 參數（訓練開始前呼叫一次，建立觸發比對基準）。"""
        self.last_broadcast_params = {}
        for domain, model in models_dict.items():
            self.last_broadcast_params[domain] = {
                name: p.data.clone() for name, p in model.named_parameters()
            }

    def _param_delta(self, domain, model):
        """(1/sqrt(n))·||w_i − ŵ_i||_2 over BACKBONE params（PersonalizedET 式(3) 左式）。
        ⚠️ 只算 backbone：diffusion_model 是註冊 submodule 故會出現在 named_parameters()
        (前綴 'diffusion_model.')，但它每步大幅漂移、不是 DG 決策模型；觸發只看 backbone
        才乾淨可解釋。push/receive 仍照常廣播/聚合 backbone+diffusion（見那兩個方法）。"""
        ref = self.last_broadcast_params.get(domain)
        if ref is None:
            return float('inf')
        sq = 0.0
        n = 0
        for name, p in model.named_parameters():
            if name.startswith('diffusion_model.'):
                continue  # 排除 diffusion；觸發只依 backbone 漂移
            if name in ref:
                d = p.data - ref[name]
                sq += float((d * d).sum().item())
                n += d.numel()
        return math.sqrt(sq / max(n, 1))

    def should_broadcast(self, domain, model, step, gamma=1.0):
        """觸發判斷（PersonalizedET 式3）：delta ≥ τ_base·γ^(k)  OR  安全閥（連續 max_interval 未廣播）。
        gamma = 時變衰減因子，由 train.py 傳入 lr(k)/lr(0)（對齊 PersonalizedET γ^(k)=step size；
        訓練後期 lr↓→門檻↓→維持敏感度、避免收斂期低通訊漂散）。gamma=1 即常數門檻。
        回 (fired, delta, forced, thresh)。"""
        delta = self._param_delta(domain, model)
        thresh = self.async_threshold * gamma
        forced = (step - self.last_broadcast_step.get(domain, 0)) >= self.async_max_interval
        fired = (delta >= thresh) or forced
        return fired, delta, forced, thresh

    def push_to_neighbors(self, domain, model, step, style_vec=None):
        """觸發後把當下 w_i(backbone+diffusion denoiser) 立即寫進每個鄰居 inbox(dedup 覆蓋);更新 ŵ_i。
        Stage 2(async_style)：同一事件夾帶呼叫端傳入的當下自風格 style_vec(=style_vecs_dict[domain]) →
        寫進鄰居持久 style_inbox(keep-latest、不 consume)。自風格不另存 communicator 狀態、免造輪子。"""
        backbone = {name: p.data.clone() for name, p in model.named_parameters()}
        diffusion = None
        if hasattr(model, 'diffusion_model') and model.diffusion_model is not None:
            diffusion = {name: p.data.clone() for name, p in model.diffusion_model.named_parameters()}
        # aggregate_bn：BN running 統計一併廣播。只送 running_mean/var——num_batches_tracked
        # 在 eval 模式下不參與正規化，送了只會多一個變因。
        bn_buf = None
        if getattr(self, 'aggregate_bn', False):
            bn_buf = {n: b.data.clone() for n, b in model.named_buffers()
                      if n.endswith(("running_mean", "running_var"))}
        msg = {'backbone': backbone, 'diffusion': diffusion, 'bn': bn_buf, 'push_step': step}
        neighbors = self.domain_adj.get(domain, [])
        style_msg = None
        if self.async_style_enabled and style_vec is not None:
            style_msg = {'style_vec': style_vec.detach().clone(), 'push_step': step}
            self.style_latest[domain] = style_vec.detach().clone()  # 記自風格最新值（供 style_buffer_dist）
        for nb in neighbors:
            self.inbox[nb][domain] = msg  # dedup by sender（保留最新）；#鄰居有界故 inbox 天然 bounded
            if style_msg is not None:
                # STYLE_NO_DELAY=1 直接寫 inbox(偷看對照)；否則寫管線頭、經 style_delay 輪 swap 才進 inbox
                (self.style_inbox[nb] if self.style_no_delay else self.style_pipeline[0][nb])[domain] = style_msg
        self.last_broadcast_params[domain] = {name: v.clone() for name, v in backbone.items()}
        self.last_broadcast_step[domain] = step
        return len(neighbors)  # 推送數（供通訊量診斷）

    def swap_style_buffers(self):
        """每個 k 開始呼叫：把 style_delay 輪前 push 的風格(管線尾)搬進 active style_inbox。
        keep-latest 覆蓋（未觸發的 sender 保留 inbox 舊值）；本輪稍後 push 的進管線頭、經 style_delay 輪才可見(staleness=style_delay)。"""
        if not self.async_style_enabled or self.style_no_delay:
            return
        oldest = self.style_pipeline[-1]  # style_delay 輪前 push 的
        for nb, box in oldest.items():
            if box:
                self.style_inbox.setdefault(nb, {}).update(box)
        self.style_pipeline.appendleft({d: {} for d in self.domain_to_idx})  # 新空格進頭；maxlen 自動擠掉剛讀完的尾格

    def style_buffer_age(self, domain, step):
        """診斷：該節點目前持有的鄰居風格 buffer 的 (數量, 平均 age=step−push_step)。age-blind 但記 age。"""
        box = self.style_inbox.get(domain, {})
        if not box:
            return 0, 0.0
        ages = [step - v['push_step'] for v in box.values()]
        return len(ages), sum(ages) / len(ages)

    def style_buffer_dist(self, domain):
        """診斷：該節點 inbox 持有的鄰居風格 vs 該 sender『最新已 push 風格』的平均 L2。
        = staleness 造成的風格值偏離；delay 越大→風格漂移越多→此值越大。
        防呆：若此值≈0 代表 delay 沒拉開風格差、『DG 沒掉』無檢驗力（非二元斷裂）。"""
        box = self.style_inbox.get(domain, {})
        if not box:
            return 0.0
        ds = []
        for sender, msg in box.items():
            latest = self.style_latest.get(sender)
            if latest is not None:
                ds.append(float(torch.norm(msg['style_vec'].to(latest.device) - latest).item()))
        return (sum(ds) / len(ds)) if ds else 0.0

    def receive_and_aggregate(self, domain, model, step):
        """inbox 有未消費鄰居模型則 MH(static-degree 權重、只對投遞邊)融合 backbone+diffusion denoiser;
        清 inbox;回 staleness ages list。無到達回 []。"""
        box = self.inbox.get(domain, {})
        if not box:
            return []
        senders = sorted(box.keys())  # 決定論
        deg_i = len(self.domain_adj.get(domain, []))
        ages = []
        with torch.no_grad():
            param_dict = dict(model.named_parameters())
            orig = {name: p.data.clone() for name, p in param_dict.items()}
            acc = {name: torch.zeros_like(orig[name]) for name in param_dict}
            has_diff = hasattr(model, 'diffusion_model') and model.diffusion_model is not None
            if has_diff:
                diff_dict = dict(model.diffusion_model.named_parameters())
                diff_orig = {name: p.data.clone() for name, p in diff_dict.items()}
                diff_acc = {name: torch.zeros_like(diff_orig[name]) for name in diff_dict}
            agg_bn = getattr(self, 'aggregate_bn', False)
            if agg_bn:
                buf_dict = {n: b for n, b in model.named_buffers()
                            if n.endswith(("running_mean", "running_var"))}
                buf_orig = {n: b.data.clone() for n, b in buf_dict.items()}
                buf_acc = {n: torch.zeros_like(buf_orig[n]) for n in buf_dict}
            w_sum = 0.0
            for s in senders:
                deg_j = len(self.domain_adj.get(s, []))
                w_ij = 1.0 / (1.0 + max(deg_i, deg_j))  # MH edge weight（static degree）
                msg = box[s]
                for name in param_dict:
                    if name in msg['backbone']:
                        acc[name] += w_ij * msg['backbone'][name]
                if has_diff and msg['diffusion'] is not None:
                    for name in diff_dict:
                        if name in msg['diffusion']:
                            diff_acc[name] += w_ij * msg['diffusion'][name]
                if agg_bn and msg.get('bn'):
                    for name in buf_dict:
                        if name in msg['bn']:
                            buf_acc[name] += w_ij * msg['bn'][name]
                w_sum += w_ij
                ages.append(step - msg['push_step'])
            selfw = 1.0 - w_sum  # MH self weight（deg_i 上限保證 >0）
            for name, p in param_dict.items():
                p.data.copy_(selfw * orig[name] + acc[name])
            if has_diff:
                for name, p in diff_dict.items():
                    p.data.copy_(selfw * diff_orig[name] + diff_acc[name])
            if agg_bn:
                for name, b in buf_dict.items():
                    b.data.copy_(selfw * buf_orig[name] + buf_acc[name])
            box.clear()  # 消費（consume-on-read）
        return ages

    def consensus_deviation(self, models_dict):
        """跨節點 backbone 參數的共識偏差診斷（取代 async 下被 skip 的 sync D1）。
        rms = 所有 backbone 參數對『節點平均』的 RMS per-param 偏差（越小越一致）；
        max_node = 偏離平均最遠的那個節點的 RMS 偏差。只算 backbone（排除 diffusion）。
        回 (rms, max_node)。"""
        with torch.no_grad():
            domains = sorted(models_dict.keys())
            N = len(domains)
            pdicts = {d: {n: p.data for n, p in models_dict[d].named_parameters()
                          if not n.startswith('diffusion_model.')} for d in domains}
            names = list(pdicts[domains[0]].keys())
            total_sq = 0.0
            total_n = 0
            node_sq = {d: 0.0 for d in domains}
            for name in names:
                stack = torch.stack([pdicts[d][name] for d in domains], dim=0)  # [N, ...]
                mean = stack.mean(dim=0, keepdim=True)
                diff = stack - mean
                total_sq += float((diff * diff).sum().item())
                total_n += pdicts[domains[0]][name].numel()
                for i, d in enumerate(domains):
                    node_sq[d] += float((diff[i] * diff[i]).sum().item())
            rms = math.sqrt(total_sq / max(N * total_n, 1))
            max_node = math.sqrt(max(node_sq.values()) / max(total_n, 1))
            return rms, max_node