import numpy as np
import time
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
            for param in model.parameters():
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
            degree = 0 # record the degree of each node
            for graph_id, flag in enumerate(active_flags):
                if flag == 0:
                    continue
                else:
                    # 如果我在這個子圖有鄰居
                    if self.topology.neighbors_info[graph_id][self.rank] != -1:
                        degree += 1
                        # 查表：我在這個子圖的鄰居是誰？
                        neighbor_rank = self.topology.neighbors_info[graph_id][self.rank]
                        
                        # 發送我的 (send_buffer) 給他，並接收他的存入 (recv_tmp)
                        self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest = neighbor_rank)
                        # 融合：把他的參數乘上權重，加到我的接收區
                        # recv_buffer += alpha * neighbor_model
                        self.recv_buffer.add_(self.neighbor_weight, self.recv_tmp)
            
            # 檢查完子圖後，計算我的權重
            selfweight = 1 - degree * self.neighbor_weight
            # compute weighted average: (1-d*alpha)x_i + alpha * sum_j x_j
            self.recv_buffer.add_(selfweight, self.send_buffer)

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
        # Style vec should already be on CPU and detached from train_mpi.py, but we ensure it here
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
            for param in model.parameters():
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
        # Style vec should already be on CPU and detached from train_mpi.py, but we ensure it here
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
            for param in model.parameters():
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