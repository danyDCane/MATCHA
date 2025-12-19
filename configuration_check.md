# 全連接去中心化學習配置檢查報告

## 當前配置總結

### 1. 數據集配置 (PACS) ✓
- **總共 4 個 domain**: art_painting, cartoon, photo, sketch
- **測試集**: art_painting (leave_out)
- **訓練 domains**: cartoon, photo, sketch (3個)
- **Rank 分配**:
  - Rank 0 → cartoon
  - Rank 1 → photo  
  - Rank 2 → sketch
- **每個 rank 的測試集**: 所有 rank 都使用 art_painting 作為測試集 ✓

### 2. 圖拓撲配置 ✓
- **graphid = -1**: 全連接 3 節點圖
- **圖分解**: 分解為 3 個 matching:
  - Matching 1: (0, 1) - Rank 0 與 Rank 1 通信
  - Matching 2: (0, 2) - Rank 0 與 Rank 2 通信
  - Matching 3: (1, 2) - Rank 1 與 Rank 2 通信
- **全連接**: 每個節點都與其他兩個節點有邊連接 ✓

### 3. 通信配置 ✓
- **budget = 1.0**: 每個 matching 的激活概率為 100%
- **FixedProcessor**: 使用固定通信圖（D-PSGD）
- **decenCommunicator**: 去中心化通信器
- **理論上**: 每個 iteration，所有 3 個 matching 都應該被激活

### 4. 模型配置 ✓
- **模型**: ResNet-18 (標準版)
- **輸入尺寸**: 224×224 (PACS)
- **Batch size**: 64
- **學習率**: 0.001
- **Momentum**: 0.9

## 潛在問題和建議

### ⚠️ 問題 1: FixedProcessor 的激活邏輯
**位置**: `graph_manager.py` line 220-231

**問題描述**:
```python
def set_flags(self, iterations):
    flags = list()
    for i in range(len(self.L_matrices)):
        flags.append(np.random.binomial(1, self.probabilities, iterations))
    return [list(x) for x in zip(*flags)]
```

即使 `budget=1.0`，代碼仍然使用 `np.random.binomial` 來生成激活標誌。雖然理論上應該總是返回 1，但為了確定性和效率，當 `budget=1.0` 時應該直接設置為全 1。

**建議修改**:
```python
def set_flags(self, iterations):
    flags = list()
    for i in range(len(self.L_matrices)):
        if self.probabilities == 1.0:
            # 如果概率為 1.0，直接設置為全 1，避免隨機性
            flags.append(np.ones(iterations, dtype=np.int32))
        else:
            flags.append(np.random.binomial(1, self.probabilities, iterations))
    return [list(x) for x in zip(*flags)]
```

### ⚠️ 問題 2: 隨機種子設置
**位置**: `train_mpi.py` line 67-68

**當前設置**:
```python
torch.manual_seed(args.randomSeed+rank)
np.random.seed(args.randomSeed)
```

**問題**: `np.random.seed` 在所有 rank 上使用相同的種子，這是正確的（確保激活標誌一致）。但 `torch.manual_seed` 使用不同的種子（`args.randomSeed+rank`），這可能導致不同 rank 的數據加載順序不同。

**建議**: 確認這是否為預期行為。如果希望數據加載也一致，可以考慮使用相同的種子。

### ✓ 正確的部分

1. **數據分配邏輯**: 正確地將 3 個 domain 分配給 3 個 rank
2. **測試集配置**: 所有 rank 都使用相同的測試集（leave_out domain）
3. **圖拓撲**: 全連接圖的分解正確
4. **通信邏輯**: `decenCommunicator` 的實現看起來正確，會根據激活的 matching 進行通信
5. **混合權重計算**: `getAlpha()` 使用拉普拉斯矩陣的特徵值計算混合權重，這是標準的 D-PSGD 方法

## 驗證建議

### 1. 檢查激活標誌
在訓練開始時，打印前幾個 iteration 的激活標誌，確認所有 matching 都被激活：
```python
# 在 train_mpi.py 中添加
if rank == 0 and k < 5:
    print(f"Iteration {k}: Active flags = {GP.active_flags[k]}")
```

### 2. 檢查通信模式
確認每個 iteration 每個 rank 都與其他兩個 rank 通信：
- Rank 0 應該與 Rank 1 和 Rank 2 通信
- Rank 1 應該與 Rank 0 和 Rank 2 通信
- Rank 2 應該與 Rank 0 和 Rank 1 通信

### 3. 檢查模型一致性
在訓練過程中，定期檢查所有 rank 的模型參數是否趨於一致（全連接圖應該使模型收斂到相同的參數）。

## 總結

整體配置**基本正確**，實現了全連接的去中心化學習。主要建議是優化 `FixedProcessor.set_flags()` 方法，當 `budget=1.0` 時直接設置為全激活，避免不必要的隨機性。

