# 對抗式困難風格生成 (z_hard) 研究歷程統整

> **目的**：完整記錄目前在「diffusion 引導的對抗式困難風格特徵生成」路線上的實驗、診斷與發現，並建立為何需要從 augmentation 角度轉向 constraint 角度的論證鏈。

---

## 目錄

1. [研究背景](#1-研究背景)
2. [z_hard 生成方法設計](#2-z_hard-生成方法設計)
3. [實驗系列與結果](#3-實驗系列與結果)
4. [綜合分析：為什麼要換方向](#4-綜合分析為什麼要換方向)
5. [方向轉換建議：從 Augmentation 到 Constraint](#5-方向轉換建議從-augmentation-到-constraint)
6. [對論文 framing 的影響](#6-對論文-framing-的影響)

---

## 1. 研究背景

### 1.1 雙軌架構

整體架構整合兩項先前工作，組成聯合訓練系統:

- **STYLEDDG**:去中心式 DG,透過 Metropolis-Hastings 加權聚合在 peer 節點間交換風格統計量 (μ, σ from layer1/2/3) 與模型參數。內部執行 `StyleShift → StyleExplore → MixStyle` 的風格擾動流程。
- **DOoD**:在 512-d avgpool 特徵空間訓練 diffusion MLP,提供 score function (`∇z log p(z)`) 作為特徵密度估計。

主軌道是分類路徑 (ResNet18 backbone + classifier head),副軌道是 diffusion MLP,兩者間以 stop-gradient 隔離後聯合訓練。

### 1.2 評估設定

- **資料集**:PACS,leave-one-domain-out (LOO)
- **Backbone**:ResNet18 (ImageNet-pretrained)
- **各 LOO 設定下的 baseline (StyleDDG-based, no z_hard)**:

  | LOO 設定 | Baseline accuracy |
  |---------|-------------------|
  | Leave-Photo-out (Photo 作 unseen test) | **~90%** (四個 domain 中最高) |
  | Leave-Art-out | ~73% 左右 |
  | Leave-Cartoon-out | ~73% 左右 |
  | Leave-Sketch-out | ~73% 左右 |

- **主要評估配置**:本研究主要以 **leave-Photo-out** 作為迭代測試平台。§3 各實驗的數據,如未特別註明,皆為此配置。

#### 為何主要選 leave-Photo-out 作迭代平台

策略性考量。在四個 LOO 設定中,leave-Photo-out 是 baseline 最高的,這意味著:

- 模型本身在這個 setting 已經能達到 ~90%,表示這對 ResNet18 backbone 是相對「友善」的測試環境
- 如果 z_hard 機制能在這裡提取出 1–2% 的提升,這個訊號相對容易被檢測 (不容易被 seed 間 variance 吞掉)
- 在 baseline 較低 (~73%) 的其他三個 domain 上,本來就有更多 headroom,預期能看到更顯著的相對提升

策略邏輯:**在「對 augmentation signal 最敏感的設定」上做 sanity check** — 如果這裡看得到提升,再推展到 baseline 較低的設定上驗證可放大效應;如果連這裡都看不到提升,augmentation 機制大概率整體性無訊號。

#### 其他三個 LOO 設定的驗證

本研究同樣在 Art / Cartoon / Sketch 作 test domain 的設定下跑過 z_hard 變體。結論一致:**沒有任何一個 domain 超過各自的 ~73% baseline**。失敗模式**不是 Photo 設定特有的個案**,而是穿透所有四個 LOO 配置。

### 1.3 研究核心假設

整合 diffusion 後的雙軌架構應該能提供超越單獨 StyleDDG 的泛化能力。具體假設是:

> 利用 diffusion 學到的 ID 密度地圖,引導生成「在 OOD 區域且分類困難」的合成風格特徵 z_hard,送入 backbone 訓練,模型應能學到對更廣風格分佈的不變性,從而在未見 domain 上有更好的表現。

---

## 2. z_hard 生成方法設計

### 2.1 核心思路

利用 diffusion 的 score function,在風格統計量 (μ, σ) 空間做對抗式擾動,生成「對 diffusion 看起來更 OOD、對 classifier 看起來更困難」的合成風格,組回特徵後送入 backbone 計算分類損失。

### 2.2 詳細流程 (MCSAD-inspired 單步 FGSM)

1. 從 layer3 的 z_style 提取 μ, σ,設定 `requires_grad=True`,detach spatial structure
2. 經 AdaIN 重建,通過 layer4 + avgpool 取得 512-d 特徵
3. 把 512-d 特徵輸入 frozen diffusion,計算 L_ood (noise prediction error)
4. 同時通過 frozen classifier 計算 L_cls
5. 組合對抗損失:`L_adv = -L_ood + λ·L_cls` (λ 從頭到尾為負,L_cls 從一開始就是對抗項而非保護項)
6. 對 μ̂, σ̂ 計算梯度,normalize 後乘上步長 α (FGSM 單步)
7. 用邊界限制 μ̂, σ̂ 相對於 post-StyleShift μ, σ 的偏移比例 (約 2%)
8. 風格抹除:`z_norm = (z_s − μ_orig) / σ_orig`
9. 套用新風格:`z_hard = σ̂ · z_norm + μ̂`
10. z_hard 送 backbone forward → CE → 更新 backbone (diffusion 與 classifier 因 stop-gradient 不參與這條更新路徑)

### 2.3 重要設計決策

- **介入位置選擇**:操作在 layer3 的 (μ, σ) 而非 512-d pooled feature。後者風格與語意混雜,前者較純粹是風格通道。
- **語意保護機制**:沒有顯式保護項,完全依賴兩個假設:(a) μ, σ 是純風格、不承載語意;(b) 小幅擾動 (相對 post-StyleShift 值約 2%) 避免破壞語意。
- **單步 vs 多步**:FGSM 單步,未採用 PGD 迭代。
- **diffusion 與 classifier 的處理**:在 z_hard 生成的 inner step 中視為 frozen;它們在 outer training loop 中正常更新。

### 2.4 早期評估與排除的替代方案

幾個方向在設計階段就因原則性問題被排除:

- **以原始 raw style statistics 作為 diffusion 直接輸入**:輸入空間維度不匹配
- **以 classifier 權重矩陣投影做語意隔離**:W 捕捉的是分類敏感度,非語意內容
- **使用 noise vector 的 per-channel spatial statistics**:512-d 空間無 spatial 維度
- **PGD 式多步迭代優化**:成本與單步收益權衡後選擇單步

---

## 3. 實驗系列與結果

> 註:本節所有 z_hard 變體與結構診斷指標,如未特別註明,皆為 **leave-Photo-out** 設定下的結果。為何選 Photo 作為主要評估配置,見 §1.2。其他三個 LOO 設定 (Art/Cartoon/Sketch as unseen) 也驗證過 z_hard 變體,結論一致 — 皆未超越各自 ~73% baseline。

### 3.1 初步收斂實驗

**目的**:確認 z_hard 生成流程能穩定運作,內部優化指標健康。

**觀察**:
- L_ood 與 L_cls_inner 在內部優化過程穩定下降
- μ, σ 的梯度 norm 隨訓練進展由初始值降至約 0.1
- 沒有 NaN、explode 或其他數值不穩

**結論**:機制本身運作正常,後續實驗結果不是 implementation bug。

---

### 3.2 超參數掃描

**目的**:找出對抗強度的最佳設定。

**改動**:不同 λ 值、step size 大小、base feature 的選擇 (z_style, z_clean, z_norm)。

**結果**:
- 多數配置落在 **87–90%** (leave-Photo-out),沒有任何一組超越 ~90% baseline
- 部分極端設定 (步長過大、邊界過寬) 崩潰到 **~60%**
- 在 Art / Cartoon / Sketch 作 test domain 時亦未超越各自 ~73% baseline

**結論**:不是「沒調好參數」的問題。z_hard 的天花板就在 baseline 附近,跨所有 LOO 配置都沒有正向訊號。

---

### 3.3 梯度方向診斷

**目的**:檢查 L_ood 與 L_cls 兩個損失各自提供什麼訊號。

**量測**:
```
cos(∇_{μ,σ} L_ood, ∇_{μ,σ} L_cls) ≈ 0
```

**發現**:兩個損失在 (μ, σ) 空間的梯度幾乎正交。diffusion 對 OoD 的概念與 classifier 對「困難」的概念,是兩個獨立的訊號 — 不會互相強化也不會互相抵銷。

---

### 3.4 z_hard 方向冗餘診斷

**目的**:檢查產生出來的 z_hard 跟 z_style 在表示空間裡的關係。

**量測** (在 512-d avgpool 空間,across 3 個源 domain,leave-Photo-out 配置):
```
cos(z_clean → z_style, z_clean → z_hard) ≈ 0.8
```

**初步假設**:z_hard 推動方向跟 StyleExplore 已經做的方向高度共線。失敗原因可能是「對抗擾動只是放大了 StyleExplore 已經在做的事,沒提供新方向」 — 也就是 **directional redundancy**。

---

### 3.5 風格正交化實驗

**目的**:既然 z_hard 方向跟 z_style 方向過於共線,主動把這個共線分量扣掉,看正交分量是否能提供新訊號。

**改動**:
- 在 (μ, σ) 串接空間裡,把 raw 梯度對 `(μ_style − μ_clean, σ_style − σ_clean)` 方向做投影並扣除
- 正交化後重新 normalize、乘步長、套邊界限制

**量測**:
- `orth_removed ≈ 0.05 – 0.10`
- 也就是 (μ, σ) 空間裡,raw gradient 已經自然 95% 與 style-clean 軸接近正交,正交化能切掉的本來就不多

**結果**:泛化成績**沒有提升**,仍卡在 leave-Photo-out 的 ~90% 附近

**含意**:「方向冗餘」假設**被否證**。即使把那個 5–10% 的平行分量切掉,剩下 90–95% 的正交分量也不能 push 表現。問題不是「沒走到正交方向」,而是「走了正交方向也沒用」。

---

### 3.6 對抗來源 Ablation

**目的**:釐清 -L_ood 與 -L_cls 各自對 z_hard 的貢獻。

**實驗組合與結果** (leave-Photo-out):

| 變體 | 結果 |
|------|------|
| -L_cls only (移除 diffusion branch),無正交化 | ~90% |
| -L_cls only (移除 diffusion branch),有正交化 | ~90% |
| -L_ood only,無正交化 | ~90% |
| -L_ood + -L_cls 雙重對抗,無正交化 | ~90% |
| -L_ood + -L_cls 雙重對抗,有正交化 | ~90% |

**發現**:**無論用哪個損失、有沒有正交化,所有變體一律卡在 leave-Photo-out 的 ~90%**。

這是比預期更強的訊號 — 失敗模式不在「梯度方向選擇」這個層級。問題在更上游。

---

### 3.7 第二次 512-d 結構診斷

**動機**:在 (μ, σ) 空間正交化沒效,但 z_hard 與 z_style 在 512-d 上 cos≈0.8。檢查 (μ, σ) → 512-d 的映射是否塌縮 (即 layer4 + avgpool 是否把正交擾動壓掉)。

**量測** (across 3 個源 domain,leave-Photo-out 配置):

| 量測項 | 值 |
|--------|-----|
| `‖vec_hard − vec_style‖ / ‖vec_style − vec_clean‖` | 0.43, 0.51, 0.56 |
| `cos(vec_hard − vec_style, vec_style − vec_clean)` | ≈ 0 (略反向) |
| `cos(vec_style − vec_clean, vec_hard − vec_clean)` | ≈ 0.8 |

**幾何詮釋**:
- z_hard 相對 z_clean 的大方向確實貼近 z_style 那條軸
- 但 z_hard 已經明顯偏離 z_style,偏離方向約 90° 略向 clean 回拉
- 偏離量是 style-clean 距離的 43–56%,**不算小**

**含意**:layer4 + avgpool **沒有**把 (μ, σ) 的正交擾動壓掉。下游確實有把這個自由度傳到 512-d。問題不是塌縮 — 而是「即使在 512-d 上產生了顯著且非平行的位移,仍不能 push 表現」。

到這裡,整個「在風格空間裡找對的對抗方向」的假設都已被否證:方向、來源、下游傳遞都驗過了,全部不是 bottleneck。

---

### 3.8 Test-Domain Style Injection 實驗 (決定性負結果)

**動機**:退一步問一個根本問題 — 如果連 test domain 的風格本身在訓練時都看到了,泛化會不會提升?如果不會,那「風格涵蓋不足」這個整個 framework 的前提就被否證了。

**改動**:在 layer3 把擾動後的 μ, σ 往測試集風格方向拉一小步,送入訓練流程。

**結果**:
1. **泛化成績沒有提升**
2. **副產物觀察**:帶有少量測試風格的特徵反而比原本的 style 擾動特徵**更好分類** (lower L_cls)

**這個結果為什麼重要**:

這是整個診斷鏈裡最強的負結果。它直接告訴我們:

- 原本 `StyleShift + StyleExplore + MixStyle` 產生的訓練樣本,從 classifier 角度看,**已經比真實測試風格更困難**
- 模型訓練時實際看到的風格分佈,已經比 test 分佈還要 OOD
- 但 test 上仍有 ~10% 錯 (leave-Photo-out) / ~27% 錯 (其他 LOO)
- 即「給模型看比 test 更困難的風格、甚至直接看 test 風格本身,都沒有用」

剩下的 gap **跟風格涵蓋無關**。可能的成因:

- ResNet18 在這個 pipeline 下的容量瓶頸
- PACS 上某些類別在特定 domain 上的本質混淆 (例:Sketch 上 elephant vs horse 線稿差異真的小)
- 標籤層級的固有困難
- 其他與 style augmentation 無關的因素

---

## 4. 綜合分析:為什麼要換方向

### 4.1 證據鏈

把所有實驗的結論串成一條收斂的證據鏈:

1. **3.4 共線發現**:z_hard 跟 z_style 在 512-d 上高度共線 (cos≈0.8)
2. **3.5 否證共線假設**:主動把 (μ, σ) 空間的共線分量扣掉,泛化仍沒提升
3. **3.6 對抗來源無關**:-L_ood、-L_cls、雙重對抗,加不加正交化,**全卡 ~90% (leave-Photo-out)**
4. **3.7 下游沒有塌縮**:正交分量確實有傳到 512-d,位移量顯著 (43–56% of style-clean magnitude),但泛化仍沒動
5. **3.8 決定性負結果**:連直接訓練測試風格都沒用;且訓練擾動已經比測試風格更困難

每一步都把可能的失敗解釋砍掉一層。

### 4.2 結論

在目前這個 method family (ResNet18 + StyleDDG-based augmentation + diffusion-guided adversarial style generation) 上:

> **在所有四個 LOO 配置下,z_hard 變體都無法突破各自的 baseline。**
> - Leave-Photo-out:baseline ~90%,z_hard 變體最佳也只到 ~90%
> - Leave-{Art, Cartoon, Sketch}-out:baseline ~73% 左右,z_hard 變體也未超過各自 baseline

這不是「某個 domain 的個別 ceiling」,是 method family 在所有四個配置下都沒有正向訊號。

進一步推論:
- 任何形式的「在 (μ, σ) 空間裡找對抗方向產生 z_hard」都不會 break 這些 ceiling
- 任何想透過「讓訓練看到更多/更難風格」來提升泛化的 augmentation 角度,都已經沒有空間
- 繼續調參、換損失函數、換步長、換邊界,只會持續累積負結果

### 4.3 為什麼不該繼續對抗式 z_hard

具體理由:

1. **機制天花板已驗證**:不是實作沒調好,是這個機制本身在這個 setting 下沒有可挖掘的訊號空間。
2. **負結果是齊次的**:各種設定的結果不是「有的好有的壞」、有調參空間;是「全部卡在同一個天花板」。在實驗統計上比噪音強得多,代表的是結構性問題。
3. **更強的反證已經存在**:Test-domain style injection 連「直接看答案」都救不回來,任何更迂迴的對抗方向更不可能。
4. **連 sanity check 平台都失敗**:策略上選 leave-Photo-out 作為主要評估平台,原本期望這裡是 augmentation signal **最容易檢測**的位置 — baseline 較高、模型已有不錯表現,若機制有訊號應能擠出 1–2% 提升。實際結果是連這個 sanity check 都拿不到。同時在 baseline 較低 (~73%) 的其他三個 LOO 設定上 — 理論上 headroom 更大 — z_hard 也未能超過 baseline。**失敗訊號穿透所有 LOO 配置,與 baseline 高低無關**,這比單一 setting 失敗強得多。
5. **時間 ROI 不對**:繼續嘗試對抗式 z_hard 的變體,期望值上不會收斂到突破任何 baseline。

---

## 5. 方向轉換建議:從 Augmentation 到 Constraint

### 5.1 兩種角度的本質差異

| 維度 | Augmentation 角度 (MCSAD / 目前路線) | Constraint 角度 (FOOGD / 提議方向) |
|------|--------------------------------------|------------------------------------|
| 訊號來源 | 擾動本身:找更困難的樣本 | Representation 約束:在現有擾動下強化結構 |
| 學習機制 | 模型被迫處理難例 → 隱式學到不變性 | 顯式約束 feature 分佈幾何 → 直接控制 representation 結構 |
| 何時失效 | 當訓練分佈已經比 test 更困難 (即現況) | 當 representation 已足夠結構化 (本架構未驗證) |
| 在現有數據下狀況 | 已撞牆 (Section 3.8 證實) | 未試,仍有探索空間 |

### 5.2 FOOGD 的 SAG 給我們的啟發

FOOGD 的 SAG (Stein Augmented Generalization) 用 Kernelized Stein Discrepancy (KSD) 度量原始與擾動特徵在 score model 估計密度下的分佈差異:

```
L = CE + λ_a · KSD(p(z_original), q(z_augmented))
```

關鍵特性:
- Augmentation `T` 是固定且 generic 的 (paper 用 Fourier augmentation),**不負責提供 generalization 的訊號**
- 訊號來自於用 score function 約束 representation 的幾何
- KSD 把擾動特徵拉向原始特徵的高密度區域,同時用 Stein operator 的第二項避免塌縮

**精神上的差異**:augmentation 角度是「給模型更難的題目讓它學」;constraint 角度是「給定現有的題目,在解題過程中強加結構性約束」。

### 5.3 我們的架構為什麼適合接這個方向

現有架構已經具備所需零件:

- **Diffusion model** 在 512-d 上學的就是 `∇z log p(z)` → 直接對應 SAG 需要的 score function (角色等同 FOOGD 的 SM3D)
- **z_clean / z_style** → 直接對應 SAG 需要的 (原始, augmented) pair
- **StyleDDG 風格交換** → 比 FOOGD 用的 Fourier augmentation 強得多,提供分佈式 cross-peer 風格多樣性

唯一的結構性改動是 diffusion 的角色:從「服務 z_hard 生成的 stop-gradient 工具」變成「主動約束 backbone 的 score 提供者」。

### 5.4 第一個 minimal 實驗

去掉整個 z_hard branch,把 loss 改為:

```
L = CE(z_clean) + CE(z_style) + λ · KSD(p(z_clean), q(z_style))
```

KSD 用現有 diffusion 提供的 score 計算。

**測試平台一樣選 leave-Photo-out** (理由同 §1.2:sanity check 訊號最敏感的設定)。跟 ~90% baseline 比較:

| 結果情境 | 詮釋 | 後續行動 |
|---------|------|---------|
| ≥ baseline | constraint signal 至少跟現有 augmentation signal 等價 | 推進到其他三個 LOO 設定驗證放大效應 |
| > baseline | 突破 leave-Photo-out 的 ceiling | 同上,且論文走 constraint 主軸 |
| < baseline | 在此 setting 下 constraint 也沒空間 | 重新評估 (見 §5.5) |

### 5.5 不確定性與風險

需要誠實列出的可能性:

- **各 LOO ceiling 可能是 backbone-level 限制** (ResNet18 在 PACS 上的物理上限)。若是,constraint 路線會撞同樣的牆。
- **差異化壓力**:FOOGD 已經佔了「FL + score-based generalization」這個 angle。我們的差異點是 (a) decentralized 而非 centralized federated;(b) StyleDDG cross-peer 風格交換取代 Fourier augmentation;(c) 雙軌聯合訓練的具體實作。要在實驗數據上把這些差異化講清楚。

若 constraint 方向也卡在各自 baseline,結論會變成「ResNet18 + PACS LOO + 風格 augmentation family 整體上的 ceiling 已到」。屆時要考慮的退路:

- 換 backbone (ResNet50 / ViT)
- 換 dataset (DomainNet 等更大的 DG benchmark)
- 重新定位貢獻 (e.g., 把 OoD detection 拉成主要評估維度)

---

## 6. 對論文 framing 的影響

即便最終 constraint 方向也不能 break ceiling,**這份診斷本身就有研究價值**:

- 完整的 augmentation-side 失敗證據鏈:cos 量測 + 正交化否證 + 對抗來源 ablation + 512-d 結構診斷 + test-style injection
- 證實在這個 method family 下 style coverage 已經過飽和 — **無論用哪個 LOO 設定**
- 直接挑戰「更多/更難的風格擾動更好」這個 DG 文獻裡常見的隱性假設

這在 DG 文獻裡是值得發聲的負結果,前提是寫得清楚、證據鏈完整。

不過這仍然只是「保底框架」。理想情況:

> 轉向 constraint 角度後找到正向訊號,把整篇論文定位成「在 decentralized DG 上,augmentation-side 已飽和,constraint-side 仍有空間 — 我們提出 XYZ 加以利用」。

兩種情境都不是死局,且都建立在已經紮實做完的對抗式 z_hard 診斷之上。

---

## 附錄:關鍵實驗指標一覽

> 註:以下指標除「各 LOO baseline」一欄外,皆於 **leave-Photo-out** 配置量測。

### Baselines (跨所有 LOO 設定)

| 設定 | StyleDDG-based baseline | z_hard 變體最佳結果 |
|------|--------------------------|---------------------|
| Leave-Photo-out | ~90% | ~90% (無提升) |
| Leave-Art-out | ~73% | 未超過 baseline |
| Leave-Cartoon-out | ~73% | 未超過 baseline |
| Leave-Sketch-out | ~73% | 未超過 baseline |

### Leave-Photo-out 配置下的診斷指標

| 指標 | 值 | 意義 |
|------|-----|------|
| z_hard 變體 accuracy 範圍 | 87–90% (部分崩潰至 ~60%) | 無一超過 baseline |
| `cos(∇L_ood, ∇L_cls)` in (μ,σ) | ≈ 0 | 兩個損失提供正交訊號 |
| `cos(z_clean→z_style, z_clean→z_hard)` in 512-d | ≈ 0.8 | z_hard 大方向與 z_style 共線 |
| `orth_removed` in (μ,σ) | 0.05–0.10 | (μ,σ) 空間裡 raw grad 與 style-clean 軸本已接近正交 |
| `‖vec_hard − vec_style‖ / ‖vec_style − vec_clean‖` | 0.43–0.56 | 512-d 位移量顯著,下游沒塌縮 |
| `cos(vec_hard − vec_style, vec_style − vec_clean)` | ≈ 0 (略反向) | 512-d 位移方向與 style-clean 軸接近正交 |
| 訓練 μ, σ 梯度 norm (後期) | ≈ 0.1 | 內部優化收斂,非 implementation 問題 |
