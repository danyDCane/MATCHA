# /ars-plan 輸入：新研究方向探索

## 我的研究目標

在現有 MATCHA 架構（StyleDDG + DOoD diffusion MLP）的基礎上，找到一個新方向。

**主要目標：提升泛化能力（DG）**
需要跟 StyleDDG 做比較，有明確的 benchmark（PACS LOO accuracy）。

**次要目標：OOD 檢測（探索性）**
去中心化 FL 上目前沒有標準 benchmark 可以做橫向比較，所以檢測是加分項，不是核心訴求。

---

## 已知前提（不需要重新討論的部分）

**研究脈絡與論文關聯性：**

```
StableFDG（集中式 FL + 風格增強）
    ↓ 繼承架構，改為去中心化
StyleDDG（去中心式 FL，MH 加權聚合風格統計量）
    ↓ MCSAD 指出 StableFDG 的弱點：
    │  風格增強只做 easy/moderate style shift，
    │  缺乏對困難 OOD 風格的暴露，泛化有上限
    ↓ 假設：StyleDDG 繼承 StableFDG 架構，應有相同問題
z_hard 實驗（用 diffusion score function 生成困難風格，試圖補上這個缺口）
    ↓ 實驗結果否證此假設（見下）
需要新方向
```

**架構現況：**
- 主軌：ResNet18 + classifier（分類）
- 副軌：DOoD diffusion MLP，學習 512-d 特徵空間的 score function
- 聯邦通訊：StyleDDG，peer 節點間交換風格統計量 (μ, σ) 與模型參數
- 風格擾動：`StyleShift → StyleExplore → MixStyle`

**已關閉的路線（z_hard，augmentation 角度）：**
在所有四個 PACS LOO 設定下，對抗式困難風格生成（z_hard）無法突破各自 baseline：
- Leave-Photo-out：baseline ~90%，z_hard 最佳也只到 ~90%
- Leave-{Art,Cartoon,Sketch}-out：baseline ~73%，z_hard 未超過各自 baseline

決定性負結果：直接把 test-domain 風格注入訓練也無提升。
**結論：MCSAD 指出的弱點在 StyleDDG 上不成立——augmentation 角度的風格涵蓋已飽和，問題不在「困難風格暴露不足」。**

---

## 我想探索的方向空間

我希望新方向能同時回答這兩個問題：

1. **泛化（DG）**：怎麼讓模型在未見 domain 上表現更好？
2. **檢測（OOD detection）**：怎麼讓模型知道自己面對的是 OOD 樣本？

目前我注意到幾個可能相關的工作，但不確定怎麼結合或哪個最值得追：
- **FOOGD**：用 score function 做 representation constraint（KSD loss）
  `research/origin_paper_pdf/foogd.pdf`
- **FedAlign**：聯邦學習下的特徵對齊
  `research/origin_paper_pdf/Gupta_FedAlign_Federated_Domain_Generalization_with_Cross-Client_Feature_Alignment_CVPRW_2025_paper.pdf`
- **FedCCRL**：對比學習 + 表示正則化
  `research/origin_paper_pdf/fedccrl.pdf`
- **StableFDG**：風格與注意力機制結合的聯邦域泛化
  `research/origin_paper_pdf/stablefdg-style-and-attention-based-learning-for-federated-domain-generalization.pdf`
- **MCSAD**：（z_hard 原始靈感來源，augmentation 角度，已確認此路線無效）
  `research/origin_paper_pdf/MCSAD_openreview.pdf`
- **StyleDDG**：目前架構的基礎，主要是泛化功能
  `research/origin_paper_pdf/styleddg2026.pdf`
- **DOoD**：目前結合的檢測模型Q1
  `research/origin_paper_pdf/DOoD.pdf`


---

## 給 ars-plan 的開放提示

請幫我從上述背景出發，探索能在 decentralized FL + DG 架構上突破 StyleDDG baseline 的新方向。
不限定方法或角度，優先找我還沒想到的可能性。

---

## 限制條件

- Backbone：ResNet18（可以換，但需要理由）
- Dataset：PACS LOO（主要 benchmark）
- 聯邦設定：decentralized（非 centralized server）
- 時間：需要在合理實驗週期內可以驗證
