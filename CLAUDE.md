# CLAUDE.md — MATCHA 專案

## 專案背景

**研究主題：** 聯邦學習下的 Domain Generalization（域泛化）

**核心程式：**
- `train.py` / `train_mpi.py` — 主訓練腳本
- `pacs_dataset.py` / `vlcs_dataset.py` — 資料集（PACS、VLCS）
- `communicator.py` / `graph_manager.py` — 聯邦通訊
- `style_transforms.py` / `style_stats.py` — 風格增強相關
- `evaluate_ood.py` / `test_domain_ood_scores.py` — OOD 評估

**架構：**
- 主軌：ResNet18 backbone + classifier head（分類路徑）
- 副軌：DOoD diffusion MLP，學習 512-d 特徵空間的 score function `∇z log p(z)`
- 聯邦通訊：StyleDDG，透過 MH 加權聚合在 peer 節點間交換風格統計量 (μ, σ) 與模型參數
- 風格擾動流程：`StyleShift → StyleExplore → MixStyle`

**關鍵概念：**
- `z_style`：風格特徵向量（layer3 AdaIN 統計量重建）
- `z_clean`：原始乾淨特徵
- `z_hard`：對抗式困難風格特徵（已證實此路線無正向訊號，見下）

**評估設定：** PACS dataset，leave-one-domain-out（LOO）

| LOO 設定 | Baseline (StyleDDG，無 z_hard) |
|---------|-------------------------------|
| Leave-Photo-out | ~90% |
| Leave-{Art,Cartoon,Sketch}-out | ~73% |

**目前研究狀態：**
z_hard（augmentation-based 對抗式風格生成）路線經完整診斷確認無正向訊號——所有四個 LOO 設定皆未突破 baseline，包含 test-domain style injection 這個最強 sanity check 也失敗。目前轉向 **constraint 角度**：以 KSD loss 約束 (z_clean, z_style) pair 的 representation 幾何（參考 FOOGD/SAG 架構）。

**實驗結果：** 儲存在 `exp_result_*/` 資料夾，以測試域命名。

---

## Research 辦公室（`research/`）

`research/` 是與 Claude 協作的辦公室，存放：
- 論文討論摘要與草稿
- 實驗設計與結果分析報告
- 與論文相關的筆記

**慣例：**
- 討論摘要用日期命名，例如 `2026-05-18_style_orthogonality.md`
- 實驗分析報告放在 `research/reports/`（量多再建）
