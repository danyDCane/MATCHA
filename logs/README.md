# logs/ — 訓練/評估 log 索引

> 結構鏡像 `research/` 的 track 資料夾（一個 track 一夾、doc↔log 一對一）。
> `*.log` 已被 `.gitignore`（純本地、不進版控）；新 run 請 tee 到對應 track 夾。

## Track 對照

| 資料夾 | 對應 research track | 內容 |
|---|---|---|
| `v1_baseline/` | `research/V1_baseline/` | V1 正典。`v2b1path/`=單步-forward V1（cartoon stage1 末20≈**75.07**）；`early/`=最初 stage1/2（0522）；`det/`=deterministic seed1234 run + OOD eval |
| `v2b1_score_reg/` | `research/V2B1_score_norm/` | V2B1 score-reg（已棄用）：det 訓練、stage1、λ0.1、OOD eval |
| `fourier_aug/` | `research/fourier_aug/` | Fourier amp aug：arm5a/b/c、v1_*_fourier |
| `ksd_coupling/` | `research/ksd_coupling/` | KSD 泛化耦合 Phase 0/1（見該夾 README 的關鍵控制臂說明）|

## 命名慣例（新 run 沿用）

- 訓練：`<exp>_<domain>_train.log` 或 `<domain>_stage1.log`
- OOD 評估：`<domain>_ood_eval.log` 或 `ood_<exp>_<domain>.log`
- 跑前先 `mkdir -p logs/<track>/`，再 `... 2>&1 | tee logs/<track>/<name>.log`
