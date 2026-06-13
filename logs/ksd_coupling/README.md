# logs/ksd_coupling/ — KSD 泛化耦合 Phase 0/1

對應 `research/ksd_coupling/`。

## 檔案

| log | 角色 | 關鍵數字 |
|---|---|---|
| **`phase0_scorediag_cartoon.log`** | ⭐ **KSD-off 同-path 控制臂（=判決基準）** + Phase 0 score_diag 見證者驗證 | **末20 test_acc = 73.294**（兩步-forward、cartoon LOO、seed1234）|
| `cartoon_ksd_r0.1.log` | KSD-on 主 run（ratio=0.1、t=25）| 末20 = 73.99；全程均值對控制臂僅 **+0.17**（正向集中在 ep151-200）|
| `cartoon_ksd_ood_eval.log` | KSD-on OOD（Textures）評估 | AUROC 0.9658 |
| `ksd_compat_eval.log` | 靜態分布相容度 eval（共同凍結裁判）| 見 `research/ksd_coupling/0605_ksd_phase1_cartoon_results.md` |
| `cartoon_smoke.log` | Phase 1 smoke test | — |

## ⚠️ 為何特別標 `phase0_scorediag_cartoon.log`

這是「KSD-on vs KSD-off」單變因對照裡的 **off 臂**，也是報告「+0.70%」的對照基準（73.294）。它**同樣經兩步 forward**（`vec_style_for_reg`），與 KSD-on 同 path、才合法。

此檔曾因放在專案頂層（非 `logs/`）而在跨 session 差點找不到、導致一度誤判「基準 log 遺失」。**勿移出本夾、勿刪。** 判決邏輯：KSD-on(73.99) − 此控制臂(73.294) = +0.70（末20）/ +0.17（全程）。
