# `scripts/posthoc/` — post-hoc 分析腳本（86 支，2026-08-18 ～ 09-18）

> **來歷**：最早的 14 支寫在 session 專屬的暫存目錄（`/tmp/.../scratchpad/`），該路徑綁 session id、
> 且 `/tmp` 會被系統清 ⇒ 2026-08-18 搬進專案並改成看得懂的檔名（原名見文末附錄）。
> 之後一個月的分析一律直接寫在本目錄。

**這批腳本是什麼**：不是工具庫，是**研究軌跡的實體化**。一支腳本 ＝ 當時要回答的一個具體問題，
很多支的檔名本身就是問句（`why_prototype_swap_has_no_effect`、`is_26pct_coverage_trustworthy`）。
因此**不要重構它們**——它們的價值在於「當時跑出那個數字的就是這份 code」。

## 怎麼跑

全部是 **post-hoc、零重訓**，讀既有 checkpoint 重新前向（少數幾支連前向都不做，直接讀落盤的 npz）。

```bash
# 一律從專案根目錄執行、用專案內的 venv
cd /home/server5090/Desktop/M11307320/MATCHA
venv_matcha/bin/python scripts/posthoc/<腳本>.py
```

**只有 10／86 支吃命令列參數，其餘把設定寫死在檔頭常數**（要換 fold／換 run 就改那幾行）。
共同前提：

- `sys.path` 插入 `scripts` 與專案根 ⇒ 直接重用 `osdg_eval`／`test_domain_ood_scores`／
  `joint_eval_mixed_stream`／`dood.prototype`／`style_transforms`
- 資料路徑 `../datasets/`、checkpoint 目錄 `exp_result_<DESC>`（**相對於專案根**）
- 預設 fold＝cartoon、9 節點、`UNK=6`（person）、checkpoint 一律 `final`
- 四臂的 `DESC` 常數：`proto_lam0`／`p1a`／`p1a…_fix`／`p1ap`
- **2026-08-25 起**：所有指標與幾何診斷一律先把 9 節點的 BN running 統計量平均再算
  （`bn_common.py`，合併變異數版 B）。理由：跨節點落差的 87% 是 BN 統計量，不平均就是在被污染的基底上診斷。

---

## §1 BN 平均（評估基底）

| 檔名 | 最後修改 | 做什麼 |
|---|---|---|
| `bn_common.py` | 08-25 | **共用實作**：`bn_avg()`／`apply_bn()`。TaskBoard §A 評估協定的單一真相源 |
| `bn_average_intervention.py` | 08-19 | 訓練後把 9 節點 BN 統計量平均，看跨風格落差與 cartoon 改不改善。起點發現：conv 權重分歧 7.7e-5、BN 仿射 6.8e-5（已達共識），但 BN running 統計量分歧 0.131（大 1700 倍）且訓練中單調上升 ⇒ 9 個模型實質是「同一個網路 + 九組不同的 BN 統計量」 |
| `bn_average_accuracy_check.py` | 08-19 | BN 平均對**分類準確率**與**誤拒率來源**的影響 ⇒ 判斷它是不是假象的關鍵 |
| `bn_avg_full_matrix.py` | 08-19 | BN 平均後的完整 9×9 矩陣：特徵端已統一、只剩原型還不同。**產出 `logs/prototype_probe/0819_bnavg_matrix_{原樣,BN平均}.npy`** |
| `baseline_ladder_bn_avg.py` | 08-29 | 完整 baseline 階梯（從「什麼都沒有」到現在）統一在 BN 平均基底上。補 TaskBoard §A 階梯 A 從沒測過的缺口 |

## §2 誤拒率根因追查（0815／0818 首批）

| 檔名 | 最後修改 | 做什麼 |
|---|---|---|
| `dump_diag_features.py` | 08-18 | 補 0815 的五個洞。**產出 `logs/prototype_probe/0815_diag_features.pkl`**（四次訓練 × 9 節點的特徵與分數，後續多數分析的共同基礎） |
| `traj_four_arms_epochs.py` | 08-18 | **雜訊基準**：四臂 × ep50/100/150/200 的方向與長度軌跡。同臂相鄰 epoch 的波動＝訓練後期時間變異，是判斷跨臂效應是否真實的最低門檻 |
| `score_distribution_stats.py` | 08-18 | 三堆的分數分布完整統計＋標準化分離度：AUROC 變差是哪一種不均勻造成的 |
| `score_variants_from_angles.py` | 08-18 | 同一組角距離矩陣導出四種分數 ⇒ **「分數相對化」被排除的依據** |
| `ood_coherence_probe.py` | 08-18 | person 的表徵一致性：std 膨脹是「散到各類別」還是「整體變鬆但方向仍一致」⇒ **推翻「person 被打散」** |
| `norm_discriminability_probe.py` | 08-18 | 角距離丟掉了範數，範數本身有沒有判別力 ⇒ 原型讀出輸給 energy 的結構性原因？ |
| `crossnode_comparability_L1L4.py` | 08-18 | L1–L4 跨節點特徵空間可比性（同一張圖餵進 9 個節點各自的完整模型，含各自的 BN，端到端比較） |
| `proto_reference_2a2b_preview.py` | 08-18 | 2b 的預演：讀出參照點從「自己 6 個原型」換成 18 格／54 個，直接算四軸 |
| `adain_intervention_angle.py` | 08-18 | cartoon 到自己類別原型的 53.57°，有多少是通道統計量造成的（確定性 AdaIN，同 P3 位置） |
| `adain_intervention_angle_v2_fpr.py` | 08-18 | **同上，角度計算逐行相同，多輸出誤拒率**（門檻＝各節點來源域分數 95 分位、來源域未被介入） |
| `inference_style_norm_fouraxis.py` | 08-18 | 推論時把**所有**測試樣本正規化到來源域統計量，算完整四軸（守門員＝部署 AUROC 不得降）。⚠️ 本質是 test-time 操作、**無法轉成訓練** |
| `attribution_ablation_2x4x2.py` | 08-18 | 歸因消融：2 模型（λ=0 未訓投影層／1a-fix）× 4 讀出 × 2 處理（原樣／推論正規化） |
| `mech_comp_vs_disp.py` | 08-18 | `L_comp` 為何讓誤拒率變差：M1（①變窄→門檻下移）vs M2（①②差距擴大） |
| `stack_angle_norm_tta.py` | 08-18 | 疊加測試（角距離＋512 維範數＋推論正規化）對上「什麼都沒有」的 baseline |

## §3 原型與讀出形式

| 檔名 | 最後修改 | 做什麼 |
|---|---|---|
| `why_prototype_swap_has_no_effect.py` | 08-19 | 原型差 7.08°、換上去卻只變 0.00° ⇒ 逐樣本看，不只看平均 |
| `prototype_vs_feature_attribution.py` | 08-19 | ①→★ 的 14.65° 是「原型的帳」還是「特徵的帳」＋檢驗 dany 的加法模型。**產出 `0819_angle_matrix_9x9x4.npy`** |
| `proto_average_and_per_node_breakdown.py` | 08-19 | 在 BN 已平均的前提下重測「原型平均有沒有用」（此時原型是節點間唯一差異，最容易看出效果） |
| `shard_level_cross_node_table.py` | 08-19 | 同一批資料換到不同節點跑差多少（同風格不同子集 vs 跨風格）。**產出 `0819_shard_cross_node.npy`** |
| `param_divergence_decomposition.py` | 08-19 | 9 節點終態參數分歧拆解＋原型漂移逐域對拆解。零前向 |
| `readout_error_overlap_probe.py` | 08-20 | 原型讀出與 energy 是不是在**同一批樣本**犯**同一種錯** ⇒ 「改讀出形式」這條路的 go/no-go |
| `readout_form_sweep.py` | 08-20 | 把原型自己的 6 個距離榨乾——`min_c` 有沒有漏掉資訊（全程不碰 energy） |
| `readout_form_sweep_saturation.py` | 08-20 | 補：α 掃描沒到飽和就下判決是錯的，掃到極限；並更正 V5 熵的符號 |
| `space_vs_readout_2x2.py` | 08-29 | energy 也是 512→6→1，為什麼它丟得比我們好？把「用哪個空間」與「用哪種讀出」拆成獨立兩軸 |
| `why_std_works.py` | 08-29 | −std(logit) 0.8656 為什麼好、能不能搬過來。把六維向量拆「共同高度」＋「輪廓形狀」，比較畫風變化與類別變化各落在哪邊 |
| `why_18_prototypes_dont_help.py` | 08-29 | 18 個原型取 min 為什麼幾乎不動 AUROC ⇒ 拆四環逐環給數字 |
| `reference_is_the_variable.py` | 08-29 | **決定性一格**：把六個參照物從「類別中心」換成「CE 學出來的判別權重」。訓練只用來源域已知類別、person 標籤全程不碰 |
| `disagreement_signal_probe.py` | 08-29 | 兩個頭吵架（分類頭 argmax vs 投影頭 argmin）為什麼吵、能不能當訊號 |
| `prototype_rank_check.py` | 09-02 | **§2.1 閘門**：原型矩陣的六個奇異值——六個類別中心張成 6 維還是 5 維？秩虧損會讓 `z⊥` 多扣一個假方向，下游全部跟著錯 |
| `proto_readout_failure_diag.py` | 09-06 | 原型讀出失效診斷：②③ 分離度＋投影前(512) vs 投影後(128) 的單一變因對照 |
| `readout_comparison.py` | 09-09 | 六個讀出的完整對照，並把「排序變好」與「門檻位置變對」拆成兩個來源 |

## §4 風格介入與分解

| 檔名 | 最後修改 | 做什麼 |
|---|---|---|
| `xcheck_adain_reproduces_4848.py` | 08-20 | 一致性核對：A 組是否重現 0818 §3.4 的 48.48° |
| `style_vs_content_decomposition.py` | 08-21 | 23.94° 裡多少是畫風、多少是內容。階梯（本地來源域／其他來源域／cartoon）＋位移向量拆共同成分 vs 類別專屬成分 |
| `style_vs_content_premise_check.py` | 08-21 | **先驗證上一支判準所依賴的前提**「畫風不挑類別」；前提為假則上一支實驗二的判讀必須撤回 |
| `adain_intervention_zperp.py` | 08-29 | 通道 μ/σ 這個軸能解釋 `‖z⊥‖` 污染的多少 ＝ `rel_loss` 的射程上界。判準事前寫死 |
| `is_26pct_coverage_trustworthy.py` | 08-29 | 質疑上一支：26% 是在**已訓練模型**上量的，若模型對通道統計量已免疫，介入本來就沒效 ⇒ 四項交叉檢驗 |
| `zperp_style_contamination.py` | 09-01 | `‖z⊥‖` 的畫風污染分解 ⇒ `rel_loss` 的事前 go/no-go。用上「來源域也有 person 圖」做完整 2×2 |
| `adain_direction_geometry.py` | 09-02 | 把畫風換掉之後，殘差**方向**的幾何會不會改善（作弊上界） |
| `adain_form_alignment.py` | 08-31 | 複驗 `rel_loss` 的 no-go：0829 判死用的介入形式（域平均一刀切）與訓練形式（batch 級統計量＋DSU 採樣）不一致 ⇒ 四形式單變因遞進 |
| `style_vector_block_energy.py` | 09-18 | 風格向量四塊的能量占比與各自的畫風分離度。整條向量分離倍率只有 2.26×，但只用 `[mu, sigma]` 兩塊外推是 6–9× |

## §5 幾何路線探路（方向／子空間／殘差）

| 檔名 | 最後修改 | 做什麼 |
|---|---|---|
| `compactness_transfer_four_arms.py` | 08-21 | `L_comp` 到底有沒有讓**未見畫風**的類內散布變小——方法設計的起點座標，四臂唯一變因是損失 |
| `deploy_auroc_ceiling_probe.py` | 08-24 | 部署 AUROC 的定錨：瓶頸是「已知類別散太開」還是「整團位置偏了」還是「person 本來就混在裡面」。目的**不是找方法，是在三條路裡刪掉兩條** |
| `scatter_structure_probe.py` | 08-25 | cartoon 比來源域多散的 9.6° 是「同樣形狀變大」還是「多了特定方向」⇒ 決定散開可不可治 |
| `synthesis_region_probe.py` | 08-25 | R1+R2：造假點之前的兩個閘門（緯度拆解：整團搬家 vs 散開） |
| `synthesis_region_probe2.py` | 08-25 | R2b+R3pre+R3：合成區裡實際住著誰 ⇒ 造點會撞到誰的下界 |
| `w_person_variance.py` | 08-26 | 探針偷看 person 標籤找到的那把刀，能不能靠「它在正常資料上變異特別小」**不看答案**地找出來 |
| `w_person_three_piles.py` | 08-27 | 沿 `w_person` 這條軸把三堆重新量一次。若換畫風也把這條軸吵起來 ⇒ 整條路死 |
| `quiet_subspace_probe.py` | 08-27 | 安靜子空間：那把好尺在不在裡面、拿不拿得到（全程零標籤，只用來源域） |
| `probe_information_content.py` | 08-26 | 投影層是「毀了資訊」還是「藏了資訊」。**產出 `0826_features_full.npz`**（BN 平均 B 基底，§3／§4 多支的共同輸入） |
| `probe_transfer_full.py` | 08-26 | 探針轉移 E0–E5：128 維那把刀是「通用陌生刀」還是「person 專屬刀」⇒ 借一個已知類別演「未知」來驗 |
| `oracle_leak1_and_person_structure.py` | 08-26 | 只修「洩漏一」的 oracle（改用預測類別、對②∪③全部套用）＋ person 的方向結構 |
| `loco_novelty_direction.py` | 08-26 | LOCO 通用新奇方向：把已知類別當可替換的角色，看方向轉不轉移得到 person。必須在 `z⊥` 上做（否則探針最省力的解是指向被訓練成好分的原型方向） |
| `probe_direction_vs_norm.py` | 09-02 | `z⊥` 的方向 vs 長度三格對照＋探針設定的復現驗證 |
| `residual_direction_probe.py` | 09-02 | 實驗 D/B/C：殘差方向的**免標籤**讀出撿不撿得到（十條陷阱逐條落實：逐節點跑、絕不跨節點池化、參考方向只用來源域算） |
| `residual_direction_structure.py` | 09-02 | 實驗 B/C：殘差方向到底有沒有結構（解釋 D 為何失敗） |
| `direction_readout_same_style.py` | 09-03 | 把方向參考法拿去跑**同畫風**任務，判定「是不是卡在風格軸」 |
| `direction_reference_transfer.py` | 09-03 | 參考方向是不是瓶頸：換三種參考方向逐節點比較。診斷＝參考方向不跨畫風遷移 |
| `class_subspace_readout.py` | 09-03 | 「離參照物遠」這族還有救嗎——把參照物從**點**換成**面**（點只能表達中心在哪，面能表達散開） |
| `subspace_gap_decomposition.py` | 09-03 | 把 0.6641 → 0.8993 的增益拆開：是 ② 靠近了面，還是 ③ 遠離了面。**這個分解決定訓練目標怎麼定** |
| `consensus_subspace_dryrun.py` | 09-04 | 共識範本 dry run：三種來源畫風各建「已知類別變化清單」投票取共識，罩不罩得住 cartoon |

## §6 門檻與操作點

| 檔名 | 最後修改 | 做什麼 |
|---|---|---|
| `miss_rate_tradeoff.py` | 08-19 | 同門檻下誤殺多少正常樣本 vs 放行多少 OOD 的完整取捨表。補 0818/0819/0819b 三份報告全都漏報的放行率 |
| `per_node_threshold_spread.py` | 08-19 | energy 與原型讀出的逐節點誤拒率離散度 ⇒ 「跨節點門檻不一致」是不是共通問題 |
| `threshold_consensus_probe.py` | 08-19 | 誤拒率@src95 不是固定操作點（模型一變門檻就跟著移）⇒ 改成**固定放行率**再比誤拒率 |
| `threshold_population_scan.py` | 08-25 | 每個門檻之上實際住著幾張 cartoon 正常圖、幾張 person。回應「均值差很多不就好了」的直覺質疑：要看**分布與張數** |
| `threshold_calibration_quality.py` | 09-11 | 門檻校準資料的品質（教授提問）：算門檻那批資料本身分對了嗎、信心高嗎、分數擁擠嗎 |
| `threshold_borderline_cases.py` | 09-14 | 門檻上的真實個案，把統計量還原成看得見的圖與 logit／softmax，**供口試回答用** |
| `threshold_tail_confidence.py` | 09-17 | 門檻捨棄／誤拒那批樣本的信心度與正確率：那 5% 是隨機犧牲還是本來就可疑 |
| `threshold_shift_marginal.py` | 09-18 | 門檻從 95 分位移到 94／96 實際翻掉哪些圖 |
| `detector_decisiveness_probe.py` | 09-10 | 檢測器有多「果斷」（分數遠離門檻、少模稜兩可）——與判得準不準是兩件獨立的事。跨讀出比較必須無尺度 |
| `detector_confidence_probe.py` | 09-10 | 檢測器自己的「信心度」有沒有用：energy 分數會不會在某些樣本上根本不可靠 |
| `decisiveness_is_scale_artifact.py` | 09-10 | **自我質疑**：果斷度是不是尺度假象？MSP 的高果斷度已被標為假象（softmax 飽和），原型讀出會不會踩到同一個機制 ⇒ 三個獨立檢驗 |

## §7 開集準確率 OSA、對外彙總與成本

| 檔名 | 最後修改 | 做什麼 |
|---|---|---|
| `summarize_baseline_vs_ourfull.py` | 09-04 | StyleDDG baseline vs 我們全套的 4-fold 對照彙總（吃 `osdg_eval` 產出的 csv） |
| `paired_attribution_from_diag_log.py` | 09-06 | 三臂 diag 批次 log → 配對分帳表。⚠️ 所有敘述性數字一律從變數帶出、不手打 |
| `inference_cost_bench.py` | 09-07 | 推論成本：三組部署形態 × 三層 × 兩種 batch |
| `zperp_oscr_summary.py` | 09-09 | 面讀出進 OSCR：四 fold × {BN 原樣, 平均B} × {baseline, 我方}。三個必須一起看的口徑 |
| `auroc_vs_osa_decomposition.py` | 09-10 | 為什麼 AUROC 較低的讀出反而 OSA 較高 ⇒ 用「作弊門檻」把排序與門檻拆開，兩張表互相印證 |
| `osa_pi_sensitivity.py` | 09-10 | OSA 對混入比例 π 的敏感度與交叉點。關鍵性質：`OSA(π)=(1−π)·a_id+π·r_ood`，兩個分量只由門檻決定、與 π 無關 |
| `osa_section4_tables.py` | 09-14 | 0905 §4「操作點各指標」四張表的完整版（含面讀出） |
| `per_class_generalization_acc.py` | 09-15 | 泛化準確率的逐類別拆解：是哪幾類在撐、哪幾類在拖。零前向，直接讀落盤 npz |
| `per_class_confusion.py` | 09-15 | 六路混淆矩陣：0/1 對錯答不出「錯成哪一類」 |
| `plot_oscr_curves.py` | 09-15 | OSCR 曲線圖（四 fold ＋ 四折平均）。面積會把「在哪個工作點贏」壓成一個數，畫出曲線才看得到低 FPR 區誰高 |
| `osa_4fold_summary.py` | 09-17 | 四 fold OSA 彙總：「逐張去向」表 ＋ 三種口徑的 Δ。吃 `scripts/open_set_accuracy.py --stage infer` 的四個 npz |

---

## 界線（引用這些腳本的輸出前必讀）

- 多數是**單 fold（cartoon）、單 seed（2026）、單一 checkpoint 系列**，訓練隨機性未量化。
  §7 的幾支是四 fold，引用時看清楚是哪一種。
- 全部是 **post-hoc 探路與上界估計**。任何一條在這裡有效的結果，**都不等於方法成立**——
  最終方法必須用訓練實現（硬約束 D1）。
- `inference_style_norm_fouraxis.py` 那個操作**本質上是 test-time 操作、無法轉成訓練**
  （理由見 0818 §3.6）。
- **落盤產物不進 git**（`.gitignore` 擋掉 `.npz`／`.pkl`／`.json`／`.csv`／`.png`）。
  `0815_diag_features.pkl`、`0826_features_full.npz` 這類共同基礎要重跑產生腳本才會有。
- 產生三臂 diag log 的 `run_art_three_arms_diag.sh` 也**不在版控內**（`*.sh` 被 ignore）。

## 附錄：首批 14 支的原名

搬進專案時改名，原名記於此（報告正文從未指名個別腳本，改名不影響任何引用）。

| 現名 | 原名 |
|---|---|
| `dump_diag_features.py` | `full_diag.py` |
| `traj_four_arms_epochs.py` | `traj.py` |
| `score_distribution_stats.py` | `score_dist.py` |
| `score_variants_from_angles.py` | `score_variants.py` |
| `ood_coherence_probe.py` | `ood_coherence.py` |
| `norm_discriminability_probe.py` | `norm_probe.py` |
| `crossnode_comparability_L1L4.py` | `crossnode.py` |
| `proto_reference_2a2b_preview.py` | `proto18.py` |
| `adain_intervention_angle.py` | `style_intervene.py` |
| `adain_intervention_angle_v2_fpr.py` | `si2.py` |
| `inference_style_norm_fouraxis.py` | `tta_full.py` |
| `attribution_ablation_2x4x2.py` | `ablation.py` |
| `mech_comp_vs_disp.py` | `mech.py` |
| `stack_angle_norm_tta.py` | `stack.py` |
