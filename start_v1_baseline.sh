#!/bin/bash
# V1 baseline parameterized launcher
# Usage: bash start_v1_baseline.sh <leave_out> <graphid>
#   leave_out: art_painting | cartoon | photo | sketch
#   graphid:   -1 (3-node K3, centralized) | 6 (9-node RGG, P2P)
# Examples:
#   bash start_v1_baseline.sh cartoon -1
#   bash start_v1_baseline.sh photo 6

set -e
LEAVE_OUT="${1:?usage: $0 <leave_out> <graphid>}"
GRAPHID="${2:?usage: $0 <leave_out> <graphid>}"

if [ "$GRAPHID" = "-1" ]; then
    STAGE_TAG="stage1"
    NUM_NODES_ARG=""
elif [ "$GRAPHID" = "6" ]; then
    STAGE_TAG="stage2"
    NUM_NODES_ARG="--num_nodes 9"
else
    echo "ERROR: graphid must be -1 or 6, got $GRAPHID"
    exit 1
fi

# Optional Fourier amplitude augmentation (Option B). Enable with: USE_FOURIER=1
# Override range via FOURIER_ALPHA / FOURIER_RATIO. Output dir/wandb get a _fourier suffix
# so the run never collides with the det V1 baseline (single-variable A/B).
USE_FOURIER="${USE_FOURIER:-0}"
FOURIER_ARGS=""
FOURIER_TAG=""
if [ "$USE_FOURIER" = "1" ]; then
    FA="${FOURIER_ALPHA:-1.0}"; FR="${FOURIER_RATIO:-1.0}"
    # grad-conflict probe on by default (cheap, every 50 steps) — measures cos(g_S,g_F)
    FOURIER_ARGS="--use_fourier_aug --fourier_alpha ${FA} --fourier_ratio ${FR} --fourier_grad_diag --fourier_grad_diag_every ${FOURIER_GRAD_DIAG_EVERY:-50}"
    FOURIER_TAG="_fourier"
    # encode non-default sweep config into the run name so output dirs never collide
    [ "$FA" != "1.0" ] && FOURIER_TAG="${FOURIER_TAG}_a${FA}"
    [ "$FR" != "1.0" ] && FOURIER_TAG="${FOURIER_TAG}_r${FR}"
    if [ "${FOURIER_TRAIN_BN:-0}" = "1" ]; then
        FOURIER_ARGS="${FOURIER_ARGS} --fourier_train_bn"
        FOURIER_TAG="${FOURIER_TAG}_trainbn"
    fi
fi

# ===== KSD generalization coupling (Phase 1; replaces the old --use_score_reg lambda0 scaffold) =====
# The backbone always uses the two-step style forward via --use_ksd_reg (grad-enabled z_aug);
# KSD_RATIO sets the gradient-balancing target. KSD_RATIO=0 (default) = KSD-off control == the
# deterministic V1 baseline (two-step forward, NO KSD loss; the coupling block is skipped). Set
# KSD_RATIO>0 to turn coupling ON.
#   baseline / KSD-off : bash start_v1_baseline.sh cartoon -1
#   KSD-on             : KSD_RATIO=0.1 bash start_v1_baseline.sh cartoon -1
KSD_RATIO="${KSD_RATIO:-0}"
KSD_SCORE_T="${KSD_SCORE_T:-25}"
KSD_BALANCE_EVERY="${KSD_BALANCE_EVERY:-20}"
KSD_ARGS="--use_ksd_reg --ksd_grad_ratio ${KSD_RATIO} --ksd_score_timestep ${KSD_SCORE_T} --ksd_balance_every ${KSD_BALANCE_EVERY}"
KSD_TAG=""
# Non-zero ratio => coupling ON; tag the run so its ckpt/wandb never collides with the baseline.
if [ "$KSD_RATIO" != "0" ]; then
    KSD_TAG="_ksd_r${KSD_RATIO}_t${KSD_SCORE_T}"
elif [ "${KSD_CTRL:-0}" = "1" ]; then
    # ratio=0 sweep control: distinct savePath so it does NOT overwrite the existing plain-V1
    # checkpoint (exp_result_v1_stage1_leave_<dom>). Used as the same-path KSD-off arm with ratio_aug.
    KSD_TAG="_ksd_r0ctrl_t${KSD_SCORE_T}"
fi

# ===== Ablation toggles (advisor experiment): no-diffusion / no-style-use / isolated-nodes =====
# USE_OOD=0        : drop diffusion training (+KSD witness, +agg_diag which need it). Faster; backbone
#                    identical since diffusion is decoupled from backbone (ksd ratio=0, latents detached).
# USE_STYLE_SHIFT=0: style stats still EXCHANGED (--use_style_stats stays) but NOT applied → "交換但不用風格".
# ISOLATED=1       : each node trains independently (skip style exchange + model aggregation).
#   Exp1 (style exchanged-but-unused):  USE_OOD=0 USE_STYLE_SHIFT=0 ... bash start_v1_baseline.sh cartoon 6
#   Exp2 (fully isolated nodes):        USE_OOD=0 USE_STYLE_SHIFT=0 ISOLATED=1 ... bash start_v1_baseline.sh cartoon 6
ABL_TAG=""
USE_OOD="${USE_OOD:-1}"
OOD_ARGS="--use_ood"
AGG_DIAG_ARGS="--enable_agg_diag --diag_t_probe 500 --diag_every_n_epoch 1 --diag_every_m_epoch 20 --diag_d4_every 1"
if [ "$USE_OOD" = "0" ]; then
    OOD_ARGS=""
    KSD_ARGS=""        # KSD needs the diffusion score witness → must drop together with --use_ood
    AGG_DIAG_ARGS=""   # agg diag needs diffusion_model → drop (avoids the auto-disable warning)
    ABL_TAG="${ABL_TAG}_nodiff"
fi
USE_STYLE_SHIFT="${USE_STYLE_SHIFT:-1}"
STYLE_SHIFT_ARGS="--use_style_shift --style_shift_prob 0.8 --style_shift_ratio 0.8"
if [ "$USE_STYLE_SHIFT" = "0" ]; then
    STYLE_SHIFT_ARGS=""
    ABL_TAG="${ABL_TAG}_nostyleuse"
fi
ISOLATED="${ISOLATED:-0}"
ISOLATED_ARGS=""
if [ "$ISOLATED" = "1" ]; then
    ISOLATED_ARGS="--isolated_nodes"
    ABL_TAG="${ABL_TAG}_isolated"
fi

# ===== Stage 1: prototype detection readout (0803 §2.2/§2.6; 0810 implementation plan) =====
# 投影層 (512->512->128, 與 fc 並聯) + (類別,畫風) 原型 buffer + 到 6 個「類別中心」的角距離。
# 前向 = 2 次（擾動含梯度 + 乾淨無梯度供原型 EMA）。與 --use_ksd_reg 互斥。
#
#   λ=0 baseline : PROTO=1 PROTO_COMP=0 PROTO_DISP=0 USE_OOD=0 ASYNC=1 ... bash start_v1_baseline.sh cartoon 6
#   1a           : PROTO=1                            USE_OOD=0 ASYNC=1 ...
#   1b (+L_rel)  : PROTO=1 PROTO_REL=0.1              USE_OOD=0 ASYNC=1 ...
#   1c (無投影層): PROTO=1 PROTO_NOPROJ=1             USE_OOD=0 ASYNC=1 ...
#   1ap(+L_pair) : PROTO=1 PROTO_PAIR=0.1            USE_OOD=0 ASYNC=1 ...   (可加 PROTO_PAIR_MARGIN=0.95)
#                  ↑ 獨立項 L_pair=mean(relu(margin-cos(z_aug,z_clean.detach())))；對照組＝1a-fix（同 code、PROTO_PAIR=0）
#
# ★ 為什麼 baseline 是 λ=0 而不是既有的 0730 run：diffusion 的 p_losses 吃 torch.randn，而
#   StyleShift/StyleExplore/MixStyle 也吃**同一條 torch 亂數流** ⇒ 拿掉 diffusion 會位移亂數、
#   改變風格增強的抽樣（0728 記錄約 1–3 分）。而 λ=0 走完全相同的程式碼路徑、乾淨那次前向傳
#   communicator=None（守衛不成立 ⇒ 零亂數消耗）⇒ **與 1a 的亂數流逐位相同**。
PROTO="${PROTO:-0}"
PROTO_ARGS=""
PROTO_TAG=""
if [ "$PROTO" = "1" ]; then
    PROTO_M="${PROTO_M:-0.95}"                 # 逐樣本 EMA；窗口約 1/(1-m) 個「樣本」(~2 批)
    PROTO_TEMP="${PROTO_TEMP:-0.1}"            # CIDER tau=0.1
    PROTO_COMP="${PROTO_COMP:-0.1}"
    PROTO_DISP="${PROTO_DISP:-0.1}"
    PROTO_STYLE="${PROTO_STYLE:-0}"            # ⚠️ 階段 1 必須為 0：單域節點的類別中心 ≡ 該畫風
                                               #   原型 ⇒ L_style 與 L_comp 是同一個量（0803 §2.6.5）
    PROTO_REL="${PROTO_REL:-0}"
    PROTO_REL_MARGIN="${PROTO_REL_MARGIN:-0}"  # 從 0 起；GorD 附錄值是 energy 尺度、不可搬
    PROTO_WARMUP="${PROTO_WARMUP:-10}"
    PROTO_BALANCE_EVERY="${PROTO_BALANCE_EVERY:-20}"
    PROTO_ARGS="--use_proto_reg --proto_m ${PROTO_M} --proto_temp ${PROTO_TEMP} \
--proto_comp_ratio ${PROTO_COMP} --proto_disp_ratio ${PROTO_DISP} \
--proto_style_ratio ${PROTO_STYLE} --proto_rel_ratio ${PROTO_REL} \
--proto_rel_margin ${PROTO_REL_MARGIN} --proto_warmup_epochs ${PROTO_WARMUP} \
--proto_balance_every ${PROTO_BALANCE_EVERY}"
    if [ "${PROTO_NOPROJ:-0}" = "1" ]; then
        PROTO_ARGS="${PROTO_ARGS} --proto_no_projection"
    fi
    # 1ap：獨立項 L_pair ⇒ 唯一直接約束 angle(z,z~) 的損失（1a/1a-fix 實測停在 27.91°、
    #      末段斜率≈0，因為現有三個損失沒有一項在管它）。
    #      ⚠️ 不用對比/softmax 形式：L_disp 已把類別中心推到 101.44° ⇒ L_comp 的 softmax 飽和
    #      （neg_sum=2.2e-04），任何加進去的正樣本梯度只有 1e-4（0814 單元測試實證）。
    PROTO_PAIR="${PROTO_PAIR:-0}"                  # L_pair 的梯度份額（0=關）
    PROTO_PAIR_MARGIN="${PROTO_PAIR_MARGIN:-0.92}" # 目標 cos；0.92→23.1°（現況 0.884→27.91°）
    PROTO_ARGS="${PROTO_ARGS} --proto_pair_ratio ${PROTO_PAIR} --proto_pair_margin ${PROTO_PAIR_MARGIN}"
    # 與 KSD 互斥（兩者都要搶那個兩步風格前向；保持單一變因）
    KSD_ARGS=""
    # 臂的標籤：確保各臂的 checkpoint / wandb 不互相覆蓋。
    # ⚠️ 必須「短」——savePath 會與 description 串接再加 _res，超過 255 bytes 就
    #    OSError: File name too long（2026-08-10 踩過）。故只有「非預設值」才編進名字。
    if [ "$PROTO_COMP" = "0" ] && [ "$PROTO_DISP" = "0" ] && [ "$PROTO_REL" = "0" ]; then
        PROTO_TAG="_p0"
    elif [ "${PROTO_NOPROJ:-0}" = "1" ]; then
        PROTO_TAG="_p1c"
    elif [ "$PROTO_REL" != "0" ]; then
        PROTO_TAG="_p1b"
    elif [ "$PROTO_PAIR" != "0" ]; then
        PROTO_TAG="_p1ap"
    else
        PROTO_TAG="_p1a"
    fi
    [ "$PROTO_COMP" != "0.1" ] && [ "$PROTO_COMP" != "0" ] && PROTO_TAG="${PROTO_TAG}c${PROTO_COMP}"
    [ "$PROTO_DISP" != "0.1" ] && [ "$PROTO_DISP" != "0" ] && PROTO_TAG="${PROTO_TAG}d${PROTO_DISP}"
    [ "$PROTO_REL"  != "0" ] && [ "$PROTO_REL" != "0.1" ] && PROTO_TAG="${PROTO_TAG}r${PROTO_REL}"
    [ "$PROTO_PAIR" != "0" ] && [ "$PROTO_PAIR" != "0.1" ] && PROTO_TAG="${PROTO_TAG}p${PROTO_PAIR}"
    [ "$PROTO_PAIR" != "0" ] && [ "$PROTO_PAIR_MARGIN" != "0.92" ] && PROTO_TAG="${PROTO_TAG}g${PROTO_PAIR_MARGIN}"
    [ "$PROTO_M"    != "0.95" ] && PROTO_TAG="${PROTO_TAG}m${PROTO_M}"
    [ "$PROTO_WARMUP" != "10" ] && PROTO_TAG="${PROTO_TAG}w${PROTO_WARMUP}"
    # ★ 2026-08-21 防呆：上面的規則在「其中一項為 0」時不加後綴，會讓 comp-only(0.1/0) 與
    #   disp-only(0/0.1) 都變成 _p1a、與既有 1a 撞名並覆蓋其 checkpoint。既有四支不受影響：
    #   1a/1a-fix 兩者皆 0.1、λ=0 兩者皆 0（走 _p0）、1ap 走 _p1ap。
    [ "$PROTO_DISP" = "0" ] && [ "$PROTO_COMP" != "0" ] && PROTO_TAG="${PROTO_TAG}_conly"
    [ "$PROTO_COMP" = "0" ] && [ "$PROTO_DISP" != "0" ] && PROTO_TAG="${PROTO_TAG}_donly"
    true
fi

# ===== Async event-triggered communication (Stage 1 baseline) =====
# Same StyleDDG config as the det V1 baseline, ONLY difference = --async_trigger (single-variable A/B).
# Enable with: ASYNC=1. Sweep the threshold via TRIGGER_THRESHOLD (0.0 => always broadcast ~ sync sanity).
#   sync baseline    : bash start_v1_baseline.sh cartoon -1
#   async sanity(τ=0): ASYNC=1 bash start_v1_baseline.sh cartoon -1
#   async event      : ASYNC=1 TRIGGER_THRESHOLD=0.01 bash start_v1_baseline.sh cartoon -1
ASYNC="${ASYNC:-0}"
ASYNC_ARGS=""
ASYNC_TAG=""
if [ "$ASYNC" = "1" ]; then
    TAU="${TRIGGER_THRESHOLD:-0.0}"
    MAXINT="${TRIGGER_MAX_INTERVAL:-50}"
    GMODE="${GAMMA_MODE:-const}"   # const(主實驗、穩定門檻) 或 lr(PersonalizedET-faithful)
    ASYNC_ARGS="--async_trigger --trigger_threshold ${TAU} --trigger_max_interval ${MAXINT} --trigger_gamma_mode ${GMODE}"
    ASYNC_TAG="_async_${GMODE}_tau${TAU}"
    # [plan 1007] 本地時鐘固定週期：TRIGGER_MODE=periodic TRIGGER_PERIOD=6.2133 [TRIGGER_PHASE_SEED=0]
    if [ "${TRIGGER_MODE:-param}" = "periodic" ]; then
        if [ -n "${TRIGGER_PERIOD_FILE:-}" ]; then
            # [plan 1009] 每節點週期表（C-bw／C-match）：TRIGGER_PERIOD_FILE=…csv TRIGGER_PERIOD_TAG=bw|match
            ASYNC_ARGS="${ASYNC_ARGS} --trigger_mode periodic --trigger_period_file ${TRIGGER_PERIOD_FILE} --trigger_phase_seed ${TRIGGER_PHASE_SEED:-0}"
            ASYNC_TAG="_async_p${TRIGGER_PERIOD_TAG:?TRIGGER_PERIOD_FILE needs TRIGGER_PERIOD_TAG}"
        else
            ASYNC_ARGS="${ASYNC_ARGS} --trigger_mode periodic --trigger_period ${TRIGGER_PERIOD:?periodic needs TRIGGER_PERIOD} --trigger_phase_seed ${TRIGGER_PHASE_SEED:-0}"
            ASYNC_TAG="_async_periodic${TRIGGER_PERIOD}"
        fi
    fi
    # [plan 1009] 頻寬不平均：BW_DIST=beta BW_SEED=0 [BW_FLOOR=0.1] [BW_RHO=1 ⇒ EF-HC 門檻 ×1/b_i]
    #   [plan 1012] 自訂頻寬表：BW_DIST=file BW_FILE=…csv BW_FILE_TAG=art0.1 ⇒ 名稱 _bwf<TAG>
    if [ "${BW_DIST:-none}" = "file" ]; then
        ASYNC_ARGS="${ASYNC_ARGS} --bw_dist file --bw_file ${BW_FILE:?BW_DIST=file needs BW_FILE}"
        BW_TAG="_bwf${BW_FILE_TAG:?BW_DIST=file needs BW_FILE_TAG}"
        if [ "${BW_RHO:-0}" = "1" ]; then
            ASYNC_ARGS="${ASYNC_ARGS} --bw_rho"
            BW_TAG="${BW_TAG}r"
        fi
        ASYNC_TAG="${ASYNC_TAG}${BW_TAG}"
    elif [ "${BW_DIST:-none}" != "none" ]; then
        ASYNC_ARGS="${ASYNC_ARGS} --bw_dist ${BW_DIST} --bw_seed ${BW_SEED:-0} --bw_floor ${BW_FLOOR:-0.1}"
        BW_TAG="_bw${BW_DIST:0:1}${BW_SEED:-0}"
        if [ "${BW_RHO:-0}" = "1" ]; then
            ASYNC_ARGS="${ASYNC_ARGS} --bw_rho"
            BW_TAG="${BW_TAG}r"
        fi
        ASYNC_TAG="${ASYNC_TAG}${BW_TAG}"
    fi
    # Stage 2: 風格也 event-trigger（綁模型觸發、持久 buffer）。Enable with: ASYNC_STYLE=1
    #   Stage1(model-only) : ASYNC=1 TRIGGER_THRESHOLD=1e-5 bash start_v1_baseline.sh cartoon 6
    #   Stage2(model+style): ASYNC=1 ASYNC_STYLE=1 TRIGGER_THRESHOLD=1e-5 bash start_v1_baseline.sh cartoon 6
    if [ "${ASYNC_STYLE:-0}" = "1" ]; then
        ASYNC_ARGS="${ASYNC_ARGS} --async_style"
        ASYNC_TAG="${ASYNC_TAG}_style"
    fi
fi

# Sparse-topology pilot: RGG connection radius (graphid=6 only). Default 0.8 = dense (unchanged).
# Lower (e.g. 0.5) => sparser graph, some nodes' 1-hop misses domains. Tags name so it won't clobber the dense anchor.
RGG_RADIUS="${RGG_RADIUS:-0.8}"
RGG_TAG=""
if [ "$RGG_RADIUS" != "0.8" ]; then
    RGG_TAG="_r${RGG_RADIUS}"
fi

# ===== BN running 統計跨節點聚合（預設 OFF＝現行 SiloBN 式 keep-local）=====
# 論證：research/bn_fusion/0729_bn_consensus_first_principles.md（§5 檢測側因果鏈）
# payload 增量 +0.086%（9,600 個數字 vs backbone 11.18M）。Enable: AGG_BN=1
AGG_BN_ARGS=""
if [ "${AGG_BN:-0}" = "1" ]; then
    AGG_BN_ARGS="--aggregate_bn"
    ABL_TAG="${ABL_TAG}_aggbn"
fi
# 只印跨節點 BN 分歧、不改行為（keep-local 對照組用；AGG_BN=1 時自動開啟不必設）
if [ "${BN_DIV_DIAG:-0}" = "1" ] && [ "${AGG_BN:-0}" != "1" ]; then
    AGG_BN_ARGS="--bn_div_diag"
fi

# ===== [唯讀診斷] 聚合【前】的逐邊偏離（EDGE_DIV=1）=====
# 問題：模型參數側有沒有畫風結構（同畫風邊 vs 跨畫風邊分不分得開）可拿來調混合權重。
# 唯讀、不改訓練行為 ⇒ 驗收條件＝準確率軌跡必須與參照 run 逐位相同。
# ABL_TAG 加尾綴 ⇒ 自成 savePath，不會覆蓋參照 run。
EDGE_DIV_ARGS=""
if [ "${EDGE_DIV:-0}" = "1" ]; then
    EDGE_DIV_ARGS="--edge_div_diag"
    ABL_TAG="${ABL_TAG}_edgediag"
fi

# ===== [唯讀診斷] 原型跨節點交換 ＋ 三訊號影子記錄（SHADOW=1）=====
# plan: research/1001_prototype_consensus_trigger_plan.md
# 問題：把「何時廣播」的判斷訊號從參數漂移換成任務層的類別原型，兩者是不是同一件事。
# 唯讀、不改任何決策（觸發仍由規則 0 ＝ _param_delta 決定）、原型表住 communicator 不進 checkpoint
# ⇒ 驗收條件＝**準確率軌跡與 checkpoint 必須與參照 run 逐位相同**（與 EDGE_DIV 同一個驗收方式）。
# ABL_TAG 加尾綴 ⇒ 自成 savePath，不會覆蓋參照 run。需要 PROTO=1 與 ASYNC=1。
SHADOW_ARGS=""
if [ "${SHADOW:-0}" = "1" ]; then
    SHADOW_ARGS="--shadow_signals"
    ABL_TAG="${ABL_TAG}_shadow"
fi

# ===== OSDG 6-way：把某類別全程排除於訓練、當 unknown（route-A 標準協定）=====
# Enable: EXCLUDE_CLASS=person  → 6-way head、person 只在測試出現
EXCLUDE_ARGS=""
if [ -n "${EXCLUDE_CLASS:-}" ]; then
    EXCLUDE_ARGS="--exclude_class ${EXCLUDE_CLASS}"
    ABL_TAG="${ABL_TAG}_osdg_excl_${EXCLUDE_CLASS}"
fi

# graphid=6 拓樸類型：rgg(預設) 或 ring(9節點環、最極端稀疏連通)。Enable: TOPOLOGY=ring
TOPOLOGY="${TOPOLOGY:-rgg}"
TOPO_TAG=""
if [ "$TOPOLOGY" != "rgg" ]; then
    TOPO_TAG="_${TOPOLOGY}"
    RGG_TAG=""  # ring 不看 radius，避免誤導的 _r tag
fi

# 隨機種子（預設 1234；改 SEED 跑不同 seed 做 robustness、tag 進 NAME 避免蓋掉 seed1234 結果）
SEED="${SEED:-1234}"
SEED_TAG=""
if [ "$SEED" != "1234" ]; then
    SEED_TAG="_seed${SEED}"
fi

# 拓樸專用 seed（拆 topo/train seed 混淆）：不設則退回用 SEED（向後相容、與現況一致）。
# 設 TOPO_SEED 可固定拓樸圖、只讓 SEED 變訓練 RNG。tag 進 NAME 避免蓋名。
TOPO_SEED_ARG=""
TOPO_SEED_TAG=""
if [ -n "${TOPO_SEED:-}" ]; then
    TOPO_SEED_ARG="--topo_seed ${TOPO_SEED}"
    TOPO_SEED_TAG="_topo${TOPO_SEED}"
fi

NAME="v1_${STAGE_TAG}_leave_${LEAVE_OUT}${FOURIER_TAG}${KSD_TAG}${PROTO_TAG}${ASYNC_TAG}${RGG_TAG}${TOPO_TAG}${ABL_TAG}${SEED_TAG}${TOPO_SEED_TAG}${NAME_SUFFIX:-}"
# 名稱過長會在存檔時崩潰（{savePath}{name}_res 超過檔名上限，架構 reference §7.11）
if [ ${#NAME} -gt 119 ]; then
    echo "[start_v1_baseline] run name too long (${#NAME} > 119): $NAME" >&2
    exit 1
fi

# Optional Phase 0 passive score-direction diagnostic. Enable with: SCORE_DIAG=1
# PASSIVE (logs only, no loss) — so the run is bit-identical to the deterministic V1 baseline.
# Reuses the same name / savePath / wandb project ($NAME) so it overwrites & reproduces the
# baseline checkpoint and lands in the same wandb project for easy comparison (dany approved).
SCORE_DIAG_ARGS=""
WANDB_PROJECT="$NAME"
RUN_NAME="$NAME"
if [ "${SCORE_DIAG:-0}" = "1" ]; then
    SCORE_DIAG_ARGS="--score_diag --score_diag_every ${SCORE_DIAG_EVERY:-50} --score_diag_t ${SCORE_DIAG_T:-10,25,50}"
fi

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

/home/server5090/Desktop/M11307320/MATCHA/venv_matcha/bin/python train.py \
--wandb_project "$WANDB_PROJECT" \
--dataset pacs \
--use_style_stats \
--datasetRoot ../datasets/ \
--leave_out "$LEAVE_OUT" \
--graphid "$GRAPHID" \
--rgg_radius "$RGG_RADIUS" \
--topology "$TOPOLOGY" \
$NUM_NODES_ARG \
$FOURIER_ARGS \
$SCORE_DIAG_ARGS \
$KSD_ARGS \
$PROTO_ARGS \
$ASYNC_ARGS \
$ISOLATED_ARGS \
$AGG_BN_ARGS \
$EDGE_DIV_ARGS \
$SHADOW_ARGS \
$EXCLUDE_ARGS \
--lr 0.001 \
--bs 64 \
--epoch ${EPOCH:-200} \
--budget 1.0 \
--momentum 0.9 \
--name "$RUN_NAME" \
--model res \
--resnet_type standard \
-p \
$OOD_ARGS \
--description "$RUN_NAME" \
--savePath "./exp_result_${RUN_NAME}" \
--randomSeed ${SEED} \
${TOPO_SEED_ARG} \
$STYLE_SHIFT_ARGS \
--pretrained \
--save_every_epoch ${SAVE_EVERY_EPOCH:-50} \
$AGG_DIAG_ARGS
