#!/bin/bash
# [1030] 啟動流程驗收：走正常 start_v1_baseline.sh（不帶任何手動環境變數，由 hw_guard 自動處理），
#        重跑 F11（scripts/run_1019_fold.sh 的 COMMON＋AR_U，5090 於 10-21 登記）前 6 epoch，與 5090 的 ep5 存檔及 async_diag 逐位比。
# 用法：setsid bash scripts/repro/accept_f11_6ep.sh <NAME_SUFFIX> > /dev/null 2>&1 < /dev/null &
SUF=${1:?需要 NAME_SUFFIX，例如 _hw80guard}
ROOT=/home/server5090/Desktop/M11307320/MATCHA; cd $ROOT || exit 9
D=$ROOT/results/proto_trigger; L=$ROOT/logs/proto_trigger
REF=v1_stage2_leave_cartoon_p1a_async_puni_bwfart0.1_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234
LOG=$L/accept_f11${SUF}_train.log; MON=$L/accept_f11${SUF}_monitor.log
COMMON="PROTO=1 USE_OOD=0 ASYNC=1 ASYNC_STYLE=1 AGG_BN=1 SHADOW=1 EXCLUDE_CLASS=person SEED=2026 TOPO_SEED=1234 SAVE_EVERY_EPOCH=5 FUSION_JUMP_LOG=1"
PER="TRIGGER_THRESHOLD=1e-5 GAMMA_MODE=const TRIGGER_MAX_INTERVAL=4000 TRIGGER_MODE=periodic TRIGGER_PHASE_SEED=0"
AR_U="$PER TRIGGER_PERIOD_FILE=$D/1019_fold/period_uniform_art0.1_17.0970.csv TRIGGER_PERIOD_TAG=uni BW_DIST=file BW_FILE=$D/1012_artslow/bandwidth_art0.1.csv BW_FILE_TAG=art0.1"
[ -d "exp_result_${REF}${SUF}" ] && { echo "abort: exp_result_${REF}${SUF} exists" > $MON; exit 3; }
setsid env -u LD_PRELOAD -u SPOOF_SM_COUNT $COMMON $AR_U NAME_SUFFIX=$SUF WANDB_MODE=offline bash start_v1_baseline.sh cartoon 6 > $LOG 2>&1 < /dev/null &
PG=$!; sleep 2
echo "start $(date '+%F %T') pid=$PG pgid=$(ps -o pgid= -p $PG | tr -d ' ')" > $MON
while kill -0 $PG 2>/dev/null; do
    [ "$(grep -c '^Epoch [0-9]*:' $LOG)" -ge 6 ] && { kill -TERM -- -$PG; sleep 5; kill -KILL -- -$PG 2>/dev/null; break; }
    sleep 5
done
echo "end $(date '+%F %T') epochs=$(grep -c '^Epoch [0-9]*:' $LOG)" >> $MON
grep -m1 "\[hw_guard\]" $LOG >> $MON
$ROOT/venv_matcha/bin/python $ROOT/scripts/repro/compare_to_registered.py --ref_dir exp_result_$REF --new_dir exp_result_${REF}${SUF} \
    --ref_name $REF --new_name ${REF}${SUF} --ckpts epoch_5 --diag consensus.csv broadcast_log.csv fusion_jump.csv shadow_signals.csv --prefix >> $MON 2>&1
echo "ALLDONE $(date '+%T')" >> $MON
