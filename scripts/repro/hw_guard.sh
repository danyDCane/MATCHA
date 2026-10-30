#!/bin/bash
# [1030] 訓練前的硬體／環境檢查。用法：在呼叫 train.py 之前 `source scripts/repro/hw_guard.sh "$RUN_NAME"`。
# 目的：同程式＋同種子＝同成績。登記值都在 RTX 5090（170 個 SM）上跑；同架構但 SM 數不同的卡（例如 5080 的 84 個），
#       要讓 PyTorch 與 cuBLAS 都看到 170（scripts/repro/sm_spoof2.so）才會選與 5090 相同的算法（1028 報告）。
# 行為：
#   ・架構不是 12.0，或 torch／cuBLAS／cuDNN／驅動版本與登記環境不同 ⇒ 停止。
#     REPRO_ALLOW_MISMATCH=1 可放行，但 log 會標明「不保證與登記值逐位相同」。
#   ・SM 數≠170 ⇒ 自動 export LD_PRELOAD＋SPOOF_SM_COUNT=170，並實測 PyTorch 與 cuBLAS 都已照 170 運作；沒生效就停止。
#   ・REPRO_SPOOF=0 ⇒ 不偽裝（想看新卡原生行為時用），log 同樣標明不保證逐位相同。
# 記錄：一行 [hw_guard] 摘要印進訓練 log；完整資訊寫 logs/hw_env/<RUN_NAME>.json。
_HG_ROOT=/home/server5090/Desktop/M11307320/MATCHA
_HG_PY=$_HG_ROOT/venv_matcha/bin/python
_HG_SO=$_HG_ROOT/scripts/repro/sm_spoof2.so
_HG_NAME=${1:-unnamed_$(date +%Y%m%d_%H%M%S)}
# 登記環境（5090 時代；1028 在 5080＋偽裝下逐位重現的版本組合）
_HG_REF="sm=170 cc=12.0 torch=2.9.1+cu128 cublas=12.8.4.1 cudnn=91002 driver=580.126.09"

_hg_fail() { echo "[hw_guard] ⛔ $*" >&2; echo "[hw_guard] ⛔ $*"; exit 4; }

# ① 不偽裝，讀真實硬體與版本
_HG_REAL=$(env -u LD_PRELOAD -u SPOOF_SM_COUNT "$_HG_PY" - <<'EOF' 2>/dev/null
import torch, importlib.metadata as md, subprocess
p = torch.cuda.get_device_properties(0)
drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()[0]
print(f"gpu={p.name.replace(' ', '_')} sm={p.multi_processor_count} cc={p.major}.{p.minor} torch={torch.__version__} "
      f"cublas={md.version('nvidia-cublas-cu12')} cudnn={torch.backends.cudnn.version()} driver={drv}")
EOF
)
[ -z "$_HG_REAL" ] && _hg_fail "讀不到 GPU（CUDA 不可用？）"
_hg_get() { echo "$1" | tr ' ' '\n' | sed -n "s/^$2=//p"; }
_HG_MISMATCH=""
for k in cc torch cublas cudnn driver; do
    [ "$(_hg_get "$_HG_REAL" $k)" != "$(_hg_get "$_HG_REF" $k)" ] && _HG_MISMATCH="$_HG_MISMATCH $k:$(_hg_get "$_HG_REAL" $k)≠$(_hg_get "$_HG_REF" $k)"
done
_HG_SM=$(_hg_get "$_HG_REAL" sm)
_HG_STATUS="可逐位重現"
if [ -n "$_HG_MISMATCH" ]; then
    if [ "${REPRO_ALLOW_MISMATCH:-0}" = "1" ]; then
        _HG_STATUS="⚠️ 環境與登記不同（${_HG_MISMATCH# }），不保證與登記值逐位相同"
    else
        _hg_fail "環境與登記不同：${_HG_MISMATCH# }。成績無法保證可重現；確定要跑請設 REPRO_ALLOW_MISMATCH=1"
    fi
fi

# ② SM 數不同 ⇒ 偽裝成 170，並實測生效
_HG_SPOOF=off
if [ "$_HG_SM" != "170" ]; then
    if [ "${REPRO_SPOOF:-1}" = "0" ]; then
        _HG_STATUS="⚠️ REPRO_SPOOF=0：以本卡原生 ${_HG_SM} 個 SM 運作，不保證與登記值逐位相同"
    else
        [ -f "$_HG_SO" ] || _hg_fail "找不到 $_HG_SO（編譯指令在 scripts/repro/sm_spoof.c 開頭）"
        _HG_CHECK=$(env LD_PRELOAD="$_HG_SO" SPOOF_SM_COUNT=170 "$_HG_PY" - <<'EOF' 2>/dev/null
import torch
torch.backends.cudnn.deterministic = True
sm = torch.cuda.get_device_properties(0).multi_processor_count
# cuBLAS 檢查：同一組權重，41 列直接算 vs 補零到 64 列算再取前 41 列。
# 以 170 個 SM 挑算法時兩者逐位相同（5090 原生、5080 偽裝後皆是）；5080 原生 84 個 SM 時不同（1028 實測）。
g = torch.Generator().manual_seed(0)
lin = torch.nn.Linear(512, 512).cuda()
with torch.no_grad():
    lin.weight.copy_(torch.randn(512, 512, generator=g) * 0.04); lin.bias.copy_(torch.randn(512, generator=g) * 0.01)
    x = torch.relu(torch.randn(64, 512, generator=g)).cuda()
    ok = torch.equal(lin(x[:41]), lin(x)[:41])
print(f"torch_sm={sm} cublas170={'yes' if ok else 'no'}")
EOF
)
        [ "$_HG_CHECK" = "torch_sm=170 cublas170=yes" ] || _hg_fail "偽裝沒有生效（$_HG_CHECK）"
        export LD_PRELOAD="$_HG_SO${LD_PRELOAD:+:$LD_PRELOAD}" SPOOF_SM_COUNT=170
        _HG_SPOOF="on($_HG_CHECK)"
    fi
fi

echo "[hw_guard] $_HG_REAL spoof=$_HG_SPOOF ⇒ $_HG_STATUS"
mkdir -p "$_HG_ROOT/logs/hw_env"
cat > "$_HG_ROOT/logs/hw_env/${_HG_NAME}.json" <<EOF
{"run": "$_HG_NAME", "time": "$(date '+%F %T')", "real": "$_HG_REAL", "reference": "$_HG_REF",
 "spoof": "$_HG_SPOOF", "status": "$_HG_STATUS", "allow_mismatch": "${REPRO_ALLOW_MISMATCH:-0}", "repro_spoof": "${REPRO_SPOOF:-1}"}
EOF
