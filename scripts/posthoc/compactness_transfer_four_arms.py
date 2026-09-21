"""`L_comp` 到底有沒有讓【未見畫風】的類內散布變小？——方法設計的起點座標

四臂（唯一變因是損失）：
  λ=0    只有交叉熵（原型照常累積但不產生梯度）
  1a     ＋`L_comp`+`L_disp`
  1a-fix 同上（修正四項實作缺陷）
  1ap    再＋`L_pair`（風格不變性約束；已知會全域塌縮）

⚠️ **必須量在兩個空間，理由是 confound**：
  λ=0 的 `proj_head` **從未被訓練**（沒有任何損失碰 `z_aug`）⇒ 它的 128 維是隨機投影
  ⇒ **128 維的跨臂比較被「訓練過的投影 vs 隨機投影」污染，不可單獨解讀**。
  **512 維 `vec` 空間才是乾淨的跨臂比較**（四臂的 backbone 都被交叉熵訓練）。

量什麼（全部只看「散布」，與「位置」分開）：
  類內散布  = 每個樣本與【自己那一類的平均方向】的夾角，跨類別平均 ⇒ 條件 (i)
  類間夾角  = 類別平均方向之間的兩兩夾角 ⇒ 尺規（`L_disp` 會改變它）
  相對緊緻度 = 類內散布 / 類間夾角 ⇒ 去掉尺規影響
  位置誤差  = cartoon 的類別平均方向 vs 來源域的類別平均方向 ⇒ 條件 (ii)

判準（事前寫死）：
  cartoon 散布隨訓練**明顯變小** ⇒ `L_comp` 有遷移，方法可在既有機制上加強
  cartoon 散布**幾乎不動**（或只有來源域變小）⇒ **遷移率是零，方法必須從頭建**
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import torch, util
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion

PACS = ["art_painting", "cartoon", "photo", "sketch"]
# ★ 2026-08-21：可用環境變數換 checkpoint 與追加臂（不設＝原行為，向後相容）
#   CKPT_TAG=epoch_100  ⇒ 讀 *_node_i_epoch_100.pth（預設 final）
#   EXTRA_ARMS="名稱:DESC[,名稱:DESC]" ⇒ 在四臂之後追加（如 comp-only 消融臂）
CKPT_TAG = os.environ.get("CKPT_TAG", "final")
ARMS = [
    ("λ=0（只有CE）", "v1_stage2_leave_cartoon_proto_lam0_m0.95w10_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234"),
    ("1a（+comp+disp）", "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234"),
    ("1a-fix", "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"),
    ("1ap（+L_pair）", "v1_stage2_leave_cartoon_p1ap_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234"),
]
leave = "cartoon"; UNK = 6; N = 9; NC = 6; DEG = 57.29577951308232
avail = [d for d in PACS if d != leave]; per = N // len(avail)
OWN = [avail[min(i // per, len(avail) - 1)] for i in range(N)]
ld = {d: TD.load_pacs_test_data("../datasets/", d, 64, 4)[0] for d in PACS}
nrm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)


def bn_avg(CK, DESC):
    S = [torch.load(os.path.join(CK, f"{DESC}_node_{i}_{CKPT_TAG}.pth"), map_location="cpu",
                    weights_only=False)["backbone_state"] for i in range(N)]
    BKs = [k for k in S[0] if k.endswith("running_mean") or k.endswith("running_var")]
    A = {}
    for k in BKs:
        if k.endswith("running_mean"):
            A[k] = torch.stack([S[i][k].float() for i in range(N)]).mean(0)
    for k in BKs:
        if k.endswith("running_var"):
            mk = k.replace("running_var", "running_mean")
            mi = torch.stack([S[i][mk].float() for i in range(N)])
            vi = torch.stack([S[i][k].float() for i in range(N)])
            A[k] = (vi + mi ** 2).mean(0) - mi.mean(0) ** 2
    del S
    return A


@torch.no_grad()
def collect(bb, loader):
    """回傳 (512維 vec, 128維投影, 標籤)"""
    V, Z, Y = [], [], []
    for b in loader:
        x, y, _ = util.unpack_batch(b)
        h = bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(x.to("cuda")))))
        h = bb.backbone.layer1(h); h = bb.backbone.layer2(h); h = bb.backbone.layer3(h)
        _, v = bb.forward_from_layer3(h)
        V.append(v.cpu().numpy()); Z.append(bb.project(v).cpu().numpy())
        Y.append(np.asarray(y).flatten())
    V, Z, Y = np.concatenate(V), np.concatenate(Z), np.concatenate(Y)
    m = Y != UNK
    return nrm(V[m]), nrm(Z[m]), Y[m].astype(int)


def scatter(X, Y):
    """類內角度散布：每個樣本與自己類別平均方向的夾角，跨類別平均"""
    out = []
    for c in range(NC):
        m = Y == c
        if m.sum() < 2: continue
        mu = nrm(X[m].mean(0))
        out.append(np.arccos(np.clip(X[m] @ mu, -1 + 1e-7, 1 - 1e-7)).mean() * DEG)
    return float(np.mean(out))


def centers(X, Y):
    return np.stack([nrm(X[Y == c].mean(0)) for c in range(NC)])


def inter(Cn):
    A = np.arccos(np.clip(Cn @ Cn.T, -1 + 1e-7, 1 - 1e-7)) * DEG
    return float(A[~np.eye(NC, dtype=bool)].mean())


R = {}
for _spec in filter(None, os.environ.get("EXTRA_ARMS", "").split(",")):
    _n, _d = _spec.split(":", 1)
    ARMS.append((_n.strip(), _d.strip()))
print(f"[cfg] checkpoint={CKPT_TAG}   臂數={len(ARMS)}：{[a[0] for a in ARMS]}", flush=True)

for name, DESC in ARMS:
    CK = "exp_result_" + DESC
    AVG = bn_avg(CK, DESC)
    acc = {k: [] for k in ["s_src512", "s_tgt512", "i_src512", "i_tgt512", "pos512",
                           "s_src128", "s_tgt128", "i_src128", "i_tgt128", "pos128"]}
    for i in range(N):
        bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_{CKPT_TAG}.pth"), 6, "cuda")
        sd = bb.state_dict()
        for k, v in AVG.items(): sd[k].copy_(v.to(sd[k].device).to(sd[k].dtype))
        Vs, Zs, Ys = collect(bb, ld[OWN[i]])
        Vt, Zt, Yt = collect(bb, ld[leave])
        del bb
        for tag, Xs, Xt in [("512", Vs, Vt), ("128", Zs, Zt)]:
            acc[f"s_src{tag}"].append(scatter(Xs, Ys)); acc[f"s_tgt{tag}"].append(scatter(Xt, Yt))
            Cs, Ct = centers(Xs, Ys), centers(Xt, Yt)
            acc[f"i_src{tag}"].append(inter(Cs)); acc[f"i_tgt{tag}"].append(inter(Ct))
            acc[f"pos{tag}"].append(float(np.arccos(np.clip((Cs * Ct).sum(1), -1, 1)).mean() * DEG))
    R[name] = {k: float(np.mean(v)) for k, v in acc.items()}
    print(f"  {name} 完成", flush=True)

W = 100
print("\n" + "=" * W); print("§0 自檢"); print("=" * W)
r = R["1a-fix"]
print(f"  1a-fix 128維：來源域散布 {r['s_src128']:.2f}°（0819b 報 31.21°）"
      f"　cartoon 散布 {r['s_tgt128']:.2f}°（報 40.78°）")
print(f"  1a-fix 128維 類間夾角（樣本均值版）{r['i_src128']:.2f}°"
      f"　（⚠️ 0818 §3.8 的 101.44° 是【原型】之間的夾角、不是同一個量）")
print("  ⚠️ λ=0 的 proj_head 從未被訓練 ⇒ **128 維的跨臂比較被污染，只能看 512 維**")

for tag, lab in [("512", "★ 512 維 vec 空間（乾淨的跨臂比較）"), ("128", "128 維投影空間（⚠️ λ=0 是隨機投影，僅供對照）")]:
    print("\n" + "=" * W); print(lab); print("=" * W)
    print(f"  {'臂':<18}{'來源域散布':>12}{'cartoon散布':>13}{'差':>9}{'類間夾角(來源)':>16}{'相對緊緻(來源)':>16}{'相對緊緻(cartoon)':>18}")
    for name, _ in ARMS:
        d = R[name]
        ss, st, ii = d[f"s_src{tag}"], d[f"s_tgt{tag}"], d[f"i_src{tag}"]
        print(f"  {name:<17}{ss:>11.2f}°{st:>12.2f}°{st-ss:>8.2f}°{ii:>15.2f}°"
              f"{ss/ii:>15.3f}{st/ii:>17.3f}")
    print(f"\n  {'臂':<18}{'位置誤差(cartoon類別均值 vs 來源域類別均值)':>44}")
    for name, _ in ARMS:
        print(f"  {name:<17}{R[name]['pos'+tag]:>40.2f}°")

print("\n" + "=" * W); print("★ 判決：`L_comp` 有沒有讓未見畫風變緊？"); print("=" * W)
base = R["λ=0（只有CE）"]; fix = R["1a-fix"]
for tag in ["512", "128"]:
    ds = fix[f"s_src{tag}"] - base[f"s_src{tag}"]
    dt = fix[f"s_tgt{tag}"] - base[f"s_tgt{tag}"]
    rs = fix[f"s_src{tag}"] / fix[f"i_src{tag}"] - base[f"s_src{tag}"] / base[f"i_src{tag}"]
    rt = fix[f"s_tgt{tag}"] / fix[f"i_src{tag}"] - base[f"s_tgt{tag}"] / base[f"i_src{tag}"]
    print(f"  {tag} 維　λ=0 → 1a-fix：來源域散布 {ds:+.2f}°　cartoon 散布 {dt:+.2f}°"
          f"　｜ 相對緊緻度 來源 {rs:+.3f}　cartoon {rt:+.3f}")
print("\n  判準：cartoon 散布明顯變小 ⇒ 有遷移，可在既有機制上加強")
print("        只有來源域變小、cartoon 幾乎不動 ⇒ 遷移率是零，方法必須從頭建")
