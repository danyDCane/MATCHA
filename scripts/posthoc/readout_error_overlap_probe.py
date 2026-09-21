"""實驗①：原型讀出與 energy 是不是在【同一批樣本】上犯【同一種錯】？

go/no-go：決定「改讀出形式」(路線 b) 值不值得投入。
  互補（有獨佔錯誤、融合能超過 energy）⇒ 原型握有 energy 沒有的資訊 ⇒ (b) 值得走
  高度重疊（融合 ≈ energy 單獨）      ⇒ 原型資訊是 energy 的子集 ⇒ (b) 收掉，全力回 (a) 幾何

「錯」的定義（兩個分數尺度不同，各用各的 95 分位＝不同操作點，不可比）：
  **把兩個分數各自校準到【相同的放行率】**（預設 0.3572＝我方目前操作點），
  在「放走一樣多的 person」前提下比較誰誤殺了哪些正常樣本。
  錯 A ＝ 誤拒（② 被判異常）　錯 B ＝ 放行（③ 被判正常）

三個基準：獨立時期望重疊 = P(A錯)·P(B錯)；完全相同時 = min(P(A錯),P(B錯))；實測落在哪。
⚠️ Spearman 必須【分堆算】——②∪③ 全體的相關會被「兩者都能分開 ②③」墊高（0818 §3.1 的 Simpson 同型）。
⚠️ 融合檢驗是【診斷不是方法】——用到 energy 就是用了信心度，撞 A4。

順帶補齊：λ=0 + msp + BN 同步的四軸；跨模型固定放行率下的誤拒率對照。
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import torch, util
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers

PACS = ["art_painting", "cartoon", "photo", "sketch"]
RUNS = {"1a-fix": "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix",
        "λ=0": "v1_stage2_leave_cartoon_proto_lam0_m0.95w10_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234"}
leave = "cartoon"; UNK = 6; N = 9; DEG = 57.29577951308232
avail = [d for d in PACS if d != leave]; per = N // len(avail)
OWN = [avail[min(i // per, len(avail) - 1)] for i in range(N)]
ld = {d: TD.load_pacs_test_data("../datasets/", d, 64, 4)[0] for d in PACS}
nrm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
TARGET_MISS = 0.3572          # 我方目前操作點的放行率


def bn_avg(CK, DESC):
    S = [torch.load(os.path.join(CK, f"{DESC}_node_{i}_final.pth"), map_location="cpu",
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
def collect(bb, loader, C):
    """回傳 (原型角距離分數, energy分數, msp分數, 標籤)。三者皆已定向為『高＝越像 OOD』"""
    AN, EN, MS, Y = [], [], [], []
    for b in loader:
        x, y, _ = util.unpack_batch(b)
        h = bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(x.to("cuda")))))
        h = bb.backbone.layer1(h); h = bb.backbone.layer2(h); h = bb.backbone.layer3(h)
        lo, v = bb.forward_from_layer3(h)
        z = nrm(bb.project(v).cpu().numpy())
        AN.append(np.arccos(np.clip(z @ C.T, -1 + 1e-7, 1 - 1e-7)).min(1) * DEG)
        EN.append((-torch.logsumexp(lo, 1)).cpu().numpy())
        MS.append((-lo.softmax(1).max(1).values).cpu().numpy())
        Y.append(np.asarray(y).flatten())
    return (np.concatenate(AN), np.concatenate(EN), np.concatenate(MS), np.concatenate(Y))


def tau_at_miss(s3, m):
    """回傳讓放行率恰為 m 的門檻（放行＝③ 的分數 ≤ τ）"""
    return float(np.quantile(s3, m))


def pct(x):
    """轉成排名百分位 [0,1]"""
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


ACC = {}
for tag, DESC in RUNS.items():
    CK = "exp_result_" + DESC
    AVG = bn_avg(CK, DESC)
    rec = []
    for i in range(N):
        bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_final.pth"), 6, "cuda")
        sd = bb.state_dict()
        for k, v in AVG.items(): sd[k].copy_(v.to(sd[k].device).to(sd[k].dtype))
        C = nrm(class_centers(bb.prototypes, bb.proto_count).cpu().numpy())
        a1, e1, m1_, y1 = collect(bb, ld[OWN[i]], C)
        a2, e2, m2_, y2 = collect(bb, ld[leave], C)
        del bb
        k1, k2, u2 = y1 != UNK, y2 != UNK, y2 == UNK
        rec.append(dict(
            proto=(a1[k1], a2[k2], a2[u2]), energy=(e1[k1], e2[k2], e2[u2]), msp=(m1_[k1], m2_[k2], m2_[u2])))
        print(f"  [{tag}] node_{i} 完成", flush=True)
    ACC[tag] = rec

m = lambda x: float(np.mean(x))
W = 96
print("\n" + "=" * W); print("§0 自檢"); print("=" * W)
for tag in RUNS:
    for nm in ["proto", "energy", "msp"]:
        r = ACC[tag]
        dep = m([roc_auc_score([0]*len(d[nm][1])+[1]*len(d[nm][2]), np.r_[d[nm][1], d[nm][2]]) for d in r])
        fpr = m([float((d[nm][1] > np.quantile(d[nm][0], .95)).mean()) for d in r])
        mis = m([float((d[nm][2] <= np.quantile(d[nm][0], .95)).mean()) for d in r])
        print(f"  {tag:<8}{nm:<8}部署 {dep:.4f}  誤拒@src95 {fpr:.4f}  放行 {mis:.4f}")
print("  （對照 1a-fix：proto .8145/.2370/.3572　energy .8380/.2372/.3152　λ=0 energy .8337/.2139/.3391）")
print("  ★ λ=0 + msp + BN 同步 為本輪首次量測（原始誤拒率靶 0.3534 是未做 BN 同步的 msp）")

print("\n" + "=" * W)
print(f"★ A：分數相關性（Spearman）——必須分堆看")
print("=" * W)
r = ACC["1a-fix"]
for lab, idx in [("②∪③ 全體（會被墊高，僅供對照）", None), ("② cartoon 已知類別 內部", 1), ("③ person 內部", 2)]:
    v = []
    for d in r:
        if idx is None:
            x = np.r_[d["proto"][1], d["proto"][2]]; y = np.r_[d["energy"][1], d["energy"][2]]
        else:
            x, y = d["proto"][idx], d["energy"][idx]
        v.append(spearmanr(x, y).statistic)
    print(f"  {lab:<34}ρ = {m(v):>7.4f}")

print("\n" + "=" * W)
print(f"★ B：錯誤集合（兩個分數皆校準到相同放行率 {TARGET_MISS}）")
print("=" * W)
for errname, pile, is_err in [("錯 A：誤拒（② 被判異常）", 1, lambda s, t: s > t),
                              ("錯 B：放行（③ 被判正常）", 2, lambda s, t: s <= t)]:
    both, only_p, only_e, neither, pa, pe = [], [], [], [], [], []
    for d in r:
        tp = tau_at_miss(d["proto"][2], TARGET_MISS); te = tau_at_miss(d["energy"][2], TARGET_MISS)
        ep_ = is_err(d["proto"][pile], tp); ee_ = is_err(d["energy"][pile], te)
        both.append((ep_ & ee_).mean()); only_p.append((ep_ & ~ee_).mean())
        only_e.append((~ep_ & ee_).mean()); neither.append((~ep_ & ~ee_).mean())
        pa.append(ep_.mean()); pe.append(ee_.mean())
    print(f"\n  ── {errname} ──")
    print(f"  {'':<20}{'energy 判對':>14}{'energy 錯':>12}")
    print(f"  {'原型判對':<18}{m(neither):>14.4f}{m(only_e):>12.4f}  ← ★只有 energy 錯")
    print(f"  {'原型錯':<19}{m(only_p):>14.4f}{m(both):>12.4f}")
    print(f"  {'':<19}  ↑★只有原型錯")
    ind = m(pa) * m(pe); ub = min(m(pa), m(pe))
    print(f"  總錯誤率：原型 {m(pa):.4f}　energy {m(pe):.4f}")
    print(f"  重疊（都錯）實測 {m(both):.4f}　｜ 獨立時期望 {ind:.4f}　｜ 完全相同時上界 {ub:.4f}")
    pos = (m(both) - ind) / max(ub - ind, 1e-9)
    print(f"  ⇒ 落在 獨立(0) → 完全相同(1) 之間的 **{pos:.2f}**")
    print(f"  ⇒ 獨佔錯誤：只有原型錯 {m(only_p):.4f}／只有 energy 錯 {m(only_e):.4f}")

print("\n" + "=" * W)
print("★ C：融合檢驗（決定性）　⚠️ 診斷用，不是方法（用到 energy 就撞 A4）")
print("=" * W)
print(f"  {'w(原型權重)':<14}{'部署 AUROC':>13}{'vs energy 0.8380':>20}")
best = (None, -1)
for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    v = []
    for d in r:
        n2, n3 = len(d["proto"][1]), len(d["proto"][2])
        fp = pct(np.r_[d["proto"][1], d["proto"][2]]); fe = pct(np.r_[d["energy"][1], d["energy"][2]])
        f = w * fp + (1 - w) * fe
        v.append(roc_auc_score([0]*n2+[1]*n3, f))
    mv = m(v)
    if mv > best[1]: best = (w, mv)
    tag2 = "  ← w=0 應等於 energy 單獨（自檢）" if w == 0 else ("  ← w=1 應等於原型單獨（自檢）" if w == 1 else "")
    print(f"  {w:<14.1f}{mv:>13.4f}{mv-0.8380:>+20.4f}{tag2}")
print(f"\n  ★ 最佳 w={best[0]}　部署 {best[1]:.4f}　vs energy 0.8380 ⇒ {best[1]-0.8380:+.4f}")
print("  判準：融合 > 0.8380 ⇒ 原型握有 energy 沒有的獨立資訊 ⇒ 路線(b) 值得走")
print("        融合 ≈ w=0 那一列 ⇒ 原型資訊是 energy 的子集 ⇒ 路線(b) 收掉")

print("\n" + "=" * W)
print("★ D：跨模型 固定放行率下的誤拒率（新目標的配套判準，首次量測）")
print("=" * W)
print(f"  {'固定放行率':<12}" + "".join(f"{t:>22}" for t in
      ["1a-fix 原型讀出", "1a-fix energy", "λ=0 energy(baseline)"]))
for tgt in [0.20, 0.30, 0.3152, 0.3391, 0.3572]:
    row = f"  {tgt:<12.4f}"
    for tag, nm in [("1a-fix", "proto"), ("1a-fix", "energy"), ("λ=0", "energy")]:
        v = [float((d[nm][1] > tau_at_miss(d[nm][2], tgt)).mean()) for d in ACC[tag]]
        row += f"{m(v):>22.4f}"
    print(row)
print("\n  ⇒ 同一列＝同一個操作點（放走一樣多的 person），此時誤拒率才可跨模型比")
