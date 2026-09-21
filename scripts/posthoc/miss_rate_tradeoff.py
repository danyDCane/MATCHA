"""同一個門檻下，誤殺多少正常樣本（誤拒率）vs 放行多少 OOD（放行率）——完整的取捨表。

dany 2026-08-19 指出：這幾輪只報了誤拒率，沒報「門檻放寬之後放行了多少 person」。
量存在（`fullspectrum_probe.py` 的 `tpr_semantic_at_src95`＝正確拒絕率，0730 §2 有報），
但 0818/0819/0819b 三份報告全都漏報。本腳本補齊。

定義（全部用同一個門檻 τ = 該節點來源域分數的 95% 分位）：
  誤拒率 = ②（cartoon 已知類別）被判成異常的比例   ← 越低越好
  放行率 = ③（cartoon 的 person）被判成正常的比例   ← 越低越好，＝ 1 − 正確拒絕率
兩者是同一個門檻的兩面：門檻放寬 ⇒ 誤拒率降、放行率升。

同時對 0819 定錨實驗的甲／乙／丙做同樣的量測（介入只作用在 ②、person 不動
⇒ 若放行率仍改變，代表是門檻經由 ① 或 ② 間接變動所致，要標明）。
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import torch, util
from sklearn.metrics import roc_auc_score
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers

PACS = ["art_painting", "cartoon", "photo", "sketch"]
RUNS = {"1a-fix": "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234",
        "λ=0": "v1_stage2_leave_cartoon_proto_lam0_m0.95w10_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234"}
RUNS["1a-fix"] += "_fix"
leave = "cartoon"; UNK = 6; N = 9; NC = 6; DEG = 57.29577951308232
avail = [d for d in PACS if d != leave]; per = N // len(avail)
OWN = [avail[min(i // per, len(avail) - 1)] for i in range(N)]
ld = {d: TD.load_pacs_test_data("../datasets/", d, 64, 4)[0] for d in PACS}
nrm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)


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
def collect(bb, loader):
    Z, Y, E = [], [], []
    for b in loader:
        x, y, _ = util.unpack_batch(b)
        h = bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(x.to("cuda")))))
        h = bb.backbone.layer1(h); h = bb.backbone.layer2(h); h = bb.backbone.layer3(h)
        lo, v = bb.forward_from_layer3(h)
        Z.append(bb.project(v).cpu().numpy()); Y.append(np.asarray(y).flatten())
        E.append((-torch.logsumexp(lo, 1)).cpu().numpy())
    return nrm(np.concatenate(Z)), np.concatenate(Y), np.concatenate(E)


def slerp(a, B, t):
    c = np.clip(B @ a, -1 + 1e-7, 1 - 1e-7); om = np.arccos(c)[:, None]; s = np.sin(om)
    out = (np.sin((1 - t) * om) * a[None, :] + np.sin(t * om) * B) / np.where(s < 1e-6, 1.0, s)
    out[(s < 1e-6).ravel()] = B[(s < 1e-6).ravel()]
    return nrm(out)


def rot_align(a, b):
    c = float(np.clip(a @ b, -1, 1)); v = b - c * a; nv = np.linalg.norm(v); D = len(a)
    if nv < 1e-8: return np.eye(D)
    v = v / nv; th = np.arccos(c)
    return (np.eye(D) + np.sin(th) * (np.outer(v, a) - np.outer(a, v))
            + (np.cos(th) - 1) * (np.outer(a, a) + np.outer(v, v)))


def sc(Z, C): return np.arccos(np.clip(Z @ C.T, -1 + 1e-7, 1 - 1e-7)).min(1) * DEG


def pair(s1, s2, s3):
    """回傳 (誤拒率, 放行率, 部署AUROC)。門檻＝①的95分位"""
    t = np.quantile(s1, 0.95)
    return float((s2 > t).mean()), float((s3 <= t).mean()), \
        roc_auc_score([0] * len(s2) + [1] * len(s3), np.r_[s2, s3])


OUT = {}
for tag, DESC in RUNS.items():
    CK = "exp_result_" + DESC
    AVG = bn_avg(CK, DESC)
    for bn in ["原樣", "BN平均B"]:
        rec = {}
        for i in range(N):
            bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_final.pth"), 6, "cuda")
            if bn == "BN平均B":
                sd = bb.state_dict()
                for k, v in AVG.items(): sd[k].copy_(v.to(sd[k].device).to(sd[k].dtype))
            C = nrm(class_centers(bb.prototypes, bb.proto_count).cpu().numpy())
            Z1, Y1, E1 = collect(bb, ld[OWN[i]]); Z2, Y2, E2 = collect(bb, ld[leave])
            del bb
            m1, mk, mu_ = Y1 != UNK, Y2 != UNK, Y2 == UNK
            Zk, Yk, Zu = Z2[mk], Y2[mk].astype(int), Z2[mu_]
            for nm, v in [("原型", pair(sc(Z1[m1], C), sc(Zk, C), sc(Zu, C))),
                          ("energy", pair(E1[m1], E2[mk], E2[mu_]))]:
                rec.setdefault(nm, []).append(v)
            if tag == "1a-fix" and bn == "BN平均B":       # 定錨介入的取捨
                MU = {c: nrm(Zk[Yk == c].mean(0)) for c in range(NC)}
                ROT = {c: rot_align(MU[c], C[c]) for c in range(NC)}
                for nm, al, ro in [("甲 收緊 α=.765", 0.765, False), ("乙 搬位置", 1.0, True),
                                   ("丙 兩者 α=.765", 0.765, True)]:
                    Zn = Zk.copy()
                    for c in range(NC):
                        m = Yk == c
                        if not m.sum(): continue
                        Wc = Zk[m]
                        if al < 1.0: Wc = slerp(MU[c], Wc, al)
                        if ro: Wc = nrm(Wc @ ROT[c].T)
                        Zn[m] = Wc
                    rec.setdefault(nm, []).append(pair(sc(Z1[m1], C), sc(Zn, C), sc(Zu, C)))
        OUT[(tag, bn)] = {k: np.mean(v, 0) for k, v in rec.items()}
        print(f"  {tag} / {bn} 完成", flush=True)

W = 92
print("\n" + "=" * W); print("§0 自檢"); print("=" * W)
a = OUT[("1a-fix", "原樣")]["原型"]
print(f"  1a-fix 原樣 原型：誤拒 {a[0]:.4f}（0818 報 0.4231）  部署 {a[2]:.4f}（0.7956）")
b = OUT[("λ=0", "原樣")]["energy"]
print(f"  λ=0   原樣 energy：誤拒 {b[0]:.4f}（靶 0.3656）  部署 {b[2]:.4f}（0.8234）")
print("  ⚠️ 放行率＝1 − 正確拒絕率；0730 §2 報過『正確拒絕』欄，但 0818/0819/0819b 三份都漏報")

print("\n" + "=" * W)
print("★ 同一個門檻的兩面：誤殺正常樣本 vs 放行 OOD")
print("=" * W)
print(f"  {'模型':<9}{'BN':<10}{'讀出':<9}{'誤拒率↓':>10}{'放行率↓':>10}{'兩者相加':>11}{'部署AUROC↑':>12}")
for tag in RUNS:
    for bn in ["原樣", "BN平均B"]:
        for nm in ["原型", "energy"]:
            f, mi, d = OUT[(tag, bn)][nm]
            print(f"  {tag:<9}{bn:<10}{nm:<9}{f:>10.4f}{mi:>10.4f}{f+mi:>11.4f}{d:>12.4f}")

print("\n" + "=" * W)
print("★ BN 同步的取捨：誤拒率降了多少、放行率漲了多少")
print("=" * W)
for tag in RUNS:
    for nm in ["原型", "energy"]:
        o = OUT[(tag, "原樣")][nm]; n = OUT[(tag, "BN平均B")][nm]
        print(f"  {tag:<9}{nm:<9}誤拒 {o[0]:.4f}→{n[0]:.4f} ({n[0]-o[0]:+.4f})   "
              f"放行 {o[1]:.4f}→{n[1]:.4f} ({n[1]-o[1]:+.4f})   "
              f"相加 {o[0]+o[1]:.4f}→{n[0]+n[1]:.4f} ({(n[0]+n[1])-(o[0]+o[1]):+.4f})")

print("\n" + "=" * W)
print("★ 定錨介入（1a-fix ＋ BN平均B）的取捨　※介入只作用在②、person 與①都沒動")
print("=" * W)
r = OUT[("1a-fix", "BN平均B")]
print(f"  {'處理':<18}{'誤拒率↓':>10}{'放行率↓':>10}{'兩者相加':>11}{'部署AUROC↑':>12}")
for nm in ["原型", "甲 收緊 α=.765", "乙 搬位置", "丙 兩者 α=.765", "energy"]:
    if nm not in r: continue
    f, mi, d = r[nm]
    lab = "未介入（原型）" if nm == "原型" else ("（參照）energy" if nm == "energy" else nm)
    print(f"  {lab:<17}{f:>10.4f}{mi:>10.4f}{f+mi:>11.4f}{d:>12.4f}")
print("\n  ⇒ 介入沒有碰 ① 也沒有碰 person ⇒ 門檻不變、放行率理應完全不變（可當自檢）")
