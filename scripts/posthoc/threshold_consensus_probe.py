"""兩個問題（dany 2026-08-19）：

Q1 把操作點固定住之後，BN 同步到底有沒有【真的】改善？
   誤拒率@src95 這個規則不是固定操作點——模型一變門檻就跟著移。
   改成【固定放行率】再比誤拒率，才是同一個操作點的比較。

Q2 跨節點門檻共識，能不能讓誤拒率下降而放行率不要漲太多？
   理論理由：BN 同步後 9 個節點的 ②vs③ 曲線幾乎完全相同（energy 部署 0.8376–0.8381），
   差別只在「各自坐在曲線上的哪一點」。ROC 曲線是凹的 ⇒ 一堆分散點的平均會落在曲線【下方】，
   把它們合併到曲線上的單一點，理應同時改善或至少不劣（Jensen）。
   ⚠️ 但這對 energy 一樣成立 ⇒ 必須同時測 energy，誠實比較。

三種門檻規則：
   L 各節點本地 q95（現況）
   C1 把 9 節點的來源域分數【合併】後取 q95（單一全域門檻）
   C2 把 9 個本地 q95 【平均】（只需交換一個純量，通訊成本極低）
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import torch, util
from sklearn.metrics import roc_auc_score
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers

PACS = ["art_painting", "cartoon", "photo", "sketch"]
DESC = "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"
CK = "exp_result_" + DESC
leave = "cartoon"; UNK = 6; N = 9; DEG = 57.29577951308232
avail = [d for d in PACS if d != leave]; per = N // len(avail)
OWN = [avail[min(i // per, len(avail) - 1)] for i in range(N)]
SH = {"art_painting": "art", "photo": "photo", "sketch": "sketch"}
ld = {d: TD.load_pacs_test_data("../datasets/", d, 64, 4)[0] for d in PACS}
nrm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)

S = [torch.load(os.path.join(CK, f"{DESC}_node_{i}_final.pth"), map_location="cpu",
                weights_only=False)["backbone_state"] for i in range(N)]
BK = [k for k in S[0] if k.endswith("running_mean") or k.endswith("running_var")]
AVG = {}
for k in BK:
    if k.endswith("running_mean"):
        AVG[k] = torch.stack([S[i][k].float() for i in range(N)]).mean(0)
for k in BK:
    if k.endswith("running_var"):
        mk = k.replace("running_var", "running_mean")
        mi = torch.stack([S[i][mk].float() for i in range(N)])
        vi = torch.stack([S[i][k].float() for i in range(N)])
        AVG[k] = (vi + mi ** 2).mean(0) - mi.mean(0) ** 2
del S


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


def sc(Z, C): return np.arccos(np.clip(Z @ C.T, -1 + 1e-7, 1 - 1e-7)).min(1) * DEG


# 收集所有節點的三堆分數
DATA = {}
for bn in ["原樣", "BN平均B"]:
    per_node = []
    for i in range(N):
        bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_final.pth"), 6, "cuda")
        if bn == "BN平均B":
            sd = bb.state_dict()
            for k, v in AVG.items(): sd[k].copy_(v.to(sd[k].device).to(sd[k].dtype))
        C = nrm(class_centers(bb.prototypes, bb.proto_count).cpu().numpy())
        Z1, Y1, E1 = collect(bb, ld[OWN[i]]); Z2, Y2, E2 = collect(bb, ld[leave])
        del bb
        m1, mk, mu_ = Y1 != UNK, Y2 != UNK, Y2 == UNK
        per_node.append({"proto": (sc(Z1[m1], C), sc(Z2[mk], C), sc(Z2[mu_], C)),
                         "energy": (E1[m1], E2[mk], E2[mu_]), "grp": SH[OWN[i]]})
        print(f"  [{bn}] node_{i} 完成", flush=True)
    DATA[bn] = per_node

W = 90
print("\n" + "=" * W); print("§0 自檢"); print("=" * W)
for bn in ["原樣", "BN平均B"]:
    for nm in ["proto", "energy"]:
        f = np.mean([float((d[nm][1] > np.quantile(d[nm][0], .95)).mean()) for d in DATA[bn]])
        mi = np.mean([float((d[nm][2] <= np.quantile(d[nm][0], .95)).mean()) for d in DATA[bn]])
        dep = np.mean([roc_auc_score([0]*len(d[nm][1])+[1]*len(d[nm][2]), np.r_[d[nm][1], d[nm][2]])
                       for d in DATA[bn]])
        print(f"  {bn:<9}{nm:<8}誤拒 {f:.4f}  放行 {mi:.4f}  部署 {dep:.4f}")
print("  （對照：1a-fix 原樣 原型 .4231/.1457/.7956；BN平均 原型 .2370/.3572/.8145）")

print("\n" + "=" * W)
print("★ Q1：固定放行率之後再比誤拒率（＝同一個操作點）")
print("=" * W)
for nm, lab in [("proto", "原型讀出"), ("energy", "energy")]:
    print(f"\n  ── {lab} ──")
    print(f"  {'固定放行率':<12}" + "".join(f"{t:>14}" for t in ["原樣的誤拒率", "BN平均的誤拒率", "改善"]))
    for tgt in [0.10, 0.1457, 0.20, 0.30, 0.3572]:
        r = []
        for bn in ["原樣", "BN平均B"]:
            v = []
            for d in DATA[bn]:
                s1, s2, s3 = d[nm]
                t = np.quantile(s3, tgt)          # 讓放行率剛好 = tgt 的門檻
                v.append(float((s2 > t).mean()))
            r.append(np.mean(v))
        print(f"  {tgt:<12.4f}{r[0]:>14.4f}{r[1]:>14.4f}{r[1]-r[0]:>+14.4f}")

print("\n" + "=" * W)
print("★ Q2：跨節點門檻共識（只在 BN 平均後測——此時 9 個節點的曲線幾乎相同）")
print("=" * W)
for nm, lab in [("proto", "原型讀出"), ("energy", "energy")]:
    d9 = DATA["BN平均B"]
    loc = [np.quantile(d[nm][0], .95) for d in d9]
    pooled = float(np.quantile(np.concatenate([d[nm][0] for d in d9]), .95))
    meanq = float(np.mean(loc))
    print(f"\n  ── {lab} ──")
    print(f"  {'門檻規則':<24}{'誤拒率↓':>10}{'放行率↓':>10}{'相加':>10}{'節點間誤拒全距':>16}")
    for rule, ths in [("L 各節點本地 q95（現況）", loc),
                      ("C1 合併後取 q95", [pooled] * N),
                      ("C2 平均 9 個本地 q95", [meanq] * N)]:
        fs = [float((d[nm][1] > t).mean()) for d, t in zip(d9, ths)]
        ms = [float((d[nm][2] <= t).mean()) for d, t in zip(d9, ths)]
        print(f"  {rule:<22}{np.mean(fs):>10.4f}{np.mean(ms):>10.4f}"
              f"{np.mean(fs)+np.mean(ms):>10.4f}{max(fs)-min(fs):>16.4f}")
    print(f"    本地門檻值：{[round(x,2) for x in loc]}")
    print(f"    C1={pooled:.2f}   C2={meanq:.2f}")
    print(f"    逐節點誤拒率（本地規則）：{[round(float((d[nm][1]>t).mean()),3) for d,t in zip(d9,loc)]}")
    print(f"    逐節點誤拒率（C1 共識）：  {[round(float((d[nm][1]>pooled).mean()),3) for d in d9]}")
