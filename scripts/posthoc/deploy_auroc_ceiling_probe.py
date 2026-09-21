"""部署 AUROC 的定錨實驗：瓶頸是「已知類別散太開」還是「整團位置偏了」還是「person 本來就混在裡面」？

dany 2026-08-19 定案的方向判斷實驗。目的**不是**找方法，是**在三條路裡刪掉兩條**。

介入只作用在 ②（cartoon 的六個已知類別），③（person）全程一根寒毛都不碰：
  甲 收緊散布：把每個樣本沿著大圓往「自己那一類的平均方向」拉近，
               縮放比例 α 直接等於角度縮放（slerp）⇒ 新散布 = α × 舊散布
  乙 搬移位置：把每一類的樣本團【剛體旋轉】，使其平均方向對齊該類原型
               （旋轉保長保角，不扭曲團內幾何）
  丙 甲＋乙

判準（事前寫死）：
  甲就衝過 energy ⇒ 瓶頸是散開程度 ⇒ 核心＝緊緻度遷移（energy 學不走）
  甲沒用、乙才有用 ⇒ 瓶頸是位置 ⇒ ⚠️ 與誤拒率同核心、energy 會一起受惠
  甲乙全做仍追不上 ⇒ person 本來就混在已知類別裡 ⇒ 核心要換成主動製造未知空隙

⚠️ 全部是 oracle（用到 cartoon 的標籤與類別均值）⇒ 給的是天花板，不是方法。
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import torch, util
from sklearn.metrics import roc_auc_score
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers

PACS = ["art_painting", "cartoon", "photo", "sketch"]
# ★ 2026-08-21：可用環境變數換臂／換 epoch（不設＝1a-fix final，向後相容）
DESC = os.environ.get("DESC", "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix")
CKPT_TAG = os.environ.get("CKPT_TAG", "final")
CK = "exp_result_" + DESC
leave = "cartoon"; UNK = 6; N = 9; NC = 6; DEG = 57.29577951308232
avail = [d for d in PACS if d != leave]; per = N // len(avail)
OWN = [avail[min(i // per, len(avail) - 1)] for i in range(N)]
ld = {d: TD.load_pacs_test_data("../datasets/", d, 64, 4)[0] for d in PACS}

S = [torch.load(os.path.join(CK, f"{DESC}_node_{i}_{CKPT_TAG}.pth"), map_location="cpu",
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

nrm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)


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
    """a:[D] 單位向量；B:[n,D] 單位向量；沿大圓把 B 拉向 a，新夾角 = t × 舊夾角"""
    c = np.clip(B @ a, -1 + 1e-7, 1 - 1e-7)
    om = np.arccos(c)[:, None]
    s = np.sin(om)
    small = (s < 1e-6).ravel()
    out = (np.sin((1 - t) * om) * a[None, :] + np.sin(t * om) * B) / np.where(s < 1e-6, 1.0, s)
    out[small] = B[small]
    return nrm(out)


def rot_align(a, b):
    """回傳把單位向量 a 轉到單位向量 b 的旋轉矩陣（在 a,b 張成的平面內；其餘方向不動）"""
    c = float(np.clip(a @ b, -1, 1))
    v = b - c * a
    nv = np.linalg.norm(v)
    D = len(a)
    if nv < 1e-8:
        return np.eye(D)
    v = v / nv
    th = np.arccos(c)
    return (np.eye(D) + np.sin(th) * (np.outer(v, a) - np.outer(a, v))
            + (np.cos(th) - 1) * (np.outer(a, a) + np.outer(v, v)))


def scatter(Z, Y):
    """類內角度散布（樣本與自己類別平均方向的夾角），跨類別平均"""
    out = []
    for c in range(NC):
        m = Y == c
        if m.sum() < 2: continue
        mu = nrm(Z[m].mean(0))
        out.append(np.arccos(np.clip(Z[m] @ mu, -1 + 1e-7, 1 - 1e-7)).mean() * DEG)
    return float(np.mean(out))


def sc(Z, C):
    return np.arccos(np.clip(Z @ C.T, -1 + 1e-7, 1 - 1e-7)).min(1) * DEG


ALPHAS = [1.0, 0.9, 0.8, 0.7, 0.662, 0.5, 0.3, 0.0]
ROWS = {}
SRC_SCAT, TGT_SCAT, EN_DEP, BASE = [], [], [], []

for bn_tag, avg in [("原樣", None), ("BN平均B", AVG)]:
    acc = {}
    for i in range(N):
        bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_{CKPT_TAG}.pth"), 6, "cuda")
        if avg is not None:
            sd = bb.state_dict()
            for k, v in avg.items():
                sd[k].copy_(v.to(sd[k].device).to(sd[k].dtype))
        C = nrm(class_centers(bb.prototypes, bb.proto_count).cpu().numpy())
        Zs, Ys, _ = collect(bb, ld[OWN[i]])
        Zt, Yt, Et = collect(bb, ld[leave])
        del bb
        mk, mu_ = Yt != UNK, Yt == UNK                 # ②已知 / ③person
        Zk, Yk, Zu = Zt[mk], Yt[mk].astype(int), Zt[mu_]
        s_unk = sc(Zu, C)                              # ③ 全程不變

        acc.setdefault("src_scat", []).append(scatter(Zs[Ys != UNK], Ys[Ys != UNK].astype(int)))
        acc.setdefault("tgt_scat", []).append(scatter(Zk, Yk))
        acc.setdefault("en_dep", []).append(
            roc_auc_score([0]*mk.sum()+[1]*mu_.sum(), np.r_[Et[mk], Et[mu_]]))
        acc.setdefault("base_dep", []).append(
            roc_auc_score([0]*len(Zk)+[1]*len(Zu), np.r_[sc(Zk, C), s_unk]))
        # ★ 2026-08-24：三堆「到最近類別中心」的分位數 ⇒ 判斷「用已知樣本尾巴當 outlier 代理」可不可行
        acc.setdefault("s1_q", []).append(np.percentile(sc(Zs[Ys != UNK], C), [50, 75, 90, 95, 99]))
        acc.setdefault("s2_q", []).append(np.percentile(sc(Zk, C), [50, 75, 90, 95, 99]))
        acc.setdefault("s3_q", []).append(np.percentile(s_unk, [1, 5, 10, 25, 50]))
        acc.setdefault("s1_mean", []).append(sc(Zs[Ys != UNK], C).mean())
        acc.setdefault("s2_mean", []).append(sc(Zk, C).mean())
        acc.setdefault("s2_std", []).append(sc(Zk, C).std())
        acc.setdefault("s3_mean", []).append(s_unk.mean())
        acc.setdefault("s3_std", []).append(s_unk.std())
        # person 最近的是哪一類
        nearc = np.arccos(np.clip(Zu @ C.T, -1+1e-7, 1-1e-7)).argmin(1)
        acc.setdefault("unk_hist", []).append(np.bincount(nearc, minlength=NC) / len(nearc))

        MU = {c: nrm(Zk[Yk == c].mean(0)) for c in range(NC)}
        ROT = {c: rot_align(MU[c], C[c]) for c in range(NC)}
        for name in ["甲 收緊", "乙 搬位置", "丙 兩者"]:
            for a in (ALPHAS if name != "乙 搬位置" else [1.0]):
                Zn = Zk.copy()
                for c in range(NC):
                    m = Yk == c
                    if not m.sum(): continue
                    W = Zk[m]
                    if name in ("甲 收緊", "丙 兩者") and a < 1.0:
                        W = slerp(MU[c], W, a)
                    if name in ("乙 搬位置", "丙 兩者"):
                        W = nrm(W @ ROT[c].T)
                    Zn[m] = W
                key = (name, a)
                acc.setdefault(key, []).append(
                    roc_auc_score([0]*len(Zn)+[1]*len(Zu), np.r_[sc(Zn, C), s_unk]))
                acc.setdefault(("scat",) + key, []).append(scatter(Zn, Yk))
        print(f"  [{bn_tag}] node_{i} 完成", flush=True)
    ROWS[bn_tag] = acc

m = lambda x: float(np.mean(x))
W = 84
for bn_tag in ["原樣", "BN平均B"]:
    a = ROWS[bn_tag]
    print("\n" + "=" * W); print(f"§0 自檢（{bn_tag}）"); print("=" * W)
    print(f"  α=1.0 的甲 = {m(a[('甲 收緊',1.0)]):.4f}   未介入基準 = {m(a['base_dep']):.4f}"
          f"   {'✅一致' if abs(m(a[('甲 收緊',1.0)])-m(a['base_dep']))<1e-9 else '❌不一致'}")
    print(f"  類內散布：①來源域 {m(a['src_scat']):.2f}°   ②cartoon {m(a['tgt_scat']):.2f}°"
          f"   ⇒ 要收到來源域水準需 α={m(a['src_scat'])/m(a['tgt_scat']):.3f}")
    print(f"  分數分布：②{m(a['s2_mean']):.2f}±{m(a['s2_std']):.2f}°   "
          f"③person {m(a['s3_mean']):.2f}±{m(a['s3_std']):.2f}°   均值差 {m(a['s3_mean'])-m(a['s2_mean']):.2f}°")
    q1, q2, q3 = np.mean(a["s1_q"], 0), np.mean(a["s2_q"], 0), np.mean(a["s3_q"], 0)
    print(f"  ★ 分位數（到最近類別中心，度）")
    print(f"    ①來源域   50%={q1[0]:.1f}  75%={q1[1]:.1f}  90%={q1[2]:.1f}  95%={q1[3]:.1f}  99%={q1[4]:.1f}   (mean {m(a['s1_mean']):.1f})")
    print(f"    ②cartoon 50%={q2[0]:.1f}  75%={q2[1]:.1f}  90%={q2[2]:.1f}  95%={q2[3]:.1f}  99%={q2[4]:.1f}   (mean {m(a['s2_mean']):.1f})")
    print(f"    ③person   1%={q3[0]:.1f}   5%={q3[1]:.1f}  10%={q3[2]:.1f}  25%={q3[3]:.1f}  50%={q3[4]:.1f}   (mean {m(a['s3_mean']):.1f})")
    print(f"    ⇒ ①的95%分位 {q1[3]:.1f}° vs person的5%分位 {q3[1]:.1f}°："
          f"{'重疊' if q1[3] >= q3[1] else f'不重疊，差 {q3[1]-q1[3]:.1f}°'}")
    h = np.mean(a["unk_hist"], 0)
    print(f"  person 最近類別分布：{[f'{x:.0%}' for x in h]}（均勻＝17%）最集中 {h.max():.0%}")

    print("\n" + "=" * W); print(f"★ 定錨結果（{bn_tag}）　③person 全程不動"); print("=" * W)
    print(f"  {'介入':<14}{'α':>7}{'②散布':>10}{'部署 AUROC':>13}{'vs 未介入':>12}{'vs energy':>12}")
    base, en = m(a["base_dep"]), m(a["en_dep"])
    print(f"  {'未介入':<14}{'—':>7}{m(a['tgt_scat']):>9.2f}°{base:>13.4f}{'—':>12}{base-en:>+12.4f}")
    for name in ["甲 收緊", "乙 搬位置", "丙 兩者"]:
        for al in (ALPHAS if name != "乙 搬位置" else [1.0]):
            if name == "甲 收緊" and al == 1.0: continue
            v = m(a[(name, al)]); s = m(a[("scat", name, al)])
            print(f"  {name:<13}{al:>8.3f}{s:>9.2f}°{v:>13.4f}{v-base:>+12.4f}{v-en:>+12.4f}")
    print(f"  {'（參照）energy':<14}{'—':>7}{'—':>10}{en:>13.4f}{en-base:>+12.4f}{'—':>12}")
