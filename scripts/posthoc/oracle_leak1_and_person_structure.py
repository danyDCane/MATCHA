"""只修「洩漏一」的 oracle ＋ person 的方向結構（plan: dany 2026-08-26 指定）

★ 甲乙丙原本藏了兩個標籤洩漏，本輪【只修洩漏一】：
   洩漏一：知道「這張是②」⇒ 只對它動手、person 不碰      ← 【本輪修掉】改用預測類別、對②∪③全部套用
   洩漏二：知道「該搬多少/該收多緊」⇒ 修正量用②的標籤算  ← 【本輪保留】MU/ROT/α 全部沿用真實標籤那組
   理由（dany）：洩漏二的可實現版本已測過（BN 平均只找回 13.5%）；兩個一起修，負值會分不清是哪個造成的。

⇒ 本輪回答一個乾淨的問題：**就算修正量完美，光是「不知道哪張是 person」會吃掉多少？**
   判準（事前寫死）：**掉到 0.81 以下 ⇒ 0.9554 是海市蜃樓，映射路線收掉。**

兩種預測來源都跑：(a) fc 分類頭 argmax  (b) 投影空間最近類別中心
順手撈：person 405 張「兩個頭是否指向同一類」的分歧率，與②的分歧率對照（零成本、可能是免費檢測訊號）

person 重合度兩個定義都算：
   (i)  person 散開方向 vs 來源域散開方向        ← 與 0825 的 cartoon 0.8281 可並排
   (ii) person 位移方向 vs cartoon 搬家 7 維子空間 ← ★決定「尺規」治不治得了

★ 逐樣本分數與特徵全部落盤（npz），之後要換讀出不必重跑。
全程 BN 平均 B（TaskBoard §A 協定）。
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import torch, util
from sklearn.metrics import roc_auc_score
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers
from bn_common import bn_avg, apply_bn

PACS = ["art_painting", "cartoon", "photo", "sketch"]
DESC = os.environ.get("DESC", "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix")
CKPT = os.environ.get("CKPT_TAG", "final"); NODES = int(os.environ.get("NODES", "9"))
OUT = os.environ.get("OUT", "logs/prototype_probe/0826_oracle_leak1_dump.npz")
CK = "exp_result_" + DESC
leave = "cartoon"; UNK = 6; N = 9; NC = 6; DEG = 57.29577951308232
avail = [d for d in PACS if d != leave]; per = N // len(avail)
OWN = [avail[min(i // per, len(avail) - 1)] for i in range(N)]
ld = {d: TD.load_pacs_test_data("../datasets/", d, 64, 4)[0] for d in PACS}
nrm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)


@torch.no_grad()
def collect(bb, loader):
    """回傳 (128維投影, 標籤, logits[B,6])"""
    Z, Y, L = [], [], []
    for b in loader:
        x, y, _ = util.unpack_batch(b)
        h = bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(x.to("cuda")))))
        h = bb.backbone.layer1(h); h = bb.backbone.layer2(h); h = bb.backbone.layer3(h)
        lo, v = bb.forward_from_layer3(h)
        Z.append(bb.project(v).cpu().numpy()); Y.append(np.asarray(y).flatten()); L.append(lo.cpu().numpy())
    return nrm(np.concatenate(Z)), np.concatenate(Y).astype(int), np.concatenate(L)


def slerp(a, B, t):
    c = np.clip(B @ a, -1 + 1e-7, 1 - 1e-7); om = np.arccos(c)[:, None]; s = np.sin(om)
    small = (s < 1e-6).ravel()
    out = (np.sin((1 - t) * om) * a[None, :] + np.sin(t * om) * B) / np.where(s < 1e-6, 1.0, s)
    out[small] = B[small]; return nrm(out)


def rot_align(a, b):
    c = float(np.clip(a @ b, -1, 1)); v = b - c * a; nv = np.linalg.norm(v); D = len(a)
    if nv < 1e-8: return np.eye(D)
    v = v / nv; th = np.arccos(c)
    return (np.eye(D) + np.sin(th) * (np.outer(v, a) - np.outer(a, v))
            + (np.cos(th) - 1) * (np.outer(a, a) + np.outer(v, v)))


sc = lambda Z, C: np.arccos(np.clip(Z @ C.T, -1 + 1e-7, 1 - 1e-7)).min(1) * DEG
scat = lambda Z, Y: float(np.mean([np.arccos(np.clip(Z[Y == c] @ nrm(Z[Y == c].mean(0)), -1+1e-7, 1-1e-7)).mean() * DEG
                                   for c in range(NC) if (Y == c).sum() > 1]))
dims90 = lambda R: int(np.searchsorted(np.cumsum(np.linalg.svd(R, compute_uv=False)**2) /
                                       (np.linalg.svd(R, compute_uv=False)**2).sum(), 0.9) + 1)


def basis(R, k):
    return np.linalg.svd(R, full_matrices=False)[2][:k].T


def overlap(E1, E2):
    """兩個正交子空間的重合度（主夾角餘弦平方和 / 較小維度）：1=同一組方向、0=完全不同"""
    return float((np.linalg.svd(E1.T @ E2, compute_uv=False) ** 2).sum() / min(E1.shape[1], E2.shape[1]))


AVG = bn_avg(CK, DESC, N=N, ckpt_tag=CKPT)
ALPHAS = [1.0, 0.9, 0.8, 0.765, 0.7, 0.5]
A = {}
DUMP = {}
print(f"[cfg] {DESC[:55]}… ckpt={CKPT} nodes={NODES} BN平均B　只修洩漏一", flush=True)

for i in range(NODES):
    bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_{CKPT}.pth"), 6, "cuda")
    apply_bn(bb, AVG)
    C = nrm(class_centers(bb.prototypes, bb.proto_count).cpu().numpy())
    Zs, Ys, Ls = collect(bb, ld[OWN[i]])
    Zt, Yt, Lt = collect(bb, ld[leave])
    SRC = {d: collect(bb, ld[d]) for d in avail}
    del bb

    mk, mu_ = Yt != UNK, Yt == UNK
    Zk, Yk, Zu = Zt[mk], Yt[mk], Zt[mu_]
    ZA = np.r_[Zk, Zu]; LA = np.r_[Lt[mk], Lt[mu_]]
    is_unk = np.r_[np.zeros(len(Zk), bool), np.ones(len(Zu), bool)]
    lab01 = is_unk.astype(int)

    # 兩種預測來源
    P_fc = LA.argmax(1)
    P_ct = np.arccos(np.clip(ZA @ C.T, -1 + 1e-7, 1 - 1e-7)).argmin(1)
    A.setdefault("dis_known", []).append(float((P_fc[~is_unk] != P_ct[~is_unk]).mean()))
    A.setdefault("dis_unk", []).append(float((P_fc[is_unk] != P_ct[is_unk]).mean()))

    # 修正量：仍用②的真實標籤算（洩漏二保留）
    MU = {c: nrm(Zk[Yk == c].mean(0)) for c in range(NC)}
    ROT = {c: rot_align(MU[c], C[c]) for c in range(NC)}
    src_s, tgt_s = scat(Zs[Ys != UNK], Ys[Ys != UNK]), scat(Zk, Yk)
    A.setdefault("src_scat", []).append(src_s); A.setdefault("tgt_scat", []).append(tgt_s)
    A.setdefault("base", []).append(roc_auc_score(lab01, sc(ZA, C)))
    A.setdefault("energy", []).append(roc_auc_score(lab01, -np.log(np.exp(LA).sum(1))))

    for ptag, P in [("fc", P_fc), ("ct", P_ct)]:
        for name in ["甲 收緊", "乙 搬位置", "丙 兩者"]:
            for a in (ALPHAS if name != "乙 搬位置" else [1.0]):
                Zn = ZA.copy()
                for c in range(NC):
                    m = P == c
                    if not m.sum(): continue
                    W = ZA[m]
                    if name in ("甲 收緊", "丙 兩者") and a < 1.0: W = slerp(MU[c], W, a)
                    if name in ("乙 搬位置", "丙 兩者"): W = nrm(W @ ROT[c].T)
                    Zn[m] = W
                s = sc(Zn, C)
                A.setdefault((ptag, name, a), []).append(roc_auc_score(lab01, s))
                # ★ 分堆看介入把誰推到哪
                A.setdefault(("d2", ptag, name, a), []).append(float(s[~is_unk].mean()))
                A.setdefault(("d3", ptag, name, a), []).append(float(s[is_unk].mean()))

    # ---- person 方向結構 ----
    # (i) 散開方向重合度
    Rs = np.concatenate([Zs[Ys != UNK][Ys[Ys != UNK] == c] -
                         np.outer(Zs[Ys != UNK][Ys[Ys != UNK] == c] @ nrm(Zs[Ys != UNK][Ys[Ys != UNK] == c].mean(0)),
                                  nrm(Zs[Ys != UNK][Ys[Ys != UNK] == c].mean(0))) for c in range(NC)])
    up = nrm(Zu.mean(0)); Ru = Zu - np.outer(Zu @ up, up)
    ks, ku = dims90(Rs), dims90(Ru)
    A.setdefault("dim_src", []).append(ks); A.setdefault("dim_unk", []).append(ku)
    A.setdefault("ov_scatter", []).append(overlap(basis(Rs, ks), basis(Ru, ku)))

    # (ii) person 位移方向 vs cartoon 搬家子空間（逐 person-最近類別分組，最多 6 個向量）
    Cd = {d: np.stack([nrm(SRC[d][0][SRC[d][1] != UNK][SRC[d][1][SRC[d][1] != UNK] == c].mean(0)) for c in range(NC)])
          for d in avail}
    Dm = np.stack([Cd[avail[x]][c] - Cd[avail[y]][c]
                   for x in range(3) for y in range(x + 1, 3) for c in range(NC)])
    Estyle = basis(Dm, 7)
    near = np.arccos(np.clip(Zu @ C.T, -1 + 1e-7, 1 - 1e-7)).argmin(1)
    Dp = np.stack([nrm(Zu[near == c].mean(0)) - C[c] for c in range(NC) if (near == c).sum() >= 5])
    Dk = np.stack([nrm(Zk[Yk == c].mean(0)) - C[c] for c in range(NC)])       # ②的位移當對照
    A.setdefault("proj_person", []).append(float((((Dp @ Estyle) ** 2).sum(1) / (Dp ** 2).sum(1)).mean()))
    A.setdefault("proj_known", []).append(float((((Dk @ Estyle) ** 2).sum(1) / (Dk ** 2).sum(1)).mean()))

    DUMP[f"n{i}"] = dict(Z=ZA.astype(np.float32), logit=LA.astype(np.float32), unk=is_unk,
                         y=np.r_[Yk, np.full(len(Zu), UNK)], C=C.astype(np.float32),
                         Zsrc=Zs.astype(np.float32), Ysrc=Ys, Lsrc=Ls.astype(np.float32))
    print(f"  node{i}({OWN[i]}) done", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(OUT, **{f"{k}_{kk}": vv for k, d in DUMP.items() for kk, vv in d.items()})
m = lambda k: float(np.mean(A[k]))
astar = m("src_scat") / m("tgt_scat")
print("\n" + "=" * 96)
print(f"§0 自檢：α=1.0 的甲(fc) = {m(('fc','甲 收緊',1.0)):.4f}   未介入基準 = {m('base'):.4f}   "
      f"{'✅一致' if abs(m(('fc','甲 收緊',1.0))-m('base'))<1e-9 else '❌不一致'}")
print(f"        ①散布 {m('src_scat'):.2f}°  ②散布 {m('tgt_scat'):.2f}°  ⇒ α* = {astar:.3f}（0819 用 0.765）")
print(f"        energy 部署 AUROC = {m('energy'):.4f}（應 ≈0.8380）　落盤 → {OUT}")
print("=" * 96)
print("\n★★ 只修洩漏一的 oracle（修正量仍用真實標籤算；②③ 全部套用、用預測類別決定套哪個）")
print(f"{'預測來源':<8}{'介入':<12}{'α':>7}{'部署AUROC':>12}{'vs 0.8145':>12}{'②均值':>10}{'③均值':>10}{'均值差':>9}")
print("-" * 96)
for ptag, pn in [("fc", "fc頭"), ("ct", "最近中心")]:
    for name in ["甲 收緊", "乙 搬位置", "丙 兩者"]:
        for a in (ALPHAS if name != "乙 搬位置" else [1.0]):
            if a == 1.0 and name == "甲 收緊": continue
            v = m((ptag, name, a)); d2 = m(("d2", ptag, name, a)); d3 = m(("d3", ptag, name, a))
            star = " ←α*" if abs(a - 0.765) < 1e-9 else ""
            print(f"{pn:<8}{name:<12}{a:>7.3f}{v:>12.4f}{v-m('base'):>+12.4f}{d2:>10.2f}{d3:>10.2f}{d3-d2:>9.2f}{star}")
print("-" * 96)
print(f"{'未介入':<20}{'':>7}{m('base'):>12.4f}{0:>+12.4f}")
print(f"\n★ 兩個頭的分歧率（fc argmax vs 投影空間最近中心）")
print(f"   ② cartoon 已知類別 {m('dis_known')*100:.1f}%    ③ person {m('dis_unk')*100:.1f}%"
      f"    ⇒ 差 {(m('dis_unk')-m('dis_known'))*100:+.1f}pp")
print(f"\n★ person 方向結構")
print(f"   (i)  散開方向重合度：person vs 來源域 = {m('ov_scatter'):.4f}"
      f"   （對照：cartoon vs 來源域 = 0.8281）　維度 ①{m('dim_src'):.0f} / person {m('dim_unk'):.0f}")
print(f"   (ii) 位移落在 cartoon 搬家 7 維子空間的能量：")
print(f"        person {m('proj_person'):.4f}     ② cartoon 已知類別 {m('proj_known'):.4f}"
      f"     無資訊基準 7/128 = {7/128:.4f}")
print("=" * 96)
