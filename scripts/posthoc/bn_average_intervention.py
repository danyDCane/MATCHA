"""訓練結束後把 9 節點的 BN running 統計量平均，看跨風格落差與 cartoon 會不會改善。

動機（2026-08-19 實測）：9 個節點的 conv 權重分歧 7.7e-5、BN 仿射 6.8e-5（＝已達共識），
但 BN running 統計量分歧 0.131（大 1700 倍），且訓練中單調上升（ep1 0.076 → ep200 0.319）
⇒ 9 個模型實質上是「同一個網路 + 九組不同的 BN 統計量」。

⚠️ 這【不是】用 target 資料作弊：平均只用到 9 個來源域節點自己的統計量，完全沒碰 cartoon。
   它是一個合法的聚合協定改動，但目前是 post-hoc（訓練後才做）⇒ 要進主線仍須搬進訓練（D1）。

兩種平均法：
  A 樸素平均      mean(running_mean), mean(running_var)
  B 合併變異數    mean(mean_i)、mean(var_i + mean_i²) − mean(mean_i)²
                 （＝把 9 份資料視為一個大批次時的正確變異數；樸素平均會低估）
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import torch, util
from sklearn.metrics import roc_auc_score
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers

PACS = ["art_painting", "cartoon", "photo", "sketch"]
DESC = os.environ.get("RUN_DESC", "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix")
CK = "exp_result_" + DESC
leave = "cartoon"; UNK = 6; N = 9; DEG = 57.29577951308232
avail = [d for d in PACS if d != leave]; per = N // len(avail)
OWN = [avail[min(i // per, len(avail) - 1)] for i in range(N)]
ld = {d: TD.load_pacs_test_data("../datasets/", d, 64, 4)[0] for d in PACS}

# ── 先算出兩種平均 BN ──
S = [torch.load(os.path.join(CK, f"{DESC}_node_{i}_final.pth"), map_location="cpu",
                weights_only=False)["backbone_state"] for i in range(N)]
BN_KEYS = [k for k in S[0] if k.endswith("running_mean") or k.endswith("running_var")]
AVG_A, AVG_B = {}, {}
for k in BN_KEYS:
    AVG_A[k] = torch.stack([S[i][k].float() for i in range(N)]).mean(0)
for k in BN_KEYS:
    if k.endswith("running_mean"):
        AVG_B[k] = AVG_A[k].clone()
    else:
        mk = k.replace("running_var", "running_mean")
        m_i = torch.stack([S[i][mk].float() for i in range(N)])
        v_i = torch.stack([S[i][k].float() for i in range(N)])
        AVG_B[k] = (v_i + m_i ** 2).mean(0) - m_i.mean(0) ** 2
del S


@torch.no_grad()
def evaluate(bb, C, loader, want_scores=False):
    A, Y, EN = [], [], []
    for b in loader:
        x, y, _ = util.unpack_batch(b)
        h = bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(x.to("cuda")))))
        h = bb.backbone.layer1(h); h = bb.backbone.layer2(h); h = bb.backbone.layer3(h)
        logits, vec = bb.forward_from_layer3(h)
        z = bb.project(vec).cpu().numpy(); z = z / np.linalg.norm(z, axis=1, keepdims=True)
        A.append(np.arccos(np.clip(z @ C.T, -1 + 1e-7, 1 - 1e-7)) * DEG)
        Y.append(np.asarray(y).flatten())
        if want_scores: EN.append((-torch.logsumexp(logits, 1)).cpu().numpy())
    A = np.concatenate(A); Y = np.concatenate(Y)
    m = Y != UNK
    own = A[m][np.arange(m.sum()), Y[m].astype(int)]
    near = A.min(1)
    return own.mean(), (near[m], near[~m], np.concatenate(EN) if want_scores else None, m)


def four_axis(s1, s2, s3):
    """①來源域已知 / ②target已知 / ③target未知 ⇒ 畫風、部署 AUROC、誤拒率@src95"""
    sty = roc_auc_score([0] * len(s1) + [1] * len(s2), np.r_[s1, s2])
    dep = roc_auc_score([0] * len(s2) + [1] * len(s3), np.r_[s2, s3])
    fpr = float((s2 > np.quantile(s1, 0.95)).mean())
    return sty, dep, fpr


VARIANTS = {"原樣(各節點自己的 BN)": None, "A 樸素平均 BN": AVG_A, "B 合併變異數 BN": AVG_B}
R = {v: {"one": [], "star": [], "cart": [], "sty": [], "dep": [], "fpr": [],
         "sty_e": [], "dep_e": [], "fpr_e": []} for v in VARIANTS}

for i in range(N):
    for vname, avg in VARIANTS.items():
        bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_final.pth"), 6, "cuda")
        if avg is not None:
            sd = bb.state_dict()
            for k, v in avg.items():
                sd[k].copy_(v.to(sd[k].device).to(sd[k].dtype))
        C = class_centers(bb.prototypes, bb.proto_count).cpu().numpy()
        C = C / np.linalg.norm(C, axis=1, keepdims=True)

        a_own, (n1, _, e1, m1) = evaluate(bb, C, ld[OWN[i]], want_scores=True)
        a_oth = np.mean([evaluate(bb, C, ld[d])[0] for d in avail if d != OWN[i]])
        a_cart, (n2, n3, e2, m2) = evaluate(bb, C, ld[leave], want_scores=True)
        R[vname]["one"].append(a_own); R[vname]["star"].append(a_oth); R[vname]["cart"].append(a_cart)
        sty, dep, fpr = four_axis(n1, n2, n3)
        R[vname]["sty"].append(sty); R[vname]["dep"].append(dep); R[vname]["fpr"].append(fpr)
        sty, dep, fpr = four_axis(e1[m1], e2[m2], e2[~m2])
        R[vname]["sty_e"].append(sty); R[vname]["dep_e"].append(dep); R[vname]["fpr_e"].append(fpr)
        del bb
    print(f"  node_{i} ({OWN[i]}) 三種設定跑完", flush=True)

m = lambda x: float(np.mean(x))
W = 88
print("\n" + "=" * W)
print("§0 自檢")
print("=" * W)
b = R["原樣(各節點自己的 BN)"]
print(f"  RUN = {DESC}")
print(f"  原樣 ①={m(b['one']):.2f}°  ★={m(b['star']):.2f}°  ②={m(b['cart']):.2f}°")
print(f"  原樣 原型讀出 誤拒={m(b['fpr']):.4f}  部署={m(b['dep']):.4f}")
print(f"  原樣 energy   誤拒={m(b['fpr_e']):.4f}  部署={m(b['dep_e']):.4f}")
print(f"  （1a-fix 應為 ①29.63 ★44.28 ②53.57／原型 .4231/.7956／energy .3971/.8242）")
print(f"  （λ=0   應為 誤拒 msp .3534 energy .3656／部署 energy .8234）")
print("  ⚠️ BN 平均只用 9 個來源域節點自己的統計量，完全沒碰 cartoon ⇒ 不是用 target 作弊")

print("\n" + "=" * W)
print("【角度】三級階梯在三種 BN 設定下")
print("=" * W)
print(f"{'BN 設定':<24}{'①本地':>10}{'★其他來源':>12}{'②cartoon':>12}{'①→★':>10}{'①→②':>10}")
for v in VARIANTS:
    r = R[v]
    print(f"{v:<22}{m(r['one']):>10.2f}°{m(r['star']):>11.2f}°{m(r['cart']):>11.2f}°"
          f"{m(r['star'])-m(r['one']):>9.2f}°{m(r['cart'])-m(r['one']):>9.2f}°")

print("\n" + "=" * W)
print("【四軸】靶＝λ=0 的 msp 誤拒 0.3534 / energy 部署 0.8234；目標誤拒 ≤0.2934")
print("=" * W)
for tag, ks in [("原型角距離讀出", ("sty", "dep", "fpr")), ("energy 讀出", ("sty_e", "dep_e", "fpr_e"))]:
    print(f"\n  ── {tag} ──")
    print(f"  {'BN 設定':<24}{'畫風AUROC':>12}{'部署AUROC':>12}{'誤拒率':>11}")
    base = None
    for v in VARIANTS:
        r = R[v]; row = (m(r[ks[0]]), m(r[ks[1]]), m(r[ks[2]]))
        d = "" if base is None else f"   Δ誤拒 {row[2]-base[2]:+.4f}  Δ部署 {row[1]-base[1]:+.4f}"
        if base is None: base = row
        print(f"  {v:<22}{row[0]:>12.4f}{row[1]:>12.4f}{row[2]:>11.4f}{d}")
print("\n★ 判準：誤拒率下降【且】部署 AUROC 不下降 ⇒ 真改善（0815 §4.3 守門員規則）")
