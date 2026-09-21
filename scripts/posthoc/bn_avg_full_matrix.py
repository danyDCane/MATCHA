"""BN running 統計量平均之後的完整矩陣：特徵端已統一，只剩原型還不同。

dany 2026-08-19 要求：
  (1) 原本 sketch ↔ art/photo 的巨大落差有沒有縮小
  (2) BN 平均後 9 個模型幾乎完全一樣（conv 7.7e-5、BN 仿射 6.8e-5、投影層 2.0e-4）
      ⇒ 表格應變成「統一的一個模型 × 不同風格輸入 × 不同節點的類別原型」
  (3) 由此判斷：各節點的類別原型到底需不需要調整

同時輸出「原樣」與「BN 平均」兩份，逐格對照。
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import torch, util
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers

PACS = ["art_painting", "cartoon", "photo", "sketch"]
DESC = os.environ.get("RUN_DESC",
    "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix")
CK = "exp_result_" + DESC
leave = "cartoon"; UNK = 6; N = 9; DEG = 57.29577951308232
avail = [d for d in PACS if d != leave]; per = N // len(avail)
OWN = [avail[min(i // per, len(avail) - 1)] for i in range(N)]
SH = {"art_painting": "art", "photo": "photo", "sketch": "sketch", "cartoon": "cartoon"}
DOMS = avail + [leave]
ld = {d: TD.load_pacs_test_data("../datasets/", d, 64, 4)[0] for d in DOMS}

# ── 合併變異數版的平均 BN ──
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
    Z, Y = [], []
    for b in loader:
        x, y, _ = util.unpack_batch(b)
        h = bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(x.to("cuda")))))
        h = bb.backbone.layer1(h); h = bb.backbone.layer2(h); h = bb.backbone.layer3(h)
        _, v = bb.forward_from_layer3(h)
        Z.append(bb.project(v).cpu().numpy()); Y.append(np.asarray(y).flatten())
    Z, Y = np.concatenate(Z), np.concatenate(Y)
    m = Y != UNK
    Z = Z[m]; Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return Z, Y[m].astype(int)


def ang(Z, Y, C):
    return float(np.arccos(np.clip((Z @ C.T)[np.arange(len(Y)), Y], -1 + 1e-7, 1 - 1e-7)).mean() * DEG)


RES = {}
for tag, avg in [("原樣", None), ("BN平均", AVG)]:
    PROTO, FEAT = [], []
    for i in range(N):
        bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_final.pth"), 6, "cuda")
        if avg is not None:
            sd = bb.state_dict()
            for k, v in avg.items():
                sd[k].copy_(v.to(sd[k].device).to(sd[k].dtype))
        C = class_centers(bb.prototypes, bb.proto_count).cpu().numpy()
        PROTO.append(C / np.linalg.norm(C, axis=1, keepdims=True))
        FEAT.append({d: collect(bb, ld[d]) for d in DOMS})
        del bb
    A = np.zeros((N, N, len(DOMS)))
    for i in range(N):
        for k, d in enumerate(DOMS):
            Z, Y = FEAT[i][d]
            for j in range(N):
                A[i, j, k] = ang(Z, Y, PROTO[j])
    PD = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            PD[i, j] = np.degrees(np.arccos(np.clip((PROTO[i] * PROTO[j]).sum(1), -1, 1))).mean()
    RES[tag] = (A, PD)
    print(f"  【{tag}】完成", flush=True)
    np.save(os.path.join(os.path.dirname(__file__), "..", "..", "logs", "prototype_probe",
                         f"0819_bnavg_matrix_{tag}.npy"), A)

m = lambda x: float(np.mean(x))
di = {d: k for k, d in enumerate(DOMS)}
home = {d: [i for i in range(N) if OWN[i] == d] for d in avail}
W = 92

print("\n" + "=" * W); print("§0 自檢"); print("=" * W)
A0 = RES["原樣"][0]
print(f"  原樣 ①={m([A0[i,i,di[OWN[i]]] for i in range(N)]):.2f}°（既有 29.63）"
      f"  ★={m([A0[i,i,di[d]] for i in range(N) for d in avail if d!=OWN[i]]):.2f}°（44.28）"
      f"  ②={m([A0[i,i,di[leave]] for i in range(N)]):.2f}°（53.57）")
A1 = RES["BN平均"][0]
print(f"  平均 ①={m([A1[i,i,di[OWN[i]]] for i in range(N)]):.2f}°（前次 34.85）"
      f"  ★={m([A1[i,i,di[d]] for i in range(N) for d in avail if d!=OWN[i]]):.2f}°（36.37）"
      f"  ②={m([A1[i,i,di[leave]] for i in range(N)]):.2f}°（50.34）")

print("\n" + "=" * W)
print("★★ 特徵端還剩多少差異？（同一批圖、同一組原型，只換誰的模型去算）")
print("=" * W)
for tag in ["原樣", "BN平均"]:
    A = RES[tag][0]
    sp = [float(np.ptp([A[i, j, k] for i in range(N)])) for j in range(N) for k in range(len(DOMS))]
    print(f"  {tag:<8} 跨 9 個模型的角度全距：平均 {np.mean(sp):>6.2f}°   最大 {np.max(sp):>6.2f}°")
print("  ⇒ BN 平均後若全距趨近 0，代表特徵端已完全統一、唯一剩下的變因是原型")

for tag in ["原樣", "BN平均"]:
    A, PD = RES[tag]
    print("\n" + "=" * W)
    print(f"【{tag}】風格 × 原型主人（特徵端取 9 個模型的平均，因為它們已幾乎相同）")
    print("=" * W)
    print(f"{'圖片的風格':<12}" + "".join(f"{'原型='+SH[d][:6]:>15}" for d in avail))
    for dd in DOMS:
        k = di[dd]; row = f"{SH[dd]:<14}"
        for d2 in avail:
            js = home[d2]
            v = m([A[i, j, k] for i in range(N) for j in js])
            mark = " ←自己" if dd == d2 else "     "
            row += f"{v:>13.2f}°{mark}"
        print(row)
    print(f"\n  原型漂移：同風格 {m([PD[i,j] for i in range(N) for j in range(N) if i!=j and OWN[i]==OWN[j]]):.2f}°"
          f"   跨風格 {m([PD[i,j] for i in range(N) for j in range(N) if OWN[i]!=OWN[j]]):.2f}°")

print("\n" + "=" * W)
print("★ sketch 的孤立程度有沒有改善（超出該風格主人多少度）")
print("=" * W)
print(f"{'風格對':<26}{'原樣':>12}{'BN平均':>12}{'改善':>11}")
for dd in avail:
    k = di[dd]
    for d2 in avail:
        if d2 == dd: continue
        r = []
        for tag in ["原樣", "BN平均"]:
            A = RES[tag][0]
            cross = m([A[i, j, k] for i in range(N) for j in home[d2]])
            hm = m([A[i, j, k] for i in home[dd] for j in home[dd]])
            r.append(cross - hm)
        print(f"  {SH[dd]+' 的圖 → '+SH[d2]+' 節點':<24}{r[0]:>+11.2f}°{r[1]:>+11.2f}°{r[1]-r[0]:>+10.2f}°")
k = di[leave]
r = []
for tag in ["原樣", "BN平均"]:
    A = RES[tag][0]
    r.append(m([A[i, j, k] for i in range(N) for j in range(N)]) -
             m([A[i, i, di[OWN[i]]] for i in range(N)]))
print(f"  {'cartoon 的圖（無主人）':<24}{r[0]:>+11.2f}°{r[1]:>+11.2f}°{r[1]-r[0]:>+10.2f}°")

print("\n" + "=" * W)
print("★ 原型還需不需要調整？（BN 平均後，只換原型的效果）")
print("=" * W)
for tag in ["原樣", "BN平均"]:
    A = RES[tag][0]
    own_p = m([A[i, i, k] for i in range(N) for k in range(len(DOMS))])
    same_p = m([A[i, j, k] for i in range(N) for j in range(N) if OWN[j] == OWN[i] and j != i
                for k in range(len(DOMS))])
    diff_p = m([A[i, j, k] for i in range(N) for j in range(N) if OWN[j] != OWN[i]
                for k in range(len(DOMS))])
    best = m([min(A[i, j, k] for j in range(N)) for i in range(N) for k in range(len(DOMS))])
    print(f"  {tag:<8} 自己的原型 {own_p:>7.2f}°  同風格他節點 {same_p:>7.2f}°  "
          f"跨風格節點 {diff_p:>7.2f}°  oracle最佳 {best:>7.2f}°")
