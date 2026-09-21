"""①→★ 的 14.65° 是「原型的帳」還是「特徵的帳」？＋ 同風格不同節點有沒有偏移？

dany 2026-08-19 提的加法模型：
    「各節點同類別原型偏移 a」＋「本地風格到自己原型 b」＝「別節點的風格過來 a+b」？
本腳本把 9×9×4 的角度矩陣全部算出來，正面檢驗它。

量測（node i 的特徵 × node j 的原型 × 域 d）：
  ①    = ang[i][i][own_d(i)]                本地風格、自己原型
  ★    = ang[i][i][d≠own_d(i)]              別的來源域、自己原型
  A'   = ang[i][j][d]  其中 own_d(j)=d      ★ 但把原型換成「該域的主人」的  ⇒ 只換原型
  A''  = ang[j][i][d]  其中 own_d(j)=d      該域主人的特徵 × node i 的原型   ⇒ 只換特徵
  同風格跨節點 = ang[i][j][own_d(i)]，own_d(j)=own_d(i)、j≠i
另量原型漂移矩陣（同域節點對 vs 跨域節點對），在 1a-fix 上重測（0803 是無原型損失的 ckpt）。

全 post-hoc、零重訓、final ckpt。
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import torch, util
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers

PACS = ["art_painting", "cartoon", "photo", "sketch"]
DESC = "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"
CK = "exp_result_" + DESC
leave = "cartoon"; UNK = 6; N = 9; NCLS = 6
DEG = 57.29577951308232
avail = [d for d in PACS if d != leave]           # art, photo, sketch
per = N // len(avail)
OWN = [avail[min(i // per, len(avail) - 1)] for i in range(N)]
DOMS = avail + [leave]
ld = {d: TD.load_pacs_test_data("../datasets/", d, 64, 4)[0] for d in DOMS}


@torch.no_grad()
def collect(bb, loader):
    Z, Y = [], []
    for b in loader:
        d, y, _ = util.unpack_batch(b)
        h = bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(d.to("cuda")))))
        h = bb.backbone.layer1(h); h = bb.backbone.layer2(h); h = bb.backbone.layer3(h)
        _, vec = bb.forward_from_layer3(h)
        Z.append(bb.project(vec).cpu().numpy()); Y.append(np.asarray(y).flatten())
    Z, Y = np.concatenate(Z), np.concatenate(Y)
    m = Y != UNK
    return Z[m], Y[m].astype(int)


def ang_own(Z, Y, C):
    zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return float(np.arccos(np.clip((zn @ C.T)[np.arange(len(Y)), Y], -1 + 1e-7, 1 - 1e-7)).mean() * DEG)


# ── 一次前向，收齊 9 節點 × 4 域的特徵與 9 組原型 ──
PROTO, FEAT = [], []
for i in range(N):
    bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_final.pth"), 6, "cuda")
    C = class_centers(bb.prototypes, bb.proto_count).cpu().numpy()
    PROTO.append(C / np.linalg.norm(C, axis=1, keepdims=True))
    FEAT.append({d: collect(bb, ld[d]) for d in DOMS})
    del bb
    print(f"  node_{i} ({OWN[i]}) 收集完成", flush=True)

# ang[i][j][d] = node i 的特徵、node j 的原型
ANG = np.zeros((N, N, len(DOMS)))
for i in range(N):
    for di, d in enumerate(DOMS):
        Z, Y = FEAT[i][d]
        for j in range(N):
            ANG[i, j, di] = ang_own(Z, Y, PROTO[j])

# 原型漂移矩陣
PD = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        cos = (PROTO[i] * PROTO[j]).sum(1)
        PD[i, j] = np.arccos(np.clip(cos, -1 + 1e-7, 1 - 1e-7)).mean() * DEG

same_pairs = [(i, j) for i in range(N) for j in range(N) if i != j and OWN[i] == OWN[j]]
diff_pairs = [(i, j) for i in range(N) for j in range(N) if OWN[i] != OWN[j]]
di_of = {d: k for k, d in enumerate(DOMS)}
home = {d: [i for i in range(N) if OWN[i] == d] for d in avail}

m = lambda x: float(np.mean(x))
W = 80
print("\n" + "=" * W)
print("§0 自檢（與既有數字對帳）")
print("=" * W)
one = m([ANG[i, i, di_of[OWN[i]]] for i in range(N)])
star = m([ANG[i, i, di_of[d]] for i in range(N) for d in avail if d != OWN[i]])
cart = m([ANG[i, i, di_of[leave]] for i in range(N)])
print(f"  ① 本地來源域 = {one:.2f}°（0819 報 29.63°）")
print(f"  ★ 其他來源域 = {star:.2f}°（0819 報 44.28°）")
print(f"  ② cartoon    = {cart:.2f}°（0819 報 53.57°）")
print(f"  類間夾角（不同類別的原型之間）= "
      f"{m([np.arccos(np.clip(PROTO[i]@PROTO[i].T,-1+1e-7,1-1e-7))[~np.eye(NCLS,dtype=bool)].mean()*DEG for i in range(N)]):.2f}°"
      f"（0818 §3.8 報 101.48°）")

print("\n" + "=" * W)
print("★ 問題一：原型跨節點漂移（在 1a-fix 上重測；0803 是無原型損失的 ckpt）")
print("=" * W)
print(f"  同域節點對（同風格、不同資料子集）  = {m([PD[i,j] for i,j in same_pairs]):>7.2f}°"
      f"   （0803 無原型損失時 5.81–6.83°）")
print(f"  跨域節點對（不同風格）              = {m([PD[i,j] for i,j in diff_pairs]):>7.2f}°"
      f"   （0803 無原型損失時 15.81–17.19°）")
print(f"  比值 = {m([PD[i,j] for i,j in diff_pairs])/max(m([PD[i,j] for i,j in same_pairs]),1e-9):.2f}x")

print("\n" + "=" * W)
print("★ 問題二：同風格、不同節點——資料端與原型端各有多少偏移")
print("=" * W)
ss_self = m([ANG[i, i, di_of[OWN[i]]] for i in range(N)])
ss_cross = m([ANG[i, j, di_of[OWN[i]]] for i, j in same_pairs])
print(f"  同風格資料 × 自己的原型      = {ss_self:>7.2f}°")
print(f"  同風格資料 × 同風格他節點原型 = {ss_cross:>7.2f}°   （偏移 {ss_cross-ss_self:+.2f}°）")
print(f"  逐節點 ①（同一批來源域測試資料）：{[round(ANG[i,i,di_of[OWN[i]]],2) for i in range(N)]}")
print(f"    ⇒ 同風格三節點之間的全距 = "
      f"{max(ANG[i,i,di_of[OWN[i]]] for i in range(N))-min(ANG[i,i,di_of[OWN[i]]] for i in range(N)):.2f}°"
      f"（僅供參考：三節點測的是同一批 held-out 資料）")

print("\n" + "=" * W)
print("★ 問題三：①→★ 的 14.65° 拆成「原型的帳」與「特徵的帳」")
print("=" * W)
rows = []
for i in range(N):
    for d in avail:
        if d == OWN[i]: continue
        k = di_of[d]; js = home[d]
        rows.append(dict(
            star=ANG[i, i, k],                                  # 特徵 i × 原型 i
            swap_p=m([ANG[i, j, k] for j in js]),               # 特徵 i × 原型 j（只換原型）
            swap_f=m([ANG[j, i, k] for j in js]),               # 特徵 j × 原型 i（只換特徵）
            home=m([ANG[j, j, k] for j in js])))                # 特徵 j × 原型 j（該域的家）
S = {k: m([r[k] for r in rows]) for k in rows[0]}
gap = S['star'] - S['home']
print(f"  ★  特徵=node i、原型=node i（現況）      {S['star']:>7.2f}°")
print(f"  A' 只換原型（換成該域主人的原型）        {S['swap_p']:>7.2f}°   Δ={S['swap_p']-S['star']:+.2f}°")
print(f"  A''只換特徵（該域主人跑的特徵）          {S['swap_f']:>7.2f}°   Δ={S['swap_f']-S['star']:+.2f}°")
print(f"  B  該域的家（特徵與原型都是主人的）      {S['home']:>7.2f}°")
print("-" * W)
print(f"  要解釋的落差 ★−B = {gap:.2f}°")
print(f"    只換原型解釋 {(S['star']-S['swap_p'])/gap*100:>6.1f}%  ｜  只換特徵解釋 {(S['star']-S['swap_f'])/gap*100:>6.1f}%")
print(f"    兩者相加 = {((S['star']-S['swap_p'])+(S['star']-S['swap_f']))/gap*100:.1f}%"
      f"（>100% 表示兩個帳重疊、非可加）")

print("\n" + "=" * W)
print("★ 問題四：dany 的加法模型檢驗")
print("=" * W)
pd_diff = m([PD[i, j] for i, j in diff_pairs])
print(f"  模型：★ ≈ 「該域在自己家的角度 B」＋「跨域原型漂移」")
print(f"        = {S['home']:.2f}° + {pd_diff:.2f}° = {S['home']+pd_diff:.2f}°")
print(f"  實測 ★ = {S['star']:.2f}°   差 {S['star']-(S['home']+pd_diff):+.2f}°")
print(f"  ⚠️ 角度是球面上的距離、滿足三角不等式 ⇒ 相加只給【上界】，貼近上界才代表兩段近乎共線")

np.save(os.path.join(os.path.dirname(__file__), "..", "..", "logs", "prototype_probe",
                     "0819_angle_matrix_9x9x4.npy"), ANG)
print("\n（9×9×4 角度矩陣已存 logs/prototype_probe/0819_angle_matrix_9x9x4.npy）")
