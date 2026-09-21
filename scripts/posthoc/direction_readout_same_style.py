"""dany 2026-09-03 設計：把方向參考法（D1）拿去跑【同畫風】任務，判定「是不是卡在風格軸」。

【邏輯】D1 在 cartoon（②vs③、跨畫風）只有 0.6114。
  若 D1 在來源域（①vs①′、同畫風、只換類別）**也低** ⇒ 方向參考法本質上就弱，不是畫風的錯
  若 D1 在來源域**很高**            ⇒ 是畫風把它毀掉的 ⇒ 風格軸假設成立
對照：‖z⊥‖ 在同一組切分上是 0.9402（同畫風）／0.8281（跨畫風）。

【必須 held-out】ū₁ 由 ① 算，直接拿去評 ①vs①′ 會讓 ① 被自己貢獻的方向偏袒
  ⇒ 2-fold：ū₁ 用 ① 的一半算、評另一半 vs ①′，交換後平均（同 plan §4.4 畫風欄處理）。
⚠️ ① 是訓練看過的、①′／③ 沒看過 ⇒ 混記憶效應，是上界。
   但該 confound 對 D1 與 ‖z⊥‖ **相同**，故兩者的比較仍然有效。
"""
import numpy as np
from scipy.stats import rankdata

N, UNK, SEED, EPS = 9, 6, 2026, 1e-8
F = np.load("logs/prototype_probe/0826_features_full.npz")
nrm = lambda X: X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), EPS)
def auroc(pos, neg):
    a = np.concatenate([pos, neg]); r = rankdata(a, method="average")
    return float((r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
def ub(u):
    m = u.mean(0); return m/max(np.linalg.norm(m), EPS)

R = {k: [] for k in ["c1_1p","c2_3","c1p_3","c1_2","D1_same","D1_cross",
                     "zp_same","zp_cross","n1p"]}
for i in range(N):
    U, _ = np.linalg.qr(F[f"n{i}_C"].astype(np.float32).T)
    d = {}
    for sp in ("src","tgt"):
        Z = nrm(F[f"n{i}_{sp}_z"].astype(np.float32))
        zp = Z-(Z@U)@U.T; npr = np.linalg.norm(zp, axis=1)
        d[sp] = dict(u=zp/np.maximum(npr,EPS)[:,None], npr=npr, y=F[f"n{i}_{sp}_y"].astype(int))
    s, t = d["src"], d["tgt"]
    m1, m1p = s["y"]!=UNK, s["y"]==UNK
    m2, m3 = t["y"]!=UNK, t["y"]==UNK
    u1, u1p, u2, u3 = ub(s["u"][m1]), ub(s["u"][m1p]), ub(t["u"][m2]), ub(t["u"][m3])
    R["c1_1p"].append(u1@u1p); R["c2_3"].append(u2@u3)
    R["c1p_3"].append(u1p@u3); R["c1_2"].append(u1@u2)
    R["n1p"].append(int(m1p.sum()))
    # ‖z⊥‖ 對照
    R["zp_same"].append(auroc(s["npr"][m1p], s["npr"][m1]))
    R["zp_cross"].append(auroc(t["npr"][m3], t["npr"][m2]))
    # D1 跨畫風（②vs③）：ū₁ 用全 ①，②③ 皆未參與 ⇒ 無洩漏
    R["D1_cross"].append(auroc((-(t["u"]@u1))[m3], (-(t["u"]@u1))[m2]))
    # D1 同畫風（①vs①′）：必須 2-fold held-out
    idx = np.where(m1)[0]; rs = np.random.default_rng(SEED+i); rs.shuffle(idx)
    h = np.array_split(idx, 2); acc = []
    for k in range(2):
        ref = ub(s["u"][h[1-k]])
        acc.append(auroc(-(s["u"][m1p]@ref), -(s["u"][h[k]]@ref)))
    R["D1_same"].append(np.mean(acc))

M = {k: float(np.mean(v)) for k,v in R.items()}
W=94; ang = lambda c: np.degrees(np.arccos(np.clip(c,-1,1)))
print("="*W); print("① 平均方向的四個關係（9 節點平均）"); print("="*W)
for k,lbl in [("c1_1p","cos(ū₁, ū₁′)  同畫風、換類別   ★本次新增"),
              ("c2_3", "cos(ū₂, ū₃)   同畫風、換類別（cartoon）"),
              ("c1_2", "cos(ū₁, ū₂)   換畫風、同類別"),
              ("c1p_3","cos(ū₁′, ū₃)  換畫風、同為 person")]:
    print(f"  {lbl:<44}{M[k]:+.4f}   {ang(M[k]):6.1f}°")
print()
print("="*W); print("② ★ 判定：同畫風 vs 跨畫風，兩種讀出各拿多少"); print("="*W)
print(f"{'讀出':<24}{'同畫風(①vs①′)':>18}{'跨畫風(②vs③)':>18}{'差':>10}")
print("-"*W)
print(f"{'D1 方向參考（held-out）':<24}{M['D1_same']:18.4f}{M['D1_cross']:18.4f}{M['D1_same']-M['D1_cross']:+10.4f}")
print(f"{'‖z⊥‖ 長度（對照）':<24}{M['zp_same']:18.4f}{M['zp_cross']:18.4f}{M['zp_same']-M['zp_cross']:+10.4f}")
print("-"*W)
print(f"  ①′ 張數（9 節點平均）：{np.mean(R['n1p']):.0f}   逐節點：{R['n1p']}")
print(f"  D1 同畫風逐節點：" + " ".join(f"{x:.3f}" for x in R["D1_same"]))
print()
print("="*W)
ds, dc = M["D1_same"], M["D1_cross"]
if ds >= 0.85:
    v = "✅ 【風格軸假設成立】D1 在同畫風上很強，是畫風把它毀掉的"
elif ds <= 0.70:
    v = "⛔ 【風格軸假設證偽】D1 在同畫風上一樣弱 ⇒ 方向參考法本質上就弱，不是畫風的錯"
else:
    v = "⚠️ 【居中】畫風有份、但方向參考法本身也不強"
print(f"判定：D1 同畫風 {ds:.4f}（跨畫風 {dc:.4f}，Δ{ds-dc:+.4f}）")
print(f"      {v}")
print(f"      對照：同一組切分上 ‖z⊥‖ 拿到 {M['zp_same']:.4f} ⇒ 這個切分本身是可分的")
print("="*W)
