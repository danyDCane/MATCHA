"""參考方向到底是不是瓶頸？—— 換三種參考方向，逐節點比較。

【由來】0903 測到 D1 在同畫風上逐節點 0.97(photo)／0.66(art)／0.42(sketch)，
  換到 cartoon 全部塌成 0.61–0.68 ⇒ 診斷：**參考方向不跨畫風遷移**（cos(ū₁,ū₂)=−0.064）。
【本輪問】如果參考方向「對了」，能拿多少？三種取法：
  ref-src     ū₁（來源域已知）              ＝ 現行 D1，免標籤
  ref-tta     ū(全部 cartoon，②∪③ 混在一起) ＝ **免標籤**（部署時測試資料本來就混在一起）
  ref-oracle  ū₂（只有 cartoon 已知類別）    ＝ 需要 ②③ 的標籤 ⇒ 上界，非方法
順帶：同畫風版本（①vs①′）也跑一次當對照。
"""
import numpy as np
from scipy.stats import rankdata

N, UNK, SEED, EPS = 9, 6, 2026, 1e-8
DOM = ["art"]*3 + ["photo"]*3 + ["sketch"]*3
F = np.load("logs/prototype_probe/0826_features_full.npz")
nrm = lambda X: X/np.maximum(np.linalg.norm(X,axis=-1,keepdims=True), EPS)
def auroc(pos, neg):
    a = np.concatenate([pos,neg]); r = rankdata(a, method="average")
    return float((r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
def ub(u):
    m = u.mean(0); return m/max(np.linalg.norm(m), EPS)

rows = []
for i in range(N):
    U,_ = np.linalg.qr(F[f"n{i}_C"].astype(np.float32).T)
    d = {}
    for sp in ("src","tgt"):
        Z = nrm(F[f"n{i}_{sp}_z"].astype(np.float32))
        zp = Z-(Z@U)@U.T; npr = np.linalg.norm(zp,axis=1)
        d[sp] = dict(u=zp/np.maximum(npr,EPS)[:,None], npr=npr, y=F[f"n{i}_{sp}_y"].astype(int))
    s,t = d["src"], d["tgt"]
    m1,m1p = s["y"]!=UNK, s["y"]==UNK
    m2,m3  = t["y"]!=UNK, t["y"]==UNK
    u1, u2, u_all = ub(s["u"][m1]), ub(t["u"][m2]), ub(t["u"])
    A = lambda ref: auroc(-(t["u"][m3]@ref), -(t["u"][m2]@ref))
    # 同畫風對照（held-out ū₁）
    idx = np.where(m1)[0]; rs = np.random.default_rng(SEED+i); rs.shuffle(idx)
    h = np.array_split(idx,2)
    same = np.mean([auroc(-(s["u"][m1p]@ub(s["u"][h[1-k]])), -(s["u"][h[k]]@ub(s["u"][h[1-k]])))
                    for k in range(2)])
    rows.append(dict(node=i, dom=DOM[i], same=same, src=A(u1), tta=A(u_all), oracle=A(u2),
                     zp=auroc(t["npr"][m3], t["npr"][m2]),
                     c12=float(u1@u2), c_tta=float(u1@u_all), n3=int(m3.sum())))

W=104; print("="*W)
print("參考方向換三種，逐節點（②vs③ 部署 AUROC）")
print("="*W)
print(f"{'node':<5}{'來源畫風':<9}{'同畫風①vs①′':>13}{'ref-src(D1)':>13}{'ref-tta':>10}{'ref-oracle':>12}"
      f"{'‖z⊥‖':>9}{'cos(ū₁,ū₂)':>12}")
print("-"*W)
for r in rows:
    print(f"{r['node']:<5}{r['dom']:<9}{r['same']:13.4f}{r['src']:13.4f}{r['tta']:10.4f}"
          f"{r['oracle']:12.4f}{r['zp']:9.4f}{r['c12']:+12.4f}")
print("-"*W)
mn = lambda k: np.mean([r[k] for r in rows])
print(f"{'平均':<14}{mn('same'):13.4f}{mn('src'):13.4f}{mn('tta'):10.4f}{mn('oracle'):12.4f}"
      f"{mn('zp'):9.4f}{mn('c12'):+12.4f}")
print()
for dm in ["art","photo","sketch"]:
    sub=[r for r in rows if r["dom"]==dm]
    g=lambda k: np.mean([r[k] for r in sub])
    print(f"  {dm:<8} 同畫風 {g('same'):.4f} │ src {g('src'):.4f} → tta {g('tta'):.4f} "
          f"→ oracle {g('oracle'):.4f} │ ‖z⊥‖ {g('zp'):.4f}")
print()
print("="*W); print("判讀"); print("="*W)
print(f"  參考方向從 ū₁ 換成【測試域自己的方向】（免標籤）：{mn('src'):.4f} → {mn('tta'):.4f}"
      f"  Δ{mn('tta')-mn('src'):+.4f}")
print(f"  再換成【只有已知類別】的方向（要標籤、上界）  ：{mn('tta'):.4f} → {mn('oracle'):.4f}"
      f"  Δ{mn('oracle')-mn('tta'):+.4f}")
print(f"  對照：同一批圖上 ‖z⊥‖ 拿 {mn('zp'):.4f}")
print(f"  cos(ū₁, ū_tta) ＝ {mn('c_tta'):+.4f}   （ū₁ 與測試域方向的一致度）")
print("="*W)
