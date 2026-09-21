"""§11：w_person 這個方向，在【來源域正常資料】上是不是特別安靜？

問題（dany 2026-08-26）：探針靠偷看 person 標籤找到的那把刀，
能不能靠「它在正常資料上變異特別小」這個線索、**不看答案**地找出來？

為什麼重要：Mahalanobis／白化做的就是「除以每個方向的標準差」⇒ 低變異方向被自動放大。
若 w_person 落在低變異區 ⇒ **不需要任何未知類別的標籤，一個共變異矩陣就找得到它。**
而共變異矩陣是可加統計量 ⇒ 與 BN 聚合同一套 gossip 論證（D2）。

⚠️ 陷阱一（本腳本檢查）：單位球面上，共變異有一個接近零的徑向方向。
   若 w_person 剛好對齊它，變異數會極低但沒有意義 ⇒ 要報 w_person 與資料均值方向的餘弦。
⚠️ 陷阱二：極低變異也可能是探針過擬合到雜訊維度 ⇒ 報 w_person 的變異數是否為數值零。
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")

F = np.load("logs/prototype_probe/0826_features_full.npz")
D = np.load("research/outputs/0826_probe_transfer/directions.npz")
N = 9; UNK = 6; NR = 1000
rng = np.random.default_rng(2026)
nrm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
A = {}

for i in range(N):
    ys = F[f"n{i}_src_y"].astype(int); Zs = nrm(F[f"n{i}_src_z"].astype(np.float32))
    Zs = Zs[ys != UNK]                                   # ① 來源域已知類別（本來就沒有 person）
    C = F[f"n{i}_C"].astype(np.float32)
    U, _ = np.linalg.qr(C.T)
    for sp in ["z", "z_perp"]:
        X = Zs if sp == "z" else nrm(Zs - (Zs @ U) @ U.T)
        d = X.shape[1]
        S = np.cov(X.T)                                   # 來源域共變異
        w = D[f"n{i}_{sp}_wp"]; w = w / np.linalg.norm(w)
        vw = float(w @ S @ w)
        Rv = nrm(rng.standard_normal((NR, d)))
        vr = np.array([float(r @ S @ r) for r in Rv])
        A.setdefault((sp, "v_w"), []).append(vw)
        A.setdefault((sp, "v_rand"), []).append(np.percentile(vr, [1, 5, 25, 50, 75, 95]).tolist())
        A.setdefault((sp, "pct"), []).append(float((vr < vw).mean() * 100))     # w 落在隨機分布的百分位
        A.setdefault((sp, "amp"), []).append(float(np.sqrt(vr.mean() / vw)))    # 白化放大倍率 σ_avg/σ_w
        # 六把刀 & 類別中心方向 當參照
        Wk = D[f"n{i}_{sp}_W"]; Wk = Wk / np.linalg.norm(Wk, axis=1, keepdims=True)
        A.setdefault((sp, "v_knife"), []).append(float(np.mean([k @ S @ k for k in Wk])))
        Cm = C if sp == "z" else nrm(C - (C @ U) @ U.T + 1e-12)
        A.setdefault((sp, "v_center"), []).append(float(np.mean([c @ S @ c for c in nrm(Cm)])))
        # 陷阱檢查
        mu = nrm(X.mean(0))
        A.setdefault((sp, "cos_mu"), []).append(float(abs(w @ mu)))
        ev = np.linalg.eigvalsh(S)
        A.setdefault((sp, "eig"), []).append([float(ev.min()), float(ev.max()), float(ev.max() / max(ev.min(), 1e-30))])
    print(f"  node{i} done", flush=True)

m = lambda k: np.mean(A[k], axis=0)
print("\n" + "=" * 92)
print("★ §11：w_person 在【來源域 ① 正常資料】上的變異數 vs 1000 條隨機方向")
for sp, nm in [("z", "z（128 維）"), ("z_perp", "z⊥（122 維）")]:
    q = m((sp, "v_rand")); ei = m((sp, "eig"))
    print(f"\n【{nm}】")
    print(f"  w_person 的變異數        {m((sp,'v_w')):.6f}")
    print(f"  隨機方向 P1/P5/P25/P50/P75/P95   {'  '.join(f'{x:.6f}' for x in q)}")
    print(f"  ⇒ w_person 落在隨機分布的第 {m((sp,'pct')):.1f} 百分位"
          f"   {'✅ 安靜' if m((sp,'pct'))<5 else ('⚠️ 中間' if m((sp,'pct'))<50 else '❌ 吵')}")
    print(f"  ⇒ 白化放大倍率 σ_rand/σ_w = {m((sp,'amp')):.3f}   （>1 才會被白化放大）")
    print(f"  參照：六把刀平均 {m((sp,'v_knife')):.6f}   六個類別中心方向 {m((sp,'v_center')):.6f}")
    print(f"  陷阱檢查：|cos(w, 資料均值方向)| = {m((sp,'cos_mu')):.4f}（大 ⇒ 對齊徑向、變異數低沒意義）")
    print(f"            共變異特徵值 min {ei[0]:.2e} / max {ei[1]:.2e} / 條件數 {ei[2]:.1e}")
print("=" * 92)
