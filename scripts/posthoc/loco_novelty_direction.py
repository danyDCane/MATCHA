"""LOCO 通用新奇方向：把「已知類別」當成可替換的角色，看方向轉不轉移得到 person
（dany 2026-08-26 設計；主 agent 修正訓練正負方向）

★ 為什麼要在 z_⊥ 上做（dany）：dog 是訓練過的類別、有原型、`comp_loss` 一直把它往原型拉；
   person 沒有原型。直接在 z 上比，探針最省力的解法是「指向 dog 的原型」——那是被明確訓練成好分的方向。
   z_⊥ ＝ z 扣掉六個原型張成的子空間（122 維）⇒ 物理性切斷那條捷徑。
   再正規化：‖z_⊥‖ 是六個距離的函數（六個中心不正交，但投影長度由那六個內積經 Gram 矩陣決定）⇒ 不是新資訊。

★ 訓練正負方向（主 agent 修正）：dany 原寫「第 k 類(正) vs 其他5類(負)」⇒ w_k 指向「k 的獨特之處」
   ⇒ 六個 w_k 指向六個不同地方 ⇒ **平均會抵銷成零，而 w̄ 正是最關鍵那一格**。
   改成「其他5類(正=已知) vs 第 k 類(負=假裝的未知)」⇒ w_k 學到「已知性」⇒ 六個都含共同成分 ⇒ 平均才放大它。
   OOD 分數取 −wᵀz。

判讀（事前寫死，dany；⚠️ 不對稱）：
   w̄ 測 person ≥ 0.87        ⇒ 通用新奇方向存在、決定性正面 ⇒ 進「怎麼免標籤找到它」
   0.75–0.87                 ⇒ 有共同成分但被稀釋 ⇒ 看 cos(w_k, w_person)
   ≈ 0.5                     ⇒ 沒轉移。⚠️ **不可直接判死**（dog 終究是訓練過的類別）⇒ 只能降級，要靠移除 dog 的重訓版定讞

★ 第 2 項同批跑：w̄ 在來源域的變異數 vs 隨機方向的平均變異數
   ⇒ LOCO 說「方向有沒有意義」，變異數說「不用標籤能不能找到它」。兩個都過才有方法。
   w̄ 落在低變異方向 ⇒ 白化/共變異度量就找得到，不需要任何未知類別標籤。
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DUMP = os.environ.get("DUMP", "logs/prototype_probe/0826_features_full.npz")
N = int(os.environ.get("NODES", "9")); UNK = 6; NC = 6
CLS = ["dog", "elephant", "giraffe", "guitar", "horse", "house"]
rng = np.random.default_rng(2026)
nrm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
F = np.load(DUMP)


def fit_w(X, y):
    """balanced 線性探針，回傳單位權重向量"""
    m = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0).fit(X, y)
    w = m.coef_.ravel(); return w / np.linalg.norm(w)


R = {}
for i in range(N):
    y = F[f"n{i}_tgt_y"].astype(int)
    z = nrm(F[f"n{i}_tgt_z"].astype(np.float32))
    C = F[f"n{i}_C"].astype(np.float32)
    zs = nrm(F[f"n{i}_src_z"].astype(np.float32)); ys = F[f"n{i}_src_y"].astype(int)
    P = np.linalg.svd(C, full_matrices=False)[2]          # 六個原型張成的子空間正交基 [6,128]

    for space in ["z", "z_perp"]:
        if space == "z":
            X, Xs = z, zs
        else:
            X, Xs = nrm(z - (z @ P.T) @ P), nrm(zs - (zs @ P.T) @ P)
        kn, un = y != UNK, y == UNK
        Wk = []
        for k in range(NC):
            tr = kn                                        # 只用 cartoon 已知類別訓練
            Xtr, ytr = X[tr], (y[tr] != k).astype(int)     # 正=其他5類(已知)、負=第k類(假裝未知)
            w = fit_w(Xtr, ytr); Wk.append(w)
            # 測：② 排除第 k 類 vs ③ person；OOD 分數 = −wᵀx
            sel = (kn & (y != k)) | un
            lab = un[sel].astype(int)
            R.setdefault((space, "loco", k), []).append(roc_auc_score(lab, -(X[sel] @ w)))
        Wk = np.stack(Wk); wbar = Wk.mean(0); wbar /= np.linalg.norm(wbar)
        lab = un.astype(int)
        R.setdefault((space, "wbar"), []).append(roc_auc_score(lab, -(X @ wbar)))
        # oracle：直接用 person 標籤訓的方向（5 折 held-out）
        sk = StratifiedKFold(5, shuffle=True, random_state=2026); o = []
        for tr, te in sk.split(X, lab):
            wp = fit_w(X[tr], 1 - lab[tr])                 # 正=已知
            o.append(roc_auc_score(lab[te], -(X[te] @ wp)))
        R.setdefault((space, "oracle"), []).append(float(np.mean(o)))
        wp_full = fit_w(X, 1 - lab)
        R.setdefault((space, "cos_kp"), []).append([float(abs(w @ wp_full)) for w in Wk])
        off = ~np.eye(NC, dtype=bool)
        R.setdefault((space, "cos_kj"), []).append(float(np.abs(Wk @ Wk.T)[off].mean()))
        # 隨機方向零點
        Rr = nrm(rng.standard_normal((20, X.shape[1])))
        R.setdefault((space, "rand"), []).append(float(np.mean([roc_auc_score(lab, -(X @ r)) for r in Rr])))
        R.setdefault((space, "rand_cos"), []).append(float(np.abs(Rr[:10] @ Rr[10:].T).mean()))
        # ★ 第2項：w̄ 在【來源域】的變異數 vs 隨機方向平均變異數
        Sc = np.cov(Xs[ys != UNK].T)
        R.setdefault((space, "var_w"), []).append(float(wbar @ Sc @ wbar))
        R.setdefault((space, "var_rand"), []).append(float(np.trace(Sc) / Sc.shape[0]))
    print(f"  node{i} done", flush=True)

m = lambda k: float(np.mean(R[k], axis=0))
print("\n" + "=" * 96)
for space, tag in [("z_perp", "★ z_⊥（扣掉六個原型子空間、再正規化；122 維）"), ("z", "對照：z（未扣原型；128 維）")]:
    print(f"\n{tag}")
    print("-" * 96)
    per = [m((space, "loco", k)) for k in range(NC)]
    for k in range(NC):
        print(f"   w_{CLS[k]:<10} 測 ②(排除{CLS[k]}) vs ③person ： {per[k]:.4f}")
    print(f"   {'六個平均':<14}                              {np.mean(per):.4f}")
    print(f"   ★ w̄（六個方向平均後）                        **{m((space,'wbar')):.4f}**")
    print(f"   oracle（直接用 person 標籤訓，5折held-out）    {m((space,'oracle')):.4f}")
    print(f"   隨機方向（無資訊零點）                        {m((space,'rand')):.4f}")
    ck = np.mean(R[(space, "cos_kp")], axis=0)
    print(f"   cos(w_k, w_person)：{[f'{x:.3f}' for x in ck]}   平均 {ck.mean():.3f}")
    print(f"   cos(w_k, w_j) 兩兩平均 {m((space,'cos_kj')):.3f}   ｜隨機兩向量 {m((space,'rand_cos')):.3f}")
    vw, vr = m((space, "var_w")), m((space, "var_rand"))
    print(f"   ★ w̄ 在來源域的變異數 {vw:.5f}  vs 隨機方向平均 {vr:.5f}  ⇒ 比值 {vw/vr:.3f}"
          f"　（<1 ＝落在低變異方向 ⇒ 白化找得到）")
print("=" * 96)
print("判讀：w̄ ≥0.87 決定性正面｜0.75–0.87 稀釋｜≈0.5 沒轉移（⚠️不可直接判死，dog 是訓練過的類別）")
