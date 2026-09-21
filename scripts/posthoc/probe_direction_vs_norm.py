"""`z⊥` 的方向 vs 長度：三格對照 ＋ 探針設定的復現驗證。

【為什麼有這支】2026-09-02 dany 問「0.9442 和 0.8281 為什麼是兩個數字」時，
讀 `probe_transfer_full.py:66-71` 才發現：

    zper = Z - zpar
    npr  = np.linalg.norm(zper, axis=1)
    Zp   = zper / npr[:, None]     # ← 長度被除掉
    for sp, X in [("z_perp", Zp), ("z", Z)]:   # 餵進探針的是 Zp

⇒ **0.9442 是「只用方向」，不是「整個 122 維向量」**（spec 的欄名 `z_perp_hat` 的
   `_hat` 就是單位向量的意思，是下游轉述時讀成了整個向量）。
⇒ 「方向 + 長度一起」那一格從來沒有人量過。本腳本補上。

【設定】逐項複製 `probe_transfer_full.py:29-52`，任何一項不同都會讓三格不可相減：
    CV      StratifiedKFold(5, shuffle=True, random_state=2026)
    線性     Pipeline(StandardScaler → LogisticRegression(class_weight="balanced",
                                                          C=1.0, max_iter=5000))
    SEED    2026｜標準化在 pipeline 內每折重 fit｜已知=+1｜Z 先 L2 正規化｜子空間用 QR
    聚合     9 節點各跑一次再平均

【復現驗證】方向 0.9442 / 完整 Z 0.9458 / 長度 0.8281 —— 三格與報告值逐位相同（差 +0.0000）。
⚠️ `0826_probe_transfer_full.log` 表 A 印的 `‖z⊥‖ 單獨 AUROC 0.1719` 是**符號修正前**的舊值，
   1−0.1719=0.8281。現行 code 的 `roc_auc_score(un, npr)` 方向正確，重跑得 0.8281。
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

N, UNK, SEED = 9, 6, 2026
nrm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
F = np.load("logs/prototype_probe/0826_features_full.npz")


def lin(C=1.0):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", C=C, max_iter=5000))


def cv(X, y, mk):
    sk = StratifiedKFold(5, shuffle=True, random_state=SEED); te = []
    for a, b in sk.split(X, y):
        m = mk(); m.fit(X[a], y[a])
        te.append(roc_auc_score(y[b], m.predict_proba(X[b])[:, 1]))
    return float(np.mean(te))


KEYS = ["方向 Zp（122維單位向量）", "完整 Z（128維）", "長度 ‖z⊥‖（1維）", "方向＋長度（123維）"]
res = {k: [] for k in KEYS}
for i in range(N):
    y = F[f"n{i}_tgt_y"].astype(int)
    Z = nrm(F[f"n{i}_tgt_z"].astype(np.float32))
    U, _ = np.linalg.qr(F[f"n{i}_C"].astype(np.float32).T)
    zper = Z - (Z @ U) @ U.T
    npr = np.linalg.norm(zper, axis=1)
    Zp = zper / npr[:, None]
    lab = (y != UNK).astype(int)
    res[KEYS[0]].append(cv(Zp, lab, lambda: lin(1.0)))
    res[KEYS[1]].append(cv(Z, lab, lambda: lin(1.0)))
    res[KEYS[2]].append(roc_auc_score((y == UNK).astype(int), npr))
    res[KEYS[3]].append(cv(np.hstack([Zp, npr[:, None]]), lab, lambda: lin(1.0)))

REF = {KEYS[0]: 0.9442, KEYS[1]: 0.9458, KEYS[2]: 0.8281, KEYS[3]: None}
d = lambda x: x - 0.5
print("=" * 84)
print("★ z⊥ 的方向 vs 長度（②vs③、線性探針 held-out AUROC、5折、9節點平均）")
print("=" * 84)
print(f"{'讀什麼':<26}{'AUROC':>9}{'判別力':>9}{'節點std':>9}{'報告值':>9}{'差':>9}")
print("-" * 84)
M = {}
for k in KEYS:
    m = float(np.mean(res[k])); M[k] = m; r = REF[k]
    tail = f"{r:9.4f}{m - r:+9.4f}" if r else f"{'—':>9}{'新':>9}"
    print(f"{k:<26}{m:9.4f}{d(m):9.4f}{np.std(res[k], ddof=1):9.4f}{tail}")
print("-" * 84)
print(f"  ★ 長度的判別力只佔方向的 {d(M[KEYS[2]]) / d(M[KEYS[0]]) * 100:.1f}%"
      f"  ⇒ 資訊在方向裡，長度是它的一個投影")
print(f"  ★ 方向之外，長度的邊際貢獻 {M[KEYS[3]] - M[KEYS[0]]:+.4f}"
      f"  ⇒ 長度幾乎完全被方向涵蓋、獨立資訊很少")
print(f"  ⚠️ 0.9442／0.9524 都是**有 person 標籤**學出來的上界；"
      f"方向裡有資訊 ≠ 該資訊有免標籤可對齊的結構。")
print("=" * 84)
