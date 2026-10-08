"""跨節點原型聚合的 post-hoc 預演（**面讀出**、四折、吃 0908_dump）。

與既有 `proto_reference_2a2b_preview.py` 的關係（2026-09-22 建檔時查證）：
  既有那支 = 點讀出（到最近原型的角距離）／吃 checkpoint／只有 cartoon 折。
  本支     = 面讀出 ‖z⊥‖（到原型張成子空間的殘差）／吃 0908_dump npz／四折。
  ⇒ 兩個不同的量、不同的資料源，**不可互相取代、也不可並列相除**（看板兩個讀出族不可混用）。

回答：把讀出的參照物從「本節點自己的 6 個 EMA 原型」換成跨節點版本，四軸會怎樣。
⚠️ 射程：這是**讀出端**換參照物（post-hoc、零重訓）。
   訓練端若真的聚合原型，comp/disp 看到的候選集會改變 ⇒ **本結果不構成對訓練端的否決**。

用法：venv_matcha/bin/python scripts/posthoc/proto_reference_plane_4fold.py
"""
import numpy as np, sys
from itertools import combinations
PERSON, N = 6, 9
DUMP = "logs/prototype_probe/0908_dump/full_{fold}.npz"
FOLDS = ["sketch", "art_painting", "cartoon", "photo"]

def auroc(s, pos):
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = pos.sum(); n0 = len(s) - n1
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def plane(z, C):
    """‖z⊥‖：單位化後對 C 張成子空間的殘差範數。"""
    u = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    B = np.linalg.svd(C.T, full_matrices=False)[0]
    return np.sqrt(np.clip(1.0 - ((u @ B) ** 2).sum(1), 0, None))

rng = np.random.default_rng(2026)
print("跨節點原型聚合 · 讀出端預演（面讀出 ‖z⊥‖、9 節點逐一算後平均）")
print("錨點：sketch 現行 = 0.7966、cartoon 現行 = 0.8281（§5.9／§A 已登記值）\n")
print(f'{"fold":13s} {"現行(自己6)":>12s} {"6個跨節點平均":>14s} {"18原型(3畫風各1套)":>18s} {"隨機6維":>9s} {"隨機18維":>10s}')
for fold in FOLDS:
    d = np.load(DUMP.format(fold=fold), allow_pickle=True)
    y = d["n0_tgt_y"]; pos = y == PERSON
    Z  = [d[f"n{i}_tgt_z"]  for i in range(N)]
    Ce = [d[f"n{i}_Cema"]   for i in range(N)]
    cur = np.mean([auroc(plane(Z[i], Ce[i]), pos) for i in range(N)])
    m = np.stack(Ce).mean(0); m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
    avg6 = np.mean([auroc(plane(Z[i], m), pos) for i in range(N)])
    # 18 原型：三個畫風組各挑一個節點，27 種組合全跑後平均（避免挑到極端值）
    cross = [np.mean([auroc(plane(Z[i], np.concatenate([Ce[a], Ce[b], Ce[c]], 0)), pos) for i in range(N)])
             for a in (0, 1, 2) for b in (3, 4, 5) for c in (6, 7, 8)]
    r6  = np.mean([auroc(plane(Z[i], rng.normal(size=(6, 128))),  pos) for i in range(N)])
    r18 = np.mean([auroc(plane(Z[i], rng.normal(size=(18, 128))), pos) for i in range(N)])
    print(f'{fold:13s} {cur:12.4f} {avg6:14.4f} {np.mean(cross):18.4f} {r6:9.4f} {r18:10.4f}')
    print(f'{"":13s} {"":12s} {"Δ"+format(avg6-cur,"+.4f"):>14s} {"Δ"+format(np.mean(cross)-cur,"+.4f"):>18s}'
          f'   〔18原型 27 種組合範圍 {min(cross):.4f}–{max(cross):.4f}〕')
