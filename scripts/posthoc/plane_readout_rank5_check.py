# -*- coding: utf-8 -*-
"""面讀出 ‖z⊥‖ 的子空間該用 6 維還是 5 維？（2026-10-02，零 GPU、用 0908_dump）

起因：觸發器路線的副產品發現六個類別中心近乎降秩 —— σ₆/σ₁ ≈ 0.068（六個隨機單位向量約 0.74），
      六中心和的長度 ≈ 0.19–0.27（隨機約 2.45）⇒ 單純形構型、實質只張 5 維。
      而現行 plane() 用 SVD 的全部 6 個左奇異向量 ⇒ 第 6 個方向幾乎不被中心撐起（σ₆ 小 16 倍），
      等於每次都多扣掉一個「近乎任意」的方向。
問題：改成只扣前 5 個主方向，AUROC 會不會變？
設計：同一批特徵、同一批節點 ⇒ **配對比較**（逐節點相減）。
      劑量：6／5／4 維；控制組：扣掉「最大」的那一維而留下最小的（應該變差）。
      判讀帶：檢測 0.01 AUROC（看板判準⑤）。
錨點：cartoon 現行必須重現登記值 0.8281、sketch 0.7966（否則管線不對、全部作廢）。
"""
import numpy as np
PERSON, N = 6, 9
DUMP = "logs/prototype_probe/0908_dump/full_{fold}.npz"
FOLDS = ["sketch", "art_painting", "cartoon", "photo"]
ANCHOR = {"cartoon": 0.8281, "sketch": 0.7966}

def auroc(s, pos):
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = pos.sum(); n0 = len(s) - n1
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def plane_basis(z, B):
    u = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    return np.sqrt(np.clip(1.0 - ((u @ B) ** 2).sum(1), 0, None))

def basis(C, keep):
    """C: [6,P]。回傳 [P,k] 正交基；keep 是要保留的奇異向量索引（0＝最大）"""
    U, S, _ = np.linalg.svd(C.T, full_matrices=False)
    return U[:, keep], S

print("面讀出：6 維 vs 5 維子空間（9 節點逐一算 AUROC，配對比較）\n")
allrows = {}
for fold in FOLDS:
    d = np.load(DUMP.format(fold=fold), allow_pickle=True)
    y = d["n0_tgt_y"]; pos = y == PERSON
    res = {k: [] for k in ("r6", "r5", "r4", "ctrl")}
    svr = []
    for i in range(N):
        Z, C = d[f"n{i}_tgt_z"], d[f"n{i}_Cema"]
        for k, keep in (("r6", [0,1,2,3,4,5]), ("r5", [0,1,2,3,4]), ("r4", [0,1,2,3]), ("ctrl", [1,2,3,4,5])):
            B, S = basis(C, keep)
            res[k].append(auroc(plane_basis(Z, B), pos))
        svr.append(S[5] / S[0])
    r = {k: np.array(v) for k, v in res.items()}
    allrows[fold] = r
    a = ANCHOR.get(fold)
    chk = "" if a is None else (f"  錨點 {a:.4f} {'✅' if abs(r['r6'].mean()-a) < 5e-5 else '🚨 對不上'}")
    print(f"【{fold}】σ₆/σ₁ 九節點平均 {np.mean(svr):.4f}｜現行(6 維) {r['r6'].mean():.4f}{chk}")
    for k, nm in (("r5", "5 維（扣掉最小那一維）"), ("r4", "4 維（再扣一維）"), ("ctrl", "控制：扣掉最大那一維")):
        diff = r[k] - r["r6"]
        print(f"   {nm:<22} {r[k].mean():.4f}   配對差 {diff.mean():+.4f}（逐節點 {diff.min():+.4f}～{diff.max():+.4f}，"
              f"變好 {int((diff>0).sum())}/9）")
    print()

print("=" * 88)
print("四折彙總（四折等權）")
print("=" * 88)
for k, nm in (("r6", "現行 6 維"), ("r5", "5 維"), ("r4", "4 維"), ("ctrl", "控制")):
    m = np.mean([allrows[f][k].mean() for f in FOLDS])
    dm = np.mean([(allrows[f][k] - allrows[f]["r6"]).mean() for f in FOLDS])
    print(f"  {nm:<10} 四折平均 {m:.4f}   相對現行 {dm:+.4f}")
diffs = np.concatenate([allrows[f]["r5"] - allrows[f]["r6"] for f in FOLDS])
print(f"\n5 維 − 6 維：36 個(折×節點)配對差 平均 {diffs.mean():+.4f}、std {diffs.std(ddof=1):.4f}、"
      f"變好 {int((diffs>0).sum())}/36")
print("判讀帶：檢測 ±0.01 AUROC。⚠️ 同一批特徵的配對差，不含訓練 seed 的變異。")
