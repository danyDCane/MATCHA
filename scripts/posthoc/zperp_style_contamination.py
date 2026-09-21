"""dany 2026-08-29：‖z⊥‖ 的畫風污染分解 —— rel_loss 的事前 go/no-go。
★ 關鍵資料：來源域也有 person 圖（訓練 excl_person 排除，資料仍在）⇒ 可做完整 2×2。
★ dany 要求加標準差：AUROC 看分布重疊，均值位移會被單調變換抵銷，但「② 變窄」會真的改變排序。
輸出：research/outputs/0829_zperp/{per_node.csv, summary.csv}
基底：0826_features_full.npz（BN 平均 B）
"""
import numpy as np, csv, os
from scipy.stats import rankdata
F = np.load("logs/prototype_probe/0826_features_full.npz")
N, UNK = 9, 6
OUT = "research/outputs/0829_zperp"; os.makedirs(OUT, exist_ok=True)
nrm = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)
def auroc(pos, neg):
    a = np.concatenate([pos, neg]); r = rankdata(a, method="average")
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))

PILES = [("1_known_source", "①已知×來源域畫風", "src", False),
         ("2_known_cartoon", "②已知×cartoon", "tgt", False),
         ("1p_person_source", "①'person×來源域畫風", "src", True),
         ("3_person_cartoon", "③person×cartoon", "tgt", True)]
D = {}
for i in range(N):
    C = nrm(F[f"n{i}_C"].astype(np.float64)); U = np.linalg.svd(C.T, full_matrices=False)[0]
    D[i] = {}
    for k in ["src", "tgt"]:
        Z = nrm(F[f"n{i}_{k}_z"].astype(np.float64))
        D[i][k] = (np.sqrt(np.maximum(1-((Z@U)**2).sum(1), 0)), F[f"n{i}_{k}_y"])

rows, agg = [], {p[0]: [] for p in PILES}
for i in range(N):
    for key, lbl, sp, isu in PILES:
        z, y = D[i][sp]; v = z[(y == UNK) if isu else (y != UNK)]
        q = np.percentile(v, [5, 25, 50, 75, 95])
        # ⚠️ 解釋率必須逐樣本算 ‖z∥‖²=1−‖z⊥‖² 再平均。
        #    拿均值算 (1−mean²) 會因 E[x²]≠(E[x])² 系統性高估 1–4pp（2026-09-01 dany 對數字時抓到）。
        r = dict(node=i, pile=key, label=lbl, n=len(v), mean=round(v.mean(), 4), std=round(v.std(ddof=1), 4),
                 explained=round(float((1 - v**2).mean()), 4),
                 min=round(v.min(), 4), p5=round(q[0], 4), p25=round(q[1], 4), p50=round(q[2], 4),
                 p75=round(q[3], 4), p95=round(q[4], 4), max=round(v.max(), 4))
        rows.append(r); agg[key].append(r)
with open(f"{OUT}/per_node.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

print("="*104); print("§0 零點（隨機 128 維向量投影到任意 6 維子空間）"); print("="*104)
rng = np.random.default_rng(2026); R = nrm(rng.standard_normal((200000, 128)))
U0 = np.linalg.svd(rng.standard_normal((128, 6)), full_matrices=False)[0]
zr = np.sqrt(np.maximum(1-((R@U0)**2).sum(1), 0))
print(f"  ‖z⊥‖ = {zr.mean():.4f} ± {zr.std():.4f}   理論 sqrt(1−6/128)={np.sqrt(1-6/128):.4f} ✅")

print(); print("="*104); print("§1 四堆完整統計（9 節點 node-mean）"); print("="*104)
print(f"{'堆':<22}{'均值':>8}{'標準差':>8}{'解釋率':>8}{'p5':>7}{'p25':>7}{'p50':>7}{'p75':>7}{'p95':>7}{'張數':>7}")
print("-"*104)
S = {}
sm_rows = []
for key, lbl, _, _ in PILES:
    a = agg[key]; m = np.mean([r["mean"] for r in a]); sd = np.mean([r["std"] for r in a])
    ex = np.mean([r["explained"] for r in a])
    qs = [np.mean([r[q] for r in a]) for q in ["p5","p25","p50","p75","p95"]]
    n_ = np.mean([r["n"] for r in a]); S[key] = (m, sd)
    print(f"{lbl:<22}{m:8.4f}{sd:8.4f}{ex*100:7.1f}%{qs[0]:7.3f}{qs[1]:7.3f}{qs[2]:7.3f}{qs[3]:7.3f}{qs[4]:7.3f}{n_:7.0f}")
    sm_rows.append(dict(pile=key, label=lbl, mean=round(m,4), std=round(sd,4),
                        explained_pct=round(ex*100,1), explained_pct_biased=round((1-m**2)*100,1),
                        p5=round(qs[0],4), p25=round(qs[1],4),
                        p50=round(qs[2],4), p75=round(qs[3],4), p95=round(qs[4],4), n_mean=round(n_,0)))
with open(f"{OUT}/summary.csv","w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=list(sm_rows[0])); w.writeheader(); w.writerows(sm_rows)

(m1,s1),(m2,s2),(m1p,s1p),(m3,s3) = [S[k] for k,_,_,_ in PILES]
print("-"*104)
print(f"  ★ 畫風對【已知類別】：均值 {m2-m1:+.4f}   標準差 {s2-s1:+.4f}（{s2/s1:.2f}×）  ⚠️ 混記憶效應、是上界")
print(f"  ★ 畫風對【person】  ：均值 {m3-m1p:+.4f}   標準差 {s3-s1p:+.4f}（{s3/s1p:.2f}×）  ✅ 兩側皆未參與訓練")
print(f"  ★ 語意（①'−①）      ：均值 {m1p-m1:+.4f}   標準差 {s1p-s1:+.4f}")
print(f"  ⇒ 均值污染比 {(m2-m1)/(m3-m1p):.1f}×    標準差污染比 {(s2-s1)/(s3-s1p+1e-12):.1f}×")
print(f"  ★★ ②/③ 標準差比 = {s2/s3:.2f}（0821b 在角距離上報 1.59）   現行 d′ = {(m3-m2)/np.sqrt((s2**2+s3**2)/2):.4f}")

print(); print("="*104)
print("§1.5 ★ 同畫風 vs 換畫風：‖z⊥‖ 當分數的 AUROC 2×2（2026-09-01 補；dany 問「0.9402 哪來的」）")
print("="*104)
pile_of = lambda i, sp, unk: (lambda z, y: z[(y == UNK) if unk else (y != UNK)])(*D[i][sp])
CASES = [("同畫風·來源域   ①已知 vs ①'person", "src", "src"),
         ("同畫風·cartoon  ②已知 vs ③person ", "tgt", "tgt"),
         ("跨畫風          ①已知 vs ③person ", "src", "tgt"),
         ("跨畫風          ②已知 vs ①'person", "tgt", "src")]
print(f"{'情境（該放行 vs 該攔下）':<40}{'AUROC':>9}{'判別力':>9}{'節點std':>9}")
print("-"*104)
A = {}
for name, sp_kn, sp_unk in CASES:
    vals = [auroc(pile_of(i, sp_unk, True), pile_of(i, sp_kn, False)) for i in range(N)]
    A[name] = float(np.mean(vals))
    print(f"{name:<40}{A[name]:9.4f}{A[name]-0.5:9.4f}{np.std(vals, ddof=1):9.4f}")
print("-"*104)
k = list(A)
print(f"  ★ 固定【該攔下】那側、只換【該放行】的畫風：")
print(f"      該攔的用 cartoon person：{A[k[2]]:.4f} → {A[k[1]]:.4f}  ({A[k[1]]-A[k[2]]:+.4f})")
print(f"      該攔的用來源域 person  ：{A[k[0]]:.4f} → {A[k[3]]:.4f}  ({A[k[3]]-A[k[0]]:+.4f})")
print(f"  ★ 固定【該放行】那側、只換【該攔下】的畫風：")
print(f"      該放的用來源域畫風：{A[k[0]]:.4f} vs {A[k[2]]:.4f}  ({A[k[2]]-A[k[0]]:+.4f})")
print(f"      該放的用 cartoon  ：{A[k[1]]:.4f} vs {A[k[3]]:.4f}  ({A[k[3]]-A[k[1]]:+.4f})")
print(f"  ⇒ AUROC 由【該放行那側是哪個畫風】決定；該攔下那側換不換畫風幾乎無影響。")
print(f"  ⇒ 判別力保留 {(A[k[1]]-0.5)/(A[k[0]]-0.5)*100:.1f}% ⇒ 丟掉 {100-(A[k[1]]-0.5)/(A[k[0]]-0.5)*100:.1f}%")
print(f"  ⚠️ ① 是訓練看過的樣本、①'／③ 都沒看過 ⇒ ①→② 混記憶效應，0.9402 是上界不是乾淨基準。")

print(); print("="*104); print("§2 ★ 分別壓「均值」與壓「寬度」，AUROC 各動多少（誠實版：對 ②∪③ 一視同仁）"); print("="*104)
def run(fn):
    return np.mean([ (lambda z,y: auroc(fn(z[y==UNK]), fn(z[y!=UNK])))(*D[i]["tgt"]) for i in range(N)])
print(f"  未介入基準                                        {run(lambda v: v):.4f}")
for g in [0.25, 0.5, 0.75, 1.0]:
    print(f"  (a) 整體平移：所有 cartoon 減去 {g:.2f}×畫風均值差      {run(lambda v,g=g: v-g*(m2-m1)):.4f}   ← 單調變換")
print()
print("  (b) ★ 單邊 hinge（＝rel_loss 的形式）：只壓超過 ① 均值的部分 z − γ·max(0, z − μ₁)")
for g in [0.1, 0.25, 0.5, 0.75, 1.0]:
    print(f"        γ={g:<5}                                     {run(lambda v,g=g: v-g*np.maximum(0, v-m1)):.4f}")
print()
print("  (c) oracle（只動 ②、③ 不碰；不可實現、僅供射程）")
for g in [0.5, 1.0]:
    d = np.mean([ (lambda z,y: auroc(z[y==UNK], z[y!=UNK]-g*(m2-m1)))(*D[i]["tgt"]) for i in range(N)])
    print(f"        只扣 ② 均值的 {g*100:3.0f}%                            {d:.4f}")
for b in [0.75, 0.5]:
    d = np.mean([ (lambda z,y: auroc(z[y==UNK], (z[y!=UNK]-z[y!=UNK].mean())*b + z[y!=UNK].mean()))(*D[i]["tgt"]) for i in range(N)])
    print(f"        只把 ② 的寬度縮成 {b:.2f}×（均值不動）            {d:.4f}")
print("="*104)
print(f"CSV 已輸出：{OUT}/per_node.csv（36 列）與 {OUT}/summary.csv（4 列）")
