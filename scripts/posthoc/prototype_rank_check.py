"""§2.1 閘門：原型矩陣 C 的六個奇異值 —— 六個類別中心到底張成 6 維還是 5 維？

【為什麼是第一個跑】若 C 秩虧損（simplex ETF 六向量和為零 ⇒ 只張成 5 維），
`U, _ = qr(C.T)` 的第六欄就是**數值噪聲**，而 `zpar = (Z@U)@U.T` 會多扣一個假方向
⇒ `z⊥` 這個向量本身就變了 ⇒ ‖z⊥‖、部署 AUROC、探針分數全部跟著變
⇒ §2 的六個自檢錨點都是「6 維扣法」下的產物，必須先重建才能當基準。

【判準（plan §2.1）】
  隨機基準  抽 6 個 128 維隨機單位向量的最小奇異值，200 次取 P5 → σ₆ 低於它 ⇒ 秩虧損
  數值地板  float32 eps × σ₁ → σ₆ 落此量級 ⇒ 純數值噪聲
  兩者之間  微弱但真實的第六維 ⇒ 維持 122 不動，報告標明 σ₆

【理論參照】完美 simplex ETF（6 個單位向量、兩兩 arccos(−1/5)=101.54°、和為零）：
  Gram 對角 1／非對角 −1/5 ⇒ 特徵值 {1.2 ×5, 0} ⇒ 奇異值 {√1.2=1.0954 ×5, 0}
"""
import numpy as np

F = np.load("logs/prototype_probe/0826_features_full.npz")
N = 9
nrm = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)

print("=" * 96)
print("§2.1 原型矩陣 C 的奇異值（逐節點）")
print("=" * 96)
print(f"理論參照｜完美 simplex ETF: σ₁..σ₅ = {np.sqrt(1.2):.4f}, σ₆ = 0")
print()

S32, S64, ANG = [], [], []
for i in range(N):
    C_raw = F[f"n{i}_C"]
    C = nrm(C_raw.astype(np.float64))
    S64.append(np.linalg.svd(C, compute_uv=False))
    S32.append(np.linalg.svd(nrm(C_raw.astype(np.float32)), compute_uv=False))
    G = C @ C.T
    off = G[~np.eye(6, dtype=bool)]
    ANG.append(np.degrees(np.arccos(np.clip(off, -1, 1))).mean())

S64, S32 = np.array(S64), np.array(S32)
print(f"{'node':<6}" + "".join(f"{'σ'+str(k+1):>9}" for k in range(6)) + f"{'σ₆/σ₁':>11}{'類間夾角':>10}")
print("-" * 96)
for i in range(N):
    print(f"{i:<6}" + "".join(f"{v:9.5f}" for v in S64[i]) + f"{S64[i][5]/S64[i][0]:11.2e}{ANG[i]:9.2f}°")
print("-" * 96)
m64 = S64.mean(0)
print(f"{'平均':<6}" + "".join(f"{v:9.5f}" for v in m64) + f"{m64[5]/m64[0]:11.2e}{np.mean(ANG):9.2f}°")
print(f"{'(f32)':<6}" + "".join(f"{v:9.5f}" for v in S32.mean(0)))

# ── 零點 1：隨機 6 個單位向量的最小奇異值
rng = np.random.default_rng(2026)
s6 = np.array([np.linalg.svd(nrm(rng.standard_normal((6, 128))), compute_uv=False)[5]
               for _ in range(200)])
# ── 零點 2：float32 數值地板
floor32 = np.finfo(np.float32).eps * m64[0]

print()
print("=" * 96)
print("判準對照")
print("=" * 96)
s6_actual = S64[:, 5]
print(f"  實測 σ₆（9 節點）      mean {s6_actual.mean():.6f}   min {s6_actual.min():.6f}   max {s6_actual.max():.6f}")
print(f"  隨機基準 σ₆            P5 {np.percentile(s6,5):.6f}   中位數 {np.median(s6):.6f}   （200 次）")
print(f"  float32 數值地板       {floor32:.3e}   (= eps {np.finfo(np.float32).eps:.2e} × σ₁ {m64[0]:.4f})")
print()
below_rand = (s6_actual < np.percentile(s6, 5)).sum()
at_floor = (s6_actual < floor32 * 10).sum()
print(f"  σ₆ 低於隨機基準 P5 的節點數：{below_rand}/9")
print(f"  σ₆ 落在數值地板量級的節點數：{at_floor}/9")
print()
if at_floor == N:
    print("  ⇒ 【純數值噪聲】確認秩虧損 ⇒ z⊥ 應為 123 維，全部錨點須用 5 維扣法重建")
elif below_rand == N:
    print("  ⇒ 【顯著低於隨機、但高於數值地板】＝ 微弱但真實的第六維")
    print("     ⇒ plan §2.1：維持 122 不動，報告標明 σ₆")
else:
    print("  ⇒ 【未低於隨機基準】六個中心確實張成 6 維，維持 122，無需重建")

# ── 影響量化：6 維扣法 vs 5 維扣法，z⊥ 差多少
print()
print("=" * 96)
print("影響量化：若改用 5 維扣法，‖z⊥‖ 會差多少（決定錨點要不要重建）")
print("=" * 96)
UNK = 6
def auroc(pos, neg):
    from scipy.stats import rankdata
    a = np.concatenate([pos, neg]); r = rankdata(a, method="average")
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))

d6, d5, a6, a5 = [], [], [], []
for i in range(N):
    Z = nrm(F[f"n{i}_tgt_z"].astype(np.float32))
    y = F[f"n{i}_tgt_y"].astype(int)
    U6, _ = np.linalg.qr(F[f"n{i}_C"].astype(np.float32).T)          # 現行：6 欄
    Uf = np.linalg.svd(nrm(F[f"n{i}_C"].astype(np.float64)).T, full_matrices=False)[0]
    U5 = Uf[:, :5]                                                    # 只取前 5 個左奇異向量
    n6 = np.linalg.norm(Z - (Z @ U6) @ U6.T, axis=1)
    n5 = np.linalg.norm(Z - (Z @ U5) @ U5.T, axis=1)
    kn = y != UNK
    d6.append(n6[kn].mean()); d5.append(n5[kn].mean())
    a6.append(auroc(n6[~kn], n6[kn])); a5.append(auroc(n5[~kn], n5[kn]))

print(f"  ② 已知×cartoon 的 ‖z⊥‖ 均值   6 維扣法 {np.mean(d6):.4f}   5 維扣法 {np.mean(d5):.4f}   Δ {np.mean(d5)-np.mean(d6):+.4f}")
print(f"  部署 AUROC（②vs③）           6 維扣法 {np.mean(a6):.4f}   5 維扣法 {np.mean(a5):.4f}   Δ {np.mean(a5)-np.mean(a6):+.4f}")
print(f"  （6 維扣法應重現 §2 錨點 0.6289 / 0.8281）")
print("=" * 96)
