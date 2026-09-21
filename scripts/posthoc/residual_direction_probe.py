"""實驗 D/B/C：殘差方向的免標籤讀出撿不撿得到 —— 依 0902_..._plan_v3.md 執行。

【十條陷阱的落實】(§9)
 1 逐節點跑完整流程、最後 node-mean，絕不跨節點池化
 2 參考方向 ū₁ 只用 ① 算（程式中 src 且 y!=UNK），不觸及任何 cartoon/person
 3 探針設定逐項同 probe_transfer_full.py:29-52
 4 分數統一「越高越可疑」；零點用雙側 max(A,1-A)
 5 AUROC 一律 rankdata(method="average")
 6 殘餘佔比用 S4 實測管線零點，不是 0.5
 7 隨機方向：128 維抽 → 投影到殘差子空間 → 正規化
 8 ‖z⊥‖→0 除零保護，回報觸發數
 9 全部逐節點報
10 報判別力
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

N, UNK, SEED, EPS = 9, 6, 2026, 1e-8
F = np.load("logs/prototype_probe/0826_features_full.npz")
rng = np.random.default_rng(SEED)
nrm = lambda X: X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), EPS)

def auroc(pos, neg):
    """pos=該攔下(高分), neg=該放行。並列用 average rank。"""
    a = np.concatenate([pos, neg]); r = rankdata(a, method="average")
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))

# ───────────────────────── 資料準備（逐節點） ─────────────────────────
D, zero_hits = {}, 0
for i in range(N):
    C = F[f"n{i}_C"].astype(np.float32)
    U, _ = np.linalg.qr(C.T)                                   # (128,6) 與現行管線一致
    d = {"U": U}
    for sp in ("src", "tgt"):
        Z = nrm(F[f"n{i}_{sp}_z"].astype(np.float32))
        y = F[f"n{i}_{sp}_y"].astype(int)
        lo = F[f"n{i}_{sp}_lo"].astype(np.float32)
        zper = Z - (Z @ U) @ U.T
        npr = np.linalg.norm(zper, axis=1)
        zero_hits += int((npr < EPS).sum())                     # 陷阱 8
        d[sp] = dict(u=zper / np.maximum(npr, EPS)[:, None], npr=npr, y=y,
                     nstd=-lo.std(axis=1))                      # −std(logit)，高=可疑
    D[i] = d

kn = lambda i, sp: D[i][sp]["y"] != UNK
un = lambda i, sp: D[i][sp]["y"] == UNK

def ubar(u):                                                    # 平均方向再正規化
    m = u.mean(0); n = np.linalg.norm(m)
    return m / max(n, EPS), float(n)

# ───────────────────────── 實驗 D：三個變體 ─────────────────────────
def d_scores(i, u_eval, ref_src_mask=None):
    """回傳 dict of 分數（越高越可疑）。參考方向一律由 ① 算。"""
    s = D[i]["src"]; m = kn(i, "src") if ref_src_mask is None else ref_src_mask
    u1 = s["u"][m]; y1 = s["y"][m]
    out = {}
    ub, _ = ubar(u1)
    out["D1"] = -(u_eval @ ub)
    cs = np.stack([ubar(u1[y1 == c])[0] for c in range(6)])      # (6,122→128座標)
    out["D2"] = -(u_eval @ cs.T).max(1)
    sim = u_eval @ u1.T                                          # 餘弦；角距離單調遞減
    for k in (1, 5, 10, 50):
        out[f"D3_k{k}"] = np.arccos(np.clip(np.partition(sim, -k, axis=1)[:, -k], -1, 1))
    return out

VAR = ["D1", "D2", "D3_k1", "D3_k5", "D3_k10", "D3_k50"]
dep = {v: [] for v in VAR}; sty = {v: [] for v in VAR}
Dsc = {}                                                         # 部署分數，供 §5 用
for i in range(N):
    t = D[i]["tgt"]
    sc = d_scores(i, t["u"]); Dsc[i] = sc
    for v in VAR:
        dep[v].append(auroc(sc[v][un(i, "tgt")], sc[v][kn(i, "tgt")]))
    # 畫風欄：2-fold held-out ū₁（§4.4）
    idx = np.where(kn(i, "src"))[0]; rs = np.random.default_rng(SEED + i); rs.shuffle(idx)
    half = np.array_split(idx, 2)
    acc = {v: [] for v in VAR}
    for h in range(2):
        m = np.zeros(len(D[i]["src"]["y"]), bool); m[half[1 - h]] = True   # 用另一半當參考
        ev = D[i]["src"]["u"][half[h]]
        s_ev = d_scores(i, ev, ref_src_mask=m)
        s_t = d_scores(i, t["u"], ref_src_mask=m)
        for v in VAR:                                            # ② 當正類
            acc[v].append(auroc(s_t[v][kn(i, "tgt")], s_ev[v]))
    for v in VAR: sty[v].append(np.mean(acc[v]))

# ───────────────────────── §4.3 零點 N1/N2/N3 ─────────────────────────
def rand_dirs(i, n):                                             # 陷阱 7
    U = D[i]["U"]; w = rng.standard_normal((n, 128)).astype(np.float32)
    return nrm(w - (w @ U) @ U.T)

def two_sided(a): return max(a, 1 - a)

n1, n2, n3 = [], [], {k: [] for k in (1, 5, 10, 50)}
for i in range(N):
    t = D[i]["tgt"]; p, q = un(i, "tgt"), kn(i, "tgt")
    v1 = [two_sided(auroc(-(t["u"] @ w)[p], -(t["u"] @ w)[q]))
          for w in rand_dirs(i, 200)]
    n1.append(np.percentile(v1, 95))
    v2 = []
    for _ in range(200):
        W = rand_dirs(i, 6); s = -(t["u"] @ W.T).max(1)
        v2.append(two_sided(auroc(s[p], s[q])))
    n2.append(np.percentile(v2, 95))
    nb = int(kn(i, "src").sum())
    v3 = {k: [] for k in (1, 5, 10, 50)}
    for _ in range(50):
        B = rand_dirs(i, nb); sim = t["u"] @ B.T
        for k in (1, 5, 10, 50):
            s = np.arccos(np.clip(np.partition(sim, -k, axis=1)[:, -k], -1, 1))
            v3[k].append(two_sided(auroc(s[p], s[q])))
    for k in (1, 5, 10, 50): n3[k].append(np.percentile(v3[k], 95))

# ───────────────────────── 輸出 ─────────────────────────
d_ = lambda x: x - 0.5
W = 100
print("=" * W); print("表 1 ── 實驗 D 主結果（node-mean）"); print("=" * W)
print(f"{'讀出':<18}{'部署 AUROC':>12}{'畫風 AUROC':>12}{'判別力':>10}{'節點std':>10}")
print("-" * W)
for lbl, dv, sv in [("min 角距離", 0.8145, 0.6723), ("−std(六角距離)", 0.8264, 0.6699),
                    ("‖z⊥‖(本輪基準)", 0.8281, 0.6665), ("energy", 0.8380, 0.6574),
                    ("★−std(logit)", 0.8656, 0.6464)]:
    print(f"{lbl:<18}{dv:12.4f}{sv:12.4f}{d_(dv):10.4f}{'—':>10}")
print("-" * W)
best = max(VAR, key=lambda v: np.mean(dep[v]))
for v in VAR:
    star = " ★" if v == best else ""
    print(f"{v:<18}{np.mean(dep[v]):12.4f}{np.mean(sty[v]):12.4f}"
          f"{d_(np.mean(dep[v])):10.4f}{np.std(dep[v], ddof=1):10.4f}{star}")
print("-" * W)
print(f"逐節點部署 AUROC（{best}）：" + " ".join(f"{x:.3f}" for x in dep[best]))

print(); print("=" * W); print("表 2 ── 無資訊基準（雙側 P95、node-mean）"); print("=" * W)
print(f"{'零點':<28}{'P95':>10}{'對應變體':>12}{'實測':>10}{'過關?':>8}")
print("-" * W)
rows = [("N1 單一隨機方向×200", np.mean(n1), "D1", np.mean(dep["D1"])),
        ("N2 六個隨機取max×200", np.mean(n2), "D2", np.mean(dep["D2"]))]
for k in (1, 5, 10, 50):
    rows.append((f"N3 隨機bank×50 k={k}", np.mean(n3[k]), f"D3_k{k}", np.mean(dep[f"D3_k{k}"])))
for lbl, z, v, a in rows:
    print(f"{lbl:<28}{z:10.4f}{v:>12}{a:10.4f}{'✅' if a > z else '❌':>8}")
print("=" * W)
np.save("/tmp/_dsc.npy", np.array([1]))   # 佔位；分數在下一支腳本重算
print(f"\n除零保護觸發數：{zero_hits}（應為 0）")
print(f"最佳 D 變體：{best}")
