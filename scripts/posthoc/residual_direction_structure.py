"""實驗 B/C ＋ 順手項：殘差方向到底有沒有結構（解釋 D 為何失敗）。依 plan v3 §6–§8。"""
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

N, UNK, SEED, EPS, P = 9, 6, 2026, 1e-8, 122
F = np.load("logs/prototype_probe/0826_features_full.npz")
rng = np.random.default_rng(SEED)
nrm = lambda X: X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), EPS)
def auroc(pos, neg):
    a = np.concatenate([pos, neg]); r = rankdata(a, method="average")
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))
def ubar(u):
    m = u.mean(0); n = float(np.linalg.norm(m)); return m / max(n, EPS), n

D = {}
for i in range(N):
    U, _ = np.linalg.qr(F[f"n{i}_C"].astype(np.float32).T)
    d = {"U": U}
    for sp in ("src", "tgt"):
        Z = nrm(F[f"n{i}_{sp}_z"].astype(np.float32))
        zp = Z - (Z @ U) @ U.T; npr = np.linalg.norm(zp, axis=1)
        d[sp] = dict(u=zp / np.maximum(npr, EPS)[:, None], npr=npr,
                     y=F[f"n{i}_{sp}_y"].astype(int),
                     nstd=-F[f"n{i}_{sp}_lo"].astype(np.float32).std(axis=1))
    D[i] = d
kn = lambda i, sp: D[i][sp]["y"] != UNK
un = lambda i, sp: D[i][sp]["y"] == UNK
def rand_dirs(i, n):
    U = D[i]["U"]; w = rng.standard_normal((n, 128)).astype(np.float32)
    return nrm(w - (w @ U) @ U.T)

W = 100
print("=" * W); print("表 5 ── B-1 平均方向長度 ‖ū‖（9 節點平均）"); print("=" * W)
print(f"{'對象':<22}{'‖ū‖':>9}{'解析1/√N':>11}{'經驗P95':>10}{'倍數(對P95)':>13}{'N':>7}")
print("-" * W)
def norm_null(i, n, reps=200):
    return np.percentile([ubar(rand_dirs(i, n))[1] for _ in range(reps)], 95)
rows = []
for lbl, sp, msk in [("① 已知×來源域", "src", kn), ("② 已知×cartoon", "tgt", kn),
                     ("①' person×來源域", "src", un), ("③ person×cartoon", "tgt", un)]:
    vs, ns, zs = [], [], []
    for i in range(N):
        u = D[i][sp]["u"][msk(i, sp)]; vs.append(ubar(u)[1]); ns.append(len(u)); zs.append(norm_null(i, len(u)))
    v, n_, z = np.mean(vs), np.mean(ns), np.mean(zs)
    rows.append((lbl, v, z))
    print(f"{lbl:<22}{v:9.4f}{1/np.sqrt(n_):11.4f}{z:10.4f}{v/z:13.2f}×{n_:7.0f}")
print("-" * W)
for c in range(6):
    vs, ns, zs = [], [], []
    for i in range(N):
        m = kn(i, "src") & (D[i]["src"]["y"] == c)
        u = D[i]["src"]["u"][m]; vs.append(ubar(u)[1]); ns.append(len(u)); zs.append(norm_null(i, len(u)))
    v, n_, z = np.mean(vs), np.mean(ns), np.mean(zs)
    print(f"{'① 類別 '+str(c):<22}{v:9.4f}{1/np.sqrt(n_):11.4f}{z:10.4f}{v/z:13.2f}×{n_:7.0f}")

print(); print("=" * W); print("表 6 ── B-2 特徵值譜（① 全部混合、9 節點平均）"); print("=" * W)
pr, mx, med, mn, top5 = [], [], [], [], []
prz, mxz = [], []
for i in range(N):
    u = D[i]["src"]["u"][kn(i, "src")]; n_ = len(u)
    lam = np.linalg.eigvalsh(np.cov(u.T, bias=True))[::-1][:P]
    pr.append(lam.sum()**2 / (lam**2).sum()); mx.append(lam[0]); med.append(np.median(lam)); mn.append(lam[-1])
    top5.append(lam[:5])
    zp, zm = [], []
    for _ in range(200):
        l2 = np.linalg.eigvalsh(np.cov(rand_dirs(i, n_).T, bias=True))[::-1][:P]
        zp.append(l2.sum()**2 / (l2**2).sum()); zm.append(l2[0])
    prz.append(np.percentile(zp, 5)); mxz.append(np.percentile(zm, 95))
print(f"  參與率 PR（等向＝122）      實測 {np.mean(pr):8.2f}    經驗零點 P5 {np.mean(prz):8.2f}"
      f"    ⇒ {'低於零點 ⇒ 有集中' if np.mean(pr) < np.mean(prz) else '未低於零點 ⇒ 接近等向'}")
print(f"  最大特徵值                  實測 {np.mean(mx):.6f}   經驗零點 P95 {np.mean(mxz):.6f}"
      f"   ⇒ {'超過 ⇒ 有主方向' if np.mean(mx) > np.mean(mxz) else '未超過 ⇒ 無主方向'}")
print(f"  最大 ÷ (1/122)              {np.mean(mx)*P:.3f}")
print(f"  前 5 大                     " + " ".join(f"{x:.5f}" for x in np.mean(top5, 0)))
print(f"  中位數 / 最小               {np.mean(med):.6f} / {np.mean(mn):.6f}")

print(); print("=" * W); print("表 7a ── 實驗 C：類內平均餘弦（帶正負號、零點＝0）"); print("=" * W)
print(f"{'類別':<10}{'實測':>10}{'經驗P95':>10}{'超過?':>8}")
print("-" * W)
for c in range(6):
    vs, zs = [], []
    for i in range(N):
        u = D[i]["src"]["u"][kn(i, "src") & (D[i]["src"]["y"] == c)]; n_ = len(u)
        vs.append((n_ * ubar(u)[1]**2 - 1) / (n_ - 1))
        zz = [(n_ * ubar(rand_dirs(i, n_))[1]**2 - 1) / (n_ - 1) for _ in range(200)]
        zs.append(np.percentile(zz, 95))
    print(f"{'類別 '+str(c):<10}{np.mean(vs):10.5f}{np.mean(zs):10.5f}{'✅' if np.mean(vs)>np.mean(zs) else '❌':>8}")

print(); print("=" * W); print("表 7b ── 順手項"); print("=" * W)
c23, c12, stab = [], [], []
for i in range(N):
    u2, _ = ubar(D[i]["tgt"]["u"][kn(i, "tgt")]); u3, _ = ubar(D[i]["tgt"]["u"][un(i, "tgt")])
    u1, _ = ubar(D[i]["src"]["u"][kn(i, "src")])
    c23.append(u2 @ u3); c12.append(u1 @ u2)
    idx = np.where(kn(i, "src"))[0]; r = np.random.default_rng(SEED + i); r.shuffle(idx)
    h = np.array_split(idx, 2)
    stab.append(ubar(D[i]["src"]["u"][h[0]])[0] @ ubar(D[i]["src"]["u"][h[1]])[0])
print(f"  cos(ū₂, ū₃)  同畫風純語意   {np.mean(c23):+.4f}   零點 1/√122 = 0.0905"
      f"   ⇒ {'遠高於零點' if abs(np.mean(c23))>0.0905 else '≈零點'}")
print(f"  cos(ū₁, ū₂)  純畫風那一段   {np.mean(c12):+.4f}   同上")
print(f"  ū₁ 穩定性 cos(半A,半B)      {np.mean(stab):+.4f}   同上"
      f"   ⇒ {'ū₁ 是族群性質' if np.mean(stab)>0.5 else '⚠️ ū₁ 不穩定'}")
print(f"  ‖z⊥‖ 最小值（① node0）      {D[0]['src']['npr'][kn(0,'src')].min():.4f}")

print(); print("=" * W); print("表 7c ── MLP「兩者一起」＋ D3_k1 與既有讀出的關係"); print("=" * W)
def cv_mlp(X, y):
    sk = StratifiedKFold(5, shuffle=True, random_state=SEED); te = []
    for a, b in sk.split(X, y):
        m = make_pipeline(StandardScaler(), MLPClassifier((256,), max_iter=500, random_state=SEED,
                                                          early_stopping=True, n_iter_no_change=15))
        m.fit(X[a], y[a]); te.append(roc_auc_score(y[b], m.predict_proba(X[b])[:, 1]))
    return float(np.mean(te))
mlp_both = []
for i in range(N):
    t = D[i]["tgt"]; X = np.hstack([t["u"], t["npr"][:, None]]); lab = (t["y"] != UNK).astype(int)
    mlp_both.append(cv_mlp(X, lab))
print(f"  MLP「方向＋長度」(123維)    {np.mean(mlp_both):.4f}   （線性版 0.9524、MLP 方向 0.9579）")

sp_n, sp_s, fus_o, fus_h = [], [], [], []
for i in range(N):
    t = D[i]["tgt"]; sim = t["u"] @ D[i]["src"]["u"][kn(i, "src")].T
    d3 = np.arccos(np.clip(sim.max(1), -1, 1))
    sp_n.append(spearmanr(d3, t["npr"]).statistic); sp_s.append(spearmanr(d3, t["nstd"]).statistic)
    s_ = D[i]["src"]; m = kn(i, "src")
    sim_s = s_["u"][m] @ s_["u"][m].T; np.fill_diagonal(sim_s, -np.inf)
    d3s = np.arccos(np.clip(sim_s.max(1), -1, 1))
    zd = (d3 - d3s.mean()) / (d3s.std() + EPS)
    zs = (t["nstd"] - s_["nstd"][m].mean()) / (s_["nstd"][m].std() + EPS)
    p, q = un(i, "tgt"), kn(i, "tgt")
    ws = np.linspace(0, 1, 21)
    a = [auroc((w*zd + (1-w)*zs)[p], (w*zd + (1-w)*zs)[q]) for w in ws]
    fus_o.append(max(a)); fus_h.append(a[10])
print(f"  Spearman(D3_k1, ‖z⊥‖)      {np.mean(sp_n):+.4f}")
print(f"  Spearman(D3_k1, −std)       {np.mean(sp_s):+.4f}")
print(f"  融合 D3_k1+−std（oracle w） {np.mean(fus_o):.4f}   ⚠️ 權重在測試集選 ⇒ 樂觀上界")
print(f"  融合 固定 w=0.5             {np.mean(fus_h):.4f}   （−std 單獨 0.8656）")
print("=" * W)
