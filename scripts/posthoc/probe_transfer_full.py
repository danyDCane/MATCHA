"""探針轉移驗證 E0–E5：完全依 research/prototype_probe/0826_probe_transfer_spec.md 規格

核心問題（§0）：128 維 Z 上 ②vs③ 線性探針 0.9495，但探針偷看了 person 標籤。
那把刀是 (A) 通用陌生刀 還是 (B) person 專屬刀？兩者數字一樣。
⇒ 借一個已知類別演「未知」，訓一把刀，切從未參與訓練的 person。

⚠️ E2 的 ρ 才是真正的答案（§4/E2）：w_k ∝ (6/5)(全域均值 − μ_k) ⇒ 對 k 平均恰好是零向量，
   與正負怎麼定無關 ⇒ w̄ 沒有判別力。ρ 符號不變，問「person 那支箭在不在六刀張成的桌面上」。
"""
import os, sys, json, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

DUMP = os.environ.get("DUMP", "logs/prototype_probe/0826_features_full.npz")
OUTD = os.environ.get("OUTD", "research/outputs/0826_probe_transfer")
N = int(os.environ.get("NODES", "9")); UNK = 6; NC = 6; SEED = 2026
NRAND = int(os.environ.get("NRAND", "1000"))
CLS = ["dog", "elephant", "giraffe", "guitar", "horse", "house"]
rng = np.random.default_rng(SEED)
nrm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
F = np.load(DUMP); os.makedirs(OUTD, exist_ok=True)


def lin(C=1.0):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", C=C, max_iter=5000))


def mlp():
    # ⚠️ sklearn 的 MLPClassifier 不支援 class_weight；AUROC 對不平衡不敏感（排序指標）⇒ 註記後照跑
    return make_pipeline(StandardScaler(),
                         MLPClassifier((256,), max_iter=500, random_state=SEED,
                                       early_stopping=True, n_iter_no_change=15))


def cv(X, y, mk):
    """5 折 stratified，回傳 (held-out AUROC, train AUROC)"""
    sk = StratifiedKFold(5, shuffle=True, random_state=SEED); te_, tr_ = [], []
    for a, b in sk.split(X, y):
        m = mk(); m.fit(X[a], y[a])
        te_.append(roc_auc_score(y[b], m.predict_proba(X[b])[:, 1]))
        tr_.append(roc_auc_score(y[a], m.predict_proba(X[a])[:, 1]))
    return float(np.mean(te_)), float(np.mean(tr_))


def wvec(X, y, C=1.0):
    m = lin(C); m.fit(X, y)
    w = m[-1].coef_.ravel() / m[0].scale_          # 還原到原始座標
    return w / np.linalg.norm(w)


R, SAN, DIRS, PS = {}, {}, {}, {}
for i in range(N):
    y = F[f"n{i}_tgt_y"].astype(int)
    Z = nrm(F[f"n{i}_tgt_z"].astype(np.float32))
    Cm = F[f"n{i}_C"].astype(np.float32)
    U, _ = np.linalg.qr(Cm.T)                       # (128,6)
    zpar = (Z @ U) @ U.T
    zper = Z - zpar
    npr = np.linalg.norm(zper, axis=1)
    Zp = zper / npr[:, None]
    kn, un = y != UNK, y == UNK
    lab_known = kn.astype(int)                      # 已知=+1

    SAN.setdefault("orth", []).append(float(np.abs(Zp @ U).max()))
    SAN.setdefault("decomp", []).append(float(np.abs((np.linalg.norm(zpar,axis=1)**2 + npr**2) - 1).max()))
    SAN.setdefault("normauc", []).append(roc_auc_score(un.astype(int), npr))   # ★ +npr：‖z⊥‖ 大＝偏離原型子空間＝OOD-like，不加負號（2026-08-26 修）
    SAN.setdefault("norm_m", []).append([float(npr[kn].mean()), float(npr[un].mean())])

    for sp, X in [("z_perp", Zp), ("z", Z)]:
        # E0 分母
        for tag, mk in [("lin", lambda: lin(1.0)), ("mlp", mlp)]:
            h, t = cv(X, lab_known, mk)
            R.setdefault((sp, "E0", tag), []).append(h)
            R.setdefault((sp, "E0", tag, "tr"), []).append(t)
        # E1 六把刀
        W = []
        for k in range(NC):
            w = wvec(X[kn], (y[kn] != k).astype(int)); W.append(w)
            sel = (kn & (y != k)) | un
            R.setdefault((sp, "E1", k), []).append(roc_auc_score(un[sel].astype(int), -(X[sel] @ w)))
            for Cc in [0.01, 0.1, 10.0]:
                w2 = wvec(X[kn], (y[kn] != k).astype(int), Cc)
                R.setdefault((sp, "E1C", k, Cc), []).append(roc_auc_score(un[sel].astype(int), -(X[sel] @ w2)))
        W = np.stack(W)
        wp = wvec(X, lab_known)                      # w_person（分析對象、非方法）
        # E2 ρ
        Q, _ = np.linalg.qr(W.T)
        R.setdefault((sp, "E2"), []).append(float(np.linalg.norm(Q @ (Q.T @ wp)) ** 2))
        # E3
        wb = W.mean(0)
        R.setdefault((sp, "E3wb"), []).append(float(np.linalg.norm(wb)))
        off = ~np.eye(NC, dtype=bool)
        R.setdefault((sp, "E3kj"), []).append([float(np.abs(W @ W.T)[off].mean()), float(np.abs(W @ W.T)[off].max())])
        ckp = np.abs(W @ wp)
        R.setdefault((sp, "E3kp"), []).append([float(ckp.mean()), float(ckp.max())])
        R.setdefault((sp, "E3kp_all"), []).append(ckp.tolist())
        # E4 隨機基準
        D = X.shape[1]
        Rv = nrm(rng.standard_normal((NRAND, D)))
        au = np.array([roc_auc_score(un.astype(int), -(X @ r)) for r in Rv])
        R.setdefault((sp, "E4a"), []).append(np.percentile(au, [50, 95, 99]).tolist() + [au.max()])
        rho = []
        for _ in range(NRAND):
            Qr, _ = np.linalg.qr(rng.standard_normal((D, NC)))
            rho.append(float(np.linalg.norm(Qr @ (Qr.T @ wp)) ** 2))
        R.setdefault((sp, "E4r"), []).append(np.percentile(rho, [50, 95, 99]).tolist() + [max(rho)])
        DIRS[f"n{i}_{sp}_W"] = W; DIRS[f"n{i}_{sp}_wp"] = wp
        PS[f"n{i}_{sp}_sE1"] = np.stack([-(X @ w) for w in W]).astype(np.float32)
        PS[f"n{i}_{sp}_sE0"] = (-(X @ wp)).astype(np.float32)   # ⚠️ 全資料訓練的分數，非 held-out，勿當 E0 用
    PS[f"n{i}_y"] = y.astype(np.int16)
    print(f"  node{i} done", flush=True)

np.savez_compressed(f"{OUTD}/directions.npz", **DIRS)
np.savez_compressed(f"{OUTD}/per_sample_scores.npz", **PS)
m = lambda k: np.mean(R[k], axis=0); sd = lambda k: np.std(R[k], axis=0)
out = {}
print("\n" + "=" * 94)
print("表 A ── 健全性檢查")
print(f"  2 ‖Uᵀz⊥‖ max            {np.max(SAN['orth']):.2e}   預期 <1e-5   {'✅' if np.max(SAN['orth'])<1e-5 else '❌'}")
print(f"  3 ‖z∥‖²+‖z⊥‖² 偏差 max   {np.max(SAN['decomp']):.2e}   預期 ≈1     {'✅' if np.max(SAN['decomp'])<1e-4 else '❌'}")
na = float(np.mean(SAN['normauc'])); print(f"  4 ‖z⊥‖ 單獨 AUROC        {na:.4f}   預期 ≤0.8264  {'✅' if na<=0.8264 else '❌'}")
nm = np.mean(SAN['norm_m'], 0); print(f"    ‖z⊥‖ 均值：②{nm[0]:.4f} / ③{nm[1]:.4f}")
gaps = [abs(m((sp,'E0',t,'tr'))-m((sp,'E0',t))) for sp in ['z_perp','z'] for t in ['lin','mlp']]
print(f"  5 最大 overfit gap       {max(gaps):.4f}   預期 <0.10   {'✅' if max(gaps)<0.10 else '❌'}")
print(f"  1 完整 Z 線性 held-out   {m(('z','E0','lin')):.4f}   預期 0.9495±0.005  {'✅' if abs(m(('z','E0','lin'))-0.9495)<=0.005 else '❌'}")
print(f"  6 標準化                 Pipeline 內 fit ✅")

for sp, nm_ in [("z_perp", "z⊥（122 維）"), ("z", "z（128 維，未扣原型）")]:
    D = 122 if sp == "z_perp" else 128
    e1 = [m((sp, "E1", k)) for k in range(NC)]; e1s = [sd((sp, "E1", k)) for k in range(NC)]
    e0 = m((sp, "E0", "lin"))
    print("\n" + "=" * 94); print(f"【{nm_}】")
    print(f"表 B  E0 分母：線性 held-out {e0:.4f}（train {m((sp,'E0','lin','tr')):.4f}）"
          f"  MLP held-out {m((sp,'E0','mlp')):.4f}（train {m((sp,'E0','mlp','tr')):.4f}）")
    print("表 C  E1 六把刀（有號 AUROC、9 節點 mean±std）")
    for k in range(NC):
        rng_c = [m((sp,'E1C',k,c)) for c in [0.01,0.1,10.0]] + [e1[k]]
        print(f"   {CLS[k]:<9} {e1[k]:.4f} ± {e1s[k]:.4f}    C 敏感度 {min(rng_c):.3f}–{max(rng_c):.3f}")
    print(f"   {'平均':<9} {np.mean(e1):.4f}      {'最大':<6} {max(e1):.4f}"
          f"      ★轉移率(max÷E0) {max(e1)/e0:.3f}")
    print(f"   符號一致性：{sum(1 for v in e1 if v>0.5)} 個 >0.5、{sum(1 for v in e1 if v<0.5)} 個 <0.5")
    r2 = m((sp, "E2")); r4r = m((sp, "E4r")); r4a = m((sp, "E4a"))
    print(f"表 D  E2 ρ = {r2:.4f} ± {sd((sp,'E2')):.4f}   解析零點 {NC/D:.4f}   倍數 {r2/(NC/D):.2f}")
    print(f"表 E  E3 ‖w̄‖ = {m((sp,'E3wb')):.4f}（隨機零點 0.408｜代數預測 ≈0）")
    kj = m((sp,'E3kj')); kp = m((sp,'E3kp'))
    print(f"       cos(w_k,w_j) 平均{kj[0]:.4f} 最大{kj[1]:.4f}   cos(w_k,w_person) 平均{kp[0]:.4f} 最大{kp[1]:.4f}"
          f"   隨機零點 {1/np.sqrt(D):.4f}")
    print(f"       cos(w_k,w_person) 逐類：{[f'{x:.3f}' for x in np.mean(R[(sp,'E3kp_all')],0)]}")
    print(f"表 F  E4 隨機方向 AUROC  P50 {r4a[0]:.4f}  P95 {r4a[1]:.4f}  P99 {r4a[2]:.4f}  max {r4a[3]:.4f}")
    print(f"       E4 隨機子空間 ρ   P50 {r4r[0]:.4f}  P95 {r4r[1]:.4f}  P99 {r4r[2]:.4f}  max {r4r[3]:.4f}")
    out[sp] = dict(E0_lin=e0, E0_lin_tr=m((sp,'E0','lin','tr')), E0_mlp=m((sp,'E0','mlp')),
                   E1=e1, E1_std=[float(x) for x in e1s], E1_max=max(e1), transfer=max(e1)/e0,
                   E2_rho=r2, E2_null=NC/D, E3_wbar=m((sp,'E3wb')), E3_kj=kj, E3_kp=kp,
                   E4_auc=r4a, E4_rho=r4r)
json.dump(json.loads(json.dumps({"sanity": SAN, "results": out}, default=lambda o: o.tolist() if hasattr(o,"tolist") else float(o))), open(f"{OUTD}/summary.json","w"), indent=1, ensure_ascii=False)
print("\n" + "=" * 94)
print(f"落盤 → {OUTD}/ (summary.json, directions.npz, per_sample_scores.npz)")
