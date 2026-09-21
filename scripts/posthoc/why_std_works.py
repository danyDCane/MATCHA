"""dany 2026-08-29：−std(logit) 0.8656 為什麼好？能不能搬到我們的讀出？
0820 提出的機制（「畫風位移＝所有 logit 一起平移，std 對此免疫」）當年沒量過。這裡量它。
六維向量 v 拆成【共同高度 m·1】＋【輪廓形狀 r】；比較畫風變化(①→②)與類別變化(②→③)各落在哪邊。
基底：0826_features_full.npz（BN 平均 B）。自檢錨點：−std(logit) 0.8656、−std(六距離) 0.8264。
"""
import numpy as np
F = np.load("logs/prototype_probe/0826_features_full.npz")
N, UNK, DEG = 9, 6, 57.29577951308232
nrm = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)
def auroc(pos, neg):
    a = np.concatenate([pos, neg]); r = a.argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))

def three(i, V_src, V_tgt, fn):
    Ys, Yt = F[f"n{i}_src_y"], F[f"n{i}_tgt_y"]
    s1, st = fn(V_src), fn(V_tgt)
    return s1[Ys != UNK], st[Yt != UNK], st[Yt == UNK]

SP = {}
for i in range(N):
    lo_s, lo_t = F[f"n{i}_src_lo"].astype(np.float64), F[f"n{i}_tgt_lo"].astype(np.float64)
    C = nrm(F[f"n{i}_C"].astype(np.float64))
    ds = np.arccos(np.clip(nrm(F[f"n{i}_src_z"].astype(np.float64)) @ C.T, -1+1e-12, 1-1e-12))*DEG
    dt = np.arccos(np.clip(nrm(F[f"n{i}_tgt_z"].astype(np.float64)) @ C.T, -1+1e-12, 1-1e-12))*DEG
    SP[i] = {"logit": (lo_s, lo_t), "dist": (ds, dt)}

print("="*98); print("§0 自檢：重現 0820 的兩個錨點"); print("="*98)
for nm_, key, f_ in [("−std(logit)", "logit", lambda V: -V.std(1)),
                     ("energy=−logsumexp(logit)", "logit", lambda V: -np.logaddexp.reduce(V, 1)),
                     ("−std(六個角距離)", "dist", lambda V: -V.std(1)),
                     ("min 角距離（現行）", "dist", lambda V: V.min(1))]:
    st, dp = [], []
    for i in range(N):
        a, b, c = three(i, *SP[i][key], f_); st.append(auroc(b, a)); dp.append(auroc(c, b))
    ref = {"−std(logit)": .8656, "energy=−logsumexp(logit)": .8380, "−std(六個角距離)": .8264, "min 角距離（現行）": .8145}[nm_]
    ok = "✅" if abs(np.mean(dp)-ref) < 0.004 else "⚠️"
    print(f"  {nm_:<26} 畫風 {np.mean(st):.4f}   ★部署 {np.mean(dp):.4f}   (0820/既有 {ref:.4f}) {ok}")

print(); print("="*98)
print("★ 機制檢驗：六維向量拆成【共同高度】＋【輪廓形狀】，各堆的變化落在哪邊")
print("="*98)
print(f"{'空間':<12}{'變化':<26}{'共同高度 |Δm|·√6':>18}{'輪廓形狀 ‖Δr‖':>16}{'形狀佔比':>10}")
print("-"*98)
for key, lbl in [("logit", "logit 空間"), ("dist", "六個角距離")]:
    rows = {}
    for i in range(N):
        Vs, Vt = SP[i][key]; Ys, Yt = F[f"n{i}_src_y"], F[f"n{i}_tgt_y"]
        g = {"①": Vs[Ys != UNK], "②": Vt[Yt != UNK], "③": Vt[Yt == UNK]}
        mu = {k: v.mean(0) for k, v in g.items()}
        for a, b, nm_ in [("①", "②", "①→② 只換畫風"), ("②", "③", "②→③ 只換類別")]:
            d = mu[b] - mu[a]; m = d.mean(); r = d - m
            rows.setdefault(nm_, []).append([abs(m)*np.sqrt(6), np.linalg.norm(r)])
    for nm_, v in rows.items():
        a, b = np.mean(v, 0)
        print(f"{lbl:<12}{nm_:<26}{a:18.4f}{b:16.4f}{b/(a+b)*100:9.1f}%")
    print("-"*98)

print(); print("="*98)
print("★★ 追問：既然畫風動的也是形狀，那 std 贏在哪？拆「共同高度」與「用幾個數字」")
print("="*98)
print(f"{'空間':<12}{'分數':<34}{'畫風↓0.5':>10}{'★部署↑':>10}")
print("-"*98)
def run(key, lbl, items):
    for nm_, f_ in items:
        st, dp = [], []
        for i in range(N):
            a, b, c = three(i, *SP[i][key], f_); st.append(auroc(b, a)); dp.append(auroc(c, b))
        print(f"{lbl:<12}{nm_:<34}{np.mean(st):10.4f}{np.mean(dp):10.4f}")
    print("-"*98)

R = lambda V: V - V.mean(1, keepdims=True)          # 扣掉共同高度＝輪廓形狀
ent = lambda V: -( lambda p: -(p*np.log(p+1e-12)).sum(1) )(np.exp(V-np.logaddexp.reduce(V,1)[:,None]))
run("logit", "logit 空間", [
    ("共同高度 m 單獨（+mean）",        lambda V: V.mean(1)),
    ("共同高度 m 單獨（−mean）",        lambda V: -V.mean(1)),
    ("−std＝形狀幅度（用全部 6 個）",     lambda V: -V.std(1)),
    ("−max(r)＝形狀最高峰（只用 1 個）",  lambda V: -R(V).max(1)),
    ("min(r)＝形狀最低谷（只用 1 個）",   lambda V: R(V).min(1)),
    ("熵（用全部 6 個）",                ent),
])
run("dist", "六個角距離", [
    ("共同高度 m 單獨（mean 距離）",     lambda V: V.mean(1)),
    ("−std＝輪廓幅度（用全部 6 個）",     lambda V: -V.std(1)),
    ("min − mean（0820 測過 .8148）",   lambda V: V.min(1) - V.mean(1)),
    ("−max(r)（只用 1 個）",            lambda V: -R(V).max(1)),
    ("min 角距離（現行、只用 1 個）",     lambda V: V.min(1)),
])

print(); print("="*98)
print("★★★ 由上表推出的沒測過的組合：距離空間裡「平均高度」本身有 0.7536，而 −std 把它扣掉了")
print("     ⇒ 該把兩者合起來（0820 掃的是 min−α·std，沒掃過 mean−α·std / ‖z⊥‖−α·std）")
print("="*98)
Zp = {}
for i in range(N):
    C = nrm(F[f"n{i}_C"].astype(np.float64)); U = np.linalg.svd(C.T, full_matrices=False)[0]
    f = lambda k: (lambda Z: np.sqrt(np.maximum(1-((Z@U)**2).sum(1), 0)))(nrm(F[f"n{i}_{k}_z"].astype(np.float64)))
    Zp[i] = (f("src"), f("tgt"))
def combo(base, alpha):
    st, dp = [], []
    for i in range(N):
        Ys, Yt = F[f"n{i}_src_y"], F[f"n{i}_tgt_y"]
        out = []
        for j, (V, P) in enumerate([(SP[i]["dist"][0], Zp[i][0]), (SP[i]["dist"][1], Zp[i][1])]):
            b = V.mean(1)/V.mean() if base == "mean" else P/P.mean()
            out.append(b - alpha*V.std(1)/V.std())
        s1, st_ = out; a, b_, c = s1[Ys != UNK], st_[Yt != UNK], st_[Yt == UNK]
        st.append(auroc(b_, a)); dp.append(auroc(c, b_))
    return np.mean(st), np.mean(dp)
print(f"{'分數':<34}{'α':>6}{'畫風↓0.5':>10}{'★部署↑':>10}")
print("-"*98)
for base, lbl in [("mean", "平均角距離 − α·std"), ("zperp", "‖z⊥‖ − α·std")]:
    for a_ in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        s, d = combo(base, a_)
        print(f"{lbl:<34}{a_:6.2f}{s:10.4f}{d:10.4f}{'   ← 贏 energy 0.8380' if d > 0.8380 else ''}")
    print("-"*98)
print("⚠️ α 在同一批資料上掃出來的、無 held-out（與 0820 同口徑）⇒ 是上界不是可宣稱的成績。")
