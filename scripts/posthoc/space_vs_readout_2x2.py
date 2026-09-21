"""dany 2026-08-29 的問題：energy 也是 512→6→1，為什麼它丟得比我們好？
把「用哪個空間」與「用哪種讀出形式」拆成獨立兩軸（兩個對角已知，補其餘格）。
基底：0826_features_full.npz（BN 平均 B）。錨點：fc energy 0.8380、現行原型 0.8145。
"""
import numpy as np
F = np.load("logs/prototype_probe/0826_features_full.npz")
N, UNK, DEG = 9, 6, 57.29577951308232
nrm = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)
def auroc(pos, neg):
    a = np.concatenate([pos, neg]); r = a.argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))

def evaluate(fn):
    st, dp = [], []
    for i in range(N):
        s1, s2, s3 = fn(i); st.append(auroc(s2, s1)); dp.append(auroc(s3, s2))
    return np.mean(st), np.mean(dp)

def pack(i, sc_src, sc_tgt):
    Ys, Yt = F[f"n{i}_src_y"], F[f"n{i}_tgt_y"]
    return sc_src[Ys != UNK], sc_tgt[Yt != UNK], sc_tgt[Yt == UNK]

def get(i, space):
    k = "h" if space == "h512" else "z"
    return nrm(F[f"n{i}_src_{k}"].astype(np.float64)), nrm(F[f"n{i}_tgt_{k}"].astype(np.float64))

def centers(i, space, kind):
    if kind == "proto": return nrm(F[f"n{i}_C"].astype(np.float64))
    Ys = F[f"n{i}_src_y"]; Hs, _ = get(i, space)
    return nrm(np.stack([Hs[Ys == c].mean(0) for c in range(6)]))

print("="*102)
print("空間 × 讀出形式（cartoon fold、9 節點 node-mean、BN 平均 B）")
print("="*102)
print(f"{'空間':<8}{'中心來源':<16}{'讀出形式':<24}{'畫風↓0.5':>10}{'★部署↑':>10}")
print("-"*102)
for nm_, f_ in [("energy = −logsumexp", lambda lo: -np.logaddexp.reduce(lo, 1)),
                ("msp", lambda lo: -np.exp(lo - np.logaddexp.reduce(lo, 1)[:, None]).max(1))]:
    st, dp = evaluate(lambda i: pack(i, f_(F[f"n{i}_src_lo"].astype(np.float64)),
                                        f_(F[f"n{i}_tgt_lo"].astype(np.float64))))
    print(f"{'512維':<8}{'fc 權重':<16}{nm_:<24}{st:10.4f}{dp:10.4f}   ← 錨點")
print("-"*102)
for space, lbl in [("h512", "512維"), ("z128", "128維")]:
    kinds = ["proto", "srcmean"] if space == "z128" else ["srcmean"]
    for kind in kinds:
        klbl = "訓練原型" if kind == "proto" else "來源域類別均值"
        C = {i: centers(i, space, kind) for i in range(N)}
        def sim(i):
            Hs, Ht = get(i, space); return Hs @ C[i].T, Ht @ C[i].T
        st, dp = evaluate(lambda i: pack(i, *[np.arccos(np.clip(S, -1+1e-12, 1-1e-12)).min(1)*DEG for S in sim(i)]))
        print(f"{lbl:<8}{klbl:<16}{'min 角距離（現行）':<24}{st:10.4f}{dp:10.4f}")
        best = None
        for T in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
            st2, dp2 = evaluate(lambda i: pack(i, *[-np.logaddexp.reduce(S/T, 1) for S in sim(i)]))
            if best is None or dp2 > best[2]: best = (T, st2, dp2)
        print(f"{'':<8}{'':<16}{f'−logsumexp (最佳T={best[0]})':<24}{best[1]:10.4f}{best[2]:10.4f}")
        st3, dp3 = evaluate(lambda i: pack(i, *[-S.max(1) for S in sim(i)]))
        print(f"{'':<8}{'':<16}{'−max 相似度':<24}{st3:10.4f}{dp3:10.4f}")
    print("-"*102)
