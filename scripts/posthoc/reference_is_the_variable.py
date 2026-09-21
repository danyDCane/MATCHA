"""決定性一格：128 維空間裡，把「六個參照物」從【類別中心】換成【CE 學出來的判別權重】。
訓練只用該節點【來源域的已知類別】(訓練時本來就有)，person 標籤全程不碰；在 cartoon 上評估。
若能逼近 0.8380 ⇒ 病灶是參照物，不是 128 維空間的形狀。
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
F = np.load("logs/prototype_probe/0826_features_full.npz")
N, UNK = 9, 6
nrm = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)
def auroc(pos, neg):
    a = np.concatenate([pos, neg]); r = a.argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))
E = lambda lo: -np.logaddexp.reduce(lo, 1)

print("="*96)
print("128 維空間、六個參照物換成 CE 學出來的判別權重（energy 讀出）｜BN 平均 B、node-mean")
print("="*96)
for space, lbl in [("z", "128 維 Z"), ("h", "512 維 h")]:
    for Cst in [0.01, 0.1, 1.0]:
        st, dp = [], []
        for i in range(N):
            Xs, Ys = nrm(F[f"n{i}_src_{space}"].astype(np.float64)), F[f"n{i}_src_y"]
            Xt, Yt = nrm(F[f"n{i}_tgt_{space}"].astype(np.float64)), F[f"n{i}_tgt_y"]
            m = Ys != UNK
            clf = LogisticRegression(C=Cst, max_iter=3000).fit(Xs[m], Ys[m])
            ss, stg = E(clf.decision_function(Xs)), E(clf.decision_function(Xt))
            st.append(auroc(stg[Yt != UNK], ss[m])); dp.append(auroc(stg[Yt == UNK], stg[Yt != UNK]))
        print(f"  {lbl}  CE 判別權重 (C={Cst:<5})   畫風 {np.mean(st):.4f}   ★部署 {np.mean(dp):.4f}")
print("-"*96)
print("  對照錨點：fc(512維) energy 0.8380 ｜ 128 維類別中心 min 角距離 0.8145 ｜ 512 維類別中心 0.8164")
print("="*96)

print()
print("="*96)
print("追問：原 fc 的 0.8380 靠的是【判別權重】還是【未被正規化掉的長度】？")
print("="*96)
for space, lbl in [("h", "512 維 h"), ("z", "128 維 Z")]:
    for raw in [True, False]:
        best = (None, -1, -1)
        for Cst in [0.01, 0.1, 1.0, 10.0]:
            st, dp = [], []
            for i in range(N):
                Xs0, Ys = F[f"n{i}_src_{space}"].astype(np.float64), F[f"n{i}_src_y"]
                Xt0, Yt = F[f"n{i}_tgt_{space}"].astype(np.float64), F[f"n{i}_tgt_y"]
                Xs, Xt = (Xs0, Xt0) if raw else (nrm(Xs0), nrm(Xt0))
                m = Ys != UNK
                clf = LogisticRegression(C=Cst, max_iter=5000).fit(Xs[m], Ys[m])
                ss, stg = E(clf.decision_function(Xs)), E(clf.decision_function(Xt))
                st.append(auroc(stg[Yt != UNK], ss[m])); dp.append(auroc(stg[Yt == UNK], stg[Yt != UNK]))
            if np.mean(dp) > best[2]: best = (Cst, np.mean(st), np.mean(dp))
        print(f"  {lbl}  {'未正規化（保留長度）' if raw else '已 L2 正規化（丟掉長度）'}  "
              f"CE 判別權重 最佳C={best[0]:<5}  畫風 {best[1]:.4f}  ★部署 {best[2]:.4f}")
print("-"*96)
print("  ★ 原 fc（與 backbone 共同訓練、吃未正規化 h）= 0.8380")
print("="*96)
