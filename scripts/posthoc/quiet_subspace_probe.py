"""安靜子空間：那把好尺在不在裡面？拿得到嗎？（dany 2026-08-27 設計 ＋ 主 agent 補 4 項）

流程（全程零標籤，只用來源域 ①）：
  Σ① = 來源域已知類別的共變異 → 特徵分解 → 128 個方向由吵到安靜排序
  取最安靜的 k 個 → 「安靜子空間」→ 問 w_person 有多少能量落在裡面

★ 主 agent 補的第 2 項（關鍵）：能量落在裡面 ≠ 拿它當偵測器有效。
  那 k 維裡除了 w_person 的成分還有 k−1 個方向的雜訊 ⇒ 必須直接測「用該子空間當分數」的 AUROC。

判準（dany）：k=10 到 70% ⇒ 訊號集中、方法成立｜k=50 才到 70% ⇒ 攤平、雜訊蓋過
              k=50 仍接近基準(39.1%) ⇒ 假線索、白化收掉
判準（主 agent 加）：若第 2 項在任何 k 都打不贏 0.8145 ⇒ 即使能量好看，也只是更精確的天花板
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from sklearn.metrics import roc_auc_score

F = np.load("logs/prototype_probe/0826_features_full.npz")
D = np.load("research/outputs/0826_probe_transfer/directions.npz")
N = 9; UNK = 6; KS = [2, 5, 10, 20, 50, 80]
nrm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
rng = np.random.default_rng(2026)
A = {}

for i in range(N):
    ys = F[f"n{i}_src_y"].astype(int); Zs = nrm(F[f"n{i}_src_z"].astype(np.float32))
    yt = F[f"n{i}_tgt_y"].astype(int); Zt = nrm(F[f"n{i}_tgt_z"].astype(np.float32))
    P1 = Zs[ys != UNK]; P2 = Zt[yt != UNK]; P3 = Zt[yt == UNK]
    lab = np.r_[np.zeros(len(P2)), np.ones(len(P3))]; PA = np.r_[P2, P3]
    C = nrm(F[f"n{i}_C"].astype(np.float32))
    w = D[f"n{i}_z_wp"]; w = w / np.linalg.norm(w)
    Wk = D[f"n{i}_z_W"]; Wk = Wk / np.linalg.norm(Wk, axis=1, keepdims=True)

    mu = P1.mean(0)
    S = np.cov(P1.T)
    ev, V = np.linalg.eigh(S)                    # 升冪：ev[0] 最安靜
    A.setdefault("eig", []).append(ev[[0, 1, 4, 9, 19, 49, 79, 127]].tolist())
    Rv = nrm(rng.standard_normal((200, 128)))
    for k in KS:
        Q = V[:, :k]                             # 最安靜的 k 個方向
        A.setdefault(("E", k), []).append(float((w @ Q) @ (Q.T @ w)))
        A.setdefault(("Ek", k), []).append(float(np.mean([(u @ Q) @ (Q.T @ u) for u in Wk])))
        A.setdefault(("Er", k), []).append(np.percentile([(r @ Q) @ (Q.T @ r) for r in Rv], [50, 95]).tolist())
        # ★ 直接當偵測器：樣本在該子空間的投影長度（先減來源域均值）
        proj = np.linalg.norm((PA - mu) @ Q, axis=1)
        A.setdefault(("AUC", k), []).append(roc_auc_score(lab, proj))
        # 白化版：該子空間內除以各自標準差
        pw = np.linalg.norm(((PA - mu) @ Q) / np.sqrt(np.maximum(ev[:k], 1e-12)), axis=1)
        A.setdefault(("AUCw", k), []).append(roc_auc_score(lab, pw))
    # 完整白化（全 128 維）：馬氏距離到最近類別中心
    Si = V @ np.diag(1.0 / np.sqrt(np.maximum(ev, 1e-10))) @ V.T
    Xw = nrm((PA - mu) @ Si); Cw = nrm((C - mu) @ Si)
    A.setdefault("full_ang", []).append(roc_auc_score(lab, np.arccos(np.clip(Xw @ Cw.T, -1+1e-7, 1-1e-7)).min(1)))
    A.setdefault("full_md", []).append(roc_auc_score(lab, np.linalg.norm((PA - mu) @ Si, axis=1)))
    # 對照：不白化的原型角距離
    A.setdefault("base", []).append(roc_auc_score(lab, np.arccos(np.clip(PA @ C.T, -1+1e-7, 1-1e-7)).min(1)))
    print(f"  node{i} done", flush=True)

m = lambda k: np.mean(A[k], axis=0)
print("\n" + "=" * 98)
print("★ 安靜子空間：w_person 落在裡面多少 ＋ 拿它當偵測器有沒有用")
print(f"{'k':>4}{'w_person 能量':>14}{'隨機 P50':>10}{'隨機 P95':>10}{'六把刀':>9}{'基準 k/128':>11} | "
      f"{'投影長度AUC':>13}{'白化後AUC':>11}{'第k小特徵值':>13}")
print("-" * 98)
eg = m("eig"); idx = {2:1, 5:2, 10:3, 20:4, 50:5, 80:6}
for k in KS:
    er = m(("Er", k))
    print(f"{k:>4}{m(('E',k))*100:>13.1f}%{er[0]*100:>9.1f}%{er[1]*100:>9.1f}%{m(('Ek',k))*100:>8.1f}%"
          f"{k/128*100:>10.1f}% | {m(('AUC',k)):>13.4f}{m(('AUCw',k)):>11.4f}{eg[idx[k]]:>13.2e}")
print("-" * 98)
print(f"完整白化（128 維）：到最近類別中心的角距離 AUROC {m('full_ang'):.4f}   純馬氏距離 {m('full_md'):.4f}")
print(f"對照（不白化）：現行原型角距離 {m('base'):.4f}   ｜靶：energy 0.8380、誠實靶 0.8656")
print(f"特徵值譜（第1/2/5/10/20/50/80/128 小）：{['%.1e'%x for x in eg]}")
print("=" * 98)
