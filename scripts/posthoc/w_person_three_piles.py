"""沿 w_person 這一條軸，把三堆重新量一次（dany 2026-08-27 提議）

★ 為什麼這個先做：前一輪只證明「w_person 在【① 來源域】上最安靜」。
   但 ② cartoon 已知類別**也是正常資料**，而它從沒被測過。
   換畫風若也把這條軸吵起來 ⇒ 白化放大它的同時，cartoon 正常圖片也沿著它移動 ⇒ 誤拒 ⇒ 整條路死。

★ 主判準（事前寫死）：**①vs② 的畫風 AUROC**（理想 0.5＝這條軸看不出畫風）
   ≈0.5 ⇒ 純語意軸、白化這條路活｜0.6–0.7 部分污染｜≥0.75 跟現行讀出(0.6723)一樣髒 ⇒ 救不了

單位：以 ① 的均值為原點、① 的標準差為 1（z-score），三堆才可比。
對照：同樣三個量在 energy 與現行原型分數上算。
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from sklearn.metrics import roc_auc_score

F = np.load("logs/prototype_probe/0826_features_full.npz")
D = np.load("research/outputs/0826_probe_transfer/directions.npz")
N = 9; UNK = 6
nrm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
A = {}

for i in range(N):
    ys = F[f"n{i}_src_y"].astype(int); Zs = nrm(F[f"n{i}_src_z"].astype(np.float32))
    yt = F[f"n{i}_tgt_y"].astype(int); Zt = nrm(F[f"n{i}_tgt_z"].astype(np.float32))
    Lt = F[f"n{i}_tgt_lo"].astype(np.float32); Ls = F[f"n{i}_src_lo"].astype(np.float32)
    C = nrm(F[f"n{i}_C"].astype(np.float32))
    P1 = Zs[ys != UNK]; P2 = Zt[yt != UNK]; P3 = Zt[yt == UNK]
    w = D[f"n{i}_z_wp"]; w = w / np.linalg.norm(w)

    for tag, s1, s2, s3 in [
        ("w_person", P1 @ w, P2 @ w, P3 @ w),
        ("energy",   -np.log(np.exp(Ls[ys != UNK]).sum(1)),
                     -np.log(np.exp(Lt[yt != UNK]).sum(1)),
                     -np.log(np.exp(Lt[yt == UNK]).sum(1))),
        ("原型角距離", np.degrees(np.arccos(np.clip(P1 @ C.T, -1+1e-7, 1-1e-7))).min(1),
                     np.degrees(np.arccos(np.clip(P2 @ C.T, -1+1e-7, 1-1e-7))).min(1),
                     np.degrees(np.arccos(np.clip(P3 @ C.T, -1+1e-7, 1-1e-7))).min(1))]:
        mu, sg = s1.mean(), s1.std()
        z1, z2, z3 = (s1-mu)/sg, (s2-mu)/sg, (s3-mu)/sg    # 以①為原點、①的std為單位
        A.setdefault((tag, "m"), []).append([z1.mean(), z2.mean(), z3.mean()])
        A.setdefault((tag, "v"), []).append([z1.var(), z2.var(), z3.var()])
        A.setdefault((tag, "auc12"), []).append(roc_auc_score([0]*len(z1)+[1]*len(z2), np.r_[z1, z2]))
        A.setdefault((tag, "auc23"), []).append(roc_auc_score([0]*len(z2)+[1]*len(z3), np.r_[z2, z3]))
        A.setdefault((tag, "auc13"), []).append(roc_auc_score([0]*len(z1)+[1]*len(z3), np.r_[z1, z3]))
        A.setdefault((tag, "dprime"), []).append((z3.mean()-z2.mean())/np.sqrt((z2.var()+z3.var())/2))
    A.setdefault("raw_var", []).append([(P1@w).var(), (P2@w).var(), (P3@w).var()])
    print(f"  node{i} done", flush=True)

m = lambda k: np.mean(A[k], axis=0)
print("\n" + "=" * 96)
print("★ 沿各條軸把三堆重新量（以 ① 的均值為 0、① 的標準差為 1）")
print(f"{'軸':<12}{'①均值':>8}{'②均值':>8}{'③均值':>8} | {'①變異':>8}{'②變異':>8}{'③變異':>8} | "
      f"{'畫風①②':>9}{'部署②③':>9}{'語意①③':>9}{'d′(②③)':>9}")
print("-" * 96)
for tag in ["w_person", "energy", "原型角距離"]:
    mm, vv = m((tag, "m")), m((tag, "v"))
    print(f"{tag:<12}{mm[0]:>8.3f}{mm[1]:>8.3f}{mm[2]:>8.3f} | {vv[0]:>8.3f}{vv[1]:>8.3f}{vv[2]:>8.3f} | "
          f"{m((tag,'auc12')):>9.4f}{m((tag,'auc23')):>9.4f}{m((tag,'auc13')):>9.4f}{m((tag,'dprime')):>9.3f}")
print("-" * 96)
rv = m("raw_var")
print(f"w_person 的原始變異數（未 z-score）：①{rv[0]:.6f}  ②{rv[1]:.6f}  ③{rv[2]:.6f}"
      f"   ⇒ ②/① = {rv[1]/rv[0]:.2f}倍")
print(f"\n判準：畫風 AUROC(①②) ≈0.5 純語意軸 ✅｜0.6–0.7 部分污染 ⚠️｜≥0.75 跟現行讀出一樣髒 ⛔")
print(f"參照：現行原型讀出的畫風 AUROC ＝ 0.6723（TaskBoard）；部署靶 energy 0.8380、誠實靶 0.8656")
print("=" * 96)
