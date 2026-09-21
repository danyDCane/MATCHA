"""把 0.6641 → 0.8993 的增益拆開：是 ② 靠近了面，還是 ③ 遠離了面？

【為什麼這個分解決定一切】
  若增益主要來自 **② 靠近**（已知類別在來源面外、在 cartoon 面內）
     ⇒ 病灶是「已知類別換畫風後跑出面外」⇒ 訓練目標＝讓已知類別跨畫風都留在面內
     ⇒ **只需要已知類別的資料，繞開了「要先知道誰是 person」的循環** ✅
  若增益主要來自 **③ 遠離**
     ⇒ 病灶跟 person 的位置有關 ⇒ 免標籤做不到 ⇒ 這條也死

同時量兩個面的主角（principal angles），確認「面到底有沒有轉」。
"""
import os
os.environ.setdefault("OMP_NUM_THREADS","4"); os.environ.setdefault("MKL_NUM_THREADS","4")
import numpy as np
from scipy.stats import rankdata
from scipy.linalg import subspace_angles

N, UNK, EPS, K = 9, 6, 1e-8, 20
DOM = ["art"]*3+["photo"]*3+["sketch"]*3
F = np.load("logs/prototype_probe/0826_features_full.npz")
nrm = lambda X: X/np.maximum(np.linalg.norm(X,axis=-1,keepdims=True),EPS)
def auroc(pos,neg):
    a=np.concatenate([pos,neg]); r=rankdata(a,method="average")
    return float((r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
basis = lambda X,k: np.linalg.svd(X, full_matrices=False)[2][:k].T
def out_frac(u,B): return np.sqrt(np.maximum(1-np.linalg.norm(u@B,axis=1)**2,0))

rows=[]
for i in range(N):
    U,_ = np.linalg.qr(F[f"n{i}_C"].astype(np.float32).T)
    d={}
    for sp in ("src","tgt"):
        Z=nrm(F[f"n{i}_{sp}_z"].astype(np.float32)); zp=Z-(Z@U)@U.T
        d[sp]=dict(u=zp/np.maximum(np.linalg.norm(zp,axis=1),EPS)[:,None], y=F[f"n{i}_{sp}_y"].astype(int))
    s,t=d["src"],d["tgt"]; m2,m3 = t["y"]!=UNK, t["y"]==UNK
    # 每類一個面，分數取 min
    def score(src, mask, ys):
        cs=[out_frac(t["u"], basis(src[mask & (ys==c)], K)) for c in range(6)
            if (mask & (ys==c)).sum()>K]
        return np.min(cs,axis=0)
    sc_src = score(s["u"], s["y"]!=UNK, s["y"])          # 用 ① 算的面
    sc_ora = score(t["u"], m2,          t["y"])          # 用 ② 算的面（oracle）
    # 主角：① 的面 vs ② 的面（整體，非逐類）
    Bs, Bo = basis(s["u"][s["y"]!=UNK],K), basis(t["u"][m2],K)
    ang = np.degrees(subspace_angles(Bs,Bo))
    rows.append(dict(node=i, dom=DOM[i],
        a_src=auroc(sc_src[m3],sc_src[m2]), a_ora=auroc(sc_ora[m3],sc_ora[m2]),
        d2_src=sc_src[m2].mean(), d3_src=sc_src[m3].mean(),
        d2_ora=sc_ora[m2].mean(), d3_ora=sc_ora[m3].mean(),
        ang_mean=ang.mean(), ang_max=ang.max(), ang_min=ang.min()))

W=100; mn=lambda k: np.mean([r[k] for r in rows])
print("="*W); print(f"面的來源換掉之後，② 和 ③ 各自到面的距離怎麼變（k={K}、9 節點平均）"); print("="*W)
print(f"{'':<26}{'用①算的面':>14}{'用②算的面':>14}{'Δ':>12}")
print("-"*W)
print(f"{'② 已知×cartoon 到面的距離':<26}{mn('d2_src'):14.4f}{mn('d2_ora'):14.4f}{mn('d2_ora')-mn('d2_src'):+12.4f}")
print(f"{'③ person 到面的距離':<26}{mn('d3_src'):14.4f}{mn('d3_ora'):14.4f}{mn('d3_ora')-mn('d3_src'):+12.4f}")
print(f"{'兩者的差距（③−②）':<26}{mn('d3_src')-mn('d2_src'):14.4f}{mn('d3_ora')-mn('d2_ora'):14.4f}"
      f"{(mn('d3_ora')-mn('d2_ora'))-(mn('d3_src')-mn('d2_src')):+12.4f}")
print(f"{'部署 AUROC':<26}{mn('a_src'):14.4f}{mn('a_ora'):14.4f}{mn('a_ora')-mn('a_src'):+12.4f}")
print("-"*W)
g2, g3 = mn('d2_src')-mn('d2_ora'), mn('d3_src')-mn('d3_ora')
tot = abs(g2)+abs(g3)
print(f"  ② 靠近面的量：{g2:+.4f}   ③ 靠近面的量：{g3:+.4f}")
print(f"  ⇒ 增益來源分解：② 貢獻 {abs(g2)/tot*100:.1f}%   ③ 貢獻 {abs(g3)/tot*100:.1f}%")
print()
print("="*W); print("兩個面之間的主角（principal angles，20 維）"); print("="*W)
print(f"  最小 {mn('ang_min'):.1f}°   平均 {mn('ang_mean'):.1f}°   最大 {mn('ang_max'):.1f}°")
print(f"  （0° ＝ 兩個面完全重合；90° ＝ 完全正交）")
print()
print("逐來源畫風：")
for dm in ["art","photo","sketch"]:
    sub=[r for r in rows if r["dom"]==dm]; g=lambda k: np.mean([r[k] for r in sub])
    print(f"  {dm:<8} AUROC {g('a_src'):.4f}→{g('a_ora'):.4f} │ ②距離 {g('d2_src'):.4f}→{g('d2_ora'):.4f}"
          f" │ ③距離 {g('d3_src'):.4f}→{g('d3_ora'):.4f} │ 主角均 {g('ang_mean'):.1f}°")
print("="*W)
