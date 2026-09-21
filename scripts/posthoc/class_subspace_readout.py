"""「離參照物遠」這一族還有救嗎？—— 把參照物從【點】換成【面】。

【由來】0903 測到：person 的方向比已知類別更集中（0.44 vs 0.27），
  所以「離平均方向遠 = 可疑」被系統性翻盤（用對的參考方向 AUROC 只有 0.4528 < 0.5）。
  但平均方向是**一個點**，只能表達「中心在哪」；子空間能表達「散開」。
【本輪問】已知類別雖散但散在一個低維面上；person 雖集中，它集中在面內還是面外？
  面外 ⇒ 子空間讀出可繞開這次的陷阱｜面內 ⇒「離參照物遠」整族死透

分數 ＝ sqrt(1 − ‖P_k u‖²)  ＝ u 落在子空間外的比例（高＝可疑）
三種子空間來源，與 0903 的三種參考方向對應，可直接比「點 vs 面」：
  sub-src    用 ①（來源域已知）算    免標籤、可部署
  sub-tta    用全部 cartoon 算       免標籤
  sub-oracle 用 ②（cartoon 已知）算  要標籤 ⇒ 上界
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
from scipy.stats import rankdata

N, UNK, EPS = 9, 6, 1e-8
DOM = ["art"]*3 + ["photo"]*3 + ["sketch"]*3
KS = [1,2,3,5,8,10,15,20]
F = np.load("logs/prototype_probe/0826_features_full.npz")
nrm = lambda X: X/np.maximum(np.linalg.norm(X,axis=-1,keepdims=True), EPS)
def auroc(pos, neg):
    a = np.concatenate([pos,neg]); r = rankdata(a, method="average")
    return float((r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
def basis(X, k):
    """X 的前 k 個主方向（不去均值：我們要的是方向張成，不是變異數結構）"""
    return np.linalg.svd(X, full_matrices=False)[2][:k].T          # (128,k)
def out_frac(u, B):
    p = np.linalg.norm(u @ B, axis=1)
    return np.sqrt(np.maximum(1 - p**2, 0))

D = []
for i in range(N):
    U,_ = np.linalg.qr(F[f"n{i}_C"].astype(np.float32).T)
    d = {}
    for sp in ("src","tgt"):
        Z = nrm(F[f"n{i}_{sp}_z"].astype(np.float32))
        zp = Z-(Z@U)@U.T; npr = np.linalg.norm(zp,axis=1)
        d[sp] = dict(u=zp/np.maximum(npr,EPS)[:,None], npr=npr, y=F[f"n{i}_{sp}_y"].astype(int))
    D.append(d)

def run(src_key, use_mask):
    """回傳 {k: [每節點 AUROC]} for 每類一個面 / 全部一個面"""
    per, whole = {k: [] for k in KS}, {k: [] for k in KS}
    for i in range(N):
        s, t = D[i][src_key], D[i]["tgt"]
        m = use_mask(s)
        ut, yt = t["u"], t["y"]; m3 = yt==UNK
        for k in KS:
            B = basis(s["u"][m], k)
            sc = out_frac(ut, B); whole[k].append(auroc(sc[m3], sc[~m3]))
            if src_key == "tgt" and use_mask is (lambda x: np.ones(len(x["y"]),bool)):
                per[k].append(np.nan); continue
            cs = []
            for c in range(6):
                mc = m & (s["y"]==c)
                if mc.sum() > k: cs.append(out_frac(ut, basis(s["u"][mc], k)))
            sc2 = np.min(cs, axis=0)                                # 到最近類別面的距離
            per[k].append(auroc(sc2[m3], sc2[~m3]))
    return per, whole

KN = lambda x: x["y"]!=UNK
ALL = lambda x: np.ones(len(x["y"]), bool)
res = {}
res["src"]    = run("src", KN)
res["tta"]    = run("tgt", ALL)
res["oracle"] = run("tgt", KN)

W=104; print("="*W); print("到【類別子空間】的距離當分數（②vs③ 部署 AUROC、9 節點平均）"); print("="*W)
print(f"{'k':<5}" + "".join(f"{n:>16}" for n in
      ["src·每類","src·整體","tta·整體","oracle·每類","oracle·整體"]))
print("-"*W)
for k in KS:
    v = [np.nanmean(res["src"][0][k]), np.nanmean(res["src"][1][k]),
         np.nanmean(res["tta"][1][k]), np.nanmean(res["oracle"][0][k]), np.nanmean(res["oracle"][1][k])]
    print(f"{k:<5}" + "".join(f"{x:16.4f}" for x in v))
print("-"*W)
print(f"{'對照':<5}{'‖z⊥‖ 0.8281':>16}{'KNN 0.6461':>16}{'D1 0.6114':>16}{'D1-oracle 0.4528':>18}")

# 逐畫風（最好的 k）
best_k = max(KS, key=lambda k: np.nanmean(res["src"][0][k]))
print(f"\n逐來源畫風（src·每類、k={best_k}）：")
for dm in ["art","photo","sketch"]:
    idx = [i for i in range(N) if DOM[i]==dm]
    print(f"  {dm:<8}{np.mean([res['src'][0][best_k][i] for i in idx]):.4f}")

print(); print("="*W); print("★ 關鍵幾何：person 的集中方向，在不在已知類別的面裡？"); print("="*W)
print(f"{'k':<5}{'ū₃ 落在面內的比例':>20}{'ū₂ 落在面內':>16}{'隨機方向零點':>16}{'判定':>12}")
print("-"*W)
rng = np.random.default_rng(2026)
for k in KS:
    a3, a2, z = [], [], []
    for i in range(N):
        s, t = D[i]["src"], D[i]["tgt"]
        B = basis(s["u"][KN(s)], k)
        u3 = t["u"][t["y"]==UNK].mean(0); u3 /= np.linalg.norm(u3)
        u2 = t["u"][t["y"]!=UNK].mean(0); u2 /= np.linalg.norm(u2)
        a3.append(np.linalg.norm(u3@B)); a2.append(np.linalg.norm(u2@B))
        U,_ = np.linalg.qr(F[f"n{i}_C"].astype(np.float32).T)
        w = rng.standard_normal((200,128)).astype(np.float32); w = nrm(w-(w@U)@U.T)
        z.append(np.mean(np.linalg.norm(w@B, axis=1)))
    m3, m2, mz = np.mean(a3), np.mean(a2), np.mean(z)
    tag = "面內" if m3 > mz*1.5 else ("面外" if m3 < mz*1.2 else "居中")
    print(f"{k:<5}{m3:20.4f}{m2:16.4f}{mz:16.4f}{tag:>12}")
print("="*W)
