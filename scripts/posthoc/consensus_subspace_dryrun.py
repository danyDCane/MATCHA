"""共識範本 dry run：三種來源畫風各建「已知類別變化清單」，投票取共識，罩不罩得住 cartoon？

【由來】0903 §6：用來源畫風的面時，② 到面 0.7033、person 0.7504——幾乎一樣遠，
  所以讀出分不出來。換成 cartoon 自己的面（oracle）→ 0.8993，但那要標籤。
  ⇒ 訓練目標假設：逼各節點把「已知類別的變化清單」寫成同一份，那份共同清單會罩得住 cartoon。
【本輪問】不開訓練，這句話裡能先驗的部分：
  P1 存在   三份清單有沒有共同條目、幾條？（得票光譜 vs 偽畫風零點）
  P2 轉移   共識清單讀 cartoon 的部署 AUROC？（vs 三個錨）
  P3 選擇性 person 會不會跟著蹭進來？（漏損拆解 + R2 前哨）
【外推等級】全程 C 級快照——量的是「當前特徵上、零訓練的共識面行不行」。
  失敗 ≠ P3 無解（訓練會改特徵）；通過 ≠ 訓練版會成（R1：學到的映射對所有輸入生效）。

分數 ＝ sqrt(1 − ‖P_k u‖²)（面外分量長度），與 class_subspace_readout.py 逐位同一支，
才與錨 0.6641（單畫風·每類）／0.8993（oracle·每類）同基底。全程零重訓、CPU。
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
from scipy.stats import rankdata

N, UNK, EPS = 9, 6, 1e-8
DOM      = ["art"]*3 + ["photo"]*3 + ["sketch"]*3
STYLES   = ["art", "photo", "sketch"]
NODES_OF = {"art": [0,1,2], "photo": [3,4,5], "sketch": [6,7,8]}
K        = 20            # 與兩個錨同維度
LAM_HI   = 0.8           # 共識面門檻：3*0.8-2=0.4 ⇒ 最弱畫風至少出 0.4 的力；「兩有一無」=0.667 進不來
NPERM    = 20            # 偽畫風零點重複次數
SEED     = 2026
F = np.load("logs/prototype_probe/0826_features_full.npz")

nrm = lambda X: X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), EPS)

def auroc(pos, neg):
    a = np.concatenate([pos, neg]); r = rankdata(a, method="average")
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))

def basis(X, k):
    """X 的前 k 個主方向（不去均值：要的是方向張成）—— 與 class_subspace_readout.py 相同"""
    return np.linalg.svd(X, full_matrices=False)[2][:k].T                      # (128,k)

def out_frac(u, B):
    """落在面外的比例（分數高＝可疑）"""
    return np.sqrt(np.maximum(1 - np.linalg.norm(u @ B, axis=1)**2, 0))

# ---------- 載入：殘差方向 u ----------
D = []
for i in range(N):
    U, _ = np.linalg.qr(F[f"n{i}_C"].astype(np.float32).T)                     # 六原型張成
    d = {}
    for sp in ("src", "tgt"):
        Z  = nrm(F[f"n{i}_{sp}_z"].astype(np.float32))
        zp = Z - (Z @ U) @ U.T
        npr = np.linalg.norm(zp, axis=1)
        d[sp] = dict(u=zp/np.maximum(npr, EPS)[:, None], npr=npr,
                     y=F[f"n{i}_{sp}_y"].astype(int))
    D.append(d)

def style_src(i, s):
    """節點 i 眼中畫風 s 的來源資料：自己的畫風用自己的，別人的用該畫風代表節點
    （同畫風三節點特徵逐樣本餘弦 >0.9998 ⇒ 用哪個代表都一樣）"""
    return D[i if DOM[i] == s else NODES_OF[s][0]]["src"]

# ---------- 面的三種建法 ----------
def vote_faces(us_by_style, k=K, lam_hi=LAM_HI):
    """三個畫風各建面 → 投影矩陣取平均 P̄ → 特徵分解
       回傳（投票面 前k、共識面 λ>lam_hi、λ 光譜）
       嚴格交集在 122 維裡一般為空（20+20-122<0）⇒ 用軟交集"""
    Pbar = np.zeros((128, 128), dtype=np.float64)
    n_ok = 0
    for X in us_by_style:
        if len(X) <= k:  continue
        B = basis(X, k); Pbar += B @ B.T; n_ok += 1
    if n_ok == 0: return None, None, None
    Pbar /= n_ok
    lam, V = np.linalg.eigh(Pbar)
    o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]
    return V[:, :k], V[:, lam > lam_hi], lam

def merged_face(us_by_style, k=K):
    """全倒一箱：三個畫風的樣本疊起來，再取前 k 個主方向"""
    X = np.concatenate([a for a in us_by_style if len(a)], 0)
    return basis(X, k) if len(X) > k else None

def score_per_class(ut, faces):
    """每類一個面、取到最近類別面的距離（與錨相同的取法）"""
    cs = [out_frac(ut, B) for B in faces if B is not None and B.shape[1] > 0]
    return np.min(cs, axis=0) if cs else None

W = 108
def hdr(t): print(); print("="*W); print(t); print("="*W)

# ================= §0 自檢 =================
hdr("§0 資料自檢")
print(f"{'節點':<5}{'畫風':<8}" + "".join(f"{c:>7}" for c in range(6)) + f"{'已知合計':>10}{'person':>8}")
print("-"*W)
for i in range(N):
    y = D[i]["src"]["y"]
    cnt = [int((y == c).sum()) for c in range(6)]
    print(f"{i:<5}{DOM[i]:<8}" + "".join(f"{c:>7}" for c in cnt)
          + f"{sum(cnt):>10}{int((y==UNK).sum()):>8}")
print("-"*W)
print(f"最小的一格：sketch·house = {int((D[6]["src"]["y"]==5).sum()):d} 張 ⇒ 建 {K} 維清單邊際，該格結論須單獨標註")
print(f"除零保護觸發數（‖z⊥‖→0）：{sum(int((D[i][sp]['npr'] < EPS).sum()) for i in range(N) for sp in ('src','tgt'))}")

# 座標系對齊（投票的前提）
C = [nrm(F[f"n{i}_C"].astype(np.float64)) for i in range(N)]
al = {f"{a}-{b}": float(np.diag(C[NODES_OF[a][0]] @ C[NODES_OF[b][0]].T).mean())
      for a, b in [("art","photo"), ("art","sketch"), ("photo","sketch")]}
print(f"跨畫風原型對齊（逐類餘弦均值，隨機零點≈0）：" + "  ".join(f"{k} {v:.4f}" for k, v in al.items()))

# 錨重現
anchor = []
for i in range(N):
    s, t = D[i]["src"], D[i]["tgt"]
    faces = [basis(s["u"][s["y"]==c], K) for c in range(6) if (s["y"]==c).sum() > K]
    sc = score_per_class(t["u"], faces)
    anchor.append(auroc(sc[t["y"]==UNK], sc[t["y"]!=UNK]))
print(f"錨重現（單畫風·每類 k=20）：{np.mean(anchor):.4f}　報告值 0.6641　差 {np.mean(anchor)-0.6641:+.4f}")

# ================= 主計算 =================
rng = np.random.default_rng(SEED)
res = {n: [] for n in ["single", "merged", "vote", "cons"]}
lam_real, m_cons, wall, leak, r2 = [], [], [], [], []

for i in range(N):
    t  = D[i]["tgt"]; ut, yt = t["u"], t["y"]; m3 = yt == UNK
    S  = {s: style_src(i, s) for s in STYLES}

    f_single, f_merged, f_vote, f_cons = [], [], [], []
    lam_i, m_i = [], []
    for c in range(6):
        by_style = [S[s]["u"][S[s]["y"] == c] for s in STYLES]
        own      = S[DOM[i]]["u"][S[DOM[i]]["y"] == c]
        if len(own) > K: f_single.append(basis(own, K))
        Bm = merged_face(by_style);              f_merged.append(Bm)
        Bv, Bc, lam = vote_faces(by_style)
        f_vote.append(Bv); f_cons.append(Bc)
        if lam is not None:
            lam_i.append(lam[:30]); m_i.append(int((lam > LAM_HI).sum()))
    lam_real.append(np.mean(lam_i, 0)); m_cons.append(m_i)

    for name, faces in [("single", f_single), ("merged", f_merged),
                        ("vote", f_vote), ("cons", f_cons)]:
        sc = score_per_class(ut, faces)
        res[name].append(auroc(sc[m3], sc[~m3]) if sc is not None else np.nan)

    # --- 漏損拆解：單畫風面 → 投票面，②③ 各自移動多少 ---
    sc_s, sc_v = score_per_class(ut, f_single), score_per_class(ut, f_vote)
    d2 = float(sc_s[~m3].mean() - sc_v[~m3].mean())        # ② 靠近面多少（正=靠近）
    d3 = float(sc_s[m3].mean()  - sc_v[m3].mean())         # ③ 跟著靠近多少
    leak.append((sc_s[~m3].mean(), sc_v[~m3].mean(), sc_s[m3].mean(), sc_v[m3].mean(), d2, d3))

    # --- R2 前哨：person 的集中方向，對單畫風面 vs 對投票面的投影 ---
    u3 = ut[m3].mean(0); u3 /= np.linalg.norm(u3)
    Bs = basis(S[DOM[i]]["u"][S[DOM[i]]["y"] != UNK], K)
    Bv_all, _, _ = vote_faces([S[s]["u"][S[s]["y"] != UNK] for s in STYLES])
    r2.append((float(np.linalg.norm(u3 @ Bs)), float(np.linalg.norm(u3 @ Bv_all))))

    # --- 家族牆：清單留 1/3 不看，held-out ① 距離 vs ② 距離（同一份清單）---
    f_ho = []
    for c in range(6):
        by = []
        for s in STYLES:
            X = S[s]["u"][S[s]["y"] == c]
            if len(X) <= K: continue
            idx = rng.permutation(len(X)); by.append(X[idx[len(X)//3:]])
        Bv2, _, _ = vote_faces(by)
        f_ho.append(Bv2)
    ho = []
    for s in STYLES:
        X = S[s]["u"][S[s]["y"] != UNK]
        idx = rng.permutation(len(X)); ho.append(X[idx[:len(X)//3]])
    sc_ho = score_per_class(np.concatenate(ho), f_ho)
    sc_t2 = score_per_class(ut[~m3], f_ho)
    sc_t3 = score_per_class(ut[m3],  f_ho)
    wall.append((float(sc_ho.mean()), float(sc_t2.mean()), float(sc_t3.mean()),
                 auroc(sc_t3, sc_t2)))

# ================= 偽畫風零點 =================
perm_auc, perm_lam, perm_m = [], [], []
for p in range(NPERM):
    a_p, l_p, m_p = [], [], []
    for i in range(N):
        t = D[i]["tgt"]; m3 = t["y"] == UNK
        S = {s: style_src(i, s) for s in STYLES}
        sizes = [int((S[s]["y"] != UNK).sum()) for s in STYLES]
        pool_u = np.concatenate([S[s]["u"][S[s]["y"] != UNK] for s in STYLES])
        pool_y = np.concatenate([S[s]["y"][S[s]["y"] != UNK] for s in STYLES])
        idx = rng.permutation(len(pool_u)); cut = np.cumsum(sizes)[:-1]
        grp = np.split(idx, cut)
        faces, lam_c, m_c = [], [], []
        for c in range(6):
            by = [pool_u[g][pool_y[g] == c] for g in grp]
            Bv, _, lam = vote_faces(by)
            faces.append(Bv)
            if lam is not None: lam_c.append(lam[:30]); m_c.append(int((lam > LAM_HI).sum()))
        sc = score_per_class(t["u"], faces)
        a_p.append(auroc(sc[m3], sc[~m3])); l_p.append(np.mean(lam_c, 0)); m_p.append(np.mean(m_c))
    perm_auc.append(np.mean(a_p)); perm_lam.append(np.mean(l_p, 0)); perm_m.append(np.mean(m_p))
perm_auc, perm_lam, perm_m = np.array(perm_auc), np.array(perm_lam), np.array(perm_m)

# ================= 輸出 =================
hdr("§1 P1 存在 —— 三份清單有沒有共同條目？（得票光譜，真畫風 vs 偽畫風零點）")
LR = np.mean(lam_real, 0)
print(f"{'名次':<6}{'真畫風 λ':>12}{'偽畫風 λ (零點)':>18}{'差':>10}   ← λ=1 三份都有、λ=0.33 只有一份")
print("-"*W)
for r in [0, 1, 2, 4, 9, 14, 19, 24, 29]:
    print(f"{r+1:<6}{LR[r]:>12.4f}{perm_lam.mean(0)[r]:>18.4f}{LR[r]-perm_lam.mean(0)[r]:>10.4f}")
print("-"*W)
mr = np.mean([np.mean(x) for x in m_cons])
print(f"λ>{LAM_HI} 的方向數 m：真畫風 {mr:.1f}　偽畫風零點 {perm_m.mean():.1f} "
      f"(p5={np.percentile(perm_m,5):.1f}, p95={np.percentile(perm_m,95):.1f})")
print(f"逐畫風 m（節點 0/3/6）：art {np.mean(m_cons[0]):.1f}｜photo {np.mean(m_cons[3]):.1f}｜sketch {np.mean(m_cons[6]):.1f}")

hdr("§2 P2 轉移 —— 共識清單罩不罩得住 cartoon？（部署 AUROC，②vs③）")
print(f"{'讀出':<28}{'AUROC':>10}{'art':>9}{'photo':>9}{'sketch':>9}   說明")
print("-"*W)
lab = {"single":"單畫風面（下錨 0.6641）", "merged":"合併面（全倒一箱）",
       "vote":f"★投票面 前{K}名", "cons":f"共識面 λ>{LAM_HI}"}
for n in ["single", "merged", "vote", "cons"]:
    v = np.array(res[n]); byd = [np.nanmean(v[NODES_OF[s]]) for s in STYLES]
    print(f"{lab[n]:<28}{np.nanmean(v):>10.4f}" + "".join(f"{x:>9.4f}" for x in byd))
print(f"{'偽畫風零點（'+str(NPERM)+'次）':<28}{perm_auc.mean():>10.4f}"
      f"   p5={np.percentile(perm_auc,5):.4f} p95={np.percentile(perm_auc,95):.4f}")
print("-"*W)
print(f"對照錨：‖z⊥‖ 0.8281｜oracle·每類 0.8993｜energy(同口徑) 0.8348｜−std 0.8656")
print(f"★ 交集 − 合併 ＝ {np.nanmean(res['vote'])-np.nanmean(res['merged']):+.4f}"
      f"  ← 唯一差別＝畫風特有條目在不在（你的假設本身）")

hdr("§3 P3 選擇性 —— person 跟著蹭進來多少？（單畫風面 → 投票面）")
L = np.array(leak)
print(f"{'量':<34}{'單畫風面':>12}{'投票面':>12}{'Δ(正=靠近面)':>16}")
print("-"*W)
print(f"{'② 已知×cartoon 到面的距離':<34}{L[:,0].mean():>12.4f}{L[:,1].mean():>12.4f}{L[:,4].mean():>16.4f}")
print(f"{'③ person 到面的距離':<34}{L[:,2].mean():>12.4f}{L[:,3].mean():>12.4f}{L[:,5].mean():>16.4f}")
print(f"{'兩者差距（③−②）':<34}{(L[:,2]-L[:,0]).mean():>12.4f}{(L[:,3]-L[:,1]).mean():>12.4f}"
      f"{((L[:,3]-L[:,1])-(L[:,2]-L[:,0])).mean():>16.4f}")
print("-"*W)
d2m, d3m = L[:,4].mean(), L[:,5].mean()
GATE = 0.1861/3
if abs(d2m) < GATE:
    print(f"⚠️ Δ② = {d2m:+.4f} 未達門檻 {GATE:.4f}（oracle 位移 0.1861 的 1/3）⇒ 無牽引力，"
          f"**漏損比率不讀**（分母趨零的假精度），只報原始值 Δ③ = {d3m:+.4f}")
else:
    print(f"漏損率 = Δ③/Δ② = {d3m/d2m:6.1%}　（對照：換 oracle 面 46.4%｜AdaIN·長度 89%｜AdaIN·方向 101%）")
R = np.array(r2)
print(f"R2 前哨：person 集中方向落在面內的比例　單畫風面 {R[:,0].mean():.4f} → 投票面 {R[:,1].mean():.4f}"
      f"（{'更進去 ⇒ 不利' if R[:,1].mean()>R[:,0].mean() else '更出來 ⇒ 有利'}）")

hdr("§4 家族牆診斷 —— 牆是「不是卡通」嗎？（清單留 1/3 不看，同一份清單比）")
Wl = np.array(wall)
print(f"{'到投票面的距離':<34}{'均值':>12}{'art':>10}{'photo':>10}{'sketch':>10}")
print("-"*W)
for j, nm in [(0,"held-out ① 來源畫風·已知"), (1,"② cartoon·已知"), (2,"③ cartoon·person")]:
    byd = [Wl[NODES_OF[s], j].mean() for s in STYLES]
    print(f"{nm:<34}{Wl[:,j].mean():>12.4f}" + "".join(f"{x:>10.4f}" for x in byd))
print("-"*W)
gap = Wl[:,1].mean() - Wl[:,0].mean()
print(f"★ 家族牆高度 ＝ ②(cartoon已知) − ①(來源已知) ＝ {gap:+.4f}")
print(f"   對比：②③ 之間的差距只有 {Wl[:,2].mean()-Wl[:,1].mean():+.4f}")
print(f"   ⇒ 牆比訊號{'大' if gap > (Wl[:,2].mean()-Wl[:,1].mean()) else '小'} "
      f"{abs(gap/(Wl[:,2].mean()-Wl[:,1].mean())):.1f} 倍" if abs(Wl[:,2].mean()-Wl[:,1].mean())>1e-6 else "")
print(f"（held-out 版投票面的部署 AUROC = {Wl[:,3].mean():.4f}，與 §2 主判準的差＝held-out 成本）")

hdr("§5 判準對照（事前寫死）")
va = np.nanmean(res["vote"]); ma = np.nanmean(res["merged"])
print(f"主判準：投票面部署 AUROC = {va:.4f}")
print(f"  三個錨：0.6641（下）／偽畫風零點 {perm_auc.mean():.4f}（無資訊）／0.8993（上，僅刻度）")
if mr < 5:
    print(f"⚠️ 第 0 關：λ>{LAM_HI} 只有 {mr:.1f} 維 ⇒ 共識稀薄，「交集−合併」該格效力打折")
if va >= 0.75 and va >= ma:
    print(f"⇒ 【發車】≥0.75 且 ≥合併面 {ma:.4f}")
elif va > 0.6641:
    print(f"⇒ 【中段】0.6641 < {va:.4f} < 0.75 ⇒ 由 R2 前哨定奪（見 §3）")
else:
    print(f"⇒ 【不發車】≤ 0.6641；不判死（C 級快照、訓練會改特徵），但不占下一個訓練檔期")
print()
print("⚠️ 只有 3 個獨立觀測點（同畫風三節點特徵逐樣本餘弦 >0.9998）⇒ 逐畫風才是真的逐節點")
print("⚠️ 偽畫風零點的每堆混了三個節點的特徵，真畫風每堆只有一個 ⇒ 零點的一致性被低估（保守方向）")
