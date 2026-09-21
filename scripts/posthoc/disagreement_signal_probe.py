"""dany 2026-08-29：兩個頭「吵架」(fc argmax vs 投影頭 argmin) ——為什麼吵？能不能當訊號？
既有：只量過吵架率 person 20.6% vs cartoon 4.8%（0826），從沒算過 AUROC、沒查過機制、
沒驗過它與 energy 是否獨立（0820 §1 的融合測試證明「內部相關高 ⇒ 融合權重 0」是常態）。
基底：0826_features_full.npz（BN 平均 B）。
"""
import numpy as np
F = np.load("logs/prototype_probe/0826_features_full.npz")
N, UNK, DEG = 9, 6, 57.29577951308232
nrm = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)
from scipy.stats import rankdata
def auroc(pos, neg):
    """⚠️ 必須用平均排名處理並列——二元分數(如吵架 flag)有大量並列，
    argsort().argsort() 會給並列樣本任意排名 ⇒ AUROC 系統性偏差。"""
    a = np.concatenate([pos, neg]); r = rankdata(a, method="average")
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))
def spearman(x, y):
    rx, ry = x.argsort().argsort().astype(float), y.argsort().argsort().astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])
sm = lambda V: np.exp(V - np.logaddexp.reduce(V, 1)[:, None])

DAT = {}
for i in range(N):
    C = nrm(F[f"n{i}_C"].astype(np.float64)); U = np.linalg.svd(C.T, full_matrices=False)[0]
    d = {}
    for k in ["src", "tgt"]:
        lo = F[f"n{i}_{k}_lo"].astype(np.float64); Z = nrm(F[f"n{i}_{k}_z"].astype(np.float64))
        cos = np.clip(Z @ C.T, -1+1e-12, 1-1e-12); ang = np.arccos(cos)*DEG
        d[k] = dict(lo=lo, ang=ang, cos=cos, y=F[f"n{i}_{k}_y"],
                    zp=np.sqrt(np.maximum(1-((Z@U)**2).sum(1), 0)))
    DAT[i] = d

print("="*100); print("§1 吵架率（自檢：0826 報 person 20.6% / cartoon 4.8%）"); print("="*100)
r2, r3, r1 = [], [], []
for i in range(N):
    for k, acc in [("src", [r1]), ("tgt", [r2, r3])]:
        D = DAT[i][k]; dis = D["lo"].argmax(1) != D["ang"].argmin(1); y = D["y"]
        if k == "src": r1.append(dis[y != UNK].mean())
        else: r2.append(dis[y != UNK].mean()); r3.append(dis[y == UNK].mean())
print(f"  ① 來源域已知 {np.mean(r1)*100:5.2f}%   ② cartoon 已知 {np.mean(r2)*100:5.2f}%   "
      f"③ person {np.mean(r3)*100:5.2f}%   （③/② = {np.mean(r3)/np.mean(r2):.1f} 倍）")

print(); print("="*100); print("§2 把吵架做成分數（四種形式）＋ 與 energy 的獨立性"); print("="*100)
SC = {
    "二元吵架 flag":        lambda D: (D["lo"].argmax(1) != D["ang"].argmin(1)).astype(float),
    "投影頭對 fc 選的類別多不服": lambda D: D["ang"][np.arange(len(D["ang"])), D["lo"].argmax(1)] - D["ang"].min(1),
    "fc 對投影頭選的類別多不服":  lambda D: D["lo"].max(1) - D["lo"][np.arange(len(D["lo"])), D["ang"].argmin(1)],
    "兩個頭機率分布的 L1 距離":  lambda D: np.abs(sm(D["lo"]) - sm(-D["ang"]/10.0)).sum(1),
    "★ 對照 energy":        lambda D: -np.logaddexp.reduce(D["lo"], 1),
    "★ 對照 ‖z⊥‖":          lambda D: D["zp"],
}
print(f"{'分數':<30}{'畫風↓0.5':>10}{'★部署↑':>10}{'與 energy 的 Spearman':>22}{'融合最佳 w':>12}")
print("-"*100)
for nm_, f_ in SC.items():
    st, dp, sp, bw = [], [], [], []
    for i in range(N):
        s = {k: f_(DAT[i][k]) for k in ["src", "tgt"]}
        e = {k: -np.logaddexp.reduce(DAT[i][k]["lo"], 1) for k in ["src", "tgt"]}
        Ys, Yt = DAT[i]["src"]["y"], DAT[i]["tgt"]["y"]
        a, b, c = s["src"][Ys != UNK], s["tgt"][Yt != UNK], s["tgt"][Yt == UNK]
        st.append(auroc(b, a)); dp.append(auroc(c, b))
        sp.append(spearman(s["tgt"], e["tgt"]))
        z = lambda v, ref: (v - ref.mean()) / (ref.std() + 1e-12)
        best = (0.0, -1)
        for w in np.arange(0, 1.01, 0.05):
            m = z(e["tgt"], e["tgt"])*(1-w) + z(s["tgt"], s["tgt"])*w
            d_ = auroc(m[Yt == UNK], m[Yt != UNK])
            if d_ > best[1]: best = (w, d_)
        bw.append(best[0])
    print(f"{nm_:<30}{np.mean(st):10.4f}{np.mean(dp):10.4f}{np.mean(sp):22.4f}{np.mean(bw):12.2f}")
print("-"*100)

print(); print("="*100)
print("§3 既然吵架訊號的畫風 AUROC 高達 0.8683 ⇒ 它其實是【畫風偵測器】")
print("    ⇒ 試 Tian 2021 式的『兩個分開的量、用一個校正另一個』(TaskBoard 標為未排除)")
print("="*100)
def cond_calib(base_fn, split_fn, nm_):
    """分組校準：按吵架與否分兩群，各自把分數標準化後再合併 ⇒ 消掉畫風造成的整體位移"""
    st, dp, dp0 = [], [], []
    for i in range(N):
        s = {k: base_fn(DAT[i][k]) for k in ["src", "tgt"]}
        g = {k: split_fn(DAT[i][k]) for k in ["src", "tgt"]}
        ref = {}
        for gg in [0, 1]:                       # 校準統計量只用【來源域】(訓練時可得)
            m = (g["src"] == gg)
            ref[gg] = (s["src"][m].mean(), s["src"][m].std() + 1e-12) if m.sum() > 20 else (s["src"].mean(), s["src"].std()+1e-12)
        adj = {k: np.array([(s[k][j] - ref[g[k][j]][0]) / ref[g[k][j]][1] for j in range(len(s[k]))]) for k in ["src","tgt"]}
        Ys, Yt = DAT[i]["src"]["y"], DAT[i]["tgt"]["y"]
        st.append(auroc(adj["tgt"][Yt != UNK], adj["src"][Ys != UNK]))
        dp.append(auroc(adj["tgt"][Yt == UNK], adj["tgt"][Yt != UNK]))
        dp0.append(auroc(s["tgt"][Yt == UNK], s["tgt"][Yt != UNK]))
    print(f"  {nm_:<34} 校正前部署 {np.mean(dp0):.4f} → 校正後 {np.mean(dp):.4f}"
          f"   ({np.mean(dp)-np.mean(dp0):+.4f})   畫風 {np.mean(st):.4f}")
split = lambda D: (D["lo"].argmax(1) != D["ang"].argmin(1)).astype(int)
cond_calib(lambda D: -np.logaddexp.reduce(D["lo"], 1), split, "energy 按吵架分組校準")
cond_calib(lambda D: D["zp"], split, "‖z⊥‖ 按吵架分組校準")
cond_calib(lambda D: D["ang"].min(1), split, "現行 min 角距離 按吵架分組校準")
print("-"*100)
print("  對照（未校正）：energy 0.8380 ｜ ‖z⊥‖ 0.8281 ｜ min 角距離 0.8145")

print(); print("="*104)
print("§4 dany 2026-08-29 設計：固定「猶豫程度」(−std) 後，吵架率是否仍有殘餘判別力？")
print("    每個節點內按 −std(logit) 切十等分，同層合併 9 節點再統計（層內 −std 幾近固定）")
print("="*104)
STD_ALL, DIS_ALL, ISU_ALL, DEC = [], [], [], []
for i in range(N):
    D = DAT[i]["tgt"]; s = -D["lo"].std(1)
    dis = (D["lo"].argmax(1) != D["ang"].argmin(1)).astype(float); isu = (D["y"] == UNK)
    q = np.quantile(s, np.linspace(0, 1, 11)); q[0] -= 1e-9
    dec = np.clip(np.searchsorted(q, s, side="left") - 1, 0, 9)
    STD_ALL.append(s); DIS_ALL.append(dis); ISU_ALL.append(isu); DEC.append(dec)
S, Dg, Iu, Dc = map(np.concatenate, (STD_ALL, DIS_ALL, ISU_ALL, DEC))
print(f"  整體：吵架 vs −std(logit) 的 Spearman = {spearman(Dg, S):+.4f}"
      f"   （對照：吵架 vs energy = {spearman(Dg, np.concatenate([-np.logaddexp.reduce(DAT[i]['tgt']['lo'],1) for i in range(N)])):+.4f}）")
print("-"*104)
print(f"{'層(−std 由低到高)':<18}{'−std 範圍':>18}{'②張數':>7}{'③張數':>7}{'②吵架率':>9}{'③吵架率':>9}{'倍數':>7}{'層內AUROC':>10}")
print("-"*104)
tot_w, tot_a = 0, 0
for d in range(10):
    m = Dc == d; m2, m3 = m & ~Iu, m & Iu
    if m3.sum() < 5 or m2.sum() < 5:
        print(f"{d+1:<18}{'':>18}{m2.sum():7d}{m3.sum():7d}   (樣本過少、略過)"); continue
    r2, r3 = Dg[m2].mean(), Dg[m3].mean()
    a = auroc(Dg[m3], Dg[m2]); tot_w += m2.sum()*m3.sum(); tot_a += a*m2.sum()*m3.sum()
    print(f"{d+1:<18}{f'{S[m].min():+.2f}~{S[m].max():+.2f}':>18}{m2.sum():7d}{m3.sum():7d}"
          f"{r2*100:8.1f}%{r3*100:8.1f}%{(r3/(r2+1e-9)):7.1f}{a:10.4f}")
print("-"*104)
print(f"  ★ 層內加權平均 AUROC = {tot_a/tot_w:.4f}   （0.5 ＝ 固定猶豫程度後、吵架完全沒有殘餘資訊）")
print(f"  ★ 不分層的吵架 AUROC = {np.mean([auroc(DIS_ALL[i][ISU_ALL[i]], DIS_ALL[i][~ISU_ALL[i]]) for i in range(N)]):.4f}")
print("-"*104)
print("  對稱測試：反過來固定吵架與否，−std 還剩多少？")
for g, nm_ in [(0, "沒吵架的樣本"), (1, "吵架的樣本")]:
    m = Dg == g; m2, m3 = m & ~Iu, m & Iu
    print(f"    {nm_:<12} ②{m2.sum():5d} ③{m3.sum():4d}   組內 −std 的 AUROC = {auroc(S[m3], S[m2]):.4f}")
print("="*104)

print(); print("="*104)
print("§5 穩健性：換三個不同的分層變數，「吵架被包含」的結論會不會翻？")
print("="*104)
BASE = {"−std(logit)": lambda D: -D["lo"].std(1),
        "energy":      lambda D: -np.logaddexp.reduce(D["lo"], 1),
        "‖z⊥‖":        lambda D: D["zp"],
        "min 角距離":    lambda D: D["ang"].min(1)}
print(f"{'分層變數':<16}{'該變數自己的部署':>14}{'層內加權 AUROC(吵架)':>22}{'殘餘判別力佔比':>16}")
print("-"*104)
raw = np.mean([auroc(DIS_ALL[i][ISU_ALL[i]], DIS_ALL[i][~ISU_ALL[i]]) for i in range(N)])
for nm_, f_ in BASE.items():
    Sx = np.concatenate([f_(DAT[i]["tgt"]) for i in range(N)])
    Dc2 = np.concatenate([np.clip(np.searchsorted(
        (lambda q: (q.__setitem__(0, q[0]-1e-9), q)[1])(np.quantile(f_(DAT[i]["tgt"]), np.linspace(0,1,11))),
        f_(DAT[i]["tgt"]), side="left")-1, 0, 9) for i in range(N)])
    tw, ta, bd = 0, 0, []
    for d in range(10):
        m = Dc2 == d; m2, m3 = m & ~Iu, m & Iu
        if m3.sum() < 5 or m2.sum() < 5: continue
        w = m2.sum()*m3.sum(); ta += auroc(Dg[m3], Dg[m2])*w; tw += w
    own = np.mean([auroc(f_(DAT[i]["tgt"])[DAT[i]["tgt"]["y"]==UNK], f_(DAT[i]["tgt"])[DAT[i]["tgt"]["y"]!=UNK]) for i in range(N)])
    a = ta/tw
    print(f"{nm_:<16}{own:14.4f}{a:22.4f}{(a-0.5)/(raw-0.5)*100:15.1f}%")
print("-"*104)
print(f"  不分層的吵架 AUROC = {raw:.4f}（判別力 {raw-0.5:.4f}）；佔比 100% ＝ 完全沒被包含、0% ＝ 完全被包含")

print(); print("="*106)
print("§6 封棺釘（dany 2026-08-29 指出的漏洞）：軟化版(L1)從沒分層過；融合只報過權重沒報過數值；")
print("   且融合測的是 energy 不是最強的 −std ⇒ 全部補上，並先做管線零點自檢")
print("="*106)
STD  = {i: -DAT[i]["tgt"]["lo"].std(1) for i in range(N)}
L1   = {i: np.abs(sm(DAT[i]["tgt"]["lo"]) - sm(-DAT[i]["tgt"]["ang"]/10.0)).sum(1) for i in range(N)}
FLAG = {i: (DAT[i]["tgt"]["lo"].argmax(1) != DAT[i]["tgt"]["ang"].argmin(1)).astype(float) for i in range(N)}
ENG  = {i: -np.logaddexp.reduce(DAT[i]["tgt"]["lo"], 1) for i in range(N)}
IU   = {i: DAT[i]["tgt"]["y"] == UNK for i in range(N)}

def strat(target, by):
    """固定 by（每節點十等分）後，target 在層內還剩多少判別力（加權平均 AUROC）"""
    tw, ta = 0, 0
    for i in range(N):
        b = by[i]; q = np.quantile(b, np.linspace(0, 1, 11)); q[0] -= 1e-9
        dec = np.clip(np.searchsorted(q, b, side="left") - 1, 0, 9)
        for d in range(10):
            m = dec == d; m2, m3 = m & ~IU[i], m & IU[i]
            if m3.sum() < 5 or m2.sum() < 5: continue
            w = m2.sum()*m3.sum(); ta += auroc(target[i][m3], target[i][m2])*w; tw += w
    return ta/tw
raw = lambda t: np.mean([auroc(t[i][IU[i]], t[i][~IU[i]]) for i in range(N)])

print("★ 管線零點自檢：用 −std 分層 −std 自己 ⇒ 若管線正確，殘餘應貼近 0.5")
z = strat(STD, STD)
print(f"    −std 分層 −std 自己：層內 AUROC = {z:.4f}   （不分層 {raw(STD):.4f}）"
      f"   {'✅ 管線正常' if z < 0.56 else '❌ 管線有問題'}")
print(f"    ⇒ ★ 這就是「殘餘多少才算雜訊」的零點：{z:.4f}")
print("-"*106)
print(f"{'分數':<22}{'不分層 AUROC':>13}{'固定 −std 後層內':>17}{'相對零點的超出量':>17}{'佔原判別力':>12}")
print("-"*106)
for nm_, t in [("二元吵架 flag", FLAG), ("★ 軟化版（L1 距離）", L1), ("(對照) energy", ENG)]:
    r, a = raw(t), strat(t, STD)
    print(f"{nm_:<22}{r:13.4f}{a:17.4f}{a-z:+17.4f}{(a-0.5)/(r-0.5)*100:11.1f}%")
print("-"*106)
print("★★ 融合後的 AUROC 數值（先前只報權重、從沒報過數值）——基準＝最強的 −std 0.8656")
print("-"*106)
zs = lambda v: (v - v.mean())/(v.std()+1e-12)
for nm_, t in [("二元吵架 flag", FLAG), ("★ 軟化版（L1 距離）", L1)]:
    for base_nm, base in [("−std(0.8656)", STD), ("energy(0.8380)", ENG)]:
        row, best = [], (0, -1)
        for w in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
            v = np.mean([auroc(*(lambda m: (m[IU[i]], m[~IU[i]]))(zs(base[i])*(1-w)+zs(t[i])*w)) for i in range(N)])
            row.append(f"w={w}:{v:.4f}")
            if v > best[1]: best = (w, v)
        print(f"  {nm_:<20} 融合 {base_nm:<15} " + "  ".join(row))
        print(f"  {'':<20} ⇒ 最佳 w={best[0]}、AUROC {best[1]:.4f}   "
              f"（vs 純 {base_nm.split('(')[0]} {raw(base):.4f}，Δ {best[1]-raw(base):+.4f}）")
print("="*106)
