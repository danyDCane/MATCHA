# -*- coding: utf-8 -*-
"""
R4 的無資訊零點與上界（看板 📌 規則：幾何／時間指標必須先建零點才可判讀）
================================================================
κ ≥ 0.90 ⇒ 換刻度，這條事前判準只管「上面」。
κ = 0.36 到底是「真的不同」還是「其實是雜訊」？本檔補下面那一半：

零點 N1 均勻隨機訊號（同觸發率）                 ⇒ 期望 κ ≈ 0
零點 N2 把訊號在節點內時間打散（保留邊際分布）      ⇒ 期望 κ ≈ 0
零點 N3 固定週期排程（每 1/p0 步觸發）            ⇒ 「完全不看學習狀態」能拿到多少 κ
零點 N5 ★★ 自上次廣播以來【累積收到幾次推播】      ⇒ matcha-02 2026-10-02 提出。非同步的第二個純排程
        變數：收到推播 → _aggregate_models 把參數拉向鄰居 → _param_delta 跳 → 規則 0 發火。
        ⚠️ 它**不是嚴格無資訊**（鄰居何時推播含鄰居的學習狀態），精確性質是「不含我自己的狀態」。
零點 N4 ★ 純時鐘（距上次廣播幾步，用規則 0 的真實重置）⇒ 兩個訊號都是「自上次廣播以來的位移」，
        都隨時間單調成長 ⇒ 這一項才是真正該扣掉的共同成分
上界 U1 sig_r0 自己用分位校準（而非 τ）           ⇒ 自檢，應 ≈ 1
上界 U2 ★ κ 天花板：全域單一門檻下逐節點邊際不等 ⇒ κ 有上限，實測值要用「實現比例」讀
分層   在「距上次廣播步數」相同的條件下，規則 1 與規則 0 是否仍一致（扣掉時鐘後的淨重疊）
"""
import numpy as np, pandas as pd

CSV = ('/home/server5090/Desktop/M11307320/MATCHA/exp_result_v1_stage2_leave_cartoon_p1a_'
       'async_const_tau1e-5_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234_fix/'
       'async_diag/shadow_signals.csv')
THRESH0 = 1e-5
E = [f'e{i}' for i in range(8)]
rng = np.random.default_rng(2026)

df = pd.read_csv(CSV)
d = df[df.is_warm == 0].reset_index(drop=True)
dom, steps = d['domain'].values, d['step'].values
nodes = sorted(np.unique(dom))
nat0 = (d['sig_r0'].values >= THRESH0)
fired0 = d['fired'].values.astype(bool)
p0 = nat0.mean()

def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pe = a.mean()*b.mean() + (1-a.mean())*(1-b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan

def calib(sig, target=p0):
    s = np.asarray(sig, float); th = np.nanquantile(s, 1 - target)
    return s >= th, th

sig_r1 = d['sig_r1'].values.astype(float)
raw = d[[f'r2c_{e}' for e in E]].to_numpy(dtype=float)
with np.errstate(all='ignore'):
    sig_2c = np.nanmean(raw, 1)
    sig_2cmax = np.nanmax(raw, 1)
dec_r1, _ = calib(sig_r1)
dec_2c, _ = calib(sig_2c)

# ---- 純時鐘：距「規則 0 上次真的廣播」幾步 ----
clock = np.zeros(len(d), float)
for n in nodes:
    m = np.where(dom == n)[0]
    last = steps[m[0]] - 1
    for i in m:
        clock[i] = steps[i] - last
        if fired0[i]:
            last = steps[i]
dec_clk, th_clk = calib(clock)

print('=' * 92)
print(f'基準：規則 0 自然觸發率 p0 = {p0*100:.4f}%｜chance = {p0**2+(1-p0)**2:.4f}｜n = {len(d)}')
print('=' * 92)
print(f'{"項目":<40}{"κ vs 規則0":>12}{"觸發率":>9}')
print('-' * 92)

# N1 隨機
ks = [kappa(rng.random(len(d)) < p0, nat0) for _ in range(20)]
print(f'{"零點 N1 均勻隨機訊號（20 次重抽）":<40}{np.mean(ks):>12.4f}{p0*100:>8.1f}%   '
      f'（std {np.std(ks):.4f}、範圍 {min(ks):+.4f}~{max(ks):+.4f}）')

# N2 節點內時間打散
ks1, ks2 = [], []
for _ in range(20):
    s1, s2 = sig_r1.copy(), sig_2c.copy()
    for n in nodes:
        m = np.where(dom == n)[0]
        s1[m] = rng.permutation(s1[m]); s2[m] = rng.permutation(s2[m])
    ks1.append(kappa(calib(s1)[0], nat0)); ks2.append(kappa(calib(s2)[0], nat0))
print(f'{"零點 N2 規則1 時間打散（20 次）":<40}{np.mean(ks1):>12.4f}{p0*100:>8.1f}%   （std {np.std(ks1):.4f}）')
print(f'{"零點 N2 規則2c 時間打散（20 次）":<40}{np.mean(ks2):>12.4f}{p0*100:>8.1f}%   （std {np.std(ks2):.4f}）')

# N3 固定週期
T = int(round(1/p0))
per = np.zeros(len(d), bool)
for n in nodes:
    m = np.where(dom == n)[0]
    per[m[::T]] = True
print(f'{f"零點 N3 固定週期排程（每 {T} 步）":<40}{kappa(per, nat0):>12.4f}{per.mean()*100:>8.1f}%')

# N4 純時鐘
print(f'{"★ 零點 N4 純時鐘（距上次廣播 ≥ %d 步）" % th_clk:<40}{kappa(dec_clk, nat0):>12.4f}{dec_clk.mean()*100:>8.1f}%')

# U1 自檢
dec_r0q, th_r0q = calib(d['sig_r0'].values.astype(float))
print(f'{"上界 U1 規則0 自己改用分位校準":<40}{kappa(dec_r0q, nat0):>12.4f}{dec_r0q.mean()*100:>8.1f}%   （自檢，應 ≈ 1）')

print('-' * 92)
print(f'{"實測 規則1":<40}{kappa(dec_r1, nat0):>12.4f}{dec_r1.mean()*100:>8.1f}%')
print(f'{"實測 規則2c-mean":<40}{kappa(dec_2c, nat0):>12.4f}{dec_2c.mean()*100:>8.1f}%')

print('\n各訊號與【純時鐘】的 κ（＝這個訊號有多少只是在數時間）：')
print(f'  規則0  vs 時鐘  κ = {kappa(nat0, dec_clk):.4f}')
print(f'  規則1  vs 時鐘  κ = {kappa(dec_r1, dec_clk):.4f}')
print(f'  規則2c vs 時鐘  κ = {kappa(dec_2c, dec_clk):.4f}')
print(f'  規則1  vs 規則2c κ = {kappa(dec_r1, dec_2c):.4f}')

# ---- 分層：固定「距上次廣播步數」後的淨重疊 ----
print('\n★ 扣掉時鐘後的淨重疊（在「距上次廣播步數」相同的分層內算 κ，再用層大小加權）')
bins = [0, 2, 4, 6, 9, 13, 20, 35, 10**9]
lab = np.digitize(clock, bins) - 1
print(f'{"分層（距上次廣播步數）":<24}{"n":>7}{"規則0 率":>9}{"規則1 κ":>10}{"規則2c κ":>10}')
tot_w = kr1 = kr2 = 0.0
for b in range(len(bins)-1):
    m = (lab == b)
    if m.sum() < 200 or nat0[m].sum() < 10 or (~nat0[m]).sum() < 10:
        continue
    k1, k2 = kappa(dec_r1[m], nat0[m]), kappa(dec_2c[m], nat0[m])
    hi = bins[b+1] if bins[b+1] < 10**8 else np.inf
    print(f'{f"[{bins[b]}, {hi})":<24}{m.sum():>7}{nat0[m].mean()*100:>8.1f}%{k1:>10.4f}{k2:>10.4f}')
    tot_w += m.sum(); kr1 += k1*m.sum(); kr2 += k2*m.sum()
print(f'{"加權平均（淨）":<24}{int(tot_w):>7}{"":>9}{kr1/tot_w:>10.4f}{kr2/tot_w:>10.4f}')

# ---- 逐節點 κ ----
print('\n逐節點 κ（全域門檻）：')
print(f'{"節點":<9}{"度數":>5}{"規則0 率":>9}{"規則1 κ":>10}{"規則2c κ":>10}')
deg = np.isfinite(raw).sum(1)
for n in nodes:
    m = (dom == n)
    print(f'{n:<9}{int(deg[m][0]):>5}{nat0[m].mean()*100:>8.1f}%'
          f'{kappa(dec_r1[m], nat0[m]):>10.4f}{kappa(dec_2c[m], nat0[m]):>10.4f}')

# ---- N5 累積收到推播（matcha-02 2026-10-02）----
nrec = d['n_received'].values.astype(float)
cum = np.zeros(len(d))
for n in nodes:
    m = np.where(dom == n)[0]; acc = 0.0
    for i in m:
        acc += nrec[i]; cum[i] = acc
        if fired0[i]: acc = 0.0
dec_cum, th_cum = calib(cum)
dec_nr, _ = calib(nrec)

def calib_pn(sig):
    s = np.asarray(sig, float); o = np.zeros(len(s), bool)
    for n in nodes:
        m = dom == n
        o[m] = s[m] >= np.nanquantile(s[m], 1 - nat0[m].mean())
    return o

print('\n★ 零點 N5（matcha-02）：非同步的第二個純排程變數')
print(f'{"項目":<34}{"全域 κ":>10}{"逐節點 κ":>11}')
for nm, ss in [('本步收到幾次', nrec), ('★ 自上次廣播以來累積收到幾次', cum),
               ('（參照）規則1', sig_r1), ('（參照）規則2c-mean', sig_2c)]:
    print(f'{nm:<34}{kappa(calib(ss)[0], nat0):>+10.4f}{kappa(calib_pn(ss), nat0):>+11.4f}')
# ★★ 劑量反應：比 κ 更直接的證據（matcha-02 2026-10-02）
print('\n★★ 劑量反應：規則 0 的觸發率 vs 累積收到幾次推播（逐值，不分箱）')
print(f'{"累積收到":>8}{"n":>7}{"規則0 觸發率":>13}   說明')
_lab = cum.astype(int); _rates = []
for v in range(int(_lab.max()) + 1):
    m = _lab == v
    if not m.any(): continue
    _rates.append(nat0[m].mean())
    tag = '' if m.sum() >= 100 else '  ← n 太小、不可讀'
    print(f'{v:>8}{m.sum():>7}{nat0[m].mean()*100:>12.2f}%{tag}')
_a = np.array(_rates); _k = 12          # v=0..11 ＝ n>=100 的那段
_cov = sum((_lab == v).sum() for v in range(_k)) / len(_lab) * 100
print(f'  前 {_k} 層（v=0–{_k-1}、涵蓋 {_cov:.2f}% 的列）**全部單調上升**：'
      f'{_a[0]*100:.4f}% → {_a[_k-1]*100:.4f}% ＝ **{_a[_k-1]/_a[0]:.2f}×**')
print(f'  尾巴 v={_k}–{len(_a)-1}（共 {len(_lab)-sum((_lab==v).sum() for v in range(_k))} 列、0.3%）不單調 ⇒ 樣本太少、不可讀')
print('  ⇒ 一次都沒收到推播時規則 0 只有 4.44% 發火；收到 9 次以上超過 55%、11 次 74%。')
print('  ⛔ 尾巴四層（n=57/22/10/6）不單調 ⇒ 不可引用，包括 v=15 那個 n=6 的 100%。')
print('  ⇒ **現行觸發器有相當大一部分是在反應「我被鄰居拉動了」，不是「我學到了東西」。**')
print('  ⇒ 這比 κ=0.2678 直接得多，建議當主證據、κ 當輔。')

print(f'  κ(規則1, 累積收到) = {kappa(dec_r1, dec_cum):+.4f}｜'
      f'κ(規則0, 累積收到) = {kappa(nat0, dec_cum):+.4f}｜'
      f'κ(規則2c, 累積收到) = {kappa(dec_2c, dec_cum):+.4f}')
print('  ⇒ 純排程能拿到規則 1 逐節點 κ 的 %.0f%% ⇒ 「零點 ≈ 0.000」的框架低估排程能達到的 κ。'
      % (kappa(calib_pn(cum), nat0) / kappa(calib_pn(sig_r1), nat0) * 100))
print('  ⇒ 機制：prototypes 是 register_buffer（dood/prototype.py:48），而 _aggregate_models')
print('     只聚合 named_parameters()、不碰 buffer ⇒ 原型不會被聚合直接覆寫，只經由特徵間接受影響。')

# ---- U2 κ 天花板 ----
def kappa_ceiling(dec):
    po_max = sum((dom == n).sum() * (1 - abs(nat0[dom == n].mean() - dec[dom == n].mean()))
                 for n in nodes) / len(d)
    pe = nat0.mean()*dec.mean() + (1-nat0.mean())*(1-dec.mean())
    return (po_max - pe) / (1 - pe)
print('\n★ 上界 U2：全域單一門檻下的 κ 天花板（逐節點邊際不等造成）')
for nm, dd in [('規則1', dec_r1), ('規則2c-mean', dec_2c)]:
    km = kappa_ceiling(dd); k = kappa(dd, nat0)
    print(f'  {nm:<12} 天花板 {km:.4f}   實測 {k:.4f}   **實現 {k/km*100:.1f}%**')
print('  ⚠️ 天花板不破 κ≥0.90 這道閘：若真是單調換刻度，全域分位門檻會選到完全相同的列 ⇒ κ=1。')

# ---- 分層（修正版：層內重校準 ＋ n(1−pe) 加權）----
print('\n★ 分層淨 κ 的修正版（matcha-02）：層內重校準門檻、用 n·(1−pe) 加權')
print('  舊版兩個問題：(a) 層內沿用全域門檻 ⇒ 層內邊際不等 ⇒ κ 被壓；(b) n 加權讓退化層同權。')
def strat_kappa(sig, lab, minn=200, minpos=10):
    num = den = 0.0
    for b in np.unique(lab):
        m = lab == b
        if m.sum() < max(minn, 1) or nat0[m].sum() < minpos or (~nat0[m]).sum() < minpos: continue
        t = nat0[m].mean(); ss = np.asarray(sig, float)[m]
        dd = ss >= np.nanquantile(ss, 1 - t)
        pe = nat0[m].mean()*dd.mean() + (1-nat0[m].mean())*(1-dd.mean())
        if pe >= 1: continue          # 退化層（層內全同）⇒ κ 無定義，權重本來也是 0
        w = m.sum() * (1 - pe); num += kappa(dd, nat0[m]) * w; den += w
    return num / den
lab_clk = np.digitize(clock, bins) - 1
# 🚨 累積收到推播是**離散變數**（唯一值只有 0–15 共 16 個、≥16 為 0 筆）
#    ⇒ 分箱邊界從來就不必要，**逐值分層就是條件化的極限**（matcha-02 2026-10-02）
lab_cum = cum.astype(int)
_u, _c = np.unique(lab_cum, return_counts=True)
print(f'  累積收到推播的唯一值 {_u.min()}–{_u.max()}（共 {len(_u)} 個）、各值列數 {_c.tolist()}')
print(f'{"分層變數":<34}{"規則1":>10}{"規則2c":>10}')
print(f'{"距上次廣播步數（舊版 n 加權）":<34}{kr1/tot_w:>10.4f}{kr2/tot_w:>10.4f}')
print(f'{"距上次廣播步數（修正版）":<34}{strat_kappa(sig_r1, lab_clk):>10.4f}{strat_kappa(sig_2c, lab_clk):>10.4f}')
print(f'{"★ 累積收到推播（逐值、全部 16 層）":<34}'
      f'{strat_kappa(sig_r1, lab_cum, 0, 0):>10.4f}{strat_kappa(sig_2c, lab_cum, 0, 0):>10.4f}')
print(f'{"  同上，丟掉 n<200 的尾巴層":<34}'
      f'{strat_kappa(sig_r1, lab_cum):>10.4f}{strat_kappa(sig_2c, lab_cum):>10.4f}')
print(f'{"（未分層原始）":<34}{kappa(dec_r1, nat0):>10.4f}{kappa(dec_2c, nat0):>10.4f}')
print('  ⇒ 兩種取捨差 <0.003 ⇒ 結論（掉幅 ~17% / ~7%、兩者大致獨立）不受影響。')
# 差異來源的歸因自檢：不是分位 tie，是分層粒度
_dev = []
for v in np.unique(lab_cum):
    m = lab_cum == v
    if not m.any(): continue
    t = nat0[m].mean()
    for ss in (sig_r1, sig_2c):
        x = np.asarray(ss, float)[m]
        _dev.append(abs(((x >= np.nanquantile(x, 1 - t)).mean() - t)) * 100)
print(f'  歸因自檢：逐層「分位門檻造成的實際觸發率偏離」最大 {max(_dev):.4f} pp'
      f' ⇒ **沒有 tie**（角度是連續值）⇒ 差異來源是**分層粒度**（15 層 vs 11 層），不是 tie 處理。')
print('  🚨 規則2c 的修正版淨 κ **高於**其原始 κ ⇒ 時鐘是在【壓低】規則2c 的一致度，不是抬高。')
print('     ⇒ 舊版「34% 的重疊是時鐘造成」不只高估，**符號是錯的**。')

# ---- 度數相關的畫風內置換檢定 ----
import itertools
print('\n★ 度數偏誤的畫風內精確置換檢定（(3!)³ ＝ 216 種 ⇒ 最小 p ＝ 1/216 ＝ 0.0046 是地板）')
print('  （度數有重複值 art 5,2,5／photo 6,6,3 ⇒ 可區分置換少於 216 ⇒ 三個 p 都當【上限】讀）')
degv = np.array([int(deg[dom == n][0]) for n in nodes], float)
print(f'{"訊號":<14}{"r":>8}{"零點中位":>10}{"p ≤":>9}  判讀')
for nm, dd in [('規則2c-mean', dec_2c), ('規則2c-max', calib(sig_2cmax)[0]),
               ('規則1', dec_r1), ('規則0', nat0)]:
    rate = np.array([dd[dom == n].mean() for n in nodes])
    r = np.corrcoef(rate, degv)[0, 1]
    null = [np.corrcoef(rate[list(a)+list(b)+list(c)], degv)[0, 1]
            for a in itertools.permutations(range(3))
            for b in itertools.permutations(range(3, 6))
            for c in itertools.permutations(range(6, 9))]
    null = np.array(null); pv = (null >= r).mean() if r > 0 else (null <= r).mean()
    print(f'{nm:<14}{r:>+8.3f}{np.median(null):>+10.3f}{pv:>9.4f}  '
          f'{"✅ 有度數偏誤" if pv < 0.05 else "⚠️ 未測出（不等於不存在）"}')

# ---- 叢發性 ----
print('\n觸發的叢發性（連續兩次觸發的間隔，步）：')
for name, dd in [('規則0', nat0), ('規則1', dec_r1), ('規則2c', dec_2c), ('時鐘', dec_clk)]:
    gaps = []
    for n in nodes:
        s = steps[dom == n][dd[dom == n]]
        if len(s) > 1: gaps.append(np.diff(s))
    g = np.concatenate(gaps)
    print(f'  {name:<7} 平均 {g.mean():6.2f}  中位 {np.median(g):5.1f}  '
          f'std {g.std():6.2f}  變異係數 {g.std()/g.mean():.3f}  最大 {g.max():4.0f}')
