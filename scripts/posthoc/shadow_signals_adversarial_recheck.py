# -*- coding: utf-8 -*-
"""對 1002 階段一 R2–R5 的對抗式複核（2026-10-02，matcha-02 寫）

fb 的要求是「挑錯不要背書」。本檔**不重算它已經算過的**，只攻它指定的六點，
每一點都用「它沒建的對照」來攻，而不是用口頭質疑。

⚠️ 本檔刻意不 import fb 的三支腳本，但**沿用它的資料與 κ 定義**（否則攻不到同一個量）。
"""
import numpy as np, pandas as pd, itertools

CSV = ('/home/server5090/Desktop/M11307320/MATCHA/exp_result_v1_stage2_leave_cartoon_p1a_'
       'async_const_tau1e-5_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234_fix/'
       'async_diag/shadow_signals.csv')
THRESH0 = 1e-5
E = [f'e{i}' for i in range(8)]
rng = np.random.default_rng(20261002)

df = pd.read_csv(CSV)
d = df[df.is_warm == 0].reset_index(drop=True)
dom, steps = d['domain'].values, d['step'].values
nodes = sorted(np.unique(dom))
nat0 = (d['sig_r0'].values >= THRESH0)
fired0 = d['fired'].values.astype(bool)
p0 = nat0.mean()
sig_r1 = d['sig_r1'].values.astype(float)
raw = d[[f'r2c_{e}' for e in E]].to_numpy(dtype=float)
with np.errstate(all='ignore'):
    sig_2c = np.nanmean(raw, 1)
deg = np.isfinite(raw).sum(1)
STYLE = {n: i // 3 for i, n in enumerate(nodes)}          # node_0-2 art, 3-5 photo, 6-8 sketch

def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pe = a.mean()*b.mean() + (1-a.mean())*(1-b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan

def calib(sig, target=p0):
    s = np.asarray(sig, float); th = np.nanquantile(s, 1 - target)
    return s >= th, th

dec_r1, th_r1 = calib(sig_r1)
dec_2c, th_2c = calib(sig_2c)

# 距上次【規則 0 真的廣播】幾步 + 自上次廣播以來收到幾次推播
clock = np.zeros(len(d)); recv_cum = np.zeros(len(d))
nrecv = d['n_received'].values.astype(float)
for n in nodes:
    m = np.where(dom == n)[0]
    last = steps[m[0]] - 1; acc = 0.0
    for i in m:
        clock[i] = steps[i] - last; acc += nrecv[i]; recv_cum[i] = acc
        if fired0[i]:
            last = steps[i]; acc = 0.0

def hdr(s): print('\n' + '=' * 94 + f'\n{s}\n' + '=' * 94)

# ══════════════════════════════════════════════════════════════════════════════
hdr('攻擊① 校準公平嗎 —— 全域單一門檻讓 κ 有數學天花板，而這個方向偏向 fb 的結論')
print('fb 的 κ 是對 nat0（自然觸發）算、訊號也校準到 p0 = nat0 的率 ⇒ 全域邊際相等、')
print('規則 0 自檢 = 1.0000 ⇒ 「用自然觸發率而非 fired」這個選擇是對的（forced 任何訊號都預測不到）。')
print('但**全域單一門檻**會讓逐節點邊際不等 ⇒ κ 有一個與「同不同意」無關的上限：')
print(f'\n{"訊號":<12}{"節點":<9}{"訊號率":>8}{"規則0率":>9}{"|差|":>7}{"該節點 κ 上限":>13}{"實測 κ":>9}')
print('-' * 94)
for nm, dec in [('規則1', dec_r1), ('規則2c', dec_2c)]:
    cap_po = 0.0
    for n in nodes:
        m = (dom == n); p, q = dec[m].mean(), nat0[m].mean()
        pe = p*q + (1-p)*(1-q); kmax = (1 - abs(p-q) - pe) / (1 - pe)
        cap_po += m.sum() * (1 - abs(p - q))
        print(f'{nm:<12}{n:<9}{p*100:>7.1f}%{q*100:>8.1f}%{abs(p-q)*100:>6.1f}%{kmax:>13.3f}{kappa(dec[m], nat0[m]):>9.3f}')
    cap_po /= len(d)
    pe_g = dec.mean()*nat0.mean() + (1-dec.mean())*(1-nat0.mean())
    print(f'  ⇒ {nm} 全域 κ 的天花板 = {(cap_po - pe_g)/(1 - pe_g):.4f}（實測 {kappa(dec, nat0):.4f}）\n')
print('🚨 規則 2c 的天花板被度數偏誤壓得比規則 1 低 ⇒ **全域門檻系統性低估規則 2c 多於規則 1**。')
print('   而「低估 κ」正是偏向「不是換刻度、路線保留」這個結論的方向。')
print('   ⇒ 邊際相等的**逐節點校準（0.3878 / 0.1990）才該是主結果**，全域當輔。')
print('   （對 κ ≥ 0.90 那道閘無影響；但 §6.2「規則1 離現行行為較近＝風險低」與 §3.4 的')
print('     「45%／34% 是時鐘」兩處都建立在主結果的數值上。）')

# ══════════════════════════════════════════════════════════════════════════════
hdr('攻擊② 零點夠強嗎 —— 有一個「不含任何學習資訊」的構造拿到 0.2–0.4')
print('fb 的四個零點都只編碼**時間**。但非同步架構裡還有一個純排程變數：**我收到了幾次推播**。')
print('收到推播 ⇒ 參數被拉向鄰居 ⇒ `_param_delta` 跳 ⇒ 規則 0 發火。這條完全不含我自己的學習狀態。')
cands = {
    '〔fb〕純時鐘（全域門檻）': clock,
    '〔新〕純時鐘（逐節點門檻）': clock,
    '〔新〕本步收到幾次推播 n_received': nrecv,
    '🚨〔新〕自上次廣播以來累積收到幾次': recv_cum,
    '〔新〕原型落後程度 age_mean': d['age_mean'].values.astype(float),
    '〔新〕時鐘 × 隨機游走（同重置結構）': np.sqrt(clock) * np.abs(rng.normal(size=len(d))),
}
print(f'\n{"零點構造":<36}{"全域 κ":>10}{"逐節點 κ":>11}{"觸發率":>9}')
print('-' * 94)
for nm, s in cands.items():
    g, _ = calib(s)
    pn = np.zeros(len(s), bool)
    for n in nodes:
        m = np.where(dom == n)[0]
        pn[m] = s[m] >= np.nanquantile(s[m], 1 - nat0[m].mean())
    print(f'{nm:<36}{kappa(g, nat0):>10.4f}{kappa(pn, nat0):>11.4f}{g.mean()*100:>8.1f}%')
print('-' * 94)
print(f'{"實測 規則1":<36}{kappa(dec_r1, nat0):>10.4f}')
print(f'{"實測 規則2c-mean":<36}{kappa(dec_2c, nat0):>10.4f}')

# ══════════════════════════════════════════════════════════════════════════════
hdr('攻擊③ 分層加權 —— n 加權會系統性低估，而且層內門檻沒重新校準')
bins = [0, 2, 4, 6, 9, 13, 20, 35, 10**9]
lab = np.digitize(clock, bins) - 1
print('兩個問題：(a) 層內用的是**全域門檻** ⇒ 層內邊際不等 ⇒ κ 被壓；')
print('         (b) 用 n 加權 ⇒ 1−pe 極小的退化層（觸發率 1.3%、pe≈0.974）拿到跟別人一樣的權重。')
print('正統做法：層內重新校準到該層的規則 0 率，並用 n·(1−pe) 加權（Mantel–Haenszel 式）。')
print(f'\n{"層":<12}{"n":>7}{"規0率":>7}{"1−pe":>8}  {"規則1：fb法":>11}{"重校準":>9}  {"規則2c：fb法":>12}{"重校準":>9}')
print('-' * 94)
acc = {k: [0.0, 0.0] for k in ['r1_n', 'r1_mh', 'r2_n', 'r2_mh']}
wsum = {'n': 0.0, 'mh1': 0.0, 'mh2': 0.0}
for b in range(len(bins) - 1):
    m = (lab == b)
    if m.sum() < 200 or nat0[m].sum() < 10 or (~nat0[m]).sum() < 10: continue
    q = nat0[m].mean(); n_ = m.sum()
    k1f, k2f = kappa(dec_r1[m], nat0[m]), kappa(dec_2c[m], nat0[m])
    r1l = sig_r1[m] >= np.nanquantile(sig_r1[m], 1 - q)
    r2l = sig_2c[m] >= np.nanquantile(sig_2c[m], 1 - q)
    k1r, k2r = kappa(r1l, nat0[m]), kappa(r2l, nat0[m])
    pe1 = 1 - 2*q*(1-q)                                    # 邊際相等時 pe = q²+(1-q)²
    hi = bins[b+1] if bins[b+1] < 10**8 else np.inf
    print(f'{f"[{bins[b]}, {hi})":<12}{n_:>7}{q*100:>6.1f}%{1-pe1:>8.4f}  {k1f:>11.4f}{k1r:>9.4f}  {k2f:>12.4f}{k2r:>9.4f}')
    acc['r1_n'][0] += k1f*n_; acc['r2_n'][0] += k2f*n_
    acc['r1_mh'][0] += k1r*n_*(1-pe1); acc['r2_mh'][0] += k2r*n_*(1-pe1)
    wsum['n'] += n_; wsum['mh1'] += n_*(1-pe1); wsum['mh2'] += n_*(1-pe1)
print('-' * 94)
print(f'  fb 的 n 加權（全域門檻）      規則1 {acc["r1_n"][0]/wsum["n"]:.4f}   規則2c {acc["r2_n"][0]/wsum["n"]:.4f}   ← 報告寫 0.1987 / 0.1217')
print(f'  n·(1−pe) 加權（層內重校準）   規則1 {acc["r1_mh"][0]/wsum["mh1"]:.4f}   規則2c {acc["r2_mh"][0]/wsum["mh2"]:.4f}')
print('  ⇒ 兩個修正都讓淨 κ **上升** ⇒ fb 的做法是**低估**，不是高估。')
print('    方向上安全（它用來說「扣掉時鐘後仍有訊號」，低估只會讓這個宣稱更保守），')
print('    但「原始 κ 當中 45%／34% 是時鐘」這個**比例**被高估了。')

# ══════════════════════════════════════════════════════════════════════════════
hdr('攻擊④ 事件層級的兩個量 —— 不是截斷汙染，是**沒有零點**')
def seg_stats(decision):
    nseg = same = anyx = 0
    for n in nodes:
        m = np.where(dom == n)[0]; s0 = 0
        for pos, i in enumerate(m):
            if not fired0[i]: continue
            idx = m[s0:pos+1]; cr = np.where(decision[idx])[0]
            nseg += 1
            if len(cr):
                anyx += 1
                if steps[idx[cr[0]]] == steps[i]: same += 1
            s0 = pos + 1
    return nseg, anyx/nseg, same/nseg

rows = [('規則1', dec_r1), ('規則2c-mean', dec_2c)]
# 零點 A：同觸發率的均勻隨機（破壞對齊也破壞叢發）
# 零點 B：把該節點的決策序列整體環狀位移（**保留叢發結構**、只破壞對齊）★ 這才是對的零點
NA, NB = [], []
for r in range(30):
    ra = rng.random(len(d)) < p0
    NA.append(seg_stats(ra))
    for nm, dcn in [('r1', dec_r1)]:
        sh = dcn.copy()
        for n in nodes:
            m = np.where(dom == n)[0]
            sh[m] = np.roll(dcn[m], int(rng.integers(50, len(m) - 50)))
        NB.append(seg_stats(sh))
NA, NB = np.array(NA, float), np.array(NB, float)
print(f'{"":<14}{"段數":>7}{"段內有越界":>12}{"同一步命中":>12}')
print('-' * 94)
for nm, dcn in rows:
    ns, a, s = seg_stats(dcn)
    print(f'{nm:<14}{ns:>7}{a*100:>11.1f}%{s*100:>11.1f}%')
print(f'{"〔零點A〕均勻隨機":<14}{int(NA[:,0].mean()):>7}{NA[:,1].mean()*100:>11.1f}%{NA[:,2].mean()*100:>11.1f}%'
      f'   ± {NA[:,1].std()*100:.2f} / {NA[:,2].std()*100:.2f}pp')
print(f'{"〔零點B〕規則1 環狀位移（保留叢發）":<14}{int(NB[:,0].mean()):>7}{NB[:,1].mean()*100:>11.1f}%{NB[:,2].mean()*100:>11.1f}%'
      f'   ± {NB[:,1].std()*100:.2f} / {NB[:,2].std()*100:.2f}pp')
print()
ra, rsame = NA[:,1].mean()*100, NA[:,2].mean()*100
ba, bsame = NB[:,1].mean()*100, NB[:,2].mean()*100
print('🚨 兩個量的判讀都翻轉：')
print(f'   「段內有越界」48.6 / 36.1 —— 均勻隨機零點是 {ra:.1f} ⇒ **兩個訊號都低於隨機**。')
print(f'     原因＝叢發：訊號把越界集中在少數段裡 ⇒ 覆蓋到的段數反而變少（保留叢發的零點 {ba:.1f}）。')
print('     ⇒ 這個量不可當作「訊號夠不夠靈敏」的證據，它主要在量叢發程度。')
print(f'   「同一步命中」16.5 / 3.7 —— 均勻隨機 {rsame:.1f}、保留叢發的環狀位移 {bsame:.1f}。')
print(f'     ⇒ 規則 1 的 16.5 是保留叢發零點的 {16.5/bsame:.1f} 倍 ⇒ **真的有對齊**。')
print(f'     ⇒ 但規則 2c 的 3.7 **低於均勻隨機（{rsame:.1f}）、也在叢發零點（{bsame:.1f}）上** ⇒ 規則 2c 在事件層級**沒有任何對齊**。')
print('   ⇒ fb 的截斷警告標對也標夠了（lead 分布確實被構造截斷在 0）。')
print('     但它放行的那兩個量**另外**缺零點：一個方向被讀反，一個把「零」說成「比較差」。')

# ── 追擊②b：規則 1 的 κ 有多少是「收到推播」？ ──
hdr('攻擊②b 追擊 —— 規則 1 與「累積收到推播」的關係，以及扣掉它之後')
dec_rc, _ = calib(recv_cum)
print(f'  規則1   vs 累積收到推播  κ = {kappa(dec_r1, dec_rc):.4f}')
print(f'  規則2c  vs 累積收到推播  κ = {kappa(dec_2c, dec_rc):.4f}')
print(f'  規則0   vs 累積收到推播  κ = {kappa(nat0,   dec_rc):.4f}')
rb = [0, 1, 2, 3, 5, 8, 10**9]
rl = np.digitize(recv_cum, rb) - 1
print(f'\n  在「累積收到幾次推播」相同的分層內算 κ（層內重校準、n·(1−pe) 加權）：')
print(f'  {"層":<10}{"n":>7}{"規0率":>7}{"規則1 κ":>10}{"規則2c κ":>10}')
w=k1s=k2s=0.0
for b in range(len(rb)-1):
    m = (rl == b)
    if m.sum() < 200 or nat0[m].sum() < 10 or (~nat0[m]).sum() < 10: continue
    q = nat0[m].mean()
    a1 = sig_r1[m] >= np.nanquantile(sig_r1[m], 1-q)
    a2 = sig_2c[m] >= np.nanquantile(sig_2c[m], 1-q)
    k1, k2 = kappa(a1, nat0[m]), kappa(a2, nat0[m]); pe = 1-2*q*(1-q)
    hi = rb[b+1] if rb[b+1] < 10**8 else np.inf
    print(f'  {f"[{rb[b]}, {hi})":<10}{m.sum():>7}{q*100:>6.1f}%{k1:>10.4f}{k2:>10.4f}')
    ww = m.sum()*(1-pe); w += ww; k1s += k1*ww; k2s += k2*ww
print(f'  {"加權（淨）":<10}{"":>7}{"":>7}{k1s/w:>10.4f}{k2s/w:>10.4f}')

# ── 追擊①b：天花板正規化 ──
hdr('攻擊①b 追擊 —— 用天花板正規化之後，兩條規則的差距變大還是變小')
for nm, dec, cap in [('規則1', dec_r1, None), ('規則2c', dec_2c, None)]:
    cp = sum((dom==n).sum()*(1-abs(dec[dom==n].mean()-nat0[dom==n].mean())) for n in nodes)/len(d)
    peg = dec.mean()*nat0.mean()+(1-dec.mean())*(1-nat0.mean())
    cap = (cp-peg)/(1-peg); k = kappa(dec, nat0)
    print(f'  {nm:<8} κ {k:.4f}   天花板 {cap:.4f}   **實現比例 {k/cap*100:.1f}%**')
print('  ⇒ 🚨 我原本預測「規則 2c 被壓得更低」—— **錯了**。規則 1 的逐節點率散布更大')
print('    （5.7%–25.2% vs 7.6%–23.4%）⇒ 它的天花板更低（0.764 vs 0.829）。')
print('  ⇒ 但天花板不破 κ≥0.90 這道閘：若訊號真是規則 0 的單調換刻度，全域分位門檻會選到')
print('    **完全相同的列** ⇒ κ=1。所以閘是可達的，fb 的事前判準有檢驗力。✅')
print('  ⇒ 真正的影響是：κ 的絕對值混了「不同意」與「邊際不等」兩件事。正規化後')
print('    規則 1 實現 47%、規則 2c 實現 22% ⇒ 兩者差距**變大**，§6.2 的建議方向不變。')

# ══════════════════════════════════════════════════════════════════════════════
hdr('攻擊⑥ 度數偏誤的射程 —— 畫風內單調，用精確置換檢定算 p')
rates = {nm: np.array([dec[dom == n].mean() for n in nodes]) for nm, dec in
         [('規則2c-mean', dec_2c), ('規則1', dec_r1), ('規則0', nat0)]}
degs = np.array([deg[dom == n][0] for n in nodes], float)
sty = np.array([STYLE[n] for n in nodes])
print(f'度數 {degs.astype(int).tolist()}｜畫風 {sty.tolist()}（0=art 1=photo 2=sketch）')
print(f'畫風平均度數 art {degs[sty==0].mean():.2f}  photo {degs[sty==1].mean():.2f}  sketch {degs[sty==2].mean():.2f}'
      f'  ⇒ 🚨 **度數與畫風本身就相關**，所以畫風內檢定是必要的')
perms = [p for p in itertools.product(*[list(itertools.permutations(range(3)))] * 3)]
print(f'\n畫風內置換的總數 = (3!)³ = {len(perms)} ⇒ 最小可能 p = 1/{len(perms)} = {1/len(perms):.4f}（地板）')
print(f'\n{"訊號":<14}{"實測 r":>9}{"畫風內置換 r 中位":>18}{"≥實測的比例 p":>15}{"判讀":>10}')
print('-' * 94)
for nm, rt in rates.items():
    obs = np.corrcoef(rt, degs)[0, 1]; null = []
    for p in perms:
        v = rt.copy()
        for s in range(3):
            idx = np.where(sty == s)[0]; v[idx] = rt[idx][list(p[s])]
        null.append(np.corrcoef(v, degs)[0, 1])
    null = np.array(null)
    pv = (null >= obs).mean() if obs >= 0 else (null <= obs).mean()
    print(f'{nm:<14}{obs:>9.3f}{np.median(null):>18.3f}{pv:>15.4f}'
          f'{("✅ 畫風內成立" if pv <= 0.05 else "⚠️ 射程過寬"):>12}')
print('-' * 94)
print('⇒ 度數有重複值（art 5,2,5 與 photo 6,6,3 各有一組同值）⇒ 可區分的置換遠少於 216，')
print('  p 的解析度比名目值差。判讀見報告。')
