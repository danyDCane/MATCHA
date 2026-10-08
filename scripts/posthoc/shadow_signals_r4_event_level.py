# -*- coding: utf-8 -*-
"""
R4c 事件層級的重疊（修正影子設定造成的「叢發」偏差）
====================================================
🚨 問題：sig_r1 的參照 proto_ref 只在【規則 0 真的廣播】時重置。
   規則 0 沉默很久時 sig_r1 持續累積 ⇒ 連續好幾步都在門檻之上 ⇒ 逐步比對會記成好幾次「多餘觸發」。
   但真部署下規則 1 一發火就會自己重置 ⇒ 不會叢發。
   ⇒ **逐步 κ 被影子設定系統性低估**。

修正做法：用規則 0 的廣播把時間軸切成「區段」。每段內兩條規則各只能發一次：
   規則 0 在段末（定義如此）；規則 1／2c 取段內【第一次越過門檻】。
   ⇒ 比的是「同一個重置週期裡，兩個訊號會不會在同一步發火」—— 這才是部署下的真問題。
   同時量 lead／lag（規則 1 比規則 0 早幾步發火）。
"""
import numpy as np, pandas as pd

CSV = ('/home/server5090/Desktop/M11307320/MATCHA/exp_result_v1_stage2_leave_cartoon_p1a_'
       'async_const_tau1e-5_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234_fix/'
       'async_diag/shadow_signals.csv')
E = [f'e{i}' for i in range(8)]
df = pd.read_csv(CSV)
print(f'全程（含暖機）規則 0 觸發率 {df.fired.mean()*100:.2f}%  ← 先前引用的 15.78% 是這個口徑')
d = df[df.is_warm == 0].reset_index(drop=True)
dom, steps = d['domain'].values, d['step'].values
nodes = sorted(np.unique(dom))
fired0 = d['fired'].values.astype(bool)
nat0 = (d['sig_r0'].values >= 1e-5)
p0 = nat0.mean()
sig = {'規則1': d['sig_r1'].values.astype(float)}
raw = d[[f'r2c_{e}' for e in E]].to_numpy(dtype=float)
with np.errstate(all='ignore'):
    sig['規則2c-mean'] = np.nanmean(raw, 1)
    sig['規則2c-max'] = np.nanmax(raw, 1)

def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pe = a.mean()*b.mean() + (1-a.mean())*(1-b.mean())
    return (po - pe)/(1 - pe) if pe < 1 else np.nan

print('=' * 94)
print(f'事件層級：用規則 0 的廣播切段（非暖機區，共 {int(fired0.sum())} 個廣播事件）')
print('=' * 94)

for name, s in sig.items():
    # 門檻掃描：讓「段內第一次越界」的平均提前量最接近 0（＝同觸發節奏）；
    # 但為可比，主結果仍用與逐步分析相同的全域分位門檻。
    th = np.nanquantile(s, 1 - p0)
    leads, same, never, nseg = [], 0, 0, 0
    collapsed_a, collapsed_b = [], []
    for n in nodes:
        m = np.where(dom == n)[0]
        seg_start = 0
        for pos, i in enumerate(m):
            if not fired0[i]:
                continue
            idx = m[seg_start:pos+1]                 # 本段（含段末的規則 0 觸發那一步）
            ss = s[idx]
            cross = np.where(np.isfinite(ss) & (ss >= th))[0]
            nseg += 1
            if len(cross) == 0:
                never += 1
                leads.append(np.nan)
            else:
                lead = steps[idx[cross[0]]] - steps[i]   # 0 ＝ 同一步；負 ＝ 提前
                leads.append(lead)
                if lead == 0: same += 1
            seg_start = pos + 1
    L = np.array(leads, float)
    fin = np.isfinite(L)
    print(f'\n【{name}】全域門檻 {th:.4f}°（對齊 {p0*100:.2f}% 逐步觸發率）')
    print(f'  段數 {nseg}｜段內有越界 {fin.sum()}（{fin.mean()*100:.1f}%）｜'
          f'整段都沒越界（要靠安全閥）{never}（{never/nseg*100:.1f}%）')
    print(f'  與規則 0 同一步發火：{same} 段（{same/nseg*100:.1f}%）')
    print(f'  提前量 lead（步；0＝同步、負＝比規則 0 早）：'
          f'中位 {np.nanmedian(L):.1f}｜平均 {np.nanmean(L):.2f}｜'
          f'p10 {np.nanquantile(L,.1):.0f}｜p90 {np.nanquantile(L,.9):.0f}')
    for lo, hi, tag in [(-0.5,0.5,'同一步'), (-1.5,-0.5,'早 1 步'), (-3.5,-1.5,'早 2–3 步'),
                        (-8.5,-3.5,'早 4–8 步'), (-10**9,-8.5,'早 >8 步')]:
        c = int(((L > lo) & (L <= hi)).sum())
        print(f'    {tag:<10}{c:>6} 段（{c/nseg*100:>5.1f}%）')

# ---- 事件層級的零點（matcha-02 2026-10-02）----
# 🚨 「段內有越界」與「同一步命中」原本都【沒有零點】⇒ 不可直接判讀。
#    零點 A＝同率均勻隨機；零點 B＝把規則1 的決策在節點內環狀位移
#    （**保留叢發結構、只破壞時間對齊**）—— B 才是對的零點。
rng = np.random.default_rng(7)

def event_stats(dec):
    seg = hit = any_ = 0
    for n in nodes:
        m = np.where(dom == n)[0]; st = 0
        for pos, i in enumerate(m):
            if not fired0[i]: continue
            idx = m[st:pos+1]; seg += 1
            any_ += bool(dec[idx].any())
            cr = np.where(dec[idx])[0]
            if len(cr): hit += int(steps[idx[cr[0]]] == steps[i])
            st = pos + 1
    return any_/seg*100, hit/seg*100

print('\n' + '=' * 94)
print('★ 事件層級的零點（這兩個量原本沒有零點、不可直接判讀）')
print('=' * 94)
print(f'{"":<24}{"段內有越界":>14}{"同一步命中":>14}')
for nm, ss in [('規則1', sig['規則1']), ('規則2c-mean', sig['規則2c-mean'])]:
    a, h = event_stats(np.asarray(ss, float) >= np.nanquantile(np.asarray(ss, float), 1 - p0))
    print(f'{nm:<24}{a:>13.1f}%{h:>13.1f}%')
A = np.array([event_stats(rng.random(len(d)) < p0) for _ in range(20)])
print(f'{"〔零點A〕均勻隨機":<24}{A[:,0].mean():>8.1f}%±{A[:,0].std():.2f}{A[:,1].mean():>8.1f}%±{A[:,1].std():.2f}')
base = np.asarray(sig['規則1'], float) >= np.nanquantile(np.asarray(sig['規則1'], float), 1 - p0)
B = []
for _ in range(20):
    dd = np.zeros(len(d), bool)
    for n in nodes:
        m = np.where(dom == n)[0]
        dd[m] = np.roll(base[m], int(rng.integers(100, len(m) - 100)))
    B.append(event_stats(dd))
B = np.array(B)
print(f'{"〔零點B〕環狀位移（保叢發）":<24}{B[:,0].mean():>8.1f}%±{B[:,0].std():.2f}{B[:,1].mean():>8.1f}%±{B[:,1].std():.2f}')
print("""
🚨 兩條判讀修正：
 ①「段內有越界」**不可當靈敏度**：均勻隨機就有 ~51%，兩個訊號都【低於】隨機
   ⇒ 那個量主要在量叢發程度，不是靈敏度。
 ②「同一步命中」要對 **零點 B**（保叢發）讀：
   規則1 16.5% ＝ 零點 B 的 ~3.5 倍 ⇒ **真的有對齊**；
   規則2c 3.7% **低於均勻隨機、落在零點 B 上** ⇒ **規則2c 在事件層級沒有任何對齊**。
   ⇒ 不是「規則1 比規則2c 好 4.5 倍」，是「**規則2c 在這個量上是零**」。""")

# 段長參照
seglen = []
for n in nodes:
    s_ = steps[dom == n][fired0[dom == n]]
    if len(s_) > 1: seglen.append(np.diff(s_))
sl = np.concatenate(seglen)
print(f'\n參照：規則 0 的段長（兩次廣播間隔）中位 {np.median(sl):.0f} 步、平均 {sl.mean():.2f} 步')
print('''
🚨 lead 的解讀限制（**不得宣稱「原型訊號比參數訊號早」**）：
   段末那一步【依定義】就是規則 0 發火的時點 ⇒ 段內任何越界都只可能是 lead ≤ 0。
   **提前量的分布被構造截斷在 0**，所以「中位早 1 步」不是提早的證據。
   這組數字唯一能讀的是「同一步」那一格（＝事件層級的命中率），**而且必須對零點 B 讀**（見下）。
   ⚠️ 2026-10-02 更正：「段內有沒有越界」**不是靈敏度**——均勻隨機就有 ~51%、兩個訊號都更低。
   真要判「誰先動」必須跑階段二（讓訊號自己驅動廣播），離線判不出來。''')
