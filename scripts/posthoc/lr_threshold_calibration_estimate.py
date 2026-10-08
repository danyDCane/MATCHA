# -*- coding: utf-8 -*-
"""D 臂（參數觸發＋門檻隨學習率衰減，EF-HC/PersonalizedET 原版 γ(k)=α(k)）的係數 r 離線推估（2026-10-07，零 GPU）

目標：找 r，使 `fired ⟺ delta ≥ r·lr(k)/lr(0)  或  距上次送出 ≥50 步` 的總傳輸份數 ≈ 29,614（＝A、C）。
難點：D 的送出時機不同 ⇒ 「距上次送出 g 步的參數距離」要用模型推估：delta(k, g) ≈ v_i(k)·g^β。
  - v_i(k)、β：用 C 的 broadcast_log 擬合（C 每 6–7 步重設參照、全程都有 g＝1…7 的觀測）。
  - β 外推到長間隔：用 A 後段（g 可到 50）檢查。
lr(k)：CosineAnnealingLR(T_max＝4000, eta_min＝0)，scheduler 在決策前已 step ⇒ γ(k)＝½(1+cos(π(k+1)/4000))。
⚠️ 這是近似：D 的融合時序與 C 不同、v 用 C 的軌跡代替 D 的軌跡。發車後用前 50 epoch 的實際份數核對（見輸出的逐段預期）。
"""
import numpy as np, pandas as pd, networkx as nx
ROOT = '/home/server5090/Desktop/M11307320/MATCHA/'
C = ROOT + 'exp_result_v1_stage2_leave_cartoon_p1a_async_periodic6.2133_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234'
A = ROOT + 'exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234_fix'
T, N, MAXI, TARGET = 4000, 9, 50, 29614
G = nx.random_geometric_graph(9, 0.8, seed=1234); deg = np.array([G.degree(i) for i in range(N)])
gam = 0.5 * (1 + np.cos(np.pi * (np.arange(T) + 1) / T))


def with_gap(path):
    b = pd.read_csv(path + '/async_diag/broadcast_log.csv'); b['i'] = b.domain.str[5:].astype(int)
    out = []
    for i, g in b.sort_values('step').groupby('i'):
        last, gaps = 0, []
        for k, f in zip(g.step.values, g.fired.values):
            gaps.append(k - last)
            if f: last = k
        g = g.copy(); g['gap'] = gaps; out.append(g)
    return pd.concat(out)


c = with_gap(C); c = c[(c.gap >= 1) & (c.delta > 0)]
# β：逐 epoch 段做 log delta ~ log gap 迴歸
print('β 擬合（C，g＝1…7）：')
for lo, hi in [(1, 10), (11, 50), (51, 100), (101, 150), (151, 200)]:
    q = c[(c.epoch >= lo) & (c.epoch <= hi)]
    beta = np.polyfit(np.log(q.gap), np.log(q.delta), 1)[0]
    print(f'  ep{lo}-{hi}: β＝{beta:.3f}（n＝{len(q)}）')
a = with_gap(A); a = a[(a.gap >= 1) & (a.delta > 0)]
qa = a[a.epoch > 150]
print(f'  A 後段 ep151-200（g 1…50）：β＝{np.polyfit(np.log(qa.gap), np.log(qa.delta), 1)[0]:.3f}（n＝{len(qa)}）')
BETA = float(np.polyfit(np.log(c.gap), np.log(c.delta), 1)[0])
print(f'  全程（C）β＝{BETA:.3f} ⇒ 模擬用此值；另跑 β＝0.5／1.0 看敏感度')

# v_i(epoch)：每節點每 epoch 的 median(delta / g^β)
def v_table(beta):
    cc = c.copy(); cc['v'] = cc.delta / cc.gap ** beta
    V = cc.groupby(['i', 'epoch']).v.median().unstack(0).reindex(range(1, 201)).ffill().bfill().values  # [200, 9]
    return V


def simulate(r, beta, V):
    fires = np.zeros(N, int); seg = np.zeros((4, N), int)
    for i in range(N):
        s = 0
        for k in range(T):
            g = k - s
            if g == 0:
                continue
            d = V[k // 20, i] * g ** beta
            if d >= r * gam[k] or g >= MAXI:
                fires[i] += 1; seg[min(k // 1000, 3), i] += 1; s = k
    return (fires * deg).sum(), fires, seg


for beta in [BETA, 0.5, 1.0]:
    V = v_table(beta)
    lo, hi = 1e-8, 1e-3
    for _ in range(40):                       # 對數二分：總份數隨 r 單調遞減
        mid = np.sqrt(lo * hi); tot = simulate(mid, beta, V)[0]
        lo, hi = (mid, hi) if tot > TARGET else (lo, mid)
    r = np.sqrt(lo * hi); tot, fires, seg = simulate(r, beta, V)
    print(f'\nβ＝{beta:.3f}：r ≈ {r:.3e} ⇒ 預估總傳輸 {tot}、送出 {fires.sum()}、逐節點 {fires.tolist()}')
    print('  逐段送出占比（ep1-50/51-100/101-150/151-200）：' + ' / '.join(f'{100*seg[q].sum()/fires.sum():.1f}%' for q in range(4))
          + f'；逐段預估傳輸份數 ' + ' / '.join(str(int((seg[q]*deg).sum())) for q in range(4)))
