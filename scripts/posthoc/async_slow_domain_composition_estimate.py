"""候選二離線估算：某畫風節點觸發變稀時，單向（現行）vs 母版式(7)雙向融合下各畫風在共識裡的份量；附 matcha-b6 數字複核。"""
import numpy as np, pandas as pd, networkx as nx
# 只取函式與資料，不重跑該檔的報表：手動重建
RUN = ('/home/server5090/Desktop/M11307320/MATCHA/exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234_fix')
G = nx.random_geometric_graph(9, 0.8, seed=1234); N = 9; NB = {i: sorted(G[i]) for i in range(N)}; deg = {i: len(NB[i]) for i in range(N)}
b = pd.read_csv(f'{RUN}/async_diag/broadcast_log.csv'); b['i'] = b['domain'].str.replace('node_', '').astype(int)
F = b.pivot(index='step', columns='i', values='fired').sort_index().values.astype(bool)
FO = b.pivot(index='step', columns='i', values='forced').sort_index().values.astype(bool)
T = F.shape[0]
# ---- 複核 b6 的數字 ----
print('== 複核 matcha-b6 ==')
for lo, hi in [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]:
    f = F[lo:hi]; fo = FO[lo:hi]; r = f.mean(0)
    print(f'  ep{lo//20+1:3d}-{hi//20:3d}: 觸發率 {f.mean()*100:5.2f}%  強制占廣播 {fo.sum()/max(f.sum(),1)*100:5.1f}%  逐節點最大/最小 {r.max()/r.min():.2f}x  最低節點 node_{r.argmin()}  ({r.min()*100:.1f}%)')
tot = F.sum(); print(f'  前 100 epoch 占全部廣播 {F[:2000].sum()/tot*100:.1f}%')
# ---- 候選二離線估算：某畫風三節點觸發變稀（逐事件以機率 q 保留） ----
W = {(i, j): 1.0 / (1 + max(deg[i], deg[j])) for i in range(N) for j in NB[i]}
DOM = ['art']*3 + ['photo']*3 + ['sketch']*3
def infl(Fm, two_way):
    r = np.full(N, 1.0 / N); acc = np.zeros(N)
    for k in range(Fm.shape[0] - 1, -1, -1):
        acc += r
        if two_way:
            # 母版式(7)：同一步內任一端觸發 ⇒ 該邊兩端以 MH 權重對稱混合（同步取當下值）⇒ P(k) 對稱雙隨機
            P = np.eye(N)
            for i in range(N):
                for j in NB[i]:
                    if Fm[k, i] or Fm[k, j]:
                        P[i, j] = W[(i, j)]
            for i in range(N): P[i, i] = 1 - (P[i].sum() - P[i, i])
            r = r @ P
        else:
            for i in reversed(range(N)):
                snd = [j for j in NB[i] if (Fm[k, j] if j < i else (k > 0 and Fm[k - 1, j]))]
                if not snd: continue
                ri = r[i]; s = 0.0
                for j in snd: r[j] += ri * W[(i, j)]; s += W[(i, j)]
                r[i] = ri * (1 - s)
    m = acc / Fm.shape[0]
    return {d: sum(m[i] for i in range(N) if DOM[i] == d) for d in ['art', 'photo', 'sketch']}
rng = np.random.default_rng(0)
print('\n== 候選二離線估算：某畫風三節點的觸發以機率 q 保留（其餘節點不變），全程平均各畫風份量（公平 0.333）==')
for slow in ['sketch', 'photo']:
    idx = [i for i in range(N) if DOM[i] == slow]
    for q in [1.0, 0.5, 0.25, 0.1]:
        Fm = F.copy()
        for i in idx: Fm[:, i] &= rng.random(T) < q
        o = infl(Fm, False); t = infl(Fm, True)
        print(f'  慢={slow:6s} q={q:4.2f}: 單向 {slow} 份量 {o[slow]:.3f}（art {o["art"]:.3f} photo {o["photo"]:.3f} sketch {o["sketch"]:.3f}）'
              f' | 雙向 {slow} 份量 {t[slow]:.3f}')
