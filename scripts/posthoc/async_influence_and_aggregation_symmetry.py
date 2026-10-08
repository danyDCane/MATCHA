"""非同步單向推播下，各節點對「全網平均模型」的長期影響力（零 GPU 探針）。
重建：沿用 trigger_relay_vs_redundant.py 的拓撲與收信規則（自檢 n_received 逐列吻合）。
每步 k 依序處理 node 0..8：node i 收信時，用 MH 權重 w_ij=1/(1+max(deg_i,deg_j)) 取用寄件人，自身權重 1-Σw。
影響力：r_T = 1ᵀ/N；反向遞推 r ← r·A_{i,k}（每步依 8..0 的反序），r_s[j] ＝ 第 s 步時節點 j 的狀態對最終全網平均的權重。
同步 MH（對稱雙隨機）下 r 恆為 1/9。
"""
import numpy as np, pandas as pd, networkx as nx, sys
RUN = ('/home/server5090/Desktop/M11307320/MATCHA/exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_'
       'osdg_excl_person_seed2026_topo1234_fix')
if len(sys.argv) > 1:          # 可指定其他 run（1007：本地時鐘臂）
    RUN = sys.argv[1]
G = nx.random_geometric_graph(9, 0.8, seed=1234); assert nx.is_connected(G)
N = 9; NB = {i: sorted(G[i]) for i in range(N)}; deg = {i: len(NB[i]) for i in range(N)}
b = pd.read_csv(f'{RUN}/async_diag/broadcast_log.csv'); b['i'] = b['domain'].str.replace('node_', '').astype(int)
F = b.pivot(index='step', columns='i', values='fired').sort_index().values.astype(bool)
R = b.pivot(index='step', columns='i', values='n_received').sort_index().values
T = F.shape[0]
pred = np.zeros_like(R)
for i in range(N):
    for j in NB[i]:
        if j < i: pred[:, i] += F[:, j]
        else:     pred[1:, i] += F[:-1, j]
print(f'度數 {deg}')
print(f'自檢：重建 n_received 吻合 {(pred == R).mean()*100:.3f}%（T={T} 步）')
W = {(i, j): 1.0 / (1 + max(deg[i], deg[j])) for i in range(N) for j in NB[i]}
DOM = {0: 'art', 1: 'art', 2: 'art', 3: 'photo', 4: 'photo', 5: 'photo', 6: 'sketch', 7: 'sketch', 8: 'sketch'}

def influence(Fm, order=None):
    """回傳 r_hist[T, N]：r_hist[s] ＝ 第 s 步結束時各節點狀態對最終平均的權重。"""
    order = list(range(N)) if order is None else order
    pos = {n: p for p, n in enumerate(order)}
    r = np.full(N, 1.0 / N); hist = np.zeros((Fm.shape[0], N))
    for k in range(Fm.shape[0] - 1, -1, -1):
        hist[k] = r
        for i in reversed(order):              # P_k = A_{last}…A_{first} ⇒ r·P_k 先乘最後處理的節點
            snd = [j for j in NB[i] if (Fm[k, j] if pos[j] < pos[i] else (k > 0 and Fm[k - 1, j]))]
            if not snd: continue
            ri = r[i]; s = 0.0
            for j in snd:
                r[j] += ri * W[(i, j)]; s += W[(i, j)]
            r[i] = ri * (1 - s)
    return hist

def report(name, H, Fm):
    rate = Fm.mean(0)
    print(f'\n== {name} ==  總觸發率 {Fm.mean()*100:.2f}%')
    for lo, hi in [(0, 1000), (1000, 2000), (2000, 3000), (3000, T)]:
        m = H[lo:hi].mean(0)
        d = {s: sum(m[i] for i in range(N) if DOM[i] == s) for s in ['art', 'photo', 'sketch']}
        print(f'  步 {lo:4d}-{hi:4d}: 逐節點影響力×9 = ' + ' '.join(f'{9*x:.3f}' for x in m)
              + f' | 各畫風占比 art {d["art"]:.3f} photo {d["photo"]:.3f} sketch {d["sketch"]:.3f}（均等＝0.333）'
              + f' | 最大/最小 {m.max()/m.min():.2f}x')
    print('  逐節點觸發率% = ' + ' '.join(f'{100*x:.1f}' for x in rate))
    mall = H.mean(0); print(f'  全程影響力×9 與觸發率的相關 r = {np.corrcoef(mall, rate)[0,1]:+.3f}；與度數 r = {np.corrcoef(mall, [deg[i] for i in range(N)])[0,1]:+.3f}')

H = influence(F); report('實際（依編號順序處理）', H, F)
rng = np.random.default_rng(0)
# 零點 1：保留每節點觸發率、打散時間（20 次）
Hs = []
for _ in range(20):
    Fp = np.stack([rng.permutation(F[:, i]) for i in range(N)], 1); Hs.append(influence(Fp).mean(0))
Hs = np.array(Hs); print(f'\n== 零點1：保留各節點觸發率、時間打散（20 次）==\n  全程影響力×9 平均 = ' + ' '.join(f'{9*x:.3f}' for x in Hs.mean(0)) + f'\n  std×9 = ' + ' '.join(f'{9*x:.3f}' for x in Hs.std(0)))
# 零點 2：所有節點同一觸發率（全體平均）、隨機時間（20 次）
p = F.mean(); Hs2 = np.array([influence(rng.random(F.shape) < p).mean(0) for _ in range(20)])
print(f'\n== 零點2：全體同觸發率 {p*100:.2f}%、隨機時間（20 次）==\n  全程影響力×9 平均 = ' + ' '.join(f'{9*x:.3f}' for x in Hs2.mean(0)) + f'\n  std×9 = ' + ' '.join(f'{9*x:.3f}' for x in Hs2.std(0)))
# 順序效應：反序處理（同一份 F）
Hr = influence(F, order=list(range(N - 1, -1, -1)))
print(f'\n== 同一份觸發紀錄改成反序處理（只看處理順序的影響）==\n  全程影響力×9 = ' + ' '.join(f'{9*x:.3f}' for x in Hr.mean(0)) + f'\n  （對照實際）  = ' + ' '.join(f'{9*x:.3f}' for x in H.mean(0)))

# ---------- 通訊量：單向推播（現行）vs 母版式(7) 雙向融合 vs 同步 ----------
E = G.number_of_edges(); deg_arr = np.array([deg[i] for i in range(N)])
sync = 2 * E * T                                  # 同步：每步每條邊雙向各傳一份
oneway = int((F * deg_arr[None, :]).sum())        # 現行：觸發節點傳給所有鄰居
edges = list(G.edges())
both = sum(int(np.logical_or(F[:, i], F[:, j]).sum()) * 2 for i, j in edges)   # 母版：任一端觸發 ⇒ 該邊雙向各一份
print(f'\n== 通訊量（模型傳輸份數，200 epoch）==  邊數 E={E}')
print(f'  同步：{sync}；現行單向：{oneway}（＝同步的 {oneway/sync*100:.2f}%，省 {100-oneway/sync*100:.2f}%）')
print(f'  母版式(7)雙向（同一份觸發紀錄）：{both}（＝同步的 {both/sync*100:.2f}%，省 {100-both/sync*100:.2f}%）；雙向/單向 ＝ {both/oneway:.3f}×')
