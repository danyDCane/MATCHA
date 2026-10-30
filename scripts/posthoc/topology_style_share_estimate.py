"""[1013 討論] 拓樸會不會改變「畫風份量此消彼長」？零 GPU 估算。
同一組送出週期（1012 ArtSlow：art 三台＋node_4 週期 32.48，其餘照表）放到三種拓樸：
  RGG（現行 seed1234）／環形・同畫風相鄰（util.py 現行 ring：cycle 0-1-…-8）／環形・同畫風分散（環上順序 0,3,6,1,4,7,2,5,8）。
⚠️ [1013 自檢] ① 的反向遞推（沿用 1003）把對方同一步已融合的內容算進來（同一步沿節點編號接力），train.py 實際是
「訓練 → 送出未融合快照 → 收」。隨機拓樸下兩者幾乎相同（--faithful_check），環形相鄰排法下差很多 ⇒ 份量一律改印照實際順序的正向值。
量兩件事：
  ① 共識份量：1003 單向遞推（同 artslow_period_and_share_estimate.infl_oneway，只是拓樸參數化）。
  ② 各節點自己模型的資料組成：正向模擬「每步本地訓練加一份自己的梯度（權重＝餘弦學習率）→ 收到就照 MH 權重融合」，
     看 ep100（step 2000）與 ep200 時 art 節點自己模型裡 art 資料佔多少。
一階理論：對稱權重的單向推播，期望混合矩陣的穩態 π_j ∝ 送出率 r_j（與拓樸無關）⇒ 畫風份量 ≈ 該畫風送出次數佔比。"""
import sys, numpy as np, networkx as nx
sys.path.insert(0, '/home/server5090/Desktop/M11307320/MATCHA/scripts/posthoc')
from artslow_period_and_share_estimate import fire_matrix
import pandas as pd
N, T = 9, 4000
DOM = ['art'] * 3 + ['photo'] * 3 + ['sketch'] * 3
PER = pd.read_csv('/home/server5090/Desktop/M11307320/MATCHA/results/proto_trigger/1012_artslow/period_Cbw_art0.1.csv').period.values


def graph(kind):
    if kind == 'rgg':
        return nx.random_geometric_graph(N, 0.8, seed=1234)
    order = list(range(N)) if kind == 'ring_blocked' else [0, 3, 6, 1, 4, 7, 2, 5, 8]
    G = nx.Graph(); G.add_nodes_from(range(N))
    G.add_edges_from((order[i], order[(i + 1) % N]) for i in range(N))
    return G


def sim(G, F):
    NB = {i: sorted(G[i]) for i in range(N)}; deg = {i: len(NB[i]) for i in range(N)}
    W = {(i, j): 1.0 / (1 + max(deg[i], deg[j])) for i in range(N) for j in NB[i]}
    snd = lambda k, i: [j for j in NB[i] if (F[k, j] if j < i else (k > 0 and F[k - 1, j]))]
    # ① 共識份量（反向遞推）
    r = np.full(N, 1.0 / N); acc = np.zeros(N)
    for k in range(T - 1, -1, -1):
        acc += r
        for i in reversed(range(N)):
            s_ = snd(k, i)
            if not s_: continue
            ri = r[i]; s = 0.0
            for j in s_: r[j] += ri * W[(i, j)]; s += W[(i, j)]
            r[i] = ri * (1 - s)
    pi = acc / T
    # ② 正向資料組成：A[i, j]＝node j 的梯度在 node i 模型裡的累積權重
    A = np.zeros((N, N)); snap = {}; msg = {}
    eta = 0.5 * (1 + np.cos(np.pi * np.arange(T) / T))
    for k in range(T):
        for i in range(N):
            A[i, i] += eta[k]
            if F[k, i]:
                msg[i] = A[i].copy()      # [1013 修正] 送出＝訓練後、未融合的快照（train.py 順序）
            s_ = snd(k, i)
            if s_:
                A[i] = (1 - sum(W[(i, j)] for j in s_)) * A[i] + sum(W[(i, j)] * msg[j] for j in s_)
        if k == 1999:
            snap['mid'] = A.copy()
    snap['end'] = A.copy()
    return pi, snap, deg


def style_frac(row):
    row = row / row.sum()
    return {d: row[[i for i in range(N) if DOM[i] == d]].sum() for d in ['art', 'photo', 'sketch']}


def dispersion(G, F, ks=(499, 999, 1999, 2999, 3999)):
    """[1013 dany 追問] 資訊流通速度的代理量：各台模型「資料組成」和全網平均差多少。
    正向模擬同 sim ②；每個時間點取各列正規化後與平均列的總變異距離（0＝完全一致、1＝完全不同），回傳九台平均。
    另回傳「跨弧」量：環形分散時三台 art 把環切成三段，每台模型裡來自『其他段』節點的成分。"""
    NB = {i: sorted(G[i]) for i in range(N)}; deg = {i: len(NB[i]) for i in range(N)}
    W = {(i, j): 1.0 / (1 + max(deg[i], deg[j])) for i in range(N) for j in NB[i]}
    snd = lambda k, i: [j for j in NB[i] if (F[k, j] if j < i else (k > 0 and F[k - 1, j]))]
    A = np.zeros((N, N)); eta = 0.5 * (1 + np.cos(np.pi * np.arange(T) / T)); out = {}; msg = {}
    for k in range(T):
        for i in range(N):
            A[i, i] += eta[k]
            if F[k, i]:
                msg[i] = A[i].copy()      # [1013 修正] 照 train.py：送出的是「訓練後、未融合」的快照
            s_ = snd(k, i)
            if s_:
                A[i] = (1 - sum(W[(i, j)] for j in s_)) * A[i] + sum(W[(i, j)] * msg[j] for j in s_)
        if k in ks:
            R_ = A / A.sum(1, keepdims=True); m = R_.mean(0)
            out[k + 1] = float(np.mean(0.5 * np.abs(R_ - m).sum(1)))
    return out


def composition_by_time(G, F, ks=(499, 999, 1999, 3999)):
    """[1013 dany 追問二] 各台模型裡「自己畫風」的資料佔多少（越高＝越只學自己畫風、各段學的東西越不一樣）。"""
    NB = {i: sorted(G[i]) for i in range(N)}; deg = {i: len(NB[i]) for i in range(N)}
    W = {(i, j): 1.0 / (1 + max(deg[i], deg[j])) for i in range(N) for j in NB[i]}
    snd = lambda k, i: [j for j in NB[i] if (F[k, j] if j < i else (k > 0 and F[k - 1, j]))]
    A = np.zeros((N, N)); eta = 0.5 * (1 + np.cos(np.pi * np.arange(T) / T)); out = {}; msg = {}
    for k in range(T):
        for i in range(N):
            A[i, i] += eta[k]
            if F[k, i]:
                msg[i] = A[i].copy()      # [1013 修正] 照 train.py：送出的是「訓練後、未融合」的快照
            s_ = snd(k, i)
            if s_:
                A[i] = (1 - sum(W[(i, j)] for j in s_)) * A[i] + sum(W[(i, j)] * msg[j] for j in s_)
        if k in ks:
            R_ = A / A.sum(1, keepdims=True); m = R_.mean(0)
            own = [R_[i, [j for j in range(N) if DOM[j] == DOM[i]]].sum() for i in range(N)]
            out[k + 1] = (own, float(np.mean(0.5 * np.abs(R_ - m).sum(1))))
    return out


def forward_share(G, F, faithful=True):
    """[1013 自檢] 正向資料組成（餘弦學習率加權），回傳 ep200 共識的畫風組成。
    faithful=True：照 train.py 實際順序（訓練 → 送出『未融合』快照 → 收鄰居融合）；收到的是對方送出當下的快照。
    faithful=False：舊寫法（讀對方『目前』狀態，含對方同一步已融合的內容 ⇒ 同一步內沿節點編號接力）。"""
    NB = {i: sorted(G[i]) for i in range(N)}; deg = {i: len(NB[i]) for i in range(N)}
    W = {(i, j): 1.0 / (1 + max(deg[i], deg[j])) for i in range(N) for j in NB[i]}
    A = np.eye(N) * 0; msg = {}; eta = 0.5 * (1 + np.cos(np.pi * np.arange(T) / T))
    for k in range(T):
        for i in range(N):
            A[i, i] += eta[k]
            if F[k, i]:
                msg[i] = A[i].copy()
            s_ = [j for j in NB[i] if (F[k, j] if j < i else (k > 0 and F[k - 1, j]))]
            if s_:
                src = {j: (msg[j] if faithful else A[j]) for j in s_}
                A[i] = (1 - sum(W[(i, j)] for j in s_)) * A[i] + sum(W[(i, j)] * src[j] for j in s_)
    row = A.mean(0); row = row / row.sum()
    return {d: row[[i for i in range(N) if DOM[i] == d]].sum() for d in ['art', 'photo', 'sketch']}


if __name__ == '__main__':
    if len(sys.argv) == 1:
        F = fire_matrix(PER)
        sends = F.sum(0)
        print('送出（三種拓樸相同）', sends.tolist(), '⇒ 一階預測 art 份量（art 送出佔比）', round(sends[:3].sum() / sends.sum(), 3))
        for kind in ['rgg', 'ring_blocked', 'ring_spread']:
            G = graph(kind); pi, snap, deg = sim(G, F)
            sh = {d: pi[[i for i in range(N) if DOM[i] == d]].sum() for d in ['art', 'photo', 'sketch']}
            same = sum(1 for a, b in G.edges() if DOM[a] == DOM[b])
            print(f'\n[{kind}] 度數 {list(deg.values())}｜同畫風邊數 {same}｜art 節點的鄰居畫風 ' +
                  '; '.join(f'n{i}:' + ','.join(DOM[j][0] for j in sorted(G[i])) for i in range(3)))
            fs = forward_share(G, F, faithful=True)
            print(f'  ① 共識份量 art/photo/sketch（照實際順序）' + '/'.join(f'{v:.3f}' for v in fs.values()) + '｜舊反向遞推 ' + '/'.join(f'{v:.3f}' for v in sh.values()))
            for tag in ['mid', 'end']:
                A = snap[tag]
                own = [style_frac(A[i])['art'] for i in range(3)]
                oth = [style_frac(A[i])['art'] for i in range(3, N)]
                spread = np.ptp([style_frac(A[i])['art'] for i in range(N)])
                print(f'  ② {"ep100" if tag == "mid" else "ep200"} art 節點自己模型裡 art 佔 ' + '/'.join(f'{x:.3f}' for x in own)
                      + f'｜其他節點模型裡 art 平均 {np.mean(oth):.3f}｜九台 art 佔比最大−最小 {spread:.3f}')
    if '--dispersion' in sys.argv:
        PERS = {'均勻送': np.full(N, 6.2133),
                '照網速送': pd.read_csv('/home/server5090/Desktop/M11307320/MATCHA/results/proto_trigger/1009_bw/period_Cbw_seed0.csv').period.values,
                'art 慢': PER,
                'sketch 慢': pd.read_csv('/home/server5090/Desktop/M11307320/MATCHA/results/proto_trigger/1013_sketchslow/period_Cbw_sketch0.1.csv').period.values}
        print('\n== 資訊流通代理量：九台模型資料組成與全網平均的差異（0＝一致；同一組送出週期換拓樸）==')
        print(f'{"拓樸":14s}{"送出規則":10s}' + ''.join(f'step{k:>5d}  ' for k in (500, 1000, 2000, 3000, 4000)))
        for kind in ['rgg', 'ring_blocked', 'ring_spread']:
            for lab, per in PERS.items():
                d = dispersion(graph(kind), fire_matrix(per))
                print(f'{kind:14s}{lab:10s}' + ''.join(f'{d[k]:9.4f}  ' for k in (500, 1000, 2000, 3000, 4000)))
    if '--blocked_cut' in sys.argv:
        # 送出總次數對齊 art 慢（5,404 次）；快:慢＝10:1（網速 1 vs 0.1，照網速送）
        def per_from(slow):
            rate = np.array([0.1 if i in slow else 1.0 for i in range(N)]); s = 5404 * rate / rate.sum()
            return T / s
        cfg = [('隨機拓樸・均勻送', 'rgg', np.full(N, 6.2133)),
               ('環形相鄰・均勻送', 'ring_blocked', np.full(N, 6.2133)),
               ('環形相鄰・各畫風交界一台慢（2,5,8）', 'ring_blocked', per_from({2, 5, 8})),
               ('環形分散・art 三台慢（前一題）', 'ring_spread', PER)]
        print('\n== 各台模型裡「自己畫風」佔多少（公平混合＝0.333）與九台差異 ==')
        for lab, kind, per in cfg:
            G = graph(kind); F = fire_matrix(per)
            out = composition_by_time(G, F)
            fs = forward_share(G, F, faithful=True)
            print(f'[{lab}] 送出 {F.sum(0).tolist()}｜共識份量（照實際順序）' + '/'.join(f'{v:.3f}' for v in fs.values()))
            for k, (own, disp) in out.items():
                print(f'   step{k:5d}（ep{k // 20:3d}）自己畫風佔比 ' + ' '.join(f'n{i}:{x:.2f}' for i, x in enumerate(own)) + f'｜九台差異 {disp:.4f}')
        G = graph('ring_blocked')
        print('\n環形相鄰時每台的鄰居（風格統計量只送直接鄰居）：' + '; '.join(f'n{i}({DOM[i][0]}):' + ','.join(f'n{j}{DOM[j][0]}' for j in sorted(G[i])) for i in range(N)))
    if '--faithful_check' in sys.argv:
        def per_from(slow):
            rate = np.array([0.1 if i in slow else 1.0 for i in range(N)]); s = 5404 * rate / rate.sum()
            return T / s
        PERS = {'均勻送': np.full(N, 6.2133),
                '照網速送': pd.read_csv('/home/server5090/Desktop/M11307320/MATCHA/results/proto_trigger/1009_bw/period_Cbw_seed0.csv').period.values,
                'art 慢': PER,
                'sketch 慢': pd.read_csv('/home/server5090/Desktop/M11307320/MATCHA/results/proto_trigger/1013_sketchslow/period_Cbw_sketch0.1.csv').period.values,
                '交界各一台慢(2,5,8)': per_from({2, 5, 8})}
        print('\n== 估算自檢：共識的畫風組成 art/photo/sketch｜舊寫法（同步接力）vs 照實際順序 vs 一階理論（送出佔比）==')
        for kind in ['rgg', 'ring_blocked', 'ring_spread']:
            for lab, per in PERS.items():
                F = fire_matrix(per); s = F.sum(0)
                th = '/'.join(f'{s[[i for i in range(N) if DOM[i] == d]].sum() / s.sum():.3f}' for d in ['art', 'photo', 'sketch'])
                old = forward_share(graph(kind), F, faithful=False); new = forward_share(graph(kind), F, faithful=True)
                print(f'  {kind:13s}{lab:16s} 舊 ' + '/'.join(f'{v:.3f}' for v in old.values())
                      + '｜實際順序 ' + '/'.join(f'{v:.3f}' for v in new.values()) + f'｜一階理論 {th}')
