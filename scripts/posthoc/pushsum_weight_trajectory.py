"""[1016 A3 發散] 推和法每台份量 w 的逐步軌跡（零 GPU）：用參照 run 實際的 fired，照 train.py 事件順序重建
（每台依序：用目前 w 訓練（梯度 ×1/w）→ 送出時 w ← w/(d+1) → 收件 w ← w＋Σ份額；編號小的鄰居同步送、編號大的上一步送）。
報：每台「訓練時用的 w」的最小值與分位數、步長倍數 1/w 超過 10／100 的步數比例、分段統計；並與 A3 訓練 log 的 [PUSHSUM] 逐 epoch 值比對（驗證重建）。
用法：venv_matcha/bin/python scripts/posthoc/pushsum_weight_trajectory.py --ref <run 目錄> [--train_log <A3 訓練 log>]
"""
import argparse, re
import numpy as np, pandas as pd
import networkx as nx

ap = argparse.ArgumentParser()
ap.add_argument('--ref', required=True); ap.add_argument('--train_log', default='')
o = ap.parse_args()
b = pd.read_csv(f'{o.ref}/async_diag/broadcast_log.csv'); b['i'] = b.domain.str[5:].astype(int)
F = b.pivot(index='step', columns='i', values='fired').sort_index().values.astype(bool); T = F.shape[0]
G = nx.random_geometric_graph(9, 0.8, seed=1234); NB = {i: sorted(G[i]) for i in range(9)}; deg = {i: len(NB[i]) for i in range(9)}
w = np.ones(9); share = {}; Wtrain = np.zeros((T, 9)); Wend = np.zeros((T, 9))
for k in range(T):
    for i in range(9):
        Wtrain[k, i] = w[i]
        if F[k, i]:
            w[i] /= (deg[i] + 1); share[(k, i)] = w[i]
        snd = [j for j in NB[i] if (F[k, j] if j < i else (k > 0 and F[k - 1, j]))]
        if snd:
            w[i] += sum(share[(k if j < i else k - 1, j)] for j in snd)
    Wend[k] = w
print(f'步數 {T}｜度數 {[deg[i] for i in range(9)]}｜送出次數 {F.sum(0).tolist()}')
if o.train_log:
    L = [l for l in open(o.train_log, encoding='utf-8', errors='ignore') if l.startswith('[PUSHSUM] epoch=')]
    ep = [int(re.search(r'epoch=(\d+)', l).group(1)) for l in L]
    Wl = np.array([[float(x) for x in re.search(r' w=([0-9.,]+) ', l).group(1).split(',')] for l in L])
    rec = np.array([Wend[20 * e - 1] for e in ep])
    print(f'與訓練 log 的 [PUSHSUM] 比對：{len(L)} 個 epoch 末、最大絕對差 {np.abs(rec.round(4) - Wl).max():.1e}（log 只印到小數 4 位）')
mult = 1.0 / Wtrain
print('\n每台「訓練時用的 w」（全程）：')
for i in range(9):
    m = mult[:, i]
    print(f'  node_{i}（{["art", "photo", "sketch"][i // 3]}、度數 {deg[i]}、送 {F[:, i].sum()} 次）w 最小 {Wtrain[:, i].min():.2e}｜'
          f'步長倍數 中位 {np.median(m):.2f}／p95 {np.percentile(m, 95):.1f}／p99 {np.percentile(m, 99):.1f}／最大 {m.max():.0f}｜'
          f'>10 倍 {100 * (m > 10).mean():.1f}%、>100 倍 {100 * (m > 100).mean():.2f}%')
print(f'  全網：步長倍數 >10 倍 {100 * (mult > 10).mean():.1f}%、>100 倍 {100 * (mult > 100).mean():.2f}%、最大 {mult.max():.0f}')
print('\n分段（epoch，每 epoch 20 步）：九台平均步長倍數／最大步長倍數／>10 倍比例')
for lo, hi in [(1, 5), (6, 25), (26, 50), (51, 100), (101, 200)]:
    s = mult[(lo - 1) * 20: hi * 20]
    print(f'  ep{lo}–{hi}: 平均 {s.mean():.1f}｜最大 {s.max():.0f}｜>10 倍 {100 * (s > 10).mean():.1f}%')
print('\n學習率 × 步長倍數（餘弦，初始學習率當 1）的最大值：ep1–25 '
      f'{(mult[:500] * (0.5 * (1 + np.cos(np.pi * np.arange(500) / T)))[:, None]).max():.0f}、'
      f'ep151–200 {(mult[3000:] * (0.5 * (1 + np.cos(np.pi * np.arange(3000, T) / T)))[:, None]).max():.1f}')
