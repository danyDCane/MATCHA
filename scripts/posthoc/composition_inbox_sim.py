"""[plan 1017] 零 GPU 組成估算（收件匣版）：逐步重演 communicator 的推送／收件匣／收件融合語意，
算共用模型裡各台梯度的占比（份量）。支援 mh、rate_pair（底座）、pushsum（SGP 派，梯度 ×1/w）、pushcen（梯度不縮放），
以及 EF-HC 雙向（bidir：觸發的台收下所有鄰居當下的組成；同一步同一連線只交換一次）。
事件順序同 train.py：每步依節點序 訓練（自己的成分 ＋ 餘弦學習率）→ 觸發就推送（＋雙向回傳）→ 收件融合（讀完清空）。
與 1013–1016 的估算器（送未融合快照、j<i 同步、j>i 上一步）等價；以 1013／A1／A3 的既有數字當錨點。
"""
import numpy as np, networkx as nx

N, T = 9, 4000
G = nx.random_geometric_graph(N, 0.8, seed=1234)
NB = {i: sorted(G[i]) for i in range(N)}
DEG = {i: len(NB[i]) for i in range(N)}
W = {(i, j): 1.0 / (1 + max(DEG[i], DEG[j])) for i in range(N) for j in NB[i]}
GRP = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])


def simulate(F, rule='mh', bidir=False, u=None):
    """F：[T, 9] bool 送出矩陣（實際紀錄）。回 dict(share＝art/photo/sketch, node＝逐台×9)。
    [plan 1019] u：rate_pair 的附的數（按群校正份量；None＝自己的送出率＝底座）。"""
    T_ = F.shape[0]; eta = 0.5 * (1 + np.cos(np.pi * np.arange(T_) / T_)); r = F.mean(0)
    ps = rule in ('pushsum', 'pushcen')
    A = np.zeros((N, N)); w = np.ones(N); inbox = {i: {} for i in range(N)}
    for k in range(T_):
        replied = set()
        for i in range(N):
            A[i, i] += eta[k] / (w[i] if rule == 'pushsum' else 1.0)
            if F[k, i]:
                msg = A[i].copy(); share = None
                if ps:
                    w[i] /= (DEG[i] + 1); share = w[i]
                for nb in NB[i]:
                    if bidir and (i, nb) in replied:
                        continue
                    if ps and i in inbox[nb]:
                        om, osh, _ = inbox[nb][i]; tot = osh + share
                        inbox[nb][i] = ((osh * om + share * msg) / tot, tot, k)
                    else:
                        inbox[nb][i] = (msg, share, k)
                if bidir:
                    for y in NB[i]:
                        prev = inbox[i].get(y)
                        if prev is not None and prev[2] == k:
                            continue
                        inbox[i][y] = (A[y].copy(), None, k); replied.add((y, i))
            if inbox[i]:
                snd = sorted(inbox[i])
                if ps:
                    tot = w[i] + sum(inbox[i][j][1] for j in snd)
                    a = {j: inbox[i][j][1] / tot for j in snd}; w[i] = tot
                elif rule == 'rate_pair':
                    uu = r if u is None else u
                    a = {j: W[(i, j)] * min(uu[i], uu[j]) / uu[j] for j in snd}
                else:
                    a = {j: W[(i, j)] for j in snd}
                A[i] = (1 - sum(a.values())) * A[i] + sum(a[j] * inbox[i][j][0] for j in snd)
                inbox[i] = {}
    row = A.mean(0); row = row / row.sum()
    return dict(share=[row[GRP == g].sum() for g in range(3)], node=row * N)
