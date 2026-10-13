"""[plan 1012] 慢設備集中在 art：產生頻寬表與 C-bw 週期表，並離線估算各畫風在共識裡的份量（零 GPU）。
  python artslow_period_and_share_estimate.py [--art_b 0.1] [--target 29613]
頻寬表＝1009 的分層表（communicator.stratified_bandwidth_table seed 0）只把 art 三台改成 --art_b，其餘不動；
週期規則＝1009 C-bw（週期 ∝ 1/b_i、係數對齊總份數 --target），算法逐行同 lr_threshold_calibration_estimate.bw_mode。
份量估算＝1003 async_slow_domain_composition_estimate.infl 的單向版（逐步反向遞推），送出序列用與
communicator.should_broadcast 相同的本地時鐘規則合成；先用 1009 C-bw 週期重現報告 R4 當自檢。"""
import sys, os, argparse
import numpy as np, networkx as nx
ROOT = '/home/server5090/Desktop/M11307320/MATCHA/'
sys.path.insert(0, ROOT)
from communicator import stratified_bandwidth_table
N, T = 9, 4000
STYLES = ['art_painting'] * 3 + ['photo'] * 3 + ['sketch'] * 3
DOM = ['art'] * 3 + ['photo'] * 3 + ['sketch'] * 3
G = nx.random_geometric_graph(N, 0.8, seed=1234); NB = {i: sorted(G[i]) for i in range(N)}
deg = np.array([len(NB[i]) for i in range(N)])
W = {(i, j): 1.0 / (1 + max(deg[i], deg[j])) for i in range(N) for j in NB[i]}


def phases(seed=0):
    return np.random.default_rng(seed).random(N)


def fire_matrix(periods, seed=0):
    """communicator.should_broadcast 本地時鐘：fired ⇔ floor((k+1)r+u) > floor(k r+u)（step 從 0 起）。"""
    u = phases(seed); r = 1.0 / np.asarray(periods)
    k = np.arange(T)[:, None]
    return np.floor((k + 1) * r + u) > np.floor(k * r + u)


def infl_oneway(Fm):
    """同 1003 infl(two_way=False)：回傳逐步各畫風份量序列。"""
    r = np.full(N, 1.0 / N); hist = np.zeros((Fm.shape[0], N))
    for k in range(Fm.shape[0] - 1, -1, -1):
        hist[k] = r
        for i in reversed(range(N)):
            snd = [j for j in NB[i] if (Fm[k, j] if j < i else (k > 0 and Fm[k - 1, j]))]
            if not snd: continue
            ri = r[i]; s = 0.0
            for j in snd: r[j] += ri * W[(i, j)]; s += W[(i, j)]
            r[i] = ri * (1 - s)
    return hist


def share(hist, lo, hi):
    m = hist[lo:hi].mean(0)
    return {d: m[[i for i in range(N) if DOM[i] == d]].sum() for d in ['art', 'photo', 'sketch']}


def cbw_periods(bw, target):
    c = target / (bw * deg).sum() / T
    return 1.0 / (c * bw)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--art_b', type=float, default=0.1)
    ap.add_argument('--target', type=int, default=29613)
    ap.add_argument('--out_dir', default=ROOT + 'results/proto_trigger/1012_artslow')
    ap.add_argument('--write', action='store_true', help='寫出頻寬表與週期表')
    o = ap.parse_args()
    print('degree', deg.tolist())
    bw0 = np.array(stratified_bandwidth_table(STYLES, seed=0, floor=0.1))
    # 自檢 1：1009 C-bw 週期逐位重現
    p0 = cbw_periods(bw0, o.target)
    ref = np.loadtxt(ROOT + 'results/proto_trigger/1009_bw/period_Cbw_seed0.csv', delimiter=',', skiprows=1,
                     usecols=1)
    print(f'自檢1 1009 C-bw 週期重算 vs 檔案 最大差 {np.abs(p0 - ref).max():.2e}')
    F0 = fire_matrix(p0)
    print(f'自檢2 1009 C-bw 合成送出 {F0.sum(0).tolist()} 總份數 {int((F0.sum(0) * deg).sum())}（報告 29,614）')
    h0 = infl_oneway(F0)
    for lo, hi in [(0, 1000), (3000, 4000)]:
        s = share(h0, lo, hi); print(f'自檢3 1009 C-bw 合成份量 {lo}-{hi}: ' + ' / '.join(f'{d} {v:.3f}' for d, v in s.items())
                                   + '（報告 R4：0–1000 0.308/0.312/0.380；3000–4000 0.302/0.311/0.387）')
    bw = bw0.copy(); bw[:3] = o.art_b
    p = cbw_periods(bw, o.target); F = fire_matrix(p); fz = F.sum(0)
    print(f'\nart_b={o.art_b}：頻寬 {np.round(bw, 4).tolist()}')
    print(f'  週期 {np.round(p, 4).tolist()}')
    print(f'  送出 {fz.tolist()} 總份數 {int((fz * deg).sum())} 總傳輸時間 {(fz / bw).sum():.0f}（1009 C-bw {(F0.sum(0) / bw0).sum():.0f}）')
    h = infl_oneway(F)
    for lo, hi in [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (0, 4000)]:
        s = share(h, lo, hi); print(f'  份量 {lo:4d}-{hi:4d}: ' + ' / '.join(f'{d} {v:.3f}' for d, v in s.items()))
    if o.write:
        os.makedirs(o.out_dir, exist_ok=True)
        tag = f'art{o.art_b:g}'
        with open(f'{o.out_dir}/bandwidth_{tag}.csv', 'w') as f:
            f.write('node,b\n'); [f.write(f'node_{i},{b:.6f}\n') for i, b in enumerate(bw)]
        with open(f'{o.out_dir}/period_Cbw_{tag}.csv', 'w') as f:
            f.write('node,period\n'); [f.write(f'node_{i},{q:.6f}\n') for i, q in enumerate(p)]
        print(f'  寫出 {o.out_dir}/bandwidth_{tag}.csv、period_Cbw_{tag}.csv')
