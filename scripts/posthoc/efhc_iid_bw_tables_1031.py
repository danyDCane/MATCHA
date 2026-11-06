"""[1031] 照 EF-HC 期刊版（PersonalizedET, arXiv 2211.12640 §V「Resource heterogeneity」）原文產生網速表，零 GPU 預估。
原文：b_i 每台設備獨立抽，(i) 均勻 U((1−σN)bM, (1+σN)bM)，bM=5000、σN=0.9 ⇒ U(500, 9500)；(ii) Beta(0.5,0.5)·bM。不分層、不設下限。
本檔只做兩件事：①把抽到的值除以該表最大值（最快＝1；只影響記帳單位，週期與門檻都依比值校準，結果不變）寫成 --bw_file；
②用既有估算器預估各畫風份量、同週期送週期、有無幾乎不送的設備。種子事先寫死（dany 1031：先跑 0、1；之後擴充依序 2、3、4）。
節點↔畫風：node_0–2 art、node_3–5 photo、node_6–8 sketch（util.assign_nodes_to_domains）。
用法：venv_matcha/bin/python scripts/posthoc/efhc_iid_bw_tables_1031.py [--seeds 0 1] [--write]
"""
import sys, os, argparse
import numpy as np
ROOT = '/home/server5090/Desktop/M11307320/MATCHA/'
sys.path.insert(0, ROOT + 'scripts/posthoc')
from artslow_period_and_share_estimate import fire_matrix, cbw_periods, NB, T
from composition_inbox_sim import simulate
N = 9; GRP = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]); STY = ['art', 'photo', 'sketch']
TARGET = 29613                                   # 總份數，同 1009／1019 照網速送
OUT = ROOT + 'results/proto_trigger/1031_efhc_bw/'


def draw(dist, seed):
    rng = np.random.default_rng(seed)
    if dist == 'uni':
        b = rng.uniform(500.0, 9500.0, N)          # U((1−0.9)·5000, (1+0.9)·5000)
    else:
        b = rng.beta(0.5, 0.5, N) * 5000.0         # Beta(0.5,0.5)·5000
    return b / b.max()


def q_true(r):  # 按群附的數＝自己＋同畫風鄰居的平均送出率（同 fold_competitors_prep_1019）
    return np.array([np.mean([r[x] for x in [i] + [j for j in NB[i] if GRP[j] == GRP[i]]]) for i in range(N)])


def uniform_period(b, target_time):
    """同週期送：九台同週期 p，使每台平均傳輸時間 Σ(送出_i/b_i)/9 ＝ 照網速送；二分搜尋（上界放寬到 T，超過＝整段不送）。"""
    lo, hi = 1.0, float(T)
    if (fire_matrix(np.full(N, hi)).sum(0) / b).mean() > target_time:
        return hi, True
    for _ in range(60):
        mid = (lo + hi) / 2; t = (fire_matrix(np.full(N, mid)).sum(0) / b).mean()
        lo, hi = (mid, hi) if t > target_time else (lo, mid)
    return (lo + hi) / 2, False


ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
ap.add_argument('--write', action='store_true'); o = ap.parse_args()
os.makedirs(OUT, exist_ok=True)
for dist in ['uni', 'beta']:
    for s in o.seeds:
        b = draw(dist, s); tag = f'{dist}_s{s}'
        p = cbw_periods(b, TARGET); F = fire_matrix(p); sends = F.sum(0); r = F.mean(0)
        tnode = sends / b; target_time = tnode.mean()
        pu, degenerate = uniform_period(b, target_time); Fu = fire_matrix(np.full(N, pu)); tu = Fu.sum(0) / b
        ssum = np.array([b[GRP == g].sum() for g in range(3)]); worst = STY[int(ssum.argmin())]
        print(f'\n== {tag}｜網速（最快＝1）：' + ' '.join(f'{STY[GRP[i]][:2]}{i}:{b[i]:.4f}' for i in range(N)))
        print(f'  各畫風網速和 ' + ' / '.join(f'{STY[g]} {ssum[g]:.3f}' for g in range(3))
              + f'｜一階份量 ' + ' / '.join(f'{x:.3f}' for x in ssum / ssum.sum())
              + f'｜最吃虧畫風（事先規則＝網速和最低）：{worst}｜最快÷最慢 {b.max() / b.min():.1f} 倍')
        print(f'  照網速送：週期 ' + ' '.join(f'{x:.1f}' for x in p) + f'｜送出 {sends.tolist()}｜0 次送出的設備 {int((sends == 0).sum())} 台'
              f'｜每台傳輸時間 {tnode.mean():.0f}（最長 {tnode.max():.0f}）')
        print(f'  同週期送：週期 {pu:.2f}{"（⚠️ 達上界＝整段幾乎不送）" if degenerate else ""}｜每台送出 {int(Fu.sum(0).mean())} 次'
              f'｜單台最長傳輸時間 {tu.max():.0f}（照網速送的 {tu.max() / tnode.max():.2f} 倍）')
        for nm, kw in [('照網速送', dict(rule='mh')), ('按群校正', dict(rule='rate_pair', u=q_true(r))),
                       ('PushCen', dict(rule='pushcen'))]:
            sh = simulate(F, **kw)['share']
            print(f'  預測份量 {nm:6s} ' + ' / '.join(f'{STY[g]} {sh[g]:.3f}' for g in range(3)))
        sh = simulate(Fu, rule='mh')['share']
        print(f'  預測份量 同週期送 ' + ' / '.join(f'{STY[g]} {sh[g]:.3f}' for g in range(3)))
        if o.write:
            with open(OUT + f'bandwidth_{tag}.csv', 'w') as f:
                f.write('node,b\n'); [f.write(f'node_{i},{b[i]:.6f}\n') for i in range(N)]
            with open(OUT + f'period_Cbw_{tag}.csv', 'w') as f:
                f.write('node,period\n'); [f.write(f'node_{i},{p[i]:.6f}\n') for i in range(N)]
            with open(OUT + f'period_uniform_{tag}_{pu:.4f}.csv', 'w') as f:
                f.write('node,period\n'); [f.write(f'node_{i},{pu:.6f}\n') for i in range(N)]
            print(f'  寫出 {OUT}bandwidth_{tag}.csv、period_Cbw_{tag}.csv、period_uniform_{tag}_{pu:.4f}.csv')
