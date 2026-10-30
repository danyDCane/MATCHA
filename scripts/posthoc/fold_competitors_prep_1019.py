"""[plan 1019 一折全貌] 零 GPU 準備：等預算均勻送的週期、各情境的組成估算預測、run 名稱長度。
- 均勻送：九台同週期 p，用 fire_matrix（相位種子 0）二分搜尋，使 Σ_i(送出_i / b_i)/9＝照網速送（C-bw 週期表）的同一量（同 1017 plan §5.1）。
- 預測：composition_inbox_sim.simulate（照 train.py 事件順序的收件匣語意）在合成送出表上算份量：不補償、底座、按群、PushCen。
  EF-HC（參數觸發）的送出時機要訓練才知道，這裡不預測。
用法：venv_matcha/bin/python scripts/posthoc/fold_competitors_prep_1019.py [--write]
"""
import sys, os, numpy as np, pandas as pd
ROOT = '/home/server5090/Desktop/M11307320/MATCHA/'
sys.path.insert(0, ROOT + 'scripts/posthoc')
from artslow_period_and_share_estimate import fire_matrix, NB
from composition_inbox_sim import simulate
N = 9; GRP = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
SUF = '_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234'
P = ROOT + 'results/proto_trigger/'
SCEN = {  # 情境: (網速表, C-bw 週期表)
    '正常網速': (ROOT + 'exp_result_v1_stage2_leave_cartoon_p1a_async_pbw_bwb0' + SUF + '/async_diag/bandwidth.csv', P + '1009_bw/period_Cbw_seed0.csv'),
    'art 慢': (P + '1012_artslow/bandwidth_art0.1.csv', P + '1012_artslow/period_Cbw_art0.1.csv'),
    'photo 慢': (P + '1019_fold/bandwidth_photo0.1.csv', P + '1019_fold/period_Cbw_photo0.1.csv'),
    'sketch 慢': (P + '1013_sketchslow/bandwidth_sketch0.1.csv', P + '1013_sketchslow/period_Cbw_sketch0.1.csv'),
}


def rd(path, col):
    t = pd.read_csv(path); t['i'] = t.node.str[5:].astype(int); return t.sort_values('i')[col].values.astype(float)


def per_node_time(F, b): return F.sum(0) / b


def q_true(r):
    return np.array([np.mean([r[x] for x in [i] + [j for j in NB[i] if GRP[j] == GRP[i]]]) for i in range(N)])


write = '--write' in sys.argv
for sc, (bf, pf) in SCEN.items():
    b, p = rd(bf, 'b'), rd(pf, 'period'); F = fire_matrix(p); r = F.mean(0); target = per_node_time(F, b).mean()
    lo, hi = 1.0, 200.0
    for _ in range(60):
        mid = (lo + hi) / 2; t = per_node_time(fire_matrix(np.full(N, mid)), b).mean()
        lo, hi = (mid, hi) if t > target else (lo, mid)
    pu = (lo + hi) / 2; Fu = fire_matrix(np.full(N, pu)); tu = per_node_time(Fu, b)
    slow = b <= 0.1 + 1e-9
    print(f'\n== {sc}｜網速 {np.round(b, 4).tolist()}')
    print(f'  照網速送：送出 {F.sum(0).tolist()}｜每台傳輸時間平均 {target:.1f}｜慢設備每台 {per_node_time(F, b)[slow].mean():.0f}')
    print(f'  等預算均勻送：週期 {pu:.4f}｜送出 {Fu.sum(0).tolist()}｜每台傳輸時間平均 {tu.mean():.1f}｜慢設備每台 {tu[slow].mean():.0f}'
          f'（照網速送的 {tu[slow].mean() / per_node_time(F, b)[slow].mean():.2f} 倍）')
    for nm, kw in [('不補償', dict(rule='mh')), ('底座', dict(rule='rate_pair')), ('按群', dict(rule='rate_pair', u=q_true(r))),
                   ('PushCen', dict(rule='pushcen')), ('等預算均勻送', dict(rule='mh', F_=Fu))]:
        FF = kw.pop('F_', F); o = simulate(FF, **kw); sh = o['share']
        print(f'  預測份量 {nm:8s} ' + '/'.join(f'{x:.3f}' for x in sh) + f'（偏差 {max(abs(x - 1 / 3) for x in sh):.3f}）｜逐台 ×9 '
              + ' '.join(f'{x:.2f}' for x in o['node']))
    if write and sc != 'sketch 慢':
        tag = {'art 慢': 'art0.1', 'photo 慢': 'photo0.1', '正常網速': 'bwb0'}[sc]
        out = P + f'1019_fold/period_uniform_{tag}_{pu:.4f}.csv'
        with open(out, 'w') as f:
            f.write('node,period\n'); [f.write(f'node_{i},{pu:.6f}\n') for i in range(N)]
        print(f'  寫出 {out}')

# 正常網速：PushCen 的預測（照網速送的送出表＝1009 C-bw）
b0 = rd(P + '1009_bw/period_Cbw_seed0.csv', 'period'); F0 = fire_matrix(b0)
for nm, kw in [('不補償', dict(rule='mh')), ('底座', dict(rule='rate_pair')), ('按群', dict(rule='rate_pair', u=q_true(F0.mean(0)))),
               ('PushCen', dict(rule='pushcen'))]:
    o = simulate(F0, **kw); sh = o['share']
    print(f'== 正常網速 預測份量 {nm:8s} ' + '/'.join(f'{x:.3f}' for x in sh) + f'（偏差 {max(abs(x - 1 / 3) for x in sh):.3f}）｜逐台 ×9 '
          + ' '.join(f'{x:.2f}' for x in o['node']))

print('\nrun 名稱長度（上限 119）：')
names = {
    '正常 EF-HC 雙向': 'v1_stage2_leave_cartoon_p1a_async_lr_tau1.368e-05_bwb0r_bd',
    '正常 PushCen': 'v1_stage2_leave_cartoon_p1a_async_pbw_bwb0_fpc',
    'art 按群': 'v1_stage2_leave_cartoon_p1a_async_pbwart_bwfart0.1_frg',
    'art EF-HC 單向': 'v1_stage2_leave_cartoon_p1a_async_lr_tau1.098e-05_bwfartr',
    'art EF-HC 雙向': 'v1_stage2_leave_cartoon_p1a_async_lr_tau1.098e-05_bwfartr_bd',
    'art PushCen': 'v1_stage2_leave_cartoon_p1a_async_pbwart_bwfart0.1_fpc',
    'art 均勻送': 'v1_stage2_leave_cartoon_p1a_async_puni_bwfart0.1',
    'photo 不補償': 'v1_stage2_leave_cartoon_p1a_async_pbwph_bwfphoto0.1',
    'photo 按群': 'v1_stage2_leave_cartoon_p1a_async_pbwph_bwfphoto0.1_frg',
    'photo EF-HC 單向': 'v1_stage2_leave_cartoon_p1a_async_lr_tau1.115e-05_bwfphr',
    'photo EF-HC 雙向': 'v1_stage2_leave_cartoon_p1a_async_lr_tau1.115e-05_bwfphr_bd',
    'photo PushCen': 'v1_stage2_leave_cartoon_p1a_async_pbwph_bwfphoto0.1_fpc',
    'photo 均勻送': 'v1_stage2_leave_cartoon_p1a_async_puni_bwfphoto0.1',
    '正常 均勻送': 'v1_stage2_leave_cartoon_p1a_async_puni_bwb0',
}
for k, v in names.items():
    print(f'  {k:16s} {len(v + SUF):3d} {v + SUF}')
