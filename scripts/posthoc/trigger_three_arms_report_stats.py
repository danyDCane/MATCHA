# -*- coding: utf-8 -*-
"""1007 三臂報告用數據：A 參數觸發（固定門檻）／C 本地時鐘／D 參數觸發＋門檻隨學習率衰減—— §0 自檢軌跡＋R1–R6 全部從原始 log／CSV 重算（零 GPU）。
來源：訓練 log 的 `Epoch N:` 行與其後的 `[proto]` 兩行；`async_diag/broadcast_log.csv`、`consensus.csv`；評估 CSV。"""
import re, numpy as np, pandas as pd
ROOT = '/home/server5090/Desktop/M11307320/MATCHA/'
ARMS = {
    'A 參數觸發': ('logs/proto_trigger/1001_shadow_stage1.log',
                'exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234_fix',
                'results/proto_trigger/1007_anchorA_cartoon_ep200_avgbnB.csv'),
    'C 本地時鐘': ('logs/proto_trigger/1007_periodic_C.log',
                'exp_result_v1_stage2_leave_cartoon_p1a_async_periodic6.2133_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234',
                'results/proto_trigger/1007_periodic_cartoon_ep200_avgbnB.csv'),
    'D 門檻衰減': ('logs/proto_trigger/1007_lrdecay_D.log',
                'exp_result_v1_stage2_leave_cartoon_p1a_async_lr_tau1.765e-05_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234',
                'results/proto_trigger/1007_lrdecay_cartoon_ep200_avgbnB.csv'),
}
PEP = re.compile(r'^Epoch (\d+): avg_loss=([-\d.]+), avg_train_acc=([\d.]+)%, avg_test_acc=([\d.]+)%')
PG = re.compile(r'\|\|g_cls\|\|=([\d.]+)')
PC = re.compile(r'comp:L=([-\d.]+)/lam=([\d.]+)/\|\|g\|\|=([\d.]+)/ratio=([\d.]+)')
PD = re.compile(r'disp:L=([-\d.]+)/lam=([\d.]+)/\|\|g\|\|=([\d.]+)/ratio=([\d.]+)')


def parse(path):
    rows, cur = [], None
    for ln in open(ROOT + path, encoding='utf-8', errors='ignore'):
        m = PEP.match(ln)
        if m:
            cur = dict(ep=int(m[1]), loss=float(m[2]), tr=float(m[3]), te=float(m[4])); rows.append(cur); continue
        if cur is None:
            continue
        if (g := PG.search(ln)) and 'g_cls' not in cur:
            cur['g_cls'] = float(g[1])
        if (c := PC.search(ln)):
            cur.update(compL=float(c[1]), compG=float(c[3]), compR=float(c[4]))
        if (d := PD.search(ln)):
            cur.update(dispL=float(d[1]), dispG=float(d[3]), dispR=float(d[4]))
    return pd.DataFrame(rows)


def nan_count(path):
    return sum(1 for ln in open(ROOT + path, encoding='utf-8', errors='ignore')
               if re.search(r'\bnan\b|NaN|Traceback', ln))


print('=' * 100); print('§0-1 全軌跡（頭／中／尾）＋ 末 20 epoch 是否平了（斜率／標準差）'); print('=' * 100)
D = {}
for arm, (lg, run, ev) in ARMS.items():
    df = parse(lg); D[arm] = df
    print(f'\n[{arm}] epoch 行數 {len(df)}；NaN/Traceback 行數 {nan_count(lg)}')
    for col, lab in [('loss', '總損失'), ('tr', '訓練 acc%'), ('te', '訓練中目標域 acc%'), ('g_cls', '‖g_cls‖'),
                     ('compL', 'comp 損失'), ('dispL', 'disp 損失'), ('compG', '‖g_comp‖'), ('dispG', '‖g_disp‖'),
                     ('compR', 'comp 份額'), ('dispR', 'disp 份額')]:
        s = df.set_index('ep')[col].dropna()
        pts = ' '.join(f'ep{e}:{s.get(e, np.nan):.4g}' for e in (1, 11, 50, 100, 150, 200))
        tail = s[s.index > 180]
        slope = np.polyfit(tail.index, tail.values, 1)[0] if len(tail) > 2 else np.nan
        post = s[s.index > 10]
        print(f'  {lab:<16} {pts} | ep11–200 min {post.min():.4g} max {post.max():.4g} mean {post.mean():.4g} std {post.std():.4g}'
              f' | 末20 斜率 {slope:+.2e}/ep std {tail.std():.3g}')

print('\n' + '=' * 100); print('§0-2 三臂訓練中目標域 acc（非正式口徑），逐段平均'); print('=' * 100)
m = D['A 參數觸發'].set_index('ep')['te'].to_frame('A').join(D['C 本地時鐘'].set_index('ep')['te'].rename('C')).join(D['D 門檻衰減'].set_index('ep')['te'].rename('D'))
for lo, hi in [(1, 10), (11, 50), (51, 100), (101, 150), (151, 200)]:
    q = m.loc[lo:hi]
    print(f'  ep{lo}-{hi}: A {q.A.mean():.2f}  C {q.C.mean():.2f}  D {q.D.mean():.2f} | C−A {(q.C-q.A).mean():+.2f}  D−A {(q.D-q.A).mean():+.2f}  C−D {(q.C-q.D).mean():+.2f}pp')

print('\n' + '=' * 100); print('R1–R6：通訊與過程'); print('=' * 100)
for arm, (lg, run, ev) in ARMS.items():
    b = pd.read_csv(ROOT + run + '/async_diag/broadcast_log.csv'); c = pd.read_csv(ROOT + run + '/async_diag/consensus.csv')
    print(f'\n[{arm}] R1 逐節點送出 {b.groupby("domain").fired.sum().astype(int).tolist()}  R2 總送出 {int(b.fired.sum())} 總傳輸 {int(b.n_pushed.sum())} 保底 {int(b.forced.sum())}')
    for lo, hi in [(1, 50), (51, 100), (101, 150), (151, 200)]:
        q = b[(b.epoch >= lo) & (b.epoch <= hi)]; f = q[q.fired == 1]; rec = q[q.n_received > 0]
        print(f'  ep{lo:>3}-{hi:<3} R3 送出占全程 {100*len(f)/b.fired.sum():5.1f}% 送出率 {100*q.fired.mean():5.1f}%'
              f' | R4 送出當下 delta 中位 {f.delta.median():.2e}、≥τ 比例 {100*(f.delta>=1e-5).mean():5.1f}%'
              f' | R6 收到模型年齡 {rec.mean_staleness.mean():.2f} 步、風格年齡 {q.mean_style_age.mean():6.2f} 步')
    print('  R5 共識 RMS ' + ' '.join(f'ep{e}:{c[c.epoch==e].consensus_rms.iloc[0]:.2e}' for e in (1, 10, 50, 100, 150, 200))
          + f' | max_node_dev ep200 {c[c.epoch==200].max_node_dev.iloc[0]:.2e}')

print('\n' + '=' * 100); print('R7／R8：評估（epoch_200、平均B、9 節點逐列平均）'); print('=' * 100)
E = {arm: pd.read_csv(ROOT + ev) for arm, (lg, run, ev) in ARMS.items()}
K = {'A': 'A 參數觸發', 'C': 'C 本地時鐘', 'D': '門檻衰減'}
K['D'] = 'D 門檻衰減'
for s_ in ['zperp', 'proto_angle', 'energy', 'msp']:
    for col in ['det_auroc', 'oscr', 'fpr95', 'closed_acc']:
        v = {x: E[K[x]][E[K[x]].score_fn == s_].set_index('node')[col] for x in 'ACD'}
        parts = []
        for p, q in [('C', 'D'), ('D', 'A'), ('C', 'A')]:
            d = v[p] - v[q]; better = int(((d < 0) if col == 'fpr95' else (d > 0)).sum())
            parts.append(f'{p}−{q} {d.mean():+.4f}[{d.min():+.4f},{d.max():+.4f}] {p}較好 {better}/9')
        print(f'  {s_:<11} {col:<10} A {v["A"].mean():.4f}  C {v["C"].mean():.4f}  D {v["D"].mean():.4f} | ' + ' | '.join(parts))
