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


def curve_section(curve_csv='results/proto_trigger/1008_curve_cartoon_avgbnB.csv',
                  fig_path='results/proto_trigger/1008_curve_cartoon.png'):
    """1008：EF-HC 第 (iii) 張圖的我們版本——準確率 vs 累積傳輸份數。
    縱軸＝osdg_eval 正式口徑（平均B、9 節點逐列平均）；橫軸＝broadcast_log 的 n_pushed 累加到該 epoch。
    另畫訓練中目標域 acc（非正式：含 person、各節點原樣 BN）的逐 epoch 密集曲線當背景。"""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ev = pd.read_csv(ROOT + curve_csv)
    ev['arm'] = ev.run.map({run.replace('exp_result_', '', 1): arm[0] for arm, (lg, run, _) in ARMS.items()})
    assert ev.arm.notna().all(), '評估 CSV 有對不上 ARMS 的 run'
    ev['ep'] = ev.ckpt.str.replace('epoch_', '').astype(int)
    cum = {}
    for arm, (lg, run, _) in ARMS.items():
        b = pd.read_csv(ROOT + run + '/async_diag/broadcast_log.csv')
        cum[arm[0]] = b.groupby('epoch').n_pushed.sum().cumsum()

    print('=' * 100); print('R9-0 錨點：ep200 逐列重現 1007 三份 CSV（同 node×score_fn，所有數值欄最大絕對差）'); print('=' * 100)
    num = ['oscr', 'h_best', 'det_auroc', 'det_aupr', 'fpr95', 'closed_acc']
    for arm, (lg, run, old) in ARMS.items():
        o = pd.read_csv(ROOT + old).set_index(['node', 'score_fn'])[num]
        n = ev[(ev.arm == arm[0]) & (ev.ep == 200)].set_index(['node', 'score_fn'])[num]
        j = o.join(n, rsuffix='_new', how='inner')
        dmax = max(float((j[c] - j[c + '_new']).abs().max()) for c in num)
        print(f'  [{arm}] 比對列數 {len(j)}（1007 {len(o)}／1008 {len(n)}）最大絕對差 {dmax:.4g}')

    print('\n' + '=' * 100); print('R9-1 各臂各 checkpoint：累積傳輸份數 → 9 節點平均（平均B）'); print('=' * 100)
    T = {}
    for a in 'ACD':
        for e in (50, 100, 150, 200):
            q = ev[(ev.arm == a) & (ev.ep == e)]
            r = dict(transfers=int(cum[a].loc[e]))
            for s_, col in [('zperp', 'det_auroc'), ('proto_angle', 'det_auroc'), ('energy', 'det_auroc'), ('zperp', 'closed_acc'), ('zperp', 'oscr')]:
                x = q[q.score_fn == s_][col]
                r[f'{s_}.{col}'] = x.mean(); r[f'{s_}.{col}.n'] = len(x)
            T[(a, e)] = r
            print(f'  {a} ep{e:<3} 累積 {r["transfers"]:>6} 份 | 泛化 closed_acc {r["zperp.closed_acc"]:.4f} | 面 {r["zperp.det_auroc"]:.4f}'
                  f' 點 {r["proto_angle.det_auroc"]:.4f} energy {r["energy.det_auroc"]:.4f} | 面 OSCR {r["zperp.oscr"]:.4f} | n={r["zperp.det_auroc.n"]}')

    print('\n' + '=' * 100); print('R9-2 通訊量相近的配對（判準：面檢測差 >0.015、泛化差 >4.92 分才算有差）'); print('=' * 100)
    for grp in [[('A', 50), ('C', 100), ('D', 100)], [('A', 100), ('C', 150), ('D', 150)], [('A', 200), ('C', 200), ('D', 200)]]:
        base = T[grp[0]]
        print('  ' + ' | '.join(f'{a} ep{e} {T[(a, e)]["transfers"]}份 面 {T[(a, e)]["zperp.det_auroc"]:.4f} 泛化 {100*T[(a, e)]["zperp.closed_acc"]:.2f}' for a, e in grp))
        for a, e in grp[1:]:
            dz = T[(a, e)]['zperp.det_auroc'] - base['zperp.det_auroc']; dg = 100 * (T[(a, e)]['zperp.closed_acc'] - base['zperp.closed_acc'])
            print(f'    {a} ep{e} − {grp[0][0]} ep{grp[0][1]}：面 {dz:+.4f}（{"有差" if abs(dz) > 0.015 else "雜訊內"}）'
                  f' 泛化 {dg:+.2f} 分（{"有差" if abs(dg) > 4.92 else "雜訊內"}）份數比 {T[(a, e)]["transfers"]/base["transfers"]:.3f}')

    print('\n' + '=' * 100); print('R9-3 同 epoch 的臂間差（差距在哪一輪拉開）'); print('=' * 100)
    for e in (50, 100, 150, 200):
        z = {a: T[(a, e)]['zperp.det_auroc'] for a in 'ACD'}; g = {a: 100 * T[(a, e)]['zperp.closed_acc'] for a in 'ACD'}
        print(f'  ep{e:<3} 面 C−A {z["C"]-z["A"]:+.4f} D−A {z["D"]-z["A"]:+.4f} C−D {z["C"]-z["D"]:+.4f}'
              f' | 泛化 C−A {g["C"]-g["A"]:+.2f} D−A {g["D"]-g["A"]:+.2f} C−D {g["C"]-g["D"]:+.2f} 分')

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    col = {'A': 'tab:red', 'C': 'tab:blue', 'D': 'tab:green'}
    for a in 'ACD':
        xs = [T[(a, e)]['transfers'] for e in (50, 100, 150, 200)]
        ax[0].plot(xs, [100 * T[(a, e)]['zperp.closed_acc'] for e in (50, 100, 150, 200)], 'o-', c=col[a], label=a)
        ax[1].plot(xs, [T[(a, e)]['zperp.det_auroc'] for e in (50, 100, 150, 200)], 'o-', c=col[a], label=a)
        for x, e in zip(xs, (50, 100, 150, 200)):
            ax[1].annotate(f'ep{e}', (x, T[(a, e)]['zperp.det_auroc']), fontsize=7, color=col[a])
        arm = [k for k in ARMS if k[0] == a][0]
        te = parse(ARMS[arm][0]).set_index('ep')['te'].rolling(5, min_periods=1).mean()
        ax[2].plot(cum[a].reindex(te.index).values, te.values, c=col[a], lw=1, label=a)
    ax[0].set_title('formal closed-set acc (%) vs cumulative transfers'); ax[1].set_title('formal face AUROC vs cumulative transfers')
    ax[2].set_title('in-training target acc (informal, 5-ep smooth)')
    for x in ax:
        x.set_xlabel('cumulative transfers'); x.legend(); x.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(ROOT + fig_path, dpi=130); print(f'\n圖：{fig_path}')


_BWS = '_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234'
BW_ARMS = {   # [plan 1009] 頻寬不平均四臂：標籤 → (訓練 log, run 目錄)
    'C0': ('logs/proto_trigger/1009_C0dense.log', f'exp_result_v1_stage2_leave_cartoon_p1a_async_periodic6.2133{_BWS}_d5'),
    'Dbw': ('logs/proto_trigger/1009_Dbw.log', f'exp_result_v1_stage2_leave_cartoon_p1a_async_lr_tau1.368e-05_bwb0r{_BWS}'),
    'Cbw': ('logs/proto_trigger/1009_Cbw.log', f'exp_result_v1_stage2_leave_cartoon_p1a_async_pbw_bwb0{_BWS}'),
    'Cmatch': ('logs/proto_trigger/1009_Cmatch.log', f'exp_result_v1_stage2_leave_cartoon_p1a_async_pmatch_bwb0{_BWS}'),
}


def bw_section(curve_csv='results/proto_trigger/1009_bw_curve_cartoon_avgbnB.csv',
               fig_path='results/proto_trigger/1009_bw_curve_cartoon.png', targets=(0.82, 0.79)):
    """[plan 1009] 四臂：R1 份數／傳輸時間、R2 終點、R3 達標所需傳輸時間、R5 訓練中 acc、R6 共識與年齡、R7 健康、R8 錨點。
    成本＝每次送出 1／b_i（EF-HC §IV 記帳式，模型大小正規化為 1）；敏感度＝每個鄰居各算一次（n_pushed／b_i）。
    R4 畫風份量另跑 async_influence_and_aggregation_symmetry.py。只處理已存在的臂。"""
    import os
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    arms = {k: v for k, v in BW_ARMS.items() if os.path.exists(ROOT + v[1] + '/async_diag/broadcast_log.csv')}
    # 頻寬表：開了頻寬旗標的臂都寫了 bandwidth.csv，須逐位相同；C0 沒開，用同一張表事後計價
    bt = {k: pd.read_csv(ROOT + v[1] + '/async_diag/bandwidth.csv') for k, v in arms.items()
          if os.path.exists(ROOT + v[1] + '/async_diag/bandwidth.csv')}
    if bt:
        ref = next(iter(bt.values()))
    else:   # 只有 C0 時：用同一支函式重算頻寬表（與 train.py 同源）
        import sys as _s; _s.path.insert(0, ROOT)
        from communicator import stratified_bandwidth_table
        _st = ['art_painting'] * 3 + ['photo'] * 3 + ['sketch'] * 3
        ref = pd.DataFrame(dict(node=[f'node_{i}' for i in range(9)], style=_st, b=stratified_bandwidth_table(_st, 0, 0.1),
                                degree=[5, 2, 5, 6, 6, 3, 8, 6, 5]))
    for k, t in bt.items():
        assert (t.b.values == ref.b.values).all(), f'{k} 的頻寬表與其他臂不同'
    bmap = dict(zip(ref.node, ref.b))
    print('=' * 100); print('頻寬表（各臂 bandwidth.csv 逐位相同：' + '、'.join(bt) + '）'); print('=' * 100)
    print('  ' + ', '.join(f'{n}({s[:5]})={b:.4f} 度數{d}' for n, s, b, d in zip(ref.node, ref['style'], ref.b, ref.degree)))
    for k, t in bt.items():
        print(f'  [{k}] rho_in_threshold={t.rho_in_threshold.round(4).tolist()} period={t.period.tolist()}')

    print('\n' + '=' * 100); print('R1 份數與傳輸時間（全程、逐段、逐節點）'); print('=' * 100)
    B, cum_t, cum_x = {}, {}, {}
    for k, (lg, run) in arms.items():
        b = pd.read_csv(ROOT + run + '/async_diag/broadcast_log.csv'); b['bw'] = b.domain.map(bmap)
        b['t'] = b.fired / b.bw; b['t_nb'] = b.n_pushed / b.bw; B[k] = b
        cum_t[k] = b.groupby('epoch').t.sum().cumsum(); cum_x[k] = b.groupby('epoch').n_pushed.sum().cumsum()
        pn = b.groupby('domain').fired.sum().astype(int)
        seg = [int(b[(b.epoch > lo) & (b.epoch <= lo + 50)].n_pushed.sum()) for lo in (0, 50, 100, 150)]
        print(f'  [{k}] epoch 數 {b.epoch.max()}｜總份數 {int(b.n_pushed.sum())}｜總送出 {int(b.fired.sum())}｜forced {int(b.forced.sum())}'
              f'｜總傳輸時間 {b.t.sum():.0f}（每鄰居計 {b.t_nb.sum():.0f}）｜逐段份數 {seg}')
        print(f'      逐節點送出 {pn.tolist()}｜逐節點傳輸時間 {b.groupby("domain").t.sum().round(0).astype(int).tolist()}')

    E = None
    if os.path.exists(ROOT + curve_csv):
        E = pd.read_csv(ROOT + curve_csv)
        E['arm'] = E.run.map({v[1].replace('exp_result_', '', 1): k for k, v in arms.items()})
        E = E[E.arm.notna()].copy(); E['ep'] = E.ckpt.str.replace('epoch_', '').astype(int)

        print('\n' + '=' * 100); print('R8 錨點：C0-dense ep200 vs 1007 C ep200（同 node×score_fn 全數值欄最大絕對差）'); print('=' * 100)
        num = ['oscr', 'h_best', 'det_auroc', 'det_aupr', 'fpr95', 'closed_acc']
        o = pd.read_csv(ROOT + 'results/proto_trigger/1007_periodic_cartoon_ep200_avgbnB.csv').set_index(['node', 'score_fn'])[num]
        n_ = E[(E.arm == 'C0') & (E.ep == 200)].set_index(['node', 'score_fn'])[num]
        if len(n_):
            j = o.join(n_, rsuffix='_new', how='inner')
            print(f'  比對列數 {len(j)}，最大絕對差 {max(float((j[c] - j[c + "_new"]).abs().max()) for c in num):.4g}')

        def m(arm, ep, s_, col):
            x = E[(E.arm == arm) & (E.ep == ep) & (E.score_fn == s_)][col]
            return x.mean() if len(x) == 9 else np.nan

        print('\n' + '=' * 100); print('R2 終點（ep200、平均B、9 節點平均）＋臂間差（面 >0.015、泛化 >4.92 分才算有差）'); print('=' * 100)
        END = {}
        for k in arms:
            END[k] = dict(acc=m(k, 200, 'zperp', 'closed_acc'), z=m(k, 200, 'zperp', 'det_auroc'), p=m(k, 200, 'proto_angle', 'det_auroc'),
                          en=m(k, 200, 'energy', 'det_auroc'), oscr=m(k, 200, 'zperp', 'oscr'))
            e = END[k]; print(f'  [{k}] 泛化 {e["acc"]:.4f}｜面 {e["z"]:.4f}｜點 {e["p"]:.4f}｜energy {e["en"]:.4f}｜面 OSCR {e["oscr"]:.4f}')
        for a, b_ in [('Dbw', 'C0'), ('Dbw', 'Cbw'), ('Dbw', 'Cmatch'), ('Cmatch', 'Cbw'), ('Cbw', 'C0')]:
            if a in END and b_ in END:
                dz = END[a]['z'] - END[b_]['z']; dg = 100 * (END[a]['acc'] - END[b_]['acc'])
                print(f'  {a} − {b_}：面 {dz:+.4f}（{"有差" if abs(dz) > 0.015 else "雜訊內"}）泛化 {dg:+.2f} 分（{"有差" if abs(dg) > 4.92 else "雜訊內"}）')

        print('\n' + '=' * 100); print(f'R3 達標所需傳輸時間（面 ≥{targets[0]}、泛化 ≥{targets[1]}；首次達到的 checkpoint；比值 <0.67 或 >1.5 才算有差）'); print('=' * 100)
        R3 = {}
        for k in arms:
            eps = sorted(E[E.arm == k].ep.unique())
            row = {}
            for key, s_, col, thr in [('z', 'zperp', 'det_auroc', targets[0]), ('acc', 'zperp', 'closed_acc', targets[1])]:
                hit = [e for e in eps if m(k, e, s_, col) >= thr]
                row[key] = (hit[0], float(cum_t[k].loc[hit[0]]), int(cum_x[k].loc[hit[0]])) if hit else None
            R3[k] = row
            fmt = lambda r: f'ep{r[0]}・時間 {r[1]:.0f}・份數 {r[2]}' if r else '未達到'
            print(f'  [{k}] 面：{fmt(row["z"])}｜泛化：{fmt(row["acc"])}')
        for a, b_ in [('Dbw', 'C0'), ('Dbw', 'Cbw'), ('Dbw', 'Cmatch'), ('Cmatch', 'Cbw'), ('Cbw', 'C0')]:
            if a in R3 and b_ in R3:
                out = []
                for key, lab in [('z', '面'), ('acc', '泛化')]:
                    if R3[a][key] and R3[b_][key]:
                        r_ = R3[a][key][1] / R3[b_][key][1]
                        out.append(f'{lab} 時間比 {r_:.3f}（{"有差" if (r_ < 0.67 or r_ > 1.5) else "範圍內"}）')
                    else:
                        out.append(f'{lab} 至少一臂未達到')
                print(f'  {a} ÷ {b_}：' + '；'.join(out))

        print('\n' + '=' * 100); print('R3 補充：同累積傳輸時間下的成績（各臂曲線線性內插到共同時間點）'); print('=' * 100)
        grid = [1000, 2000, 3000, 4000, 6000, 8000]
        for k in arms:
            eps = sorted(E[E.arm == k].ep.unique()); tt = [float(cum_t[k].loc[e]) for e in eps]
            zz = [m(k, e, 'zperp', 'det_auroc') for e in eps]; aa = [m(k, e, 'zperp', 'closed_acc') for e in eps]
            print(f'  [{k}] ' + ' '.join(f't{g}: 面 {np.interp(g, tt, zz, left=np.nan, right=np.nan):.4f}/泛化 {np.interp(g, tt, aa, left=np.nan, right=np.nan):.4f}' for g in grid))

    print('\n' + '=' * 100); print('R5／R7 訓練中 acc（非正式）與健康（頭／中／尾；末 20 epoch 斜率）'); print('=' * 100)
    for k, (lg, run) in arms.items():
        if not os.path.exists(ROOT + lg):
            continue
        df = parse(lg); print(f'  [{k}] epoch 行數 {len(df)}；NaN/Traceback 行數 {nan_count(lg)}')
        for col, lab in [('loss', '總損失'), ('te', '訓練中目標域 acc%'), ('g_cls', '‖g_cls‖'), ('compL', 'comp 損失'), ('dispL', 'disp 損失')]:
            s = df.set_index('ep')[col].dropna() if col in df else pd.Series(dtype=float)
            if not len(s):
                continue
            pts = ' '.join(f'ep{e}:{s.get(e, np.nan):.4g}' for e in (1, 11, 50, 100, 150, 200))
            tail = s[s.index > 180]; post = s[s.index > 10]
            slope = np.polyfit(tail.index, tail.values, 1)[0] if len(tail) > 2 else np.nan
            print(f'      {lab:<16} {pts} | ep11+ min {post.min():.4g} max {post.max():.4g} mean {post.mean():.4g} std {post.std():.4g} | 末20 斜率 {slope:+.2e}')

    print('\n' + '=' * 100); print('R6 共識與年齡'); print('=' * 100)
    for k, (lg, run) in arms.items():
        b = B[k]; c = pd.read_csv(ROOT + run + '/async_diag/consensus.csv')
        ages = ' '.join(f'ep{lo + 1}-{lo + 50}: 模型 {b[(b.epoch > lo) & (b.epoch <= lo + 50) & (b.n_received > 0)].mean_staleness.mean():.2f}／風格 '
                        f'{b[(b.epoch > lo) & (b.epoch <= lo + 50)].mean_style_age.mean():.2f}' for lo in (0, 50, 100, 150))
        cons = ' '.join(f'ep{e}:{c[c.epoch == e].consensus_rms.iloc[0]:.2e}' for e in (50, 100, 150, 200) if (c.epoch == e).any())
        print(f'  [{k}] 共識 RMS {cons}\n      年齡（步）{ages}')
        slow = min(bmap, key=bmap.get)
        q = b[b.domain == slow]
        print(f'      最慢節點 {slow}：送出 {int(q.fired.sum())}、收到模型平均年齡 {q[q.n_received > 0].mean_staleness.mean():.2f}')

    if E is not None:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
        col = {'C0': 'tab:gray', 'Dbw': 'tab:green', 'Cbw': 'tab:blue', 'Cmatch': 'tab:orange'}
        for k in arms:
            eps = sorted(E[E.arm == k].ep.unique()); tt = [float(cum_t[k].loc[e]) for e in eps]
            ax[0].plot(tt, [100 * m(k, e, 'zperp', 'closed_acc') for e in eps], '-', c=col[k], label=k, lw=1.2)
            ax[1].plot(tt, [m(k, e, 'zperp', 'det_auroc') for e in eps], '-', c=col[k], label=k, lw=1.2)
            if os.path.exists(ROOT + arms[k][0]):
                te = parse(arms[k][0]).set_index('ep')['te'].rolling(5, min_periods=1).mean()
                ax[2].plot(cum_t[k].reindex(te.index).values, te.values, c=col[k], lw=1, label=k)
        ax[0].axhline(100 * targets[1], ls=':', c='k', lw=.8); ax[1].axhline(targets[0], ls=':', c='k', lw=.8)
        ax[0].set_title('formal closed-set acc (%) vs cumulative tx time'); ax[1].set_title('formal face AUROC vs cumulative tx time')
        ax[2].set_title('in-training target acc (informal, 5-ep smooth)')
        for x in ax:
            x.set_xlabel('cumulative transmission time (1/b per send)'); x.legend(); x.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(ROOT + fig_path, dpi=130); print(f'\n圖：{fig_path}')


def efhc_figure(curve_csv='results/proto_trigger/1009_bw_curve_cartoon_avgbnB.csv',
                fig_path='results/proto_trigger/1009_bw_efhc_style.png'):
    """[1012] 照 EF-HC 原文圖 2 的子圖 (i)(ii)(iii) 重畫四臂（cartoon、單 seed）。
    原文 §IV 記帳式：第 k 步的傳輸時間 ＝ (1/m)Σ_i (Σ_j v_ij/d_i)·n/b_i；我們每次推播送所有鄰居 ⇒ Σ_j v_ij/d_i ＝ fired_i(k)，
    n 正規化為 1、m＝9 ⇒ 每步傳輸時間 ＝ (1/9)Σ_i fired_i(k)/b_i；(i) 以每 epoch（20 步）平均畫；(iii) 橫軸為其累積。
    準確率：原文是各設備平均準確率；我們用正式評估（平均B、9 節點平均）的泛化與面檢測，每 5 epoch 一點。"""
    import os
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    bt = pd.read_csv(ROOT + BW_ARMS['Cbw'][1] + '/async_diag/bandwidth.csv'); bmap = dict(zip(bt.node, bt.b))
    E = pd.read_csv(ROOT + curve_csv)
    E['arm'] = E.run.map({v[1].replace('exp_result_', '', 1): k for k, v in BW_ARMS.items()})
    E = E[E.arm.notna()].copy(); E['ep'] = E.ckpt.str.replace('epoch_', '').astype(int)
    g = E[E.score_fn == 'zperp'].groupby(['arm', 'ep'])[['det_auroc', 'closed_acc']].mean()
    col = {'C0': 'tab:gray', 'Dbw': 'tab:green', 'Cbw': 'tab:blue', 'Cmatch': 'tab:orange'}
    lab = {'C0': 'uniform periodic (C0)', 'Dbw': 'EF-HC (D-bw)', 'Cbw': 'bandwidth-fair periodic (C-bw)', 'Cmatch': 'C-match'}
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
    print('=' * 100); print('EF-HC 圖 2 式：(i) 每步傳輸時間（每 epoch 平均）、(iii) 累積傳輸時間（單位同原文：(1/9)Σ fired/b）'); print('=' * 100)
    for k, (lg, run) in BW_ARMS.items():
        b = pd.read_csv(ROOT + run + '/async_diag/broadcast_log.csv'); b['t'] = b.fired / b.domain.map(bmap) / 9.0
        per_step = b.groupby('step').t.sum()                       # 第 k 步的 (1/9)Σ_i fired_i(k)/b_i
        per_ep = per_step.groupby(per_step.index // 20).mean()     # 每 epoch 平均
        cum_ep = b.groupby('epoch').t.sum().cumsum()
        ax[0].plot((per_ep.index + 1) * 20, per_ep.values, c=col[k], lw=1, label=lab[k])
        s = g.loc[k]; it = s.index.values * 20; tt = cum_ep.loc[s.index].values
        ax[1].plot(it, 100 * s.closed_acc.values, c=col[k], lw=1.2, label=lab[k])
        ax[2].plot(tt, 100 * s.closed_acc.values, c=col[k], lw=1.2, label=lab[k])
        ax[3].plot(tt, s.det_auroc.values, c=col[k], lw=1.2, label=lab[k])
        print(f'  [{k}] 每步傳輸時間：全程平均 {per_step.mean():.3f}｜ep1–50 {per_step[per_step.index < 1000].mean():.3f}'
              f'｜ep151–200 {per_step[per_step.index >= 3000].mean():.3f}｜累積總計 {cum_ep.iloc[-1]:.0f}')
    ax[0].set_title('(i) transmission time per iteration (epoch avg)'); ax[0].set_xlabel('iteration'); ax[0].set_ylabel('time units')
    ax[1].set_title('(ii) closed-set acc (%) vs iteration'); ax[1].set_xlabel('iteration')
    ax[2].set_title('(iii) closed-set acc (%) vs cumulative tx time'); ax[2].set_xlabel('cumulative transmission time units')
    ax[3].set_title('(iii\') face AUROC vs cumulative tx time'); ax[3].set_xlabel('cumulative transmission time units')
    for x in ax:
        x.grid(alpha=.3); x.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ROOT + fig_path, dpi=130); print(f'\n圖：{fig_path}')


ARTSLOW_RUN = f'exp_result_v1_stage2_leave_cartoon_p1a_async_pbwart_bwfart0.1{_BWS}'
ARTSLOW_LOG = 'logs/proto_trigger/1012_artslow_train.log'
# [plan 1012 §6] 事前寫死的 R3 門檻（1009 四臂同一隨機起點的範圍；面 ep200 用雜訊門檻 0.015，其餘用範圍寬度）
ARTSLOW_THR = {('z', 'end'): 0.8031, ('z', 'win'): 0.7955, ('acc', 'end'): 0.7900, ('acc', 'win'): 0.7888}


def artslow_section(curve_csv='results/proto_trigger/1012_artslow_curve_cartoon_avgbnB.csv',
                    ref_csv='results/proto_trigger/1009_bw_curve_cartoon_avgbnB.csv',
                    fig_path='results/proto_trigger/1012_artslow_curve_cartoon.png'):
    """[plan 1012] 慢設備集中在 art：R1 送出、R2 art 份量（實際送出紀錄、1003 單向遞推）、R3 事前門檻四級判決、R4 健康、R5 輔助。
    對照＝1009 四臂（同一隨機起點；C-bw 為主對照），按 checkpoint（步數）對齊——兩者總份數已對齊。"""
    import os, sys
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    sys.path.insert(0, ROOT + 'scripts/posthoc')
    from artslow_period_and_share_estimate import infl_oneway, share, fire_matrix
    arms = dict(BW_ARMS); arms['ArtSlow'] = (ARTSLOW_LOG, ARTSLOW_RUN)
    B = {}
    for k, (lg, run) in arms.items():
        b = pd.read_csv(ROOT + run + '/async_diag/broadcast_log.csv'); b['i'] = b.domain.str[5:].astype(int); B[k] = b
    bt = pd.read_csv(ROOT + ARTSLOW_RUN + '/async_diag/bandwidth.csv'); bmap = dict(zip(bt.node, bt.b))
    bt9 = pd.read_csv(ROOT + BW_ARMS['Cbw'][1] + '/async_diag/bandwidth.csv'); bmap9 = dict(zip(bt9.node, bt9.b))

    print('=' * 100); print('R1 送出（ArtSlow 用自己的頻寬表計價；1009 臂用 1009 表）'); print('=' * 100)
    print('  ArtSlow 頻寬表 ' + ', '.join(f'{n}={b:.4f}/週期{p:.4f}' for n, b, p in zip(bt.node, bt.b, bt.period)))
    for k in ['Cbw', 'ArtSlow']:
        b = B[k]; bm = bmap if k == 'ArtSlow' else bmap9; b['t'] = b.fired / b.domain.map(bm)
        print(f'  [{k}] epoch {b.epoch.max()}｜總份數 {int(b.n_pushed.sum())}｜forced {int(b.forced.sum())}｜總傳輸時間 {b.t.sum():.0f}'
              f'｜逐節點送出 {b.groupby("i").fired.sum().astype(int).tolist()}')
    Fe = B['ArtSlow'].pivot(index='step', columns='i', values='fired').sort_index().values.astype(bool)
    Fs = fire_matrix(bt.period.values)[:Fe.shape[0]]
    print(f'  全程實際 vs 合成送出不一致格數 {int((Fe != Fs).sum())}（{Fe.shape[0]} 步）')

    print('\n' + '=' * 100); print('R2 操作檢查：各畫風在共識裡的份量（實際送出紀錄、1003 單向遞推；公平 0.333）'); print('=' * 100)
    SH = {}
    for k in ['C0', 'Cbw', 'ArtSlow']:
        F = B[k].pivot(index='step', columns='i', values='fired').sort_index().values.astype(bool)
        h = infl_oneway(F); SH[k] = share(h, 0, F.shape[0])
        segs = ' '.join(f'{lo}-{lo + 1000}: ' + '/'.join(f'{v:.3f}' for v in share(h, lo, lo + 1000).values()) for lo in (0, 1000, 2000, 3000))
        print(f'  [{k}] 全程 art/photo/sketch ' + '/'.join(f'{v:.3f}' for v in SH[k].values()) + f'｜{segs}')
    a = SH['ArtSlow']['art']
    print(f'  ⇒ ArtSlow art 份量 {a:.3f}：' + ('≤0.10 操作成立' if a <= 0.10 else ('0.10–0.15 操作偏弱' if a <= 0.15 else '>0.15 無檢驗力、判決作廢')))

    E = pd.concat([pd.read_csv(ROOT + ref_csv), pd.read_csv(ROOT + curve_csv)], ignore_index=True)
    E['arm'] = E.run.map({v[1].replace('exp_result_', '', 1): k for k, v in arms.items()})
    E = E[E.arm.notna()].copy(); E['ep'] = E.ckpt.str.replace('epoch_', '').astype(int)
    G = E.groupby(['arm', 'score_fn', 'ep']).agg(n=('node', 'size'), z=('det_auroc', 'mean'), acc=('closed_acc', 'mean')).reset_index()
    assert (G.n == 9).all(), '有 checkpoint 不是 9 節點'
    Z = G[G.score_fn == 'zperp'].pivot(index='ep', columns='arm', values='z')
    A = G[G.score_fn == 'zperp'].pivot(index='ep', columns='arm', values='acc')
    eps_e = sorted(E[E.arm == 'ArtSlow'].ep.unique())
    print(f'\n  ArtSlow 評估 checkpoint {len(eps_e)} 個（{eps_e[0]}…{eps_e[-1]}）')

    print('\n' + '=' * 100); print('R3 主指標（zperp 列、九台平均、BN 平均B）＋事前門檻四級判決'); print('=' * 100)
    win = lambda P, k: P.loc[25:200, k].mean() if P.loc[25:200, k].notna().sum() == 36 else np.nan
    for lab, P in [('面檢測', Z), ('泛化', A)]:
        print(f'  {lab}：' + '｜'.join(f'{k} ep200 {P.loc[200, k]:.4f}／ep25–200 {win(P, k):.4f}' for k in ['C0', 'Dbw', 'Cbw', 'Cmatch', 'ArtSlow']))
        four = [k for k in ['C0', 'Dbw', 'Cbw', 'Cmatch']]
        e_ = P.loc[200, four]; w_ = pd.Series({k: win(P, k) for k in four})
        print(f'    1009 四臂範圍（重算）：ep200 {e_.min():.4f}–{e_.max():.4f}（寬 {e_.max() - e_.min():.4f}）'
              f'；ep25–200 {w_.min():.4f}–{w_.max():.4f}（寬 {w_.max() - w_.min():.4f}）')
    v = {('z', 'end'): Z.loc[200, 'ArtSlow'], ('z', 'win'): win(Z, 'ArtSlow'),
         ('acc', 'end'): A.loc[200, 'ArtSlow'], ('acc', 'win'): win(A, 'ArtSlow')}
    below = {key: v[key] < thr for key, thr in ARTSLOW_THR.items()}
    for key, thr in ARTSLOW_THR.items():
        print(f'  {"面檢測" if key[0] == "z" else "泛化"} {"ep200" if key[1] == "end" else "ep25–200"}：{v[key]:.4f} vs 門檻 {thr:.4f} ⇒ '
              f'{"低於門檻" if below[key] else "未低於"}（距 C-bw {v[key] - (Z if key[0] == "z" else A).loc[200 if key[1] == "end" else slice(25, 200), "Cbw"].mean():+.4f}）')
    hurt = {m_: below[(m_, 'end')] and below[(m_, 'win')] for m_ in ('z', 'acc')}
    nb = sum(below.values())
    verdict = ('強訊號（兩指標都傷）' if all(hurt.values()) else '中等（' + ('面檢測' if hurt['z'] else '泛化') + '傷）' if any(hurt.values())
               else '邊際（只有一個數低於門檻）' if nb >= 1 else '退化（不傷）')
    if not any(hurt.values()) and nb >= 2:
        verdict = f'邊際（{nb} 個數低於門檻但不成對）'
    print(f'  ⇒ 四級判決：{verdict}')

    print('\n' + '=' * 100); print('R3 補充：逐 checkpoint ArtSlow − C-bw（面檢測／泛化）與分段平均'); print('=' * 100)
    for e in [5, 10, 15, 25, 50, 75, 100, 125, 150, 175, 200]:
        print(f'  ep{e:3d}: 面 {Z.loc[e, "ArtSlow"]:.4f}（{Z.loc[e, "ArtSlow"] - Z.loc[e, "Cbw"]:+.4f}）泛化 {A.loc[e, "ArtSlow"]:.4f}（{A.loc[e, "ArtSlow"] - A.loc[e, "Cbw"]:+.4f}）')
    for lo, hi in [(5, 50), (55, 100), (105, 150), (155, 200)]:
        dz = (Z.loc[lo:hi, 'ArtSlow'] - Z.loc[lo:hi, 'Cbw']).mean(); da = (A.loc[lo:hi, 'ArtSlow'] - A.loc[lo:hi, 'Cbw']).mean()
        rz = Z.loc[lo:hi, ['C0', 'Dbw', 'Cbw', 'Cmatch']].mean(); ra = A.loc[lo:hi, ['C0', 'Dbw', 'Cbw', 'Cmatch']].mean()
        print(f'  ep{lo}–{hi} 平均差：面 {dz:+.4f}（四臂範圍寬 {rz.max() - rz.min():.4f}）泛化 {da:+.4f}（寬 {ra.max() - ra.min():.4f}）')
    for fn in ['energy', 'proto_angle']:
        P = G[G.score_fn == fn].pivot(index='ep', columns='arm', values='z')
        print(f'  次要 {fn}：' + '｜'.join(f'{k} ep200 {P.loc[200, k]:.4f}／ep25–200 {win(P, k):.4f}' for k in ['Cbw', 'ArtSlow']))

    print('\n' + '=' * 100); print('R4 訓練健康'); print('=' * 100)
    for k in ['Cbw', 'ArtSlow']:
        lg = arms[k][0]; df = parse(lg)
        pts = ' '.join(f'ep{e}:{df.set_index("ep").loss.get(e, np.nan):.4g}' for e in (1, 11, 50, 100, 150, 200))
        tes = ' '.join(f'ep{e}:{df.set_index("ep").te.get(e, np.nan):.2f}' for e in (1, 11, 50, 100, 150, 200))
        print(f'  [{k}] epoch 行 {len(df)}｜NaN/Traceback {nan_count(lg)}｜總損失 {pts}｜訓練中目標 acc% {tes}')

    print('\n' + '=' * 100); print('R5 輔助：art 三台 vs 其他、最差那台、風格年齡、共識'); print('=' * 100)
    zz = E[E.score_fn == 'zperp']
    for k in ['Cbw', 'ArtSlow']:
        q = zz[zz.arm == k]; q = q.assign(art=q.node.isin(['node_0', 'node_1', 'node_2']))
        for lab, col in [('面', 'det_auroc'), ('泛化', 'closed_acc')]:
            w = q[(q.ep >= 25)].groupby('art')[col].mean(); e2 = q[q.ep == 200].groupby('art')[col].mean()
            p = q.pivot(index='ep', columns='node', values=col)
            worst = p.idxmin(axis=1).value_counts().head(3).to_dict()
            print(f'  [{k}] {lab}：art 三台 ep200 {e2[True]:.4f}／其他 {e2[False]:.4f}｜ep25–200 art {w[True]:.4f}／其他 {w[False]:.4f}｜最差次數 {worst}')
        b = B[k]; c = pd.read_csv(ROOT + arms[k][1] + '/async_diag/consensus.csv')
        ages = ' '.join(f'ep{lo + 1}-{lo + 50}:{b[(b.epoch > lo) & (b.epoch <= lo + 50)].mean_style_age.mean():.2f}' for lo in (0, 50, 100, 150))
        cons = ' '.join(f'ep{e}:{c[c.epoch == e].consensus_rms.iloc[0]:.2e}' for e in (50, 100, 150, 200) if (c.epoch == e).any())
        print(f'  [{k}] 收件匣風格平均年齡（步）{ages}｜共識 RMS {cons}')
    ba = B['ArtSlow']
    print('  ArtSlow 逐節點收件匣風格年齡（全程平均）：' + ' '.join(f'n{i}:{ba[ba.i == i].mean_style_age.mean():.1f}' for i in range(9)))

    print('\n' + '=' * 100); print('R6（事後，非事前判準）逐節點訓練準確率：<run>_res/…-r{i}-tacc.log（風格增強後的訓練批次、訓練中模型；r＝節點序）'); print('=' * 100)
    for k, (lg, run) in arms.items():
        n_ = run.replace('exp_result_', '', 1)
        T = np.array([np.loadtxt(f'{ROOT}{run}{n_}_res/dsgd-lr0.001-budget1.0-r{i}-tacc.log') for i in range(9)])
        w = T[:, 100:200].mean(1); t20 = T[:, 180:200].mean(1)
        seg = ' '.join(f'ep{a + 1}-{b}:{T[:3, a:b].mean():.2f}' for a, b in [(0, 10), (10, 50), (50, 100), (100, 150), (150, 200)])
        print(f'  [{k:7s}] ep101–200 逐節點 ' + ' '.join(f'{x:.2f}' for x in w)
              + f'｜art {w[:3].mean():.2f} photo {w[3:6].mean():.2f} sketch {w[6:].mean():.2f}｜ep181–200 art {t20[:3].mean():.2f} 其他 {t20[3:].mean():.2f}')
        print(f'             art 三台分段 {seg}｜ep200 逐節點 ' + ' '.join(f'{x:.1f}' for x in T[:, 199]))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    col = {'C0': 'tab:gray', 'Dbw': 'tab:green', 'Cbw': 'tab:blue', 'Cmatch': 'tab:orange', 'ArtSlow': 'tab:red'}
    for k in col:
        ax[0].plot(A.index, 100 * A[k], c=col[k], lw=1.8 if k in ('Cbw', 'ArtSlow') else 0.9, label=k)
        ax[1].plot(Z.index, Z[k], c=col[k], lw=1.8 if k in ('Cbw', 'ArtSlow') else 0.9, label=k)
    ax[0].axhline(100 * ARTSLOW_THR[('acc', 'end')], ls=':', c='k', lw=.8); ax[1].axhline(ARTSLOW_THR[('z', 'end')], ls=':', c='k', lw=.8)
    ax[0].set_title('formal closed-set acc (%) vs epoch'); ax[1].set_title('formal face AUROC (zperp) vs epoch')
    for x in ax:
        x.set_xlabel('epoch (20 steps each)'); x.grid(alpha=.3); x.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT + fig_path, dpi=130); print(f'\n圖：{fig_path}')


if '--curve' in __import__('sys').argv:
    curve_section(); raise SystemExit(0)
if '--bw' in __import__('sys').argv:
    bw_section(); raise SystemExit(0)
if '--efhc' in __import__('sys').argv:
    efhc_figure(); raise SystemExit(0)
if '--artslow' in __import__('sys').argv:
    artslow_section(); raise SystemExit(0)

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
