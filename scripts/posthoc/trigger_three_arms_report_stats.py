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
SKETCHSLOW_RUN = f'exp_result_v1_stage2_leave_cartoon_p1a_async_pbwsk_bwfsketch0.1{_BWS}'   # [plan 1013]
# [plan 1012 §6] 事前寫死的 R3 門檻（1009 四臂同一隨機起點的範圍；面 ep200 用雜訊門檻 0.015，其餘用範圍寬度）
ARTSLOW_THR = {('z', 'end'): 0.8031, ('z', 'win'): 0.7955, ('acc', 'end'): 0.7900, ('acc', 'win'): 0.7888}


def artslow_section(curve_csv='results/proto_trigger/1012_artslow_curve_cartoon_avgbnB.csv',
                    ref_csv='results/proto_trigger/1009_bw_curve_cartoon_avgbnB.csv',
                    fig_path='results/proto_trigger/1012_artslow_curve_cartoon.png',
                    X='ArtSlow', run_x=ARTSLOW_RUN, log_x=ARTSLOW_LOG, style='art', extra=None):
    """[plan 1012] 慢設備集中在 art：R1 送出、R2 art 份量（實際送出紀錄、1003 單向遞推）、R3 事前門檻四級判決、R4 健康、R5 輔助。
    對照＝1009 四臂（同一隨機起點；C-bw 為主對照），按 checkpoint（步數）對齊——兩者總份數已對齊。
    [1013] 參數化：X／run_x／log_x／style＝哪一臂、哪個畫風慢；extra＝{標籤: (log, run, curve_csv)} 額外並列的臂（如 ArtSlow）。"""
    import os, sys
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    sys.path.insert(0, ROOT + 'scripts/posthoc')
    from artslow_period_and_share_estimate import infl_oneway, share, fire_matrix
    extra = extra or {}
    arms = dict(BW_ARMS); arms.update({k: (v[0], v[1]) for k, v in extra.items()}); arms[X] = (log_x, run_x)
    si = [i for i in range(9) if ['art', 'photo', 'sketch'][i // 3] == style]; slow_nodes = [f'node_{i}' for i in si]
    B = {}
    for k, (lg, run) in arms.items():
        b = pd.read_csv(ROOT + run + '/async_diag/broadcast_log.csv'); b['i'] = b.domain.str[5:].astype(int); B[k] = b
    bt = pd.read_csv(ROOT + run_x + '/async_diag/bandwidth.csv'); bmap = dict(zip(bt.node, bt.b))
    bt9 = pd.read_csv(ROOT + BW_ARMS['Cbw'][1] + '/async_diag/bandwidth.csv'); bmap9 = dict(zip(bt9.node, bt9.b))

    print('=' * 100); print(f'R1 送出（{X} 用自己的頻寬表計價；1009 臂用 1009 表）'); print('=' * 100)
    print(f'  {X} 頻寬表 ' + ', '.join(f'{n}={b:.4f}/週期{p:.4f}' for n, b, p in zip(bt.node, bt.b, bt.period)))
    for k in ['Cbw', X]:
        b = B[k]; bm = bmap if k == X else bmap9; b['t'] = b.fired / b.domain.map(bm)
        print(f'  [{k}] epoch {b.epoch.max()}｜總份數 {int(b.n_pushed.sum())}｜forced {int(b.forced.sum())}｜總傳輸時間 {b.t.sum():.0f}'
              f'｜逐節點送出 {b.groupby("i").fired.sum().astype(int).tolist()}')
    Fe = B[X].pivot(index='step', columns='i', values='fired').sort_index().values.astype(bool)
    Fs = fire_matrix(bt.period.values)[:Fe.shape[0]]
    print(f'  全程實際 vs 合成送出不一致格數 {int((Fe != Fs).sum())}（{Fe.shape[0]} 步）')

    print('\n' + '=' * 100); print('R2 操作檢查：各畫風在共識裡的份量（實際送出紀錄、1003 單向遞推；公平 0.333）'); print('=' * 100)
    SH = {}
    for k in ['C0', 'Cbw', *extra, X]:
        F = B[k].pivot(index='step', columns='i', values='fired').sort_index().values.astype(bool)
        h = infl_oneway(F); SH[k] = share(h, 0, F.shape[0])
        segs = ' '.join(f'{lo}-{lo + 1000}: ' + '/'.join(f'{v:.3f}' for v in share(h, lo, lo + 1000).values()) for lo in (0, 1000, 2000, 3000))
        print(f'  [{k}] 全程 art/photo/sketch ' + '/'.join(f'{v:.3f}' for v in SH[k].values()) + f'｜{segs}')
    a = SH[X][style]
    print(f'  ⇒ {X} {style} 份量 {a:.3f}：' + ('≤0.10 操作成立' if a <= 0.10 else ('0.10–0.15 操作偏弱' if a <= 0.15 else '>0.15 無檢驗力、判決作廢')))

    E = pd.concat([pd.read_csv(ROOT + ref_csv)] + [pd.read_csv(ROOT + v[2]) for v in extra.values()]
                  + [pd.read_csv(ROOT + curve_csv)], ignore_index=True)
    E['arm'] = E.run.map({v[1].replace('exp_result_', '', 1): k for k, v in arms.items()})
    E = E[E.arm.notna()].copy(); E['ep'] = E.ckpt.str.replace('epoch_', '').astype(int)
    G = E.groupby(['arm', 'score_fn', 'ep']).agg(n=('node', 'size'), z=('det_auroc', 'mean'), acc=('closed_acc', 'mean')).reset_index()
    assert (G.n == 9).all(), '有 checkpoint 不是 9 節點'
    Z = G[G.score_fn == 'zperp'].pivot(index='ep', columns='arm', values='z')
    A = G[G.score_fn == 'zperp'].pivot(index='ep', columns='arm', values='acc')
    eps_e = sorted(E[E.arm == X].ep.unique())
    print(f'\n  {X} 評估 checkpoint {len(eps_e)} 個（{eps_e[0]}…{eps_e[-1]}）')

    print('\n' + '=' * 100); print('R3 主指標（zperp 列、九台平均、BN 平均B）＋事前門檻四級判決'); print('=' * 100)
    win = lambda P, k: P.loc[25:200, k].mean() if P.loc[25:200, k].notna().sum() == 36 else np.nan
    for lab, P in [('面檢測', Z), ('泛化', A)]:
        print(f'  {lab}：' + '｜'.join(f'{k} ep200 {P.loc[200, k]:.4f}／ep25–200 {win(P, k):.4f}' for k in ['C0', 'Dbw', 'Cbw', 'Cmatch', *extra, X]))
        four = [k for k in ['C0', 'Dbw', 'Cbw', 'Cmatch']]
        e_ = P.loc[200, four]; w_ = pd.Series({k: win(P, k) for k in four})
        print(f'    1009 四臂範圍（重算）：ep200 {e_.min():.4f}–{e_.max():.4f}（寬 {e_.max() - e_.min():.4f}）'
              f'；ep25–200 {w_.min():.4f}–{w_.max():.4f}（寬 {w_.max() - w_.min():.4f}）')
    v = {('z', 'end'): Z.loc[200, X], ('z', 'win'): win(Z, X),
         ('acc', 'end'): A.loc[200, X], ('acc', 'win'): win(A, X)}
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

    print('\n' + '=' * 100); print(f'R3 補充：逐 checkpoint {X} − C-bw（面檢測／泛化）與分段平均'); print('=' * 100)
    for e in [5, 10, 15, 25, 50, 75, 100, 125, 150, 175, 200]:
        print(f'  ep{e:3d}: 面 {Z.loc[e, X]:.4f}（{Z.loc[e, X] - Z.loc[e, "Cbw"]:+.4f}）泛化 {A.loc[e, X]:.4f}（{A.loc[e, X] - A.loc[e, "Cbw"]:+.4f}）')
    for lo, hi in [(5, 50), (55, 100), (105, 150), (155, 200)]:
        dz = (Z.loc[lo:hi, X] - Z.loc[lo:hi, 'Cbw']).mean(); da = (A.loc[lo:hi, X] - A.loc[lo:hi, 'Cbw']).mean()
        rz = Z.loc[lo:hi, ['C0', 'Dbw', 'Cbw', 'Cmatch']].mean(); ra = A.loc[lo:hi, ['C0', 'Dbw', 'Cbw', 'Cmatch']].mean()
        print(f'  ep{lo}–{hi} 平均差：面 {dz:+.4f}（四臂範圍寬 {rz.max() - rz.min():.4f}）泛化 {da:+.4f}（寬 {ra.max() - ra.min():.4f}）')
    for fn in ['energy', 'proto_angle']:
        P = G[G.score_fn == fn].pivot(index='ep', columns='arm', values='z')
        print(f'  次要 {fn}：' + '｜'.join(f'{k} ep200 {P.loc[200, k]:.4f}／ep25–200 {win(P, k):.4f}' for k in ['Cbw', *extra, X]))

    print('\n' + '=' * 100); print('R4 訓練健康'); print('=' * 100)
    for k in ['Cbw', *extra, X]:
        lg = arms[k][0]; df = parse(lg)
        pts = ' '.join(f'ep{e}:{df.set_index("ep").loss.get(e, np.nan):.4g}' for e in (1, 11, 50, 100, 150, 200))
        tes = ' '.join(f'ep{e}:{df.set_index("ep").te.get(e, np.nan):.2f}' for e in (1, 11, 50, 100, 150, 200))
        print(f'  [{k}] epoch 行 {len(df)}｜NaN/Traceback {nan_count(lg)}｜總損失 {pts}｜訓練中目標 acc% {tes}')

    print('\n' + '=' * 100); print(f'R5 輔助：{style} 三台 vs 其他、最差那台、風格年齡、共識'); print('=' * 100)
    zz = E[E.score_fn == 'zperp']
    for k in ['Cbw', *extra, X]:
        q = zz[zz.arm == k]; q = q.assign(art=q.node.isin(slow_nodes))
        for lab, col in [('面', 'det_auroc'), ('泛化', 'closed_acc')]:
            w = q[(q.ep >= 25)].groupby('art')[col].mean(); e2 = q[q.ep == 200].groupby('art')[col].mean()
            p = q.pivot(index='ep', columns='node', values=col)
            worst = p.idxmin(axis=1).value_counts().head(3).to_dict()
            print(f'  [{k}] {lab}：{style} 三台 ep200 {e2[True]:.4f}／其他 {e2[False]:.4f}｜ep25–200 {style} {w[True]:.4f}／其他 {w[False]:.4f}｜最差次數 {worst}')
        b = B[k]; c = pd.read_csv(ROOT + arms[k][1] + '/async_diag/consensus.csv')
        ages = ' '.join(f'ep{lo + 1}-{lo + 50}:{b[(b.epoch > lo) & (b.epoch <= lo + 50)].mean_style_age.mean():.2f}' for lo in (0, 50, 100, 150))
        cons = ' '.join(f'ep{e}:{c[c.epoch == e].consensus_rms.iloc[0]:.2e}' for e in (50, 100, 150, 200) if (c.epoch == e).any())
        print(f'  [{k}] 收件匣風格平均年齡（步）{ages}｜共識 RMS {cons}')
    ba = B[X]
    print(f'  {X} 逐節點收件匣風格年齡（全程平均）：' + ' '.join(f'n{i}:{ba[ba.i == i].mean_style_age.mean():.1f}' for i in range(9)))

    print('\n' + '=' * 100); print('R6（事後，非事前判準）逐節點訓練準確率：<run>_res/…-r{i}-tacc.log（風格增強後的訓練批次、訓練中模型；r＝節點序）'); print('=' * 100)
    for k, (lg, run) in arms.items():
        n_ = run.replace('exp_result_', '', 1)
        T = np.array([np.loadtxt(f'{ROOT}{run}{n_}_res/dsgd-lr0.001-budget1.0-r{i}-tacc.log') for i in range(9)])
        w = T[:, 100:200].mean(1); t20 = T[:, 180:200].mean(1)
        seg = ' '.join(f'ep{a + 1}-{b}:{T[si, a:b].mean():.2f}' for a, b in [(0, 10), (10, 50), (50, 100), (100, 150), (150, 200)])
        print(f'  [{k:7s}] ep101–200 逐節點 ' + ' '.join(f'{x:.2f}' for x in w)
              + f'｜art {w[:3].mean():.2f} photo {w[3:6].mean():.2f} sketch {w[6:].mean():.2f}｜ep181–200 {style} {t20[si].mean():.2f} 其他 {np.delete(t20, si).mean():.2f}')
        print(f'             {style} 三台分段 {seg}｜ep200 逐節點 ' + ' '.join(f'{x:.1f}' for x in T[:, 199]))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    col = {'C0': 'tab:gray', 'Dbw': 'tab:green', 'Cbw': 'tab:blue', 'Cmatch': 'tab:orange'}
    col.update({k: 'tab:purple' for k in extra}); col[X] = 'tab:red'
    for k in col:
        ax[0].plot(A.index, 100 * A[k], c=col[k], lw=1.8 if k in ('Cbw', X) else 0.9, label=k)
        ax[1].plot(Z.index, Z[k], c=col[k], lw=1.8 if k in ('Cbw', X) else 0.9, label=k)
    ax[0].axhline(100 * ARTSLOW_THR[('acc', 'end')], ls=':', c='k', lw=.8); ax[1].axhline(ARTSLOW_THR[('z', 'end')], ls=':', c='k', lw=.8)
    ax[0].set_title('formal closed-set acc (%) vs epoch'); ax[1].set_title('formal face AUROC (zperp) vs epoch')
    for x in ax:
        x.set_xlabel('epoch (20 steps each)'); x.grid(alpha=.3); x.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT + fig_path, dpi=130); print(f'\n圖：{fig_path}')


def efhc_scenario_figure(fig_path='results/proto_trigger/1014_efhc_style_art_sketch_slow.png'):
    """[1014 dany] 慢設備集中情境的 EF-HC 圖 2 式呈現：每個情境用自己的頻寬表計價（記帳式，送一次＝1/b_i）。
    臂：均勻送 C0（訓練與頻寬無關 ⇒ 送出紀錄直接用該情境的表重新計價）vs 該情境的照網速送（ArtSlow／SketchSlow）。
    (i) 每步傳輸時間＝(1/9)Σ fired_i(k)/b_i（每 epoch 平均）；(ii) 準確率 vs epoch；(iii) 準確率 vs 累積傳輸時間。"""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ev = pd.concat([pd.read_csv(ROOT + 'results/proto_trigger/1009_bw_curve_cartoon_avgbnB.csv'),
                    pd.read_csv(ROOT + 'results/proto_trigger/1012_artslow_curve_cartoon_avgbnB.csv'),
                    pd.read_csv(ROOT + 'results/proto_trigger/1013_sketchslow_curve_cartoon_avgbnB.csv')], ignore_index=True)
    ev['ep'] = ev.ckpt.str.replace('epoch_', '').astype(int)
    ev = ev[ev.score_fn == 'zperp']
    SC = {'art 慢': ('results/proto_trigger/1012_artslow/bandwidth_art0.1.csv', ARTSLOW_RUN, 'ArtSlow'),
          'sketch 慢': ('results/proto_trigger/1013_sketchslow/bandwidth_sketch0.1.csv', SKETCHSLOW_RUN, 'SketchSlow')}
    c0 = BW_ARMS['C0'][1]
    fig, ax = plt.subplots(2, 4, figsize=(20, 8.5))
    for r, (scen, (tab, run_s, lab_s)) in enumerate(SC.items()):
        bmap = dict(pd.read_csv(ROOT + tab)[['node', 'b']].values)
        print('=' * 100); print(f'情境：{scen}（頻寬表 {tab}）'); print('=' * 100)
        res = {}
        for lab, run in [('均勻送 C0', c0), (f'照網速送 {lab_s}', run_s)]:
            b = pd.read_csv(ROOT + run + '/async_diag/broadcast_log.csv')
            b['t'] = b.fired / b.domain.map(bmap)
            per_step = b.groupby('step').t.sum() / 9.0
            ep_t = (b.groupby('epoch').t.sum() / 9.0)            # 每 epoch 累積（20 步）
            cum = ep_t.cumsum()
            q = ev[ev.run == run.replace('exp_result_', '', 1)].groupby('ep')[['closed_acc', 'det_auroc']].mean()
            q['cum_t'] = cum.reindex(q.index).values
            res[lab] = (per_step, ep_t, q)
            print(f'  [{lab}] (i) 每步傳輸時間 全程平均 {per_step.mean():.3f}｜ep1–50 {per_step.loc[:999].mean():.3f}｜ep151–200 {per_step.loc[3000:].mean():.3f}'
                  f'｜累積總計 {cum.iloc[-1]:.0f}｜逐節點送出 {b.groupby("domain").fired.sum().astype(int).tolist()}')
            print(f'      (ii) ep200 泛化 {q.closed_acc.loc[200]:.4f} 面 {q.det_auroc.loc[200]:.4f}｜ep25–200 平均 泛化 {q.closed_acc.loc[25:].mean():.4f} 面 {q.det_auroc.loc[25:].mean():.4f}')
            pl = 'uniform C0' if lab.startswith('均勻') else f'bandwidth-aware {lab_s}'   # 圖例用英文（字型無中文）
            ax[r, 0].plot(ep_t.index, ep_t.values / 20, label=pl)
            ax[r, 1].plot(q.index, 100 * q.closed_acc, label=pl)
            ax[r, 2].plot(q.cum_t, 100 * q.closed_acc, label=pl)
            ax[r, 3].plot(q.cum_t, q.det_auroc, label=pl)
        (l0, (ps0, et0, q0)), (l1, (ps1, et1, q1)) = list(res.items())
        T1 = q1.cum_t.iloc[-1]
        print(f'  (iii) 同樣累積傳輸時間下（＝照網速送的全程 {T1:.0f}）：均勻送內插 泛化 {np.interp(T1, q0.cum_t, q0.closed_acc):.4f} 面 {np.interp(T1, q0.cum_t, q0.det_auroc):.4f}'
              f'｜照網速送 泛化 {q1.closed_acc.iloc[-1]:.4f} 面 {q1.det_auroc.iloc[-1]:.4f}')
        for g in [0.25, 0.5, 1.0, 2.0]:
            t = g * T1
            print(f'      時間 {t:7.0f}：均勻送 泛化 {np.interp(t, q0.cum_t, q0.closed_acc, right=np.nan):.4f} 面 {np.interp(t, q0.cum_t, q0.det_auroc, right=np.nan):.4f}'
                  f'｜照網速送 泛化 {np.interp(t, q1.cum_t, q1.closed_acc, right=np.nan):.4f} 面 {np.interp(t, q1.cum_t, q1.det_auroc, right=np.nan):.4f}')
        for key, thr, nm in [('closed_acc', 0.79, '泛化 ≥0.79'), ('closed_acc', 0.77, '泛化 ≥0.77'), ('det_auroc', 0.80, '面 ≥0.80'), ('det_auroc', 0.78, '面 ≥0.78')]:
            out = []
            for lab, (_, _, q) in res.items():
                hit = q[q[key] >= thr]
                out.append(f'{lab} ' + (f'ep{hit.index[0]}／時間 {hit.cum_t.iloc[0]:.0f}' if len(hit) else '未達到'))
            print(f'      首次達到 {nm}：' + '｜'.join(out))
        for c, t_ in enumerate(['(i) tx time per step', '(ii) closed acc vs epoch', '(iii) closed acc vs cum. tx time', "(iii') face AUROC vs cum. tx time"]):
            ax[r, c].set_title(f'{"art-slow" if r == 0 else "sketch-slow"}: {t_}', fontsize=9); ax[r, c].grid(alpha=.3); ax[r, c].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ROOT + fig_path, dpi=120); print(f'\n圖：{fig_path}')


DIAG_RUNS = {'DiagA': f'exp_result_v1_stage2_leave_cartoon_p1a_async_pbw_bwb0_lrsk0.13{_BWS}',
             'DiagC': f'exp_result_v1_stage2_leave_cartoon_p1a_async_pbw_bwb0_fw0.22{_BWS}'}
DIAG_LOGS = {'DiagA': 'logs/proto_trigger/1014_diagA_train.log', 'DiagC': 'logs/proto_trigger/1014_diagC_train.log'}


def diag_section(curve_csv='results/proto_trigger/1014_diag_curve_cartoon_avgbnB.csv',
                 fig_path='results/proto_trigger/1014_diag_curve_cartoon.png', runs=None):
    """[plan 1014] sketch 慢拆解：DiagA（sketch 三台 lr×0.13）／DiagC（融合權重×0.22）vs C-bw（對照）與 SketchSlow（要解釋的掉分）。
    R2 檢驗力、R3 四級判決（門檻同 1012）、R3b 解釋比例、R5 分組面檢測／原圖準確率／份量（(a) 用學習率加權正向組成）、共識離散度。"""
    import os, sys
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    runs = runs or DIAG_RUNS
    ref = {'C-bw': BW_ARMS['Cbw'][1], 'SketchSlow': SKETCHSLOW_RUN}
    allr = {**ref, **runs}
    E = pd.concat([pd.read_csv(ROOT + 'results/proto_trigger/1009_bw_curve_cartoon_avgbnB.csv'),
                   pd.read_csv(ROOT + 'results/proto_trigger/1013_sketchslow_curve_cartoon_avgbnB.csv'),
                   pd.read_csv(ROOT + curve_csv)], ignore_index=True)
    E['arm'] = E.run.map({v.replace('exp_result_', '', 1): k for k, v in allr.items()})
    E = E[E.arm.notna() & (E.score_fn == 'zperp')].copy(); E['ep'] = E.ckpt.str.replace('epoch_', '').astype(int)
    assert (E.groupby(['arm', 'ep']).size() == 9).all(), '有 checkpoint 不是 9 節點'
    Z = E.groupby(['ep', 'arm']).det_auroc.mean().unstack(); A = E.groupby(['ep', 'arm']).closed_acc.mean().unstack()
    arms = [k for k in allr if k in Z.columns]
    win = lambda P, k, lo=25, hi=200: P.loc[lo:hi, k].mean()
    print('=' * 100); print('R3 主指標（zperp、九台平均、BN 平均B）＋四級判決（門檻同 1012：0.8031／0.7955／0.7900／0.7888）'); print('=' * 100)
    for k in arms:
        v = {('z', 'end'): Z.loc[200, k], ('z', 'win'): win(Z, k), ('acc', 'end'): A.loc[200, k], ('acc', 'win'): win(A, k)}
        below = {kk: v[kk] < thr for kk, thr in ARTSLOW_THR.items()}
        hurt = {m_: below[(m_, 'end')] and below[(m_, 'win')] for m_ in ('z', 'acc')}; nb = sum(below.values())
        verdict = ('強訊號（兩指標都傷）' if all(hurt.values()) else ('中等（' + ('面檢測' if hurt['z'] else '泛化') + '傷）') if any(hurt.values())
                   else (f'邊際（{nb} 個數低於門檻）' if nb else '退化（不傷）'))
        print(f'  [{k:10s}] 面 ep200 {v[("z","end")]:.4f}／ep25–200 {v[("z","win")]:.4f}｜泛化 ep200 {v[("acc","end")]:.4f}／ep25–200 {v[("acc","win")]:.4f}｜ep5–50 泛化 {win(A,k,5,50):.4f} 面 {win(Z,k,5,50):.4f} ⇒ {verdict}')
    print('\n' + '=' * 100); print('R3b 解釋比例＝(該臂 − C-bw)／(SketchSlow − C-bw)'); print('=' * 100)
    for k in [a for a in arms if a not in ref]:
        out = []
        for lab, P, lo, hi in [('泛化 ep25–200', A, 25, 200), ('泛化 ep5–50', A, 5, 50), ('泛化 ep155–200', A, 155, 200), ('泛化 ep200', A, 200, 200),
                               ('面 ep25–200', Z, 25, 200), ('面 ep55–200', Z, 55, 200), ('面 ep200', Z, 200, 200)]:
            d = win(P, k, lo, hi) - win(P, 'C-bw', lo, hi); dS = win(P, 'SketchSlow', lo, hi) - win(P, 'C-bw', lo, hi)
            out.append(f'{lab} {d:+.4f}／{dS:+.4f}＝{d / dS:.2f}')
        print(f'  [{k}] ' + '｜'.join(out))
    print('\n逐 checkpoint（泛化／面）：')
    for e in [5, 10, 15, 25, 50, 75, 100, 125, 150, 175, 200]:
        print(f'  ep{e:3d} ' + '  '.join(f'{k} {A.loc[e, k]:.4f}/{Z.loc[e, k]:.4f}' for k in arms))
    print('\n' + '=' * 100); print('R2 檢驗力＋R5 輔助'); print('=' * 100)
    for k in arms:
        c = pd.read_csv(ROOT + allr[k] + '/async_diag/consensus.csv').set_index('epoch').consensus_rms
        print(f'  [{k:10s}] 共識離散度 ' + ' '.join(f'ep{lo}-{hi}:{c.loc[lo:hi].mean():.2e}' for lo, hi in [(1, 25), (26, 50), (51, 100), (101, 150), (151, 190)])
              + f' ep200:{c.loc[200]:.2e}｜ep1–100 平均÷C-bw ' + f'{c.loc[1:100].mean() / pd.read_csv(ROOT + ref["C-bw"] + "/async_diag/consensus.csv").set_index("epoch").consensus_rms.loc[1:100].mean():.2f}')
    for k in arms:
        g = E[E.arm == k].assign(grp=lambda d: d.node.map(lambda n: ['art', 'photo', 'sketch'][int(n[5:]) // 3]))
        t = g[g.ep == 200].groupby('grp').det_auroc.mean(); w = g[g.ep >= 25].groupby('grp').det_auroc.mean()
        print(f'  [{k:10s}] 面檢測分組 ep200 art/photo/sketch ' + '/'.join(f'{t[x]:.4f}' for x in ['art', 'photo', 'sketch'])
              + '｜ep25–200 ' + '/'.join(f'{w[x]:.4f}' for x in ['art', 'photo', 'sketch']))
    src = {'C-bw': 'results/proto_trigger/1012_artslow_source_clean_acc.csv', 'SketchSlow': 'results/proto_trigger/1013_sketchslow_source_clean_acc.csv',
           'DiagA': 'results/proto_trigger/1014_diagA_source_clean_acc.csv', 'DiagC': 'results/proto_trigger/1014_diagC_source_clean_acc.csv'}
    keymap = {'C-bw': 'Cbw'}
    for k in arms:
        f = ROOT + src.get(k, '')
        if not os.path.exists(f):
            print(f'  [{k:10s}] 原圖準確率：檔案不存在'); continue
        d = pd.read_csv(f); d = d[(d.arm == keymap.get(k, k)) & (d.bn == 'avgB')]
        t = d.groupby('eval_domain').closed_acc.mean() * 100
        print(f'  [{k:10s}] 原圖準確率（BN 平均B、九台平均 %）art {t["art_painting"]:.2f} photo {t["photo"]:.2f} sketch {t["sketch"]:.2f} cartoon {t["cartoon"]:.2f}')
    # (a) 的份量：學習率加權正向組成（照 train.py 順序），用實際送出紀錄
    sys.path.insert(0, ROOT + 'scripts/posthoc')
    import networkx as nx
    G = nx.random_geometric_graph(9, 0.8, seed=1234); NB = {i: sorted(G[i]) for i in range(9)}; deg = {i: len(NB[i]) for i in range(9)}
    for k, lrm, kap in [('C-bw', [1] * 9, 1.0), ('DiagA', [1] * 6 + [0.13] * 3, 1.0), ('DiagC', [1] * 9, 0.22)]:
        if k not in arms: continue
        b = pd.read_csv(ROOT + allr[k] + '/async_diag/broadcast_log.csv'); b['i'] = b.domain.str[5:].astype(int)
        F = b.pivot(index='step', columns='i', values='fired').sort_index().values.astype(bool); T_ = F.shape[0]
        Am = np.zeros((9, 9)); msg = {}; eta = 0.5 * (1 + np.cos(np.pi * np.arange(T_) / T_))
        for kk in range(T_):
            for i in range(9):
                Am[i, i] += eta[kk] * lrm[i]
                if F[kk, i]: msg[i] = Am[i].copy()
                sn = [j for j in NB[i] if (F[kk, j] if j < i else (kk > 0 and F[kk - 1, j]))]
                if sn:
                    w = {j: kap / (1 + max(deg[i], deg[j])) for j in sn}
                    Am[i] = (1 - sum(w.values())) * Am[i] + sum(w[j] * msg[j] for j in sn)
        row = Am.mean(0); row /= row.sum()
        print(f'  [{k:10s}] 份量（學習率加權正向組成，實際送出紀錄）art/photo/sketch {row[:3].sum():.3f}/{row[3:6].sum():.3f}/{row[6:].sum():.3f}')
    print('\n' + '=' * 100); print('R4 健康與全軌跡'); print('=' * 100)
    logs = {'C-bw': BW_ARMS['Cbw'][0], 'SketchSlow': 'logs/proto_trigger/1013_sketchslow_train.log', **DIAG_LOGS}
    for k in arms:
        if not os.path.exists(ROOT + logs[k]): continue
        df = parse(logs[k]).set_index('ep'); print(f'  [{k}] epoch 行 {len(df)}｜NaN/Traceback {nan_count(logs[k])}')
        for col in ['loss', 'tr', 'te', 'g_cls', 'compL', 'compG', 'dispL', 'dispG']:
            s_ = df[col].dropna(); post = s_[s_.index > 10]; tail = s_[s_.index > 180]
            pts = ' '.join(f'{e}:{s_.get(e, np.nan):.4g}' for e in (1, 11, 50, 100, 150, 200))
            sl = np.polyfit(tail.index, tail.values, 1)[0] if len(tail) > 2 else np.nan
            print(f'      {col:6s} {pts} | ep11+ min {post.min():.4g} max {post.max():.4g} mean {post.mean():.4g} std {post.std():.4g} | ep101-200 {s_[s_.index > 100].mean():.4g} | 末20斜率 {sl:+.2e}')
        n_ = allr[k].replace('exp_result_', '', 1)
        T = np.array([np.loadtxt(f'{ROOT}{allr[k]}{n_}_res/dsgd-lr0.001-budget1.0-r{i}-tacc.log') for i in range(9)])
        w = T[:, 100:200].mean(1)
        print(f'      訓練準確率 ep101–200 art/photo/sketch {w[:3].mean():.2f}/{w[3:6].mean():.2f}/{w[6:].mean():.2f}｜sketch 三台分段 '
              + ' '.join(f'ep{a + 1}-{b_}:{T[6:, a:b_].mean():.2f}' for a, b_ in [(0, 10), (10, 50), (50, 100), (100, 200)]))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    col = {'C-bw': 'tab:blue', 'SketchSlow': 'tab:red', 'DiagA': 'tab:green', 'DiagC': 'tab:purple'}
    for k in arms:
        ax[0].plot(A.index, 100 * A[k], c=col.get(k, 'k'), label=k); ax[1].plot(Z.index, Z[k], c=col.get(k, 'k'), label=k)
    ax[0].set_title('formal closed-set acc (%) vs epoch'); ax[1].set_title('formal face AUROC (zperp) vs epoch')
    for x in ax:
        x.set_xlabel('epoch'); x.grid(alpha=.3); x.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT + fig_path, dpi=130); print(f'\n圖：{fig_path}')


SHARE_RUNS = {'A1': f'exp_result_v1_stage2_leave_cartoon_p1a_async_pbwsk_bwfsketch0.1_frp{_BWS}',   # [plan 1016]
              'A2': f'exp_result_v1_stage2_leave_cartoon_p1a_async_pbw_bwb0_frp{_BWS}',
              'A3': f'exp_result_v1_stage2_leave_cartoon_p1a_async_pbwsk_bwfsketch0.1_fps{_BWS}'}
SHARE_REF = {'A1': 'SketchSlow', 'A2': 'C-bw', 'A3': 'SketchSlow'}


def share_section(curve_csv='results/proto_trigger/1016_share_rules_curve_cartoon_avgbnB.csv',
                  fig_path='results/proto_trigger/1016_share_rules_curve_cartoon.png'):
    """[plan 1016] 份量平衡規則：A1 底座×sketch 慢／A2 底座×正常網速／A3 推和法×sketch 慢 vs C-bw（正常）與 SketchSlow（要修的掉分）。
    R1 送出逐位相同、R2 融合跳動與共識離散度、R3 門檻判決＋事前三級判準、R3b 回收比例、R4 健康全軌跡與推和法份量軌跡、R5 原圖準確率與分組面檢測。
    只納入評估 checkpoint 已齊 40 個的臂（鏈還在跑時可先看已完成的）。"""
    import os, re as _re
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ref = {'C-bw': BW_ARMS['Cbw'][1], 'SketchSlow': SKETCHSLOW_RUN}
    logs = {'C-bw': BW_ARMS['Cbw'][0], 'SketchSlow': 'logs/proto_trigger/1013_sketchslow_train.log',
            **{k: f'logs/proto_trigger/1016_{k}_train.log' for k in SHARE_RUNS}}
    allr = {**ref, **SHARE_RUNS}
    E = pd.concat([pd.read_csv(ROOT + 'results/proto_trigger/1009_bw_curve_cartoon_avgbnB.csv'),
                   pd.read_csv(ROOT + 'results/proto_trigger/1013_sketchslow_curve_cartoon_avgbnB.csv')]
                  + ([pd.read_csv(ROOT + curve_csv)] if os.path.exists(ROOT + curve_csv) else []), ignore_index=True)
    E['arm'] = E.run.map({v.replace('exp_result_', '', 1): k for k, v in allr.items()})
    E = E[E.arm.notna() & (E.score_fn == 'zperp')].copy(); E['ep'] = E.ckpt.str.replace('epoch_', '').astype(int)
    E = E.drop_duplicates(['arm', 'ep', 'node'], keep='last')
    n_ck = E.groupby('arm').ep.nunique()
    assert (E.groupby(['arm', 'ep']).size() == 9).all(), '有 checkpoint 不是 9 節點'
    arms = [k for k in allr if n_ck.get(k, 0) == 40]
    test = [k for k in SHARE_RUNS if k in arms]
    print(f'評估 checkpoint 數：' + ' '.join(f'{k}={n_ck.get(k, 0)}' for k in allr) + f'｜納入 {arms}')
    Z = E.groupby(['ep', 'arm']).det_auroc.mean().unstack(); A = E.groupby(['ep', 'arm']).closed_acc.mean().unstack()
    win = lambda P, k, lo=25, hi=200: P.loc[lo:hi, k].mean()

    print('=' * 100); print('R1 送出序列：全程 fired 與對照組逐格比對（傳輸時間逐位相同的前提）'); print('=' * 100)
    B = {}
    for k in arms:
        b = pd.read_csv(ROOT + allr[k] + '/async_diag/broadcast_log.csv'); b['i'] = b.domain.str[5:].astype(int); B[k] = b
    for k in test:
        r = SHARE_REF[k]
        Fa = B[k].pivot(index='step', columns='i', values='fired').sort_index(); Fr = B[r].pivot(index='step', columns='i', values='fired').sort_index()
        n = min(len(Fa), len(Fr))
        print(f'  [{k}] vs {r}：{n} 步、不同格數 {int((Fa.values[:n] != Fr.values[:n]).sum())}｜總份數 {int(B[k].n_pushed.sum())} vs {int(B[r].n_pushed.sum())}'
              f'｜forced {int(B[k].forced.sum())}｜n_received 不同格數 '
              f'{int((B[k].pivot(index="step", columns="i", values="n_received").values[:n] != B[r].pivot(index="step", columns="i", values="n_received").values[:n]).sum())}')

    print('\n' + '=' * 100); print('R3 主指標（zperp、九台平均、BN 平均B）＋門檻（0.8031／0.7955／0.7900／0.7888；「傷」＝ep200 與 ep25–200 都低於）'); print('=' * 100)
    HURT = {}
    for k in arms:
        v = {('z', 'end'): Z.loc[200, k], ('z', 'win'): win(Z, k), ('acc', 'end'): A.loc[200, k], ('acc', 'win'): win(A, k)}
        below = {kk: v[kk] < thr for kk, thr in ARTSLOW_THR.items()}
        HURT[k] = {m_: below[(m_, 'end')] and below[(m_, 'win')] for m_ in ('z', 'acc')}
        print(f'  [{k:10s}] 面 ep200 {v[("z","end")]:.4f}／ep25–200 {v[("z","win")]:.4f}｜泛化 ep200 {v[("acc","end")]:.4f}／ep25–200 {v[("acc","win")]:.4f}'
              f'｜ep5–50 泛化 {win(A, k, 5, 50):.4f} 面 {win(Z, k, 5, 50):.4f}｜低於門檻 {[f"{a_}{b_}" for (a_, b_), x in below.items() if x]}'
              f'｜傷：面 {HURT[k]["z"]} 泛化 {HURT[k]["acc"]}')

    print('\n' + '=' * 100); print('R3b 回收比例＝(該臂 − SketchSlow)／(C-bw − SketchSlow)；A2 改報 A2 − C-bw'); print('=' * 100)
    REC = {}
    segs = [('ep25–200', 25, 200), ('ep200', 200, 200), ('ep5–50', 5, 50), ('ep55–100', 55, 100), ('ep105–150', 105, 150), ('ep155–200', 155, 200)]
    for k in test:
        out = []
        for lab, P in [('泛化', A), ('面', Z)]:
            for sl, lo, hi in segs:
                if k == 'A2':
                    out.append(f'{lab} {sl} {win(P, k, lo, hi) - win(P, "C-bw", lo, hi):+.4f}')
                else:
                    d = win(P, k, lo, hi) - win(P, 'SketchSlow', lo, hi); g = win(P, 'C-bw', lo, hi) - win(P, 'SketchSlow', lo, hi)
                    REC[(k, lab, sl)] = d / g
                    out.append(f'{lab} {sl} {d:+.4f}／{g:+.4f}＝{d / g:.2f}')
        print(f'  [{k}] ' + '｜'.join(out))

    print('\n' + '=' * 100); print('事前判準（plan §6）'); print('=' * 100)
    for k in test:
        if k == 'A2':
            print(f'  [A2] ' + ('不傷' if not any(HURT[k].values()) else '弄壞（' + '、'.join(n_ for m_, n_ in [('z', '面檢測'), ('acc', '泛化')] if HURT[k][m_]) + ' 傷）'))
            continue
        rz, ra = REC[(k, '面', 'ep25–200')], REC[(k, '泛化', 'ep25–200')]
        vd = ('修好' if not any(HURT[k].values()) else '部分修好' if min(rz, ra) >= 0.5 else '沒修好')
        print(f'  [{k}] 傷：面 {HURT[k]["z"]} 泛化 {HURT[k]["acc"]}｜ep25–200 回收 面 {rz:.2f} 泛化 {ra:.2f} ⇒ {vd}')
    if 'A1' in test and 'A3' in test:
        m1 = (REC[('A1', '面', 'ep25–200')] + REC[('A1', '泛化', 'ep25–200')]) / 2
        m3 = (REC[('A3', '面', 'ep25–200')] + REC[('A3', '泛化', 'ep25–200')]) / 2
        print(f'  底座 vs 推和法：ep25–200 兩指標平均回收 A1 {m1:.2f}／A3 {m3:.2f}，差 {m1 - m3:+.2f} ⇒ '
              + ('平手（<0.1）→ 改比 R2 穩定度' if abs(m1 - m3) < 0.1 else ('底座較好' if m1 > m3 else '推和法較好')))

    print('\n' + '=' * 100); print('R2 穩定度：融合跳動（fusion_jump.csv）與共識離散度（consensus.csv）'); print('=' * 100)
    q_ = lambda s: f'中位 {s.median():.3g}／p99 {s.quantile(.99):.3g}／最大 {s.max():.3g}'
    for k in test:
        f = ROOT + allr[k] + '/async_diag/fusion_jump.csv'
        if not os.path.exists(f):
            print(f'  [{k}] fusion_jump.csv 不存在'); continue
        j = pd.read_csv(f)
        print(f'  [{k}] 融合 {len(j)} 次（epoch {j.epoch.min()}–{j.epoch.max()}）｜收進來的總比重 w_in {q_(j.w_in)}｜單一送方最大比重 w_max {q_(j.w_max)}')
        print(f'        跳動 L2 {q_(j.jump_l2)}｜相對跳動 {q_(j.jump_rel)}')
        for lo, hi in [(1, 25), (26, 50), (51, 100), (101, 150), (151, 200)]:
            s = j[(j.epoch >= lo) & (j.epoch <= hi)]
            print(f'        ep{lo}–{hi}: w_in {q_(s.w_in)}｜相對跳動 {q_(s.jump_rel)}')
        top = j.nlargest(5, 'jump_rel')[['step', 'epoch', 'domain', 'n_senders', 'w_in', 'jump_rel']]
        print('        相對跳動最大五次：' + '；'.join(f'step{r.step}/ep{r.epoch}/{r.domain}/{r.n_senders}份/w_in{r.w_in:.3f}/{r.jump_rel:.2e}' for r in top.itertuples()))
    cref = pd.read_csv(ROOT + ref['C-bw'] + '/async_diag/consensus.csv').set_index('epoch').consensus_rms
    for k in arms:
        c = pd.read_csv(ROOT + allr[k] + '/async_diag/consensus.csv').set_index('epoch').consensus_rms
        print(f'  [{k:10s}] 共識離散度 ' + ' '.join(f'ep{lo}-{hi}:{c.loc[lo:hi].mean():.2e}' for lo, hi in [(1, 25), (26, 50), (51, 100), (101, 150), (151, 190)])
              + f' ep200:{c.loc[200]:.2e}｜ep1–100 平均÷C-bw {c.loc[1:100].mean() / cref.loc[1:100].mean():.2f}')

    print('\n' + '=' * 100); print('R4 健康與全軌跡'); print('=' * 100)
    for k in arms:
        if not os.path.exists(ROOT + logs[k]): continue
        df = parse(logs[k]).set_index('ep'); print(f'  [{k}] epoch 行 {len(df)}｜NaN/Traceback {nan_count(logs[k])}')
        for col in ['loss', 'tr', 'te', 'g_cls', 'compL', 'compG', 'dispL', 'dispG']:
            if col not in df: continue
            s_ = df[col].dropna(); post = s_[s_.index > 10]; tail = s_[s_.index > 180]
            pts = ' '.join(f'{e}:{s_.get(e, np.nan):.4g}' for e in (1, 11, 50, 100, 150, 200))
            sl = np.polyfit(tail.index, tail.values, 1)[0] if len(tail) > 2 else np.nan
            print(f'      {col:6s} {pts} | ep11+ min {post.min():.4g} max {post.max():.4g} mean {post.mean():.4g} std {post.std():.4g} | ep101-200 {s_[s_.index > 100].mean():.4g} | 末20斜率 {sl:+.2e}')
        n_ = allr[k].replace('exp_result_', '', 1)
        T = np.array([np.loadtxt(f'{ROOT}{allr[k]}{n_}_res/dsgd-lr0.001-budget1.0-r{i}-tacc.log') for i in range(9)])
        w = T[:, 100:200].mean(1)
        print(f'      訓練準確率 ep101–200 art/photo/sketch {w[:3].mean():.2f}/{w[3:6].mean():.2f}/{w[6:].mean():.2f}｜sketch 三台分段 '
              + ' '.join(f'ep{a + 1}-{b_}:{T[6:, a:b_].mean():.2f}' for a, b_ in [(0, 10), (10, 50), (50, 100), (100, 200)]))
    if 'A3' in test:
        L = [l for l in open(ROOT + logs['A3'], encoding='utf-8', errors='ignore') if l.startswith('[PUSHSUM] epoch=')]
        W = np.array([[float(x) for x in _re.search(r' w=([0-9.,]+) ', l).group(1).split(',')] for l in L])
        tot = np.array([float(_re.search(r'total_incl_inflight=([0-9.]+)', l).group(1)) for l in L])
        mg = max(int(_re.search(r'merges=(\d+)', l).group(1)) for l in L)
        print(f'  [A3] 推和法份量：{len(L)} 行｜總和（含在途）最小 {tot.min():.9f} 最大 {tot.max():.9f}｜合併次數 {mg}')
        print(f'        全程各台 w 最小 ' + ' '.join(f'n{i}:{W[:, i].min():.3f}' for i in range(9)) + '（步長倍數最大＝1/最小 w）')
        print(f'        全程各台 w 最大 ' + ' '.join(f'n{i}:{W[:, i].max():.3f}' for i in range(9)))
        for e in (1, 10, 25, 50, 100, 150, 200):
            if e <= len(W):
                print(f'        ep{e:3d} w ' + ' '.join(f'{x:.3f}' for x in W[e - 1]) + f'｜art/photo/sketch 加總 {W[e - 1, :3].sum():.2f}/{W[e - 1, 3:6].sum():.2f}/{W[e - 1, 6:].sum():.2f}')
        print(f'        逐 epoch 末的 w（只是每 epoch 一個取樣點，不是全步最小）：sketch 三台平均 {W[:, 6:].mean():.3f}、其他六台平均 {W[:, :6].mean():.3f}')

    print('\n' + '=' * 100); print('R5 輔助：原圖準確率、分組面檢測'); print('=' * 100)
    src = {'C-bw': ('results/proto_trigger/1012_artslow_source_clean_acc.csv', 'Cbw'),
           'SketchSlow': ('results/proto_trigger/1013_sketchslow_source_clean_acc.csv', 'SketchSlow'),
           **{k: (f'results/proto_trigger/1016_{k}_source_clean_acc.csv', f'Share{k}') for k in SHARE_RUNS}}
    for k in arms:
        f, key = src[k]
        if not os.path.exists(ROOT + f):
            print(f'  [{k:10s}] 原圖準確率：檔案不存在'); continue
        d = pd.read_csv(ROOT + f); d = d[(d.arm == key) & (d.bn == 'avgB')]
        t = d.groupby('eval_domain').closed_acc.mean() * 100
        print(f'  [{k:10s}] 原圖準確率（BN 平均B、九台平均 %）art {t["art_painting"]:.2f} photo {t["photo"]:.2f} sketch {t["sketch"]:.2f} cartoon {t["cartoon"]:.2f}')
    for k in arms:
        g = E[E.arm == k].assign(grp=lambda d: d.node.map(lambda n: ['art', 'photo', 'sketch'][int(n[5:]) // 3]))
        t = g[g.ep == 200].groupby('grp').det_auroc.mean(); w = g[g.ep >= 25].groupby('grp').det_auroc.mean()
        print(f'  [{k:10s}] 面檢測分組 ep200 art/photo/sketch ' + '/'.join(f'{t[x]:.4f}' for x in ['art', 'photo', 'sketch'])
              + '｜ep25–200 ' + '/'.join(f'{w[x]:.4f}' for x in ['art', 'photo', 'sketch']))

    print('\n逐 checkpoint（泛化／面）：')
    for e in [5, 10, 15, 25, 50, 75, 100, 125, 150, 175, 200]:
        print(f'  ep{e:3d} ' + '  '.join(f'{k} {A.loc[e, k]:.4f}/{Z.loc[e, k]:.4f}' for k in arms))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    col = {'C-bw': 'tab:blue', 'SketchSlow': 'tab:red', 'A1': 'tab:green', 'A2': 'tab:cyan', 'A3': 'tab:purple'}
    for k in arms:
        ax[0].plot(A.index, 100 * A[k], c=col[k], label=k); ax[1].plot(Z.index, Z[k], c=col[k], label=k)
    ax[0].axhline(100 * ARTSLOW_THR[('acc', 'end')], ls=':', c='k', lw=.8); ax[1].axhline(ARTSLOW_THR[('z', 'end')], ls=':', c='k', lw=.8)
    ax[0].set_title('formal closed-set acc (%) vs epoch'); ax[1].set_title('formal face AUROC (zperp) vs epoch')
    for x in ax:
        x.set_xlabel('epoch'); x.grid(alpha=.3); x.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT + fig_path, dpi=130); print(f'\n圖：{fig_path}')


def share_efhc_figure(fig_path='results/proto_trigger/1017_share_rules_efhc_style.png'):
    """[1017 dany] 貢獻二的 EF-HC 圖 2 式呈現：(i) 每步傳輸時間 (ii) 準確率 vs epoch (iii) 準確率 vs 累積傳輸時間。
    記帳式：送一次＝1/b_i（b＝該情境的網速表），每步＝(1/9)Σ fired_i/b_i。均勻送 C0 的訓練與網速無關 ⇒ 送出紀錄用該情境的表重新計價。
    情境一 sketch 慢（表＝1013 bandwidth_sketch0.1.csv）：C0、不補償（1013）、底座 A1、推和法 A3。
    情境二 正常網速（表＝1009 C-bw 的 bandwidth.csv）：C0、C-bw、D-bw、C-match、底座 A2。"""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ev = pd.concat([pd.read_csv(ROOT + f) for f in ['results/proto_trigger/1009_bw_curve_cartoon_avgbnB.csv',
                    'results/proto_trigger/1013_sketchslow_curve_cartoon_avgbnB.csv', 'results/proto_trigger/1016_share_rules_curve_cartoon_avgbnB.csv']],
                   ignore_index=True)
    ev = ev[ev.score_fn == 'zperp'].copy(); ev['ep'] = ev.ckpt.str.replace('epoch_', '').astype(int)
    ev = ev.drop_duplicates(['run', 'ep', 'node'], keep='last')
    SC = {'sketch-slow': ('results/proto_trigger/1013_sketchslow/bandwidth_sketch0.1.csv',
                          {'uniform C0': BW_ARMS['C0'][1], 'no-comp (1013)': SKETCHSLOW_RUN, 'base A1': SHARE_RUNS['A1'], 'push-sum A3': SHARE_RUNS['A3']}),
          'normal': (BW_ARMS['Cbw'][1] + '/async_diag/bandwidth.csv',
                     {'uniform C0': BW_ARMS['C0'][1], 'EF-HC rule D-bw': BW_ARMS['Dbw'][1], 'no-comp C-bw': BW_ARMS['Cbw'][1],
                      'C-match': BW_ARMS['Cmatch'][1], 'base A2': SHARE_RUNS['A2']})}
    fig, ax = plt.subplots(2, 4, figsize=(21, 8.8))
    for r, (scen, (tab, arms)) in enumerate(SC.items()):
        bt = pd.read_csv(ROOT + tab); bmap = dict(zip(bt.node, bt.b))
        print('=' * 100); print(f'情境：{scen}（網速表 {tab}）'); print('=' * 100)
        Q = {}
        for lab, run in arms.items():
            b = pd.read_csv(ROOT + run + '/async_diag/broadcast_log.csv'); b['t'] = b.fired / b.domain.map(bmap)
            per_step = b.groupby('step').t.sum() / 9.0; ep_t = b.groupby('epoch').t.sum() / 9.0; cum = ep_t.cumsum()
            q = ev[ev.run == run.replace('exp_result_', '', 1)].groupby('ep')[['closed_acc', 'det_auroc']].mean()
            assert len(q) == 40, (lab, len(q))
            q['cum_t'] = cum.reindex(q.index).values; Q[lab] = q
            print(f'  [{lab:16s}] (i) 每步傳輸時間 {per_step.mean():.3f}｜累積總計 {cum.iloc[-1]:.0f}｜逐節點送出 {b.groupby("domain").fired.sum().astype(int).tolist()}')
            print(f'      (ii) 泛化 ep10 {q.closed_acc.loc[10]:.4f} ep25 {q.closed_acc.loc[25]:.4f} ep50 {q.closed_acc.loc[50]:.4f} ep200 {q.closed_acc.loc[200]:.4f}'
                  f'｜面 ep10 {q.det_auroc.loc[10]:.4f} ep25 {q.det_auroc.loc[25]:.4f} ep50 {q.det_auroc.loc[50]:.4f} ep200 {q.det_auroc.loc[200]:.4f}')
            pk = dict(lw=2.2 if lab.startswith('base') else 1.1)
            ax[r, 0].plot(ep_t.index, ep_t.values / 20, label=lab, **pk)
            ax[r, 1].plot(q.index, 100 * q.closed_acc, label=lab, **pk)
            ax[r, 2].plot(q.cum_t, 100 * q.closed_acc, label=lab, **pk)
            ax[r, 3].plot(q.cum_t, q.det_auroc, label=lab, **pk)
        ref = [k for k in Q if k.startswith('base')][0]; Tb = Q[ref].cum_t.iloc[-1]
        print(f'  (iii) 同樣累積傳輸時間下（內插；底座全程 {Tb:.0f}）：')
        for g in [0.1, 0.25, 0.5, 1.0]:
            t = g * Tb
            print(f'      時間 {t:6.0f}：' + '｜'.join(f'{k} 泛化 {np.interp(t, q.cum_t, q.closed_acc, left=np.nan, right=np.nan):.4f} 面 {np.interp(t, q.cum_t, q.det_auroc, left=np.nan, right=np.nan):.4f}' for k, q in Q.items()))
        for key, thr, nm in [('closed_acc', 0.75, '泛化≥0.75'), ('closed_acc', 0.78, '泛化≥0.78'), ('closed_acc', 0.79, '泛化≥0.79'), ('closed_acc', 0.80, '泛化≥0.80'),
                             ('det_auroc', 0.79, '面≥0.79'), ('det_auroc', 0.80, '面≥0.80'), ('det_auroc', 0.81, '面≥0.81'), ('det_auroc', 0.82, '面≥0.82')]:
            out = []
            for k, q in Q.items():
                hit = q[q[key] >= thr]
                out.append(f'{k} ' + (f'ep{hit.index[0]}／時間 {hit.cum_t.iloc[0]:.0f}' if len(hit) else '未達'))
            print(f'      首次達到 {nm}：' + '｜'.join(out))
        for c, t_ in enumerate(['(i) tx time per step', '(ii) closed acc vs epoch', '(iii) closed acc vs cum. tx time', "(iii') face AUROC vs cum. tx time"]):
            ax[r, c].set_title(f'{scen}: {t_}', fontsize=9); ax[r, c].grid(alpha=.3); ax[r, c].legend(fontsize=7)
        if scen == 'sketch-slow':
            for c in (1, 2): ax[r, c].set_ylim(55, 82)
            ax[r, 3].set_ylim(0.5, 0.84)
    fig.tight_layout(); fig.savefig(ROOT + fig_path, dpi=120); print(f'\n圖：{fig_path}')


COMP_RUNS = {  # [plan 1017] sketch 慢的主表：標籤 → (run 目錄, 估算規則, 雙向, 要多存 46 MB, 訓練 log)
    'no-comp': (SKETCHSLOW_RUN, 'mh', False, False, 'logs/proto_trigger/1013_sketchslow_train.log'),
    'base A1': (SHARE_RUNS['A1'], 'rate_pair', False, False, 'logs/proto_trigger/1016_A1_train.log'),
    'uniform-full C0': (BW_ARMS['C0'][1], 'mh', False, False, BW_ARMS['C0'][0]),
    'uniform-matched': (f'exp_result_v1_stage2_leave_cartoon_p1a_async_puni_bwfsketch0.1{_BWS}', 'mh', False, False, 'logs/proto_trigger/1017_U_train.log'),
    'D-bw': (f'exp_result_v1_stage2_leave_cartoon_p1a_async_lr_tau8.7e-06_bwfskr{_BWS}', 'mh', False, True, 'logs/proto_trigger/1017_D_train.log'),
    'PushCen': (f'exp_result_v1_stage2_leave_cartoon_p1a_async_pbwsk_bwfsketch0.1_fpc{_BWS}', 'pushcen', False, False, 'logs/proto_trigger/1017_P_train.log'),
    'EF-HC bidir': (f'exp_result_v1_stage2_leave_cartoon_p1a_async_lr_tau8.7e-06_bwfskr_bd{_BWS}', 'mh', True, True, 'logs/proto_trigger/1017_E_train.log'),
    'SGP A3': (SHARE_RUNS['A3'], 'pushsum', False, False, 'logs/proto_trigger/1016_A3_train.log'),
}
COMP_SRC = {'no-comp': ('results/proto_trigger/1013_sketchslow_source_clean_acc.csv', 'SketchSlow'),
            'uniform-full C0': ('results/proto_trigger/1012_artslow_source_clean_acc.csv', 'C0'),
            'base A1': ('results/proto_trigger/1016_A1_source_clean_acc.csv', 'ShareA1'),
            'uniform-matched': ('results/proto_trigger/1017_U_source_clean_acc.csv', 'CompU'),
            'D-bw': ('results/proto_trigger/1017_D_source_clean_acc.csv', 'CompD'),
            'PushCen': ('results/proto_trigger/1017_P_source_clean_acc.csv', 'CompP'),
            'EF-HC bidir': ('results/proto_trigger/1017_E_source_clean_acc.csv', 'CompE'),
            'SGP A3': ('results/proto_trigger/1016_A3_source_clean_acc.csv', 'ShareA3')}


def comp_section(fig_path='results/proto_trigger/1017_competitors_efhc_style.png'):
    """[plan 1017] sketch 慢主表：每個對手 vs 底座 A1。傳輸記帳＝EF-HC 原文（每步 (1/9)Σ_i(Σ_j v_ij/d_i)/b_i；
    單向 v＝fired_i；雙向 v_ij＝max(fired_i, fired_j)）。主比較＝同累積傳輸時間 T*（＝底座全程）下的泛化／面檢測；
    四級判決照計畫 §6.2（點值雜訊寬度：泛化 0.0109、面 0.0144；成本較高＝總傳輸 ≥1.3×、或慢設備每台傳輸 ≥1.3×、或要多存 46 MB）。
    只納入 40 個 checkpoint 都評估完的組。"""
    import os, sys
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import networkx as nx
    sys.path.insert(0, ROOT + 'scripts/posthoc')
    from composition_inbox_sim import simulate
    ev = pd.concat([pd.read_csv(ROOT + f) for f in ['results/proto_trigger/1009_bw_curve_cartoon_avgbnB.csv',
                    'results/proto_trigger/1013_sketchslow_curve_cartoon_avgbnB.csv', 'results/proto_trigger/1016_share_rules_curve_cartoon_avgbnB.csv']
                    + (['results/proto_trigger/1017_competitors_curve_cartoon_avgbnB.csv']
                       if os.path.exists(ROOT + 'results/proto_trigger/1017_competitors_curve_cartoon_avgbnB.csv') else [])], ignore_index=True)
    ev = ev[ev.score_fn == 'zperp'].copy(); ev['ep'] = ev.ckpt.str.replace('epoch_', '').astype(int)
    ev = ev.drop_duplicates(['run', 'ep', 'node'], keep='last')
    bt = pd.read_csv(ROOT + 'results/proto_trigger/1013_sketchslow/bandwidth_sketch0.1.csv'); bt['i'] = bt.node.str[5:].astype(int)
    b_ = bt.sort_values('i').b.values
    G = nx.random_geometric_graph(9, 0.8, seed=1234); NBR = {i: sorted(G[i]) for i in range(9)}; DEG = {i: len(NBR[i]) for i in range(9)}
    Q, COST, SH = {}, {}, {}
    for lab, (run, rule, bd, store, lg) in COMP_RUNS.items():
        q = ev[ev.run == run.replace('exp_result_', '', 1)].groupby('ep')[['closed_acc', 'det_auroc']].mean()
        if len(q) != 40:
            print(f'  [{lab}] 評估 checkpoint {len(q)}／40，略過'); continue
        b = pd.read_csv(ROOT + run + '/async_diag/broadcast_log.csv'); b['i'] = b.domain.str[5:].astype(int)
        Fm = b.pivot(index='step', columns='i', values='fired').sort_index().values.astype(bool)
        if bd:
            V = np.zeros_like(Fm, dtype=float)
            for i in range(9):
                V[:, i] = sum(np.maximum(Fm[:, i], Fm[:, j]) for j in NBR[i]) / DEG[i]
        else:
            V = Fm.astype(float)
        cost = V / b_[None, :]                                # [T, 9] 每台每步傳輸時間
        ep_t = pd.Series(cost.sum(1) / 9).groupby(np.arange(len(cost)) // 20).sum(); ep_t.index += 1
        q['cum_t'] = ep_t.cumsum().reindex(q.index).values
        Q[lab] = q
        COST[lab] = dict(total=ep_t.sum(), per_node=cost.sum(0), slow=cost.sum(0)[b_ <= 0.1 + 1e-9].mean(), store=store,
                         sends=Fm.sum(0), ep_t=ep_t)
        SH[lab] = simulate(Fm, rule, bd)
    ref = 'base A1'; Tb = COST[ref]['total']; slow_ref = COST[ref]['slow']
    print('=' * 110); print(f'§1 成本與份量（sketch 慢網速表；傳輸照 EF-HC 記帳；T*＝底座全程 {Tb:.1f}）'); print('=' * 110)
    for lab in Q:
        c = COST[lab]; sh = SH[lab]['share']
        print(f'  [{lab:16s}] 總傳輸 {c["total"]:7.1f}（{c["total"] / Tb:.2f}×）｜慢設備每台 {c["slow"]:6.0f}（{c["slow"] / slow_ref:.2f}×）'
              f'｜多存 46 MB {"是" if c["store"] else "否"}｜逐台送出 {c["sends"].tolist()}'
              f'｜份量 art/photo/sketch ' + '/'.join(f'{x:.3f}' for x in sh) + f'｜單台×9 最小 {SH[lab]["node"].min():.2f}')
    print('\n' + '=' * 110); print('§2 主比較：同累積傳輸時間下（內插；對手總時間不到就用最終值）＋ ep200／ep25–200 ＋ 達標時間'); print('=' * 110)
    at = lambda q, t, col: float(np.interp(t, q.cum_t, q[col])) if t <= q.cum_t.iloc[-1] else float(q[col].iloc[-1])
    for lab, q in Q.items():
        hit = lambda col, thr: (lambda h: f'{h.cum_t.iloc[0]:.0f}（ep{h.index[0]}）' if len(h) else '未達')(q[q[col] >= thr])
        print(f'  [{lab:16s}] T*：泛化 {at(q, Tb, "closed_acc"):.4f} 面 {at(q, Tb, "det_auroc"):.4f}｜T 400：{at(q, 400, "closed_acc"):.4f}／{at(q, 400, "det_auroc"):.4f}'
              f'｜T 800：{at(q, 800, "closed_acc"):.4f}／{at(q, 800, "det_auroc"):.4f}'
              f'｜ep200 {q.closed_acc.loc[200]:.4f}／{q.det_auroc.loc[200]:.4f}｜ep25–200 {q.closed_acc.loc[25:].mean():.4f}／{q.det_auroc.loc[25:].mean():.4f}'
              f'｜ep5–50 泛化 {q.closed_acc.loc[5:50].mean():.4f}｜達到泛化 0.78 {hit("closed_acc", 0.78)}、面 0.80 {hit("det_auroc", 0.80)}')
    print('\n' + '=' * 110); print('§3 四級判決（計畫 §6.2；差＝底座 − 對手，在 T*）'); print('=' * 110)
    wg, wf = 0.0109, 0.0144
    qa = Q[ref]
    for lab, q in Q.items():
        if lab == ref: continue
        da = at(qa, Tb, 'closed_acc') - at(q, Tb, 'closed_acc'); dz = at(qa, Tb, 'det_auroc') - at(q, Tb, 'det_auroc')
        c = COST[lab]; higher = (c['total'] >= 1.3 * Tb) or (c['slow'] >= 1.3 * slow_ref) or c['store']
        better = (da > wg) or (dz > wf); worse = (da < -wg) or (dz < -wf)
        if better and not worse: v = '底座勝'
        elif not better and not worse: v = '持平、底座成本較低' if higher else '持平'
        elif worse and not better: v = '底座輸' if not higher else '對手成績較好、但成本較高（取捨）'
        else: v = '取捨（一高一低）'
        why = [n for n, f in [('總傳輸 ≥1.3×', c['total'] >= 1.3 * Tb), ('慢設備 ≥1.3×', c['slow'] >= 1.3 * slow_ref), ('多存 46 MB', c['store'])] if f]
        print(f'  [{lab:16s}] 泛化差 {da:+.4f}（寬 {wg}）面差 {dz:+.4f}（寬 {wf}）｜對手成本較高：{"、".join(why) if why else "否"} ⇒ {v}')
    print('\n' + '=' * 110); print('§4 健康與全軌跡（1017 新組）'); print('=' * 110)
    for lab in [l for l in ('uniform-matched', 'D-bw', 'PushCen', 'EF-HC bidir') if l in Q]:
        lg = COMP_RUNS[lab][4]; df = parse(lg).set_index('ep'); print(f'  [{lab}] epoch 行 {len(df)}｜NaN/Traceback {nan_count(lg)}')
        for col in ['loss', 'tr', 'te', 'g_cls', 'compL', 'compG', 'dispL', 'dispG']:
            s_ = df[col].dropna(); post = s_[s_.index > 10]; tail = s_[s_.index > 180]
            pts = ' '.join(f'{e}:{s_.get(e, np.nan):.4g}' for e in (1, 11, 50, 100, 150, 200))
            sl = np.polyfit(tail.index, tail.values, 1)[0] if len(tail) > 2 else np.nan
            print(f'      {col:6s} {pts} | ep11+ min {post.min():.4g} max {post.max():.4g} mean {post.mean():.4g} std {post.std():.4g} | ep101-200 {s_[s_.index > 100].mean():.4g} | 末20斜率 {sl:+.2e}')
        run = COMP_RUNS[lab][0]; n_ = run.replace('exp_result_', '', 1)
        T_ = np.array([np.loadtxt(f'{ROOT}{run}{n_}_res/dsgd-lr0.001-budget1.0-r{i}-tacc.log') for i in range(9)]); w = T_[:, 100:200].mean(1)
        print(f'      訓練準確率 ep101–200 art/photo/sketch {w[:3].mean():.2f}/{w[3:6].mean():.2f}/{w[6:].mean():.2f}')
        c = pd.read_csv(ROOT + run + '/async_diag/consensus.csv').set_index('epoch').consensus_rms
        print(f'      共識離散度 ep1–25 {c.loc[1:25].mean():.2e} ep51–100 {c.loc[51:100].mean():.2e}')
        if os.path.exists(ROOT + run + '/async_diag/fusion_jump.csv'):
            j = pd.read_csv(ROOT + run + '/async_diag/fusion_jump.csv')
            print(f'      融合 {len(j)} 次｜收進總比重 中位 {j.w_in.median():.3f} p99 {j.w_in.quantile(.99):.3f} 最大 {j.w_in.max():.3f}｜相對跳動 p99 {j.jump_rel.quantile(.99):.2e} 最大 {j.jump_rel.max():.2e}')
        et = COST[lab]['ep_t']; print(f'      每 epoch 傳輸時間 ep1 {et.loc[1]:.2f} ep100 {et.loc[100]:.2f} ep199 {et.loc[199]:.2f} ep200 {et.loc[200]:.2f}（最後一個 epoch 占全程 {et.loc[200] / et.sum():.1%}）')
    if 'EF-HC bidir' in Q:
        run = COMP_RUNS['EF-HC bidir'][0]; rp = pd.read_csv(ROOT + run + '/async_diag/reply_log.csv')
        rp['y'] = rp.replier.str[5:].astype(int); nrep = rp.groupby('y').size().reindex(range(9)).fillna(0).astype(int).values
        own = COST['EF-HC bidir']['sends']
        print(f'  [EF-HC bidir] 每台上傳：自己送出 {own.tolist()}＋回傳 {nrep.tolist()}；慢設備（網速 0.1）回傳／自己送出 '
              + ' '.join(f'n{i}:{nrep[i] / max(own[i], 1):.1f}×' for i in range(9) if b_[i] <= 0.1 + 1e-9))
    print('\n' + '=' * 110); print('§5 原圖準確率（BN 平均B、九台平均 %）'); print('=' * 110)
    for lab, (f, key) in COMP_SRC.items():
        if lab not in Q or not os.path.exists(ROOT + f): continue
        d = pd.read_csv(ROOT + f); d = d[(d.arm == key) & (d.bn == 'avgB')]; t = d.groupby('eval_domain').closed_acc.mean() * 100
        print(f'  [{lab:16s}] art {t["art_painting"]:.2f} photo {t["photo"]:.2f} sketch {t["sketch"]:.2f} cartoon {t["cartoon"]:.2f}')
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.8))
    for lab, q in Q.items():
        pk = dict(lw=2.4 if lab == ref else 1.2, ls='--' if lab in ('uniform-full C0', 'SGP A3') else '-')
        et = COST[lab]['ep_t']
        ax[0].plot(et.index, et.values / 20, label=lab, **pk); ax[1].plot(q.index, 100 * q.closed_acc, label=lab, **pk)
        ax[2].plot(q.cum_t, 100 * q.closed_acc, label=lab, **pk); ax[3].plot(q.cum_t, q.det_auroc, label=lab, **pk)
    for c_, t_ in enumerate(['(i) tx time per step', '(ii) closed acc vs epoch', '(iii) closed acc vs cum. tx time', "(iii') face AUROC vs cum. tx time"]):
        ax[c_].set_title(f'sketch-slow: {t_}', fontsize=9); ax[c_].grid(alpha=.3); ax[c_].legend(fontsize=7)
    ax[1].set_ylim(55, 83); ax[2].set_ylim(55, 83); ax[3].set_ylim(0.5, 0.85)
    for c_ in (2, 3): ax[c_].axvline(Tb, c='k', ls=':', lw=.8)
    fig.tight_layout(); fig.savefig(ROOT + fig_path, dpi=120); print(f'\n圖：{fig_path}')


R3B_RUNS = {'GN': f'exp_result_v1_stage2_leave_cartoon_p1a_async_pbw_bwb0_frg{_BWS}',          # [plan 1019] 按群 × 正常網速
            'GS': f'exp_result_v1_stage2_leave_cartoon_p1a_async_pbwsk_bwfsketch0.1_frg{_BWS}'}  # [plan 1019] 按群 × sketch 慢
R3B_PAIR = {'GN': 'A2', 'GS': 'A1'}   # 配對的底座臂（同送出表、只差附的數）


def r3b_section(fig_path='results/proto_trigger/1019_r3b_curve_cartoon.png'):
    """[plan 1019] 按群校正份量（R3b）vs 底座：R0 組、R1 送出逐格、R2 node_4 隔離、R3 共識離散度、R4 正常不傷、R5 sketch 慢不輸底座、
    R6 份量（組成估算、實際送出表）、R7 記錄＋健康全軌跡、四級判決（plan §7）。只納入評估 checkpoint 已齊 40 個的臂。"""
    import os, sys as _sys
    _sys.path.insert(0, ROOT + 'scripts/posthoc')
    from composition_inbox_sim import simulate
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ctrl = {'C0': BW_ARMS['C0'][1], 'D-bw': BW_ARMS['Dbw'][1], 'C-match': BW_ARMS['Cmatch'][1]}
    allr = {'C-bw': BW_ARMS['Cbw'][1], 'SketchSlow': SKETCHSLOW_RUN, 'A1': SHARE_RUNS['A1'], 'A2': SHARE_RUNS['A2'], **ctrl, **R3B_RUNS}
    logs = {'C-bw': BW_ARMS['Cbw'][0], 'SketchSlow': 'logs/proto_trigger/1013_sketchslow_train.log',
            'A1': 'logs/proto_trigger/1016_A1_train.log', 'A2': 'logs/proto_trigger/1016_A2_train.log',
            **{k: f'logs/proto_trigger/1019_{k}_train.log' for k in R3B_RUNS}}
    csvs = ['1009_bw_curve_cartoon_avgbnB.csv', '1013_sketchslow_curve_cartoon_avgbnB.csv', '1016_share_rules_curve_cartoon_avgbnB.csv',
            '1019_r3b_curve_cartoon_avgbnB.csv']
    E = pd.concat([pd.read_csv(ROOT + 'results/proto_trigger/' + c) for c in csvs if os.path.exists(ROOT + 'results/proto_trigger/' + c)],
                  ignore_index=True)
    E['arm'] = E.run.map({v.replace('exp_result_', '', 1): k for k, v in allr.items()})
    E = E[E.arm.notna() & (E.score_fn == 'zperp')].copy(); E['ep'] = E.ckpt.str.replace('epoch_', '').astype(int)
    E = E.drop_duplicates(['arm', 'ep', 'node'], keep='last'); E['i'] = E.node.str[5:].astype(int)
    n_ck = E.groupby('arm').ep.nunique()
    assert (E.groupby(['arm', 'ep']).size() == 9).all(), '有 checkpoint 不是 9 節點'
    arms = [k for k in allr if n_ck.get(k, 0) == 40]; test = [k for k in R3B_RUNS if k in arms]
    print('評估 checkpoint 數：' + ' '.join(f'{k}={n_ck.get(k, 0)}' for k in allr) + f'｜納入 {arms}')
    Z = E.groupby(['ep', 'arm']).det_auroc.mean().unstack(); A = E.groupby(['ep', 'arm']).closed_acc.mean().unstack()
    win = lambda P, k, lo=25, hi=200: P.loc[lo:hi, k].mean()
    GRP = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    print('=' * 100); print('R0 組與附的數（async_diag/rate_group.csv）'); print('=' * 100)
    U = {}
    for k in R3B_RUNS:
        f = ROOT + allr[k] + '/async_diag/rate_group.csv'
        if not os.path.exists(f):
            print(f'  [{k}] rate_group.csv 不存在'); continue
        g = pd.read_csv(f); first = g.groupby('node').first()
        U[k] = np.array([first.loc[f'node_{i}', 'u'] for i in range(9)])
        same = g[g['style'] == g['neighbor_style']]; cross = g[g['style'] != g['neighbor_style']]
        ok = bool(((g.in_group == 1) == (g['style'] == g['neighbor_style'])).all())
        print(f'  [{k}] 組＝真實同畫風：{ok}｜組 ' + ' '.join(f'{i}:[{first.loc[f"node_{i}", "group"] if isinstance(first.loc[f"node_{i}", "group"], str) else ""}]' for i in range(9)))
        print(f'        r ' + ' '.join(f'{first.loc[f"node_{i}", "r"]:.4f}' for i in range(9)) + '｜u ' + ' '.join(f'{x:.4f}' for x in U[k]))
        print(f'        距離比 同組最大 {same.ratio.max():.2f}（中位 {same.ratio.median():.2f}）｜跨組最小 {cross.ratio.min():.2f}（中位 {cross.ratio.median():.2f}）｜c＝{g.c.iloc[0]}')

    print('\n' + '=' * 100); print('R1 送出序列：fired 與配對底座逐格比對'); print('=' * 100)
    Fd = {}
    for k in arms:
        b = pd.read_csv(ROOT + allr[k] + '/async_diag/broadcast_log.csv'); b['i'] = b.domain.str[5:].astype(int)
        Fd[k] = b.pivot(index='step', columns='i', values='fired').sort_index().values.astype(bool)
    for k in test:
        r_ = R3B_PAIR[k]; n = min(len(Fd[k]), len(Fd[r_]))
        print(f'  [{k}] vs {r_}：{n} 步、不同格數 {int((Fd[k][:n] != Fd[r_][:n]).sum())}｜總送出 {int(Fd[k].sum())} vs {int(Fd[r_].sum())}')

    print('\n' + '=' * 100); print('R2 機制一：逐台泛化（各組 − C-bw），node_4 − 其他八台；門檻 ≥ −0.005（A2 −0.0231）'); print('=' * 100)
    seg_ = lambda k, lo, hi: E[(E.arm == k) & (E.ep >= lo) & (E.ep <= hi)].groupby('i').closed_acc.mean().values
    N4 = {}
    for k in [x for x in ['C0', 'D-bw', 'C-match', 'A2', 'GN'] if x in arms]:
        for lo, hi in [(5, 50), (55, 100), (25, 200)]:
            d = seg_(k, lo, hi) - seg_('C-bw', lo, hi); n4 = d[4] - np.delete(d, 4).mean()
            if (lo, hi) == (5, 50): N4[k] = n4
            print(f'  [{k:8s}] ep{lo}–{hi} 逐台 ' + ' '.join(f'{x:+.4f}' for x in d) + f'｜九台平均 {d.mean():+.4f}｜node_4 − 其他八台 {n4:+.4f}')
    if 'GN' in arms:
        for lo, hi in [(5, 50), (25, 200)]:
            d = seg_('GN', lo, hi) - seg_('A2', lo, hi)
            print(f'  [GN − A2] ep{lo}–{hi} 逐台 ' + ' '.join(f'{x:+.4f}' for x in d) + f'｜九台平均 {d.mean():+.4f}｜正的台數 {int((d > 0).sum())}/9')

    print('\n' + '=' * 100); print('R3 機制二：共識離散度 ep1–100 ÷ C-bw；門檻 ≤ 1.30（A2 1.59）'); print('=' * 100)
    cref = pd.read_csv(ROOT + allr['C-bw'] + '/async_diag/consensus.csv').set_index('epoch').consensus_rms
    CR = {}
    for k in [x for x in ['C-bw', 'A2', 'GN', 'SketchSlow', 'A1', 'GS'] if x in arms]:
        c = pd.read_csv(ROOT + allr[k] + '/async_diag/consensus.csv').set_index('epoch').consensus_rms
        CR[k] = c.loc[1:100].mean() / cref.loc[1:100].mean()
        print(f'  [{k:10s}] ' + ' '.join(f'ep{lo}-{hi}:{c.loc[lo:hi].mean():.2e}' for lo, hi in [(1, 25), (26, 50), (51, 100), (101, 150), (151, 200)])
              + f'｜ep1–100 ÷ C-bw {CR[k]:.2f}')

    print('\n' + '=' * 100); print('R4／R5 主指標（zperp、九台平均、BN 平均B）'); print('=' * 100)
    HURT, LOSE = {}, {}
    for k in arms:
        if k in ctrl: continue
        v = {('z', 'end'): Z.loc[200, k], ('z', 'win'): win(Z, k), ('acc', 'end'): A.loc[200, k], ('acc', 'win'): win(A, k)}
        HURT[k] = {m_: v[(m_, 'end')] < ARTSLOW_THR[(m_, 'end')] and v[(m_, 'win')] < ARTSLOW_THR[(m_, 'win')] for m_ in ('z', 'acc')}
        print(f'  [{k:10s}] 面 ep200 {v[("z","end")]:.4f}／ep25–200 {v[("z","win")]:.4f}｜泛化 ep200 {v[("acc","end")]:.4f}／ep25–200 {v[("acc","win")]:.4f}'
              f'｜ep5–50 泛化 {win(A, k, 5, 50):.4f} 面 {win(Z, k, 5, 50):.4f}｜傷（1016 門檻）：面 {HURT[k]["z"]} 泛化 {HURT[k]["acc"]}')
    PT, WD = {'acc': 0.0109, 'z': 0.0144}, {'acc': 0.0069, 'z': 0.0141}
    for k in test:
        r_ = R3B_PAIR[k]; out = []
        for m_, P in [('acc', A), ('z', Z)]:
            de, dw = P.loc[200, k] - P.loc[200, r_], win(P, k) - win(P, r_)
            if k == 'GS':
                LOSE[m_] = (de < -PT[m_]) and (dw < -WD[m_])
            out.append(f'{"泛化" if m_ == "acc" else "面"} ep200 {de:+.4f} ep25–200 {dw:+.4f}'
                       + ' 分段 ' + ' '.join(f'ep{lo}-{hi}:{win(P, k, lo, hi) - win(P, r_, lo, hi):+.4f}' for lo, hi in [(5, 50), (55, 100), (105, 150), (155, 200)]))
        print(f'  [{k} − {r_}] ' + '｜'.join(out) + (f'｜輸（ep200 超過點值寬度且 ep25–200 超過區間寬度）：泛化 {LOSE["acc"]} 面 {LOSE["z"]}' if k == 'GS' else ''))
        for m_, P in [('acc', A), ('z', Z)]:
            hi_ = int((P.loc[25:200, k] > P.loc[25:200, r_]).sum())
            print(f'        {"泛化" if m_ == "acc" else "面"} 36 個評估點中 {k} 較高 {hi_}')

    print('\n' + '=' * 100); print('R6 份量（組成估算 composition_inbox_sim、實際送出表）'); print('=' * 100)
    for k in [x for x in ['C-bw', 'A2', 'GN', 'SketchSlow', 'A1', 'GS'] if x in Fd]:
        F = Fd[k][:4000]
        o = simulate(F, 'mh') if k in ('C-bw', 'SketchSlow') else simulate(F, 'rate_pair', u=U.get(k))
        sh = o['share']; print(f'  [{k:10s}] 份量 ' + '/'.join(f'{x:.3f}' for x in sh) + f'（偏差 {max(abs(x - 1 / 3) for x in sh):.3f}）'
                               + '｜逐台 ×9 ' + ' '.join(f'{x:.2f}' for x in o['node']))

    print('\n' + '=' * 100); print('R7 記錄：融合跳動、收入縮放、原圖準確率、健康全軌跡'); print('=' * 100)
    q_ = lambda s: f'中位 {s.median():.3g}／p99 {s.quantile(.99):.3g}／最大 {s.max():.3g}'
    for k in [x for x in ['A2', 'GN', 'A1', 'GS'] if x in arms]:
        j = pd.read_csv(ROOT + allr[k] + '/async_diag/fusion_jump.csv')
        print(f'  [{k}] 融合 {len(j)} 次｜w_in {q_(j.w_in)}｜w_max {q_(j.w_max)}｜相對跳動 {q_(j.jump_rel)}')
        jj = j.assign(i=j.domain.str[5:].astype(int)).groupby('i').w_in.mean()
        print(f'        逐台平均收進比重 ' + ' '.join(f'{jj.get(i, np.nan):.3f}' for i in range(9)))
    src = {'C-bw': ('results/proto_trigger/1012_artslow_source_clean_acc.csv', 'Cbw'),
           'SketchSlow': ('results/proto_trigger/1013_sketchslow_source_clean_acc.csv', 'SketchSlow'),
           'A1': ('results/proto_trigger/1016_A1_source_clean_acc.csv', 'ShareA1'), 'A2': ('results/proto_trigger/1016_A2_source_clean_acc.csv', 'ShareA2'),
           'GN': ('results/proto_trigger/1019_GN_source_clean_acc.csv', 'R3bN'), 'GS': ('results/proto_trigger/1019_GS_source_clean_acc.csv', 'R3bS')}
    SRC = {}
    for k in [x for x in src if x in arms]:
        f, key = src[k]
        if not os.path.exists(ROOT + f):
            print(f'  [{k:10s}] 原圖準確率：檔案不存在'); continue
        d = pd.read_csv(ROOT + f); d = d[(d.arm == key) & (d.bn == 'avgB')]
        t = d.groupby('eval_domain').closed_acc.mean() * 100; SRC[k] = t
        print(f'  [{k:10s}] 原圖準確率（BN 平均B、九台平均 %）art {t["art_painting"]:.2f} photo {t["photo"]:.2f} sketch {t["sketch"]:.2f} cartoon {t["cartoon"]:.2f}')
    for k in [x for x in ['A2', 'GN', 'A1', 'GS'] if x in arms]:
        if not os.path.exists(ROOT + logs[k]): continue
        df = parse(logs[k]).set_index('ep'); print(f'  [{k}] epoch 行 {len(df)}｜NaN/Traceback {nan_count(logs[k])}')
        for col in ['loss', 'tr', 'te', 'g_cls', 'compL', 'compG', 'dispL', 'dispG']:
            if col not in df: continue
            s_ = df[col].dropna(); post = s_[s_.index > 10]; tail = s_[s_.index > 180]
            pts = ' '.join(f'{e}:{s_.get(e, np.nan):.4g}' for e in (1, 11, 50, 100, 150, 200))
            sl = np.polyfit(tail.index, tail.values, 1)[0] if len(tail) > 2 else np.nan
            print(f'      {col:6s} {pts} | ep11+ min {post.min():.4g} max {post.max():.4g} mean {post.mean():.4g} std {post.std():.4g} | 末20斜率 {sl:+.2e}')
        n_ = allr[k].replace('exp_result_', '', 1)
        T = np.array([np.loadtxt(f'{ROOT}{allr[k]}{n_}_res/dsgd-lr0.001-budget1.0-r{i}-tacc.log') for i in range(9)])
        print(f'      訓練準確率（增強批次）ep101–200 逐台 ' + ' '.join(f'{x:.2f}' for x in T[:, 100:200].mean(1))
              + '｜ep1–50 node_4 ' + f'{T[4, :50].mean():.2f} vs photo 其他兩台 {T[[3, 5], :50].mean():.2f}')

    print('\n' + '=' * 100); print('四級判決（plan §7）'); print('=' * 100)
    if set(test) == {'GN', 'GS'}:
        r2 = N4['GN'] >= -0.005; r3 = CR['GN'] <= 1.30; r4 = not any(HURT['GN'].values())
        r5 = not any(LOSE.values()) and SRC.get('GS', {}).get('sketch', 0) >= 90
        r5_both = all(LOSE.values())
        lv = ('強' if r2 and r3 and r4 and r5 else '中' if r4 and r5 and (r2 or r3) else '退化' if r5_both else '邊際')
        print(f'  R2 node_4 {N4["GN"]:+.4f}（≥−0.005：{r2}）｜R3 共識 {CR["GN"]:.2f}（≤1.30：{r3}）｜R4 正常不傷：{r4}'
              f'｜R5 sketch 慢不輸 A1：{r5}（輸：泛化 {LOSE["acc"]} 面 {LOSE["z"]}；sketch 原圖 {SRC.get("GS", {}).get("sketch", float("nan")):.2f}）⇒ {lv}'
              '（R0／R1 見上，不過則為退化）')
    else:
        print(f'  尚未齊：{test}')

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.4))
    col = {'C-bw': 'tab:blue', 'A2': 'tab:cyan', 'GN': 'tab:orange', 'SketchSlow': 'tab:red', 'A1': 'tab:green', 'GS': 'tab:purple'}
    for r_, ks in enumerate([['C-bw', 'A2', 'GN'], ['SketchSlow', 'A1', 'GS']]):
        for k in [x for x in ks if x in arms]:
            ax[r_, 0].plot(A.index, 100 * A[k], c=col[k], label=k); ax[r_, 1].plot(Z.index, Z[k], c=col[k], label=k)
        ax[r_, 0].set_title(('normal bw' if r_ == 0 else 'sketch slow') + ': closed-set acc (%)')
        ax[r_, 1].set_title(('normal bw' if r_ == 0 else 'sketch slow') + ': face AUROC (zperp)')
    for x in ax.flat:
        x.set_xlabel('epoch'); x.grid(alpha=.3); x.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT + fig_path, dpi=130); print(f'\n圖：{fig_path}')


_FD = 'results/proto_trigger/'
_FL = 'logs/proto_trigger/'
FOLD_SCEN = {  # [plan 1019 一折全貌] 情境 → (網速表, {組: (run 目錄, 估算規則, 雙向, 要多存 46 MB, 訓練 log, (原圖 csv, 鍵))})
    'normal': (BW_ARMS['Cbw'][1] + '/async_diag/bandwidth.csv', {
        'no-comp': (BW_ARMS['Cbw'][1], 'mh', False, False, BW_ARMS['Cbw'][0], (_FD + '1012_artslow_source_clean_acc.csv', 'Cbw')),
        'ours (group)': (R3B_RUNS['GN'], 'group', False, False, _FL + '1019_GN_train.log', (_FD + '1019_GN_source_clean_acc.csv', 'R3bN')),
        'EF-HC one-way': (BW_ARMS['Dbw'][1], 'mh', False, True, BW_ARMS['Dbw'][0], (_FD + '1012_artslow_source_clean_acc.csv', 'Dbw')),
        'EF-HC two-way': (f'exp_result_v1_stage2_leave_cartoon_p1a_async_lr_tau1.368e-05_bwb0r_bd{_BWS}', 'mh', True, True, _FL + '1019_fold_F13_train.log', (_FD + '1019_fold_F13_source_clean_acc.csv', 'FoldF13')),
        'PushCen': (f'exp_result_v1_stage2_leave_cartoon_p1a_async_pbw_bwb0_fpc{_BWS}', 'pushcen', False, False, _FL + '1019_fold_F12_train.log', (_FD + '1019_fold_F12_source_clean_acc.csv', 'FoldF12')),
        'uniform-matched': (f'exp_result_v1_stage2_leave_cartoon_p1a_async_puni_bwb0{_BWS}', 'mh', False, False, _FL + '1019_fold_F14_train.log', (_FD + '1019_fold_F14_source_clean_acc.csv', 'FoldF14')),
        'uniform-full C0': (BW_ARMS['C0'][1], 'mh', False, False, BW_ARMS['C0'][0], (_FD + '1012_artslow_source_clean_acc.csv', 'C0')),
        'base (ablation)': (SHARE_RUNS['A2'], 'rate_pair', False, False, _FL + '1016_A2_train.log', (_FD + '1016_A2_source_clean_acc.csv', 'ShareA2')),
    }),
    'art-slow': (_FD + '1012_artslow/bandwidth_art0.1.csv', {
        'no-comp': (ARTSLOW_RUN, 'mh', False, False, ARTSLOW_LOG, (_FD + '1012_artslow_source_clean_acc.csv', 'ArtSlow')),
        'ours (group)': (f'exp_result_v1_stage2_leave_cartoon_p1a_async_pbwart_bwfart0.1_frg{_BWS}', 'group', False, False, _FL + '1019_fold_F7_train.log', (_FD + '1019_fold_F7_source_clean_acc.csv', 'FoldF7')),
        'EF-HC one-way': (f'exp_result_v1_stage2_leave_cartoon_p1a_async_lr_tau1.1e-05_bwfartr{_BWS}', 'mh', False, True, _FL + '1019_fold_F8_train.log', (_FD + '1019_fold_F8_source_clean_acc.csv', 'FoldF8')),
        'EF-HC two-way': (f'exp_result_v1_stage2_leave_cartoon_p1a_async_lr_tau1.1e-05_bwfartr_bd{_BWS}', 'mh', True, True, _FL + '1019_fold_F9_train.log', (_FD + '1019_fold_F9_source_clean_acc.csv', 'FoldF9')),
        'PushCen': (f'exp_result_v1_stage2_leave_cartoon_p1a_async_pbwart_bwfart0.1_fpc{_BWS}', 'pushcen', False, False, _FL + '1019_fold_F10_train.log', (_FD + '1019_fold_F10_source_clean_acc.csv', 'FoldF10')),
        'uniform-matched': (f'exp_result_v1_stage2_leave_cartoon_p1a_async_puni_bwfart0.1{_BWS}', 'mh', False, False, _FL + '1019_fold_F11_train.log', (_FD + '1019_fold_F11_source_clean_acc.csv', 'FoldF11')),
    }),
    'photo-slow': (_FD + '1019_fold/bandwidth_photo0.1.csv', {
        'no-comp': (f'exp_result_v1_stage2_leave_cartoon_p1a_async_pbwph_bwfphoto0.1{_BWS}', 'mh', False, False, _FL + '1019_fold_F1_train.log', (_FD + '1019_fold_F1_source_clean_acc.csv', 'FoldF1')),
        'ours (group)': (f'exp_result_v1_stage2_leave_cartoon_p1a_async_pbwph_bwfphoto0.1_frg{_BWS}', 'group', False, False, _FL + '1019_fold_F2_train.log', (_FD + '1019_fold_F2_source_clean_acc.csv', 'FoldF2')),
        'EF-HC one-way': (f'exp_result_v1_stage2_leave_cartoon_p1a_async_lr_tau1.11e-05_bwfphr{_BWS}', 'mh', False, True, _FL + '1019_fold_F3_train.log', (_FD + '1019_fold_F3_source_clean_acc.csv', 'FoldF3')),
        'EF-HC two-way': (f'exp_result_v1_stage2_leave_cartoon_p1a_async_lr_tau1.11e-05_bwfphr_bd{_BWS}', 'mh', True, True, _FL + '1019_fold_F4_train.log', (_FD + '1019_fold_F4_source_clean_acc.csv', 'FoldF4')),
        'PushCen': (f'exp_result_v1_stage2_leave_cartoon_p1a_async_pbwph_bwfphoto0.1_fpc{_BWS}', 'pushcen', False, False, _FL + '1019_fold_F5_train.log', (_FD + '1019_fold_F5_source_clean_acc.csv', 'FoldF5')),
        'uniform-matched': (f'exp_result_v1_stage2_leave_cartoon_p1a_async_puni_bwfphoto0.1{_BWS}', 'mh', False, False, _FL + '1019_fold_F6_train.log', (_FD + '1019_fold_F6_source_clean_acc.csv', 'FoldF6')),
    }),
    'sketch-slow': (_FD + '1013_sketchslow/bandwidth_sketch0.1.csv', {
        'no-comp': (SKETCHSLOW_RUN, 'mh', False, False, _FL + '1013_sketchslow_train.log', (_FD + '1013_sketchslow_source_clean_acc.csv', 'SketchSlow')),
        'ours (group)': (R3B_RUNS['GS'], 'group', False, False, _FL + '1019_GS_train.log', (_FD + '1019_GS_source_clean_acc.csv', 'R3bS')),
        'EF-HC one-way': (COMP_RUNS['D-bw'][0], 'mh', False, True, COMP_RUNS['D-bw'][4], COMP_SRC['D-bw']),
        'EF-HC two-way': (COMP_RUNS['EF-HC bidir'][0], 'mh', True, True, COMP_RUNS['EF-HC bidir'][4], COMP_SRC['EF-HC bidir']),
        'PushCen': (COMP_RUNS['PushCen'][0], 'pushcen', False, False, COMP_RUNS['PushCen'][4], COMP_SRC['PushCen']),
        'uniform-matched': (COMP_RUNS['uniform-matched'][0], 'mh', False, False, COMP_RUNS['uniform-matched'][4], COMP_SRC['uniform-matched']),
        'uniform-full C0': (BW_ARMS['C0'][1], 'mh', False, False, BW_ARMS['C0'][0], COMP_SRC['uniform-full C0']),
        'base (ablation)': (SHARE_RUNS['A1'], 'rate_pair', False, False, _FL + '1016_A1_train.log', COMP_SRC['base A1']),
        'SGP': (SHARE_RUNS['A3'], 'pushsum', False, False, _FL + '1016_A3_train.log', COMP_SRC['SGP A3']),
    }),
}
FOLD_REF_SRC = (_FD + '1012_artslow_source_clean_acc.csv', 'Cbw')   # 公平的參照：正常網速照網速送的原圖準確率


def fold_section(fig_path='results/proto_trigger/1019_fold_main.png', collect=None):
    """[plan 1019 一折全貌] 四情境 × 各組 vs 我們（按群）。傳輸記帳＝EF-HC 原文（同 comp_section）。
    主判決（plan §7.2）：同累積傳輸時間 T*（＝我們全程）下泛化／面檢測的四級判決＋成本；公平＝各畫風原圖準確率相對正常網速 C-bw 的最大掉分
    （寬 1.54）；兩難＝EF-HC 雙向「持平、成本較低」或「勝」且 EF-HC 單向「勝」或「較公平」；PushCen「勝」或「較公平」。只納入 40 個 checkpoint 都評估完的組。"""
    import os, sys
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
    sys.path.insert(0, ROOT + 'scripts/posthoc')
    from composition_inbox_sim import simulate
    csvs = ['1009_bw', '1012_artslow', '1013_sketchslow', '1016_share_rules', '1017_competitors', '1019_r3b', '1019_fold']
    ev = pd.concat([pd.read_csv(ROOT + _FD + f'{c}_curve_cartoon_avgbnB.csv') for c in csvs
                    if os.path.exists(ROOT + _FD + f'{c}_curve_cartoon_avgbnB.csv')], ignore_index=True)
    ev = ev[ev.score_fn == 'zperp'].copy(); ev['ep'] = ev.ckpt.str.replace('epoch_', '').astype(int)
    ev = ev.drop_duplicates(['run', 'ep', 'node'], keep='last')
    G = nx.random_geometric_graph(9, 0.8, seed=1234); NBR = {i: sorted(G[i]) for i in range(9)}; DEG = {i: len(NBR[i]) for i in range(9)}
    wg, wf, wfair = 0.0109, 0.0144, 1.54
    rs = pd.read_csv(ROOT + FOLD_REF_SRC[0]); rs = rs[(rs.arm == FOLD_REF_SRC[1]) & (rs.bn == 'avgB')]
    REF = rs.groupby('eval_domain').closed_acc.mean() * 100
    at = lambda q, t, col: float(np.interp(t, q.cum_t, q[col])) if t <= q.cum_t.iloc[-1] else float(q[col].iloc[-1])
    OUT = {}
    fig, ax = plt.subplots(4, 3, figsize=(19, 17))
    for r_, (sc, (bwf, arms)) in enumerate(FOLD_SCEN.items()):
        bt = pd.read_csv(ROOT + bwf); bt['i'] = bt.node.str[5:].astype(int); b_ = bt.sort_values('i').b.values.astype(float)
        slow_m = b_ <= 0.1 + 1e-9
        Q, COST, SH, SRC = {}, {}, {}, {}
        for lab, (run, rule, bd, store, lg, (sf, sk)) in arms.items():
            q = ev[ev.run == run.replace('exp_result_', '', 1)].groupby('ep')[['closed_acc', 'det_auroc']].mean()
            if len(q) != 40: continue
            b = pd.read_csv(ROOT + run + '/async_diag/broadcast_log.csv'); b['i'] = b.domain.str[5:].astype(int)
            Fm = b.pivot(index='step', columns='i', values='fired').sort_index().values.astype(bool)
            if bd:
                V = np.zeros_like(Fm, dtype=float)
                for i in range(9): V[:, i] = sum(np.maximum(Fm[:, i], Fm[:, j]) for j in NBR[i]) / DEG[i]
            else:
                V = Fm.astype(float)
            cost = V / b_[None, :]
            ep_t = pd.Series(cost.sum(1) / 9).groupby(np.arange(len(cost)) // 20).sum(); ep_t.index += 1
            q['cum_t'] = ep_t.cumsum().reindex(q.index).values; Q[lab] = q
            COST[lab] = dict(total=ep_t.sum(), per_node=cost.sum(0), slow=cost.sum(0)[slow_m].mean() if slow_m.any() else np.nan,
                             store=store, sends=Fm.sum(0), ep_t=ep_t, run=run, lg=lg)
            if rule == 'group':
                g = pd.read_csv(ROOT + run + '/async_diag/rate_group.csv').groupby('node').first()
                SH[lab] = simulate(Fm[:4000], 'rate_pair', False, u=np.array([g.loc[f'node_{i}', 'u'] for i in range(9)]))
            else:
                SH[lab] = simulate(Fm[:4000], rule, bd)
            if os.path.exists(ROOT + sf):
                d = pd.read_csv(ROOT + sf); d = d[(d.arm == sk) & (d.bn == 'avgB')]
                if len(d): SRC[lab] = d.groupby('eval_domain').closed_acc.mean() * 100
        ref = 'ours (group)'
        if collect is not None: collect[sc] = dict(Q=Q, COST=COST, SH=SH, SRC=SRC, b=b_, slow=slow_m, V={})
        print('\n' + '#' * 110); print(f'# 情境 {OV_SC.get(sc, sc)}｜網速 {np.round(b_, 4).tolist()}｜納入 {list(Q)}'); print('#' * 110)
        if ref not in Q:
            print('  按群校正（我們）尚未評估完，略過判決'); continue
        Tb = COST[ref]['total']; slow_ref = COST[ref]['slow']
        print(f'§1 成本、份量、原圖準確率（同傳輸時間點＝按群校正全程每台平均傳輸時間 {Tb:.1f}）')
        md = {}
        for lab in Q:
            c = COST[lab]; sh = SH[lab]['share']
            src = SRC.get(lab)
            if src is not None:
                drop = {k: REF[k] - src[k] for k in ['art_painting', 'photo', 'sketch']}; md[lab] = max(drop.values())
                s_txt = f'｜原圖 art {src["art_painting"]:.2f} photo {src["photo"]:.2f} sketch {src["sketch"]:.2f} cartoon {src["cartoon"]:.2f}｜最大掉分 {md[lab]:+.2f}'
            else:
                s_txt = '｜原圖：檔案不存在'
            sl = f'{c["slow"]:6.0f}（{c["slow"] / slow_ref:.2f}×）' if slow_m.any() else '—'
            print(f'  [{OV_NAME.get(lab, lab)}] 每台平均傳輸 {c["total"]:7.1f}（{c["total"] / Tb:.2f}×）｜慢設備每台 {sl}｜多存 46 MB {"是" if c["store"] else "否"}'
                  f'｜份量 ' + '/'.join(f'{x:.3f}' for x in sh) + s_txt)
            print(f'        逐台送出 {c["sends"].tolist()}')
        print(f'§2 主指標（同傳輸時間點內插；對手總時間不到就用最終值）')
        for lab, q in Q.items():
            print(f'  [{OV_NAME.get(lab, lab)}] 同傳輸時間點：泛化 {at(q, Tb, "closed_acc"):.4f} 面 {at(q, Tb, "det_auroc"):.4f}｜ep200 {q.closed_acc.loc[200]:.4f}／{q.det_auroc.loc[200]:.4f}'
                  f'｜ep25–200 {q.closed_acc.loc[25:].mean():.4f}／{q.det_auroc.loc[25:].mean():.4f}｜ep155–195 {q.closed_acc.loc[155:195].mean():.4f}／{q.det_auroc.loc[155:195].mean():.4f}'
                  f'｜ep5–50 泛化 {q.closed_acc.loc[5:50].mean():.4f}｜最後 epoch 傳輸占全程 {COST[lab]["ep_t"].iloc[-1] / COST[lab]["ep_t"].sum():.1%}')
        print(f'§3 判決（差＝按群校正 − 對手，在同傳輸時間點；雜訊 泛化 {wg} 面 {wf}；公平寬 {wfair}）')
        V_ = {}
        qa = Q[ref]
        for lab, q in Q.items():
            if lab == ref: continue
            da = at(qa, Tb, 'closed_acc') - at(q, Tb, 'closed_acc'); dz = at(qa, Tb, 'det_auroc') - at(q, Tb, 'det_auroc')
            c = COST[lab]; hs = slow_m.any() and c['slow'] >= 1.3 * slow_ref
            higher = (c['total'] >= 1.3 * Tb) or hs or c['store']
            better = (da > wg) or (dz > wf); worse = (da < -wg) or (dz < -wf)
            if better and not worse: v = '按群勝'
            elif not better and not worse: v = '持平、按群成本較低' if higher else '持平'
            elif worse and not better: v = '按群輸' if not higher else '對手成績較好、但成本較高（取捨）'
            else: v = '取捨（一高一低）'
            fair = '—'
            if lab in md and ref in md:
                fd = md[lab] - md[ref]; fair = '按群較公平' if fd > wfair else ('對手較公平' if fd < -wfair else '相同')
                fair += f'（最大掉分 對手 {md[lab]:+.2f} vs 我們 {md[ref]:+.2f}）'
            why = [n for n, f in [('傳輸 ≥1.3×', c['total'] >= 1.3 * Tb), ('慢設備 ≥1.3×', hs), ('多存 46 MB', c['store'])] if f]
            V_[lab] = (v, fair)
            print(f'  [{OV_NAME.get(lab, lab)}] 泛化差 {da:+.4f} 面差 {dz:+.4f}｜對手成本較高：{"、".join(why) if why else "否"} ⇒ {v}｜公平：{fair}')
        if sc != 'normal' and 'EF-HC one-way' in V_ and 'EF-HC two-way' in V_:
            two = V_['EF-HC two-way'][0] in ('持平、按群成本較低', '按群勝')
            one = V_['EF-HC one-way'][0] == '按群勝' or V_['EF-HC one-way'][1].startswith('按群較公平')
            print(f'  ⇒ EF-HC 兩難成立：{two and one}（雙向 {V_["EF-HC two-way"][0]}；單向 {V_["EF-HC one-way"][0]}／{V_["EF-HC one-way"][1][:5]}）')
        if 'PushCen' in V_:
            pc = V_['PushCen'][0] == '按群勝' or V_['PushCen'][1].startswith('按群較公平')
            print(f'  ⇒ 勝 PushCen（成績或公平）：{pc}')
        print(f'  ⇒ 有「按群輸」：{any(v[0] == "按群輸" for v in V_.values())}')
        OUT[sc] = V_
        if collect is not None: collect[sc]['V'] = V_; collect[sc]['Tb'] = Tb
        print('§4 健康與過程（本次新跑的組；EF-HC 雙向回傳倍數）')
        for lab in Q:
            lg = COST[lab]['lg']; run = COST[lab]['run']
            if '1019_fold_' not in lg or not os.path.exists(ROOT + lg): continue
            df = parse(lg).set_index('ep'); print(f'  [{lab}] epoch 行 {len(df)}｜NaN/Traceback {nan_count(lg)}')
            for col in ['loss', 'tr', 'g_cls', 'compL', 'compG', 'dispL']:
                s_ = df[col].dropna(); post = s_[s_.index > 10]; tail = s_[s_.index > 180]
                sl_ = np.polyfit(tail.index, tail.values, 1)[0] if len(tail) > 2 else np.nan
                print(f'      {col:6s} ' + ' '.join(f'{e}:{s_.get(e, np.nan):.4g}' for e in (1, 11, 50, 100, 150, 200))
                      + f' | ep11+ min {post.min():.4g} max {post.max():.4g} mean {post.mean():.4g} std {post.std():.4g} | 末20斜率 {sl_:+.2e}')
            c = pd.read_csv(ROOT + run + '/async_diag/consensus.csv').set_index('epoch').consensus_rms
            cref = pd.read_csv(ROOT + BW_ARMS['Cbw'][1] + '/async_diag/consensus.csv').set_index('epoch').consensus_rms
            print(f'      共識離散度 ep1–100 ÷ 正常 C-bw {c.loc[1:100].mean() / cref.loc[1:100].mean():.2f}')
            if os.path.exists(ROOT + run + '/async_diag/fusion_jump.csv'):
                j = pd.read_csv(ROOT + run + '/async_diag/fusion_jump.csv')
                print(f'      融合 {len(j)} 次｜收進總比重 中位 {j.w_in.median():.3f} p99 {j.w_in.quantile(.99):.3f} 最大 {j.w_in.max():.3f}｜相對跳動 p99 {j.jump_rel.quantile(.99):.2e} 最大 {j.jump_rel.max():.2e}')
            if os.path.exists(ROOT + run + '/async_diag/reply_log.csv'):
                rp = pd.read_csv(ROOT + run + '/async_diag/reply_log.csv'); rp['y'] = rp.replier.str[5:].astype(int)
                nrep = rp.groupby('y').size().reindex(range(9)).fillna(0).astype(int).values; own = COST[lab]['sends']
                print(f'      每台上傳：自己送出 {own.tolist()}＋回傳 {nrep.tolist()}；慢設備回傳／自己送出 '
                      + ' '.join(f'n{i}:{nrep[i] / max(own[i], 1):.1f}×' for i in range(9) if slow_m[i]))
        for lab, q in Q.items():
            pk = dict(lw=2.6 if lab == ref else 1.2, ls='--' if lab in ('uniform-full C0', 'SGP', 'base (ablation)') else '-')
            ax[r_, 0].plot(q.cum_t, 100 * q.closed_acc, label=OV_NAME.get(lab, lab), **pk); ax[r_, 1].plot(q.cum_t, q.det_auroc, label=OV_NAME.get(lab, lab), **pk)
        labs = list(Q); xs = np.arange(9); wdt = 0.8 / len(labs)
        for k_, lab in enumerate(labs):
            ax[r_, 2].bar(xs + k_ * wdt, COST[lab]['per_node'], wdt, label=OV_NAME.get(lab, lab))
        ax[r_, 0].set_title(f'{OV_SC.get(sc, sc)}：泛化 vs 累積傳輸時間', fontsize=9); ax[r_, 1].set_title(f'{OV_SC.get(sc, sc)}：面檢測 vs 累積傳輸時間', fontsize=9)
        ax[r_, 2].set_title(f'{OV_SC.get(sc, sc)}：逐台傳輸時間（node_0–8；網速 0.1＝慢）', fontsize=9)
        ax[r_, 0].set_ylim(55, 83); ax[r_, 1].set_ylim(0.5, 0.86)
        for c_ in (0, 1): ax[r_, c_].axvline(Tb, c='k', ls=':', lw=.8)
        for c_ in range(3): ax[r_, c_].grid(alpha=.3); ax[r_, c_].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(ROOT + fig_path, dpi=110); print(f'\n圖：{fig_path}')
    print('\n' + '=' * 110); print('整份計畫四級判決（plan §7.2）'); print('=' * 110)
    slow_sc = [s for s in ('art-slow', 'photo-slow', 'sketch-slow') if s in OUT]
    if len(slow_sc) == 3 and all('EF-HC one-way' in OUT[s] and 'EF-HC two-way' in OUT[s] and 'PushCen' in OUT[s] for s in slow_sc):
        ok = [s for s in slow_sc if (OUT[s]['EF-HC two-way'][0] in ('持平、按群成本較低', '按群勝'))
              and (OUT[s]['EF-HC one-way'][0] == '按群勝' or OUT[s]['EF-HC one-way'][1].startswith('按群較公平'))]
        pc = all(OUT[s]['PushCen'][0] == '按群勝' or OUT[s]['PushCen'][1].startswith('按群較公平') for s in slow_sc)
        lose = any(v[0] == '按群輸' for s in OUT for v in OUT[s].values())
        lv = '退化' if lose else ('強' if len(ok) == 3 and pc else '中' if len(ok) >= 2 else '邊際')
        print(f'  兩難成立的情境 {ok}｜三情境都勝 PushCen：{pc}｜任何「按群輸」：{lose} ⇒ {lv}')
    else:
        print(f'  尚未齊：{ {s: list(OUT[s]) for s in OUT} }')


OV_NAME = {'ours (group)': '按群校正（我們）', 'no-comp': '照網速送（不校正）', 'EF-HC one-way': 'EF-HC 單向',
           'EF-HC two-way': 'EF-HC 雙向', 'PushCen': 'PushCen', 'uniform-matched': '同週期送（同傳輸時間）',
           'uniform-full C0': '全員同速（理想）', 'base (ablation)': '只看自己（消融）', 'SGP': '推和法 SGP'}   # dany 1020 名詞
OV_SC = {'sketch-slow': 'sketch 集中慢', 'photo-slow': 'photo 集中慢', 'art-slow': 'art 集中慢', 'normal': '慢設備分散'}
OV_DN = {'art_painting': 'art', 'photo': 'photo', 'sketch': 'sketch'}
OV_SLOW = {'sketch-slow': 'sketch', 'photo-slow': 'photo', 'art-slow': 'art_painting', 'normal': None}
OV_NOISE = {'acc': 1.09, 'z': 0.0144, 'src': 1.54}   # plan §7.2：泛化／面＝1009 四臂點值；原圖＝1009 四臂 sketch 範圍


def _ov_gap(x, ideal, noise, f):
    """離理想的差距；超過雜訊寬度加粗。"""
    g = x - ideal; s = f'{g:+{f}}'
    return f'**{s}**' if abs(g) > noise else s


def overview_section(fig_path='results/proto_trigger/1019_contrib2_overview.png'):
    """[1019 整理；dany 1020 改成以「全員同速（理想）」為基準] 理想能到多少 → 慢設備發生後照網速送掉多少 → 各方法離理想多少。
    方法都在同傳輸時間點（我們全程每台平均傳輸時間）取值；理想＝全員同速（九台同週期、總份數相同那支）ep200。
    所有數字都由 fold_section 的同一套計算收集（collect），不另算；原圖準確率＝ep200、BN 平均B、九台平均。"""
    import io, os, contextlib
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'font.size': 9})
    C = {}
    with contextlib.redirect_stdout(io.StringIO()): fold_section(collect=C)
    at = lambda q, t, col: float(np.interp(t, q.cum_t, q[col])) if t <= q.cum_t.iloc[-1] else float(q[col].iloc[-1])
    DOM3 = ['art_painting', 'photo', 'sketch']
    _iq = C['normal']['Q']['uniform-full C0']; IA, IZ = 100 * _iq.closed_acc.loc[200], _iq.det_auroc.loc[200]
    IS = C['normal']['SRC']['uniform-full C0']
    print(f'理想＝全員同速（沒人慢）：泛化 {IA:.2f}｜面檢測 {IZ:.4f}｜原圖 art {IS["art_painting"]:.2f}／photo {IS["photo"]:.2f}／sketch {IS["sketch"]:.2f}')
    print('　　理想＝1007 C（本地時鐘）；EF-HC 原版在全員同速下（1007 D）泛化 79.53、面檢測 0.8324；第一貢獻的 A（固定門檻）79.07／0.8164 不當理想')
    print(f'雜訊寬度：泛化 {OV_NOISE["acc"]}、面檢測 {OV_NOISE["z"]}、原圖 {OV_NOISE["src"]}（離理想超過雜訊寬度者加粗）')
    for sc in ['sketch-slow', 'photo-slow', 'art-slow', 'normal']:
        d_ = C.get(sc)
        if not d_ or not d_['Q']:
            print(f'\n## {OV_SC[sc]}：尚無評估完的組'); continue
        Q, COST, SH, V, SRC = d_['Q'], d_['COST'], d_['SH'], d_['V'], d_['SRC']; Tb = d_.get('Tb'); b_ = d_['b']; slow = d_['slow']
        ref = 'ours (group)' if 'ours (group)' in Q else None; sl = OV_SLOW[sc]
        Tr = COST[ref]['total'] if ref else None; Sr = COST[ref]['slow'] if ref else None
        val = lambda lab, col: at(Q[lab], Tb, col) if Tb else float(Q[lab][col].loc[200])
        labs = [l for l in OV_NAME if l in Q and l != 'uniform-full C0']
        print(f'\n## {OV_SC[sc]}｜網速 node_0–8 = {np.round(b_, 2).tolist()}' + (f'｜同傳輸時間點 = {Tb:.1f}' if Tb else '（還沒有按群校正，取 ep200）'))
        print('\n表一 離理想多少（同傳輸時間點）')
        oth = '其他畫風最多掉' if sl else '各畫風最多掉'
        print(f'| 方法 | 泛化（離理想） | 面檢測（離理想） | {OV_DN.get(sl, "—")} 原圖（離理想） | {oth}（離理想） |')
        print('|---|---|---|---|---|')
        print(f'| 全員同速（理想） | {IA:.2f} | {IZ:.4f} | {IS[sl]:.2f} | — |' if sl else f'| 全員同速（理想） | {IA:.2f} | {IZ:.4f} | — | — |')
        for lab in labs:
            src = SRC.get(lab); xa = 100 * val(lab, 'closed_acc'); xz = val(lab, 'det_auroc')
            c1 = f'{xa:.2f}（{_ov_gap(xa, IA, OV_NOISE["acc"], ".2f")}）'; c2 = f'{xz:.4f}（{_ov_gap(xz, IZ, OV_NOISE["z"], ".4f")}）'
            if src is None: print(f'| {OV_NAME[lab]} | {c1} | {c2} | 檔案不存在 | |'); continue
            c3 = f'{src[sl]:.2f}（{_ov_gap(src[sl], IS[sl], OV_NOISE["src"], ".2f")}）' if sl else '—'
            od = {k: src[k] - IS[k] for k in DOM3 if k != sl}; ko = min(od, key=od.get)
            c4 = f'{_ov_gap(src[ko], IS[ko], OV_NOISE["src"], ".2f")}（{OV_DN[ko]}）' if od[ko] < 0 else '無（都不低於理想）'
            nm = OV_NAME[lab] + ('　←慢設備造成的損失' if lab == 'no-comp' and sl else '')
            print(f'| {nm} | {c1} | {c2} | {c3} | {c4} |')
        if sl and 'no-comp' in Q and ref:
            parts = []
            for key, x_of, I_, n_, nmk in [('acc', lambda l: 100 * val(l, 'closed_acc'), IA, OV_NOISE['acc'], '泛化'),
                                          ('z', lambda l: val(l, 'det_auroc'), IZ, OV_NOISE['z'], '面檢測'),
                                          ('src', lambda l: SRC[l][sl], IS[sl], OV_NOISE['src'], f'{OV_DN[sl]} 原圖')]:
                L = I_ - x_of('no-comp')
                if L > 2 * n_:
                    parts.append(f'{nmk}（照網速送掉 {L:.4g}）：' + '、'.join(f'{OV_NAME[l]} {(x_of(l) - x_of("no-comp")) / L:.0%}' for l in labs if l != 'no-comp' and (key != 'src' or l in SRC)))
            print('　　補回比例＝(方法 − 照網速送)÷(理想 − 照網速送)，只列照網速送的損失超過雜訊寬度兩倍的指標：' + ('；'.join(parts) if parts else '無（損失都不夠大）'))
        print('\n表二 成本')
        print('| 方法 | 每台平均傳輸時間 | 慢設備每台傳輸時間 | 多存 46 MB | 份量 art/photo/sketch | 跑完全程 泛化／面檢測 | 計畫判決（我們 vs 它） |')
        print('|---|---|---|---|---|---|---|')
        for lab in labs:
            c = COST[lab]; q = Q[lab]; sh = SH[lab]['share']
            tt = f'{c["total"]:.0f}' + (f'（{c["total"] / Tr:.2f}×）' if Tr else '')
            st = (f'{c["slow"]:.0f}' + (f'（{c["slow"] / Sr:.2f}×）' if Sr else '')) if slow.any() else '—'
            vd = '（基準）' if lab == ref else (V[lab][0] + '；公平' + V[lab][1].split('（')[0] if lab in V else '—')
            print(f'| {OV_NAME[lab]} | {tt} | {st} | {"是" if c["store"] else "否"} | ' + '/'.join(f'{x:.3f}' for x in sh)
                  + f' | {100 * q.closed_acc.loc[200]:.2f}／{q.det_auroc.loc[200]:.4f} | {vd} |')
        print('\n表三 各節點「自己畫風」的原圖準確率（每台自己手上的模型＋自己的 BN，在自己畫風的原圖上；ep200）'
              '\n　　（BN 平均B 下九台幾乎是同一個模型，最大−最小 ≤0.13 分，所以這裡用自己的 BN）')
        hdr = None
        for lab in ['uniform-full C0'] + labs:
            sf, sk = FOLD_SCEN['normal'][1][lab][5] if lab == 'uniform-full C0' else FOLD_SCEN[sc][1][lab][5]
            if not os.path.exists(ROOT + sf): continue
            t = pd.read_csv(ROOT + sf); t = t[(t.arm == sk) & (t.bn == 'own') & (t.node_style == t.eval_domain)]
            t['i'] = t.node.str[5:].astype(int); t = t.sort_values('i')
            if hdr is None:
                hdr = [f'n{i} {s[:2] if s != "photo" else "ph"}{"·慢" if slow[i] else ""}' for i, s in zip(t.i, t.node_style)]
                print('| 方法 | ' + ' | '.join(hdr) + ' | 最低 |'); print('|---' * (len(hdr) + 2) + '|')
            v = (t.closed_acc.values * 100)
            print(f'| {OV_NAME[lab]} | ' + ' | '.join(f'{x:.1f}' for x in v) + f' | {v.min():.1f}（n{t.i.values[v.argmin()]}） |')
    # 圖：每個有按群校正的情境一列：泛化 vs 累積傳輸時間、面檢測 vs 累積傳輸時間、各畫風原圖離理想多少
    rows = [sc for sc in ['sketch-slow', 'photo-slow', 'art-slow', 'normal'] if sc in C and 'ours (group)' in C[sc]['Q']]
    col = {'ours (group)': '#1f3b73', 'no-comp': '#9a9a9a', 'EF-HC one-way': '#d62728', 'EF-HC two-way': '#ff7f0e',
           'PushCen': '#2ca02c'}   # 圖只放主對手；同週期送、只看自己只進表格（dany 1020）
    fig, ax = plt.subplots(len(rows), 3, figsize=(17, 5.2 * len(rows)), squeeze=False)
    for r_, sc in enumerate(rows):
        Q, COST, SRC, Tb = C[sc]['Q'], C[sc]['COST'], C[sc]['SRC'], C[sc]['Tb']
        labs = [l for l in col if l in Q]
        for lab in labs:
            q = Q[lab]; pk = dict(c=col[lab], lw=2.8 if lab == 'ours (group)' else 1.3, label=OV_NAME[lab])
            ax[r_, 0].plot(q.cum_t, 100 * q.closed_acc, **pk); ax[r_, 1].plot(q.cum_t, q.det_auroc, **pk)
        for c_, (yl, tt, I_, n_) in enumerate([('泛化（目標域 cartoon 閉集準確率 %）', '泛化 vs 累積傳輸時間', IA, OV_NOISE['acc']),
                                               ('面檢測（AUROC）', '面檢測 vs 累積傳輸時間', IZ, OV_NOISE['z'])]):
            a = ax[r_, c_]; a.axvline(Tb, c='k', ls=':', lw=1); a.text(Tb, 0.98, ' 同傳輸時間點', fontsize=8, va='top', transform=a.get_xaxis_transform())
            a.axhspan(I_ - n_, I_ + n_, color='k', alpha=0.07, lw=0); a.axhline(I_, c='k', ls='--', lw=1, label='全員同速（理想）± 雜訊寬度')
            a.set_xlabel('每台平均累積傳輸時間（送一次＝1÷自己網速）'); a.set_ylabel(yl); a.set_title(f'{OV_SC[sc]}：{tt}', loc='left'); a.grid(alpha=.3)
            a.set_xlim(0, max(COST[l]['total'] for l in labs) * 1.03)
        ax[r_, 0].set_ylim(60, 83); ax[r_, 1].set_ylim(0.62, 0.86); ax[r_, 0].legend(fontsize=7.5, loc='lower right')
        sl = OV_SLOW[sc]; a = ax[r_, 2]; xs = np.arange(3); w = 0.8 / len(labs); gl = []
        a.axhspan(-OV_NOISE['src'], OV_NOISE['src'], color='k', alpha=0.07, lw=0)
        for k, lab in enumerate(labs):
            gl.append(f'{OV_NAME[lab]}：泛化 {100 * at(Q[lab], Tb, "closed_acc") - IA:+.2f}｜面 {at(Q[lab], Tb, "det_auroc") - IZ:+.4f}')
            if lab not in SRC: continue
            dv = [SRC[lab][d] - IS[d] for d in DOM3]; xb = xs + (k - len(labs) / 2 + .5) * w
            a.bar(xb, dv, w, color=col[lab], label=OV_NAME[lab])
            for x_, v_ in zip(xb, dv):
                if abs(v_) >= 1: a.text(x_, v_ + (0.3 if v_ > 0 else -0.3), f'{v_:+.1f}', ha='center', va='bottom' if v_ > 0 else 'top', fontsize=7)
        a.axhline(0, c='k', lw=.8); a.set_xticks(xs); a.set_xticklabels([OV_DN[d] + ('（慢）' if d == sl else '') for d in DOM3])
        a.set_ylim(-16, 6.5); a.grid(alpha=.3, axis='y')
        a.text(0.02, 0.03, '同傳輸時間點離理想\n' + '\n'.join(gl), transform=a.transAxes, fontsize=7.5, va='bottom',
               bbox=dict(boxstyle='round', fc='white', ec='0.7'))
        a.set_ylabel('離理想的差距（分；0＝全員同速，灰帶＝雜訊寬度）'); a.set_title(f'{OV_SC[sc]}：各畫風原圖準確率離理想多少（顏色同左圖）', loc='left')
    fig.tight_layout(); fig.savefig(ROOT + fig_path, dpi=120); print(f'\n圖：{fig_path}')


def swing_section(fig_path='results/proto_trigger/1021_contrib2_swing.png'):
    """[1021 三情境擺盪圖，dany 1021] 三個集中慢情境下，各方法離「全員同速（理想）」多少：泛化、面檢測、最差畫風原圖。
    主張「目標未知時照網速送是在賭」用。數字與 overview_section 同源（fold_section 的 collect），同傳輸時間點取值。"""
    import io, contextlib
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'font.size': 10})
    C = {}
    with contextlib.redirect_stdout(io.StringIO()): fold_section(collect=C)
    at = lambda q, t, col: float(np.interp(t, q.cum_t, q[col])) if t <= q.cum_t.iloc[-1] else float(q[col].iloc[-1])
    DOM3 = ['art_painting', 'photo', 'sketch']
    _iq = C['normal']['Q']['uniform-full C0']; IA, IZ = 100 * _iq.closed_acc.loc[200], _iq.det_auroc.loc[200]
    IS = C['normal']['SRC']['uniform-full C0']
    SCS = ['sketch-slow', 'photo-slow', 'art-slow']
    sty = {'ours (group)': dict(c='#1f3b73', lw=3.2, ls='-', marker='o', ms=8, zorder=5),
           'no-comp': dict(c='#7f7f7f', lw=2.2, ls='-', marker='s', ms=7, zorder=4),
           'EF-HC one-way': dict(c='#d62728', lw=1.3, ls='-', marker='^', ms=6),
           'EF-HC two-way': dict(c='#ff7f0e', lw=1.3, ls='-', marker='v', ms=6),
           'PushCen': dict(c='#2ca02c', lw=1.3, ls='-', marker='D', ms=5),
           'uniform-matched': dict(c='#9467bd', lw=1.3, ls='--', marker='x', ms=6)}
    note = {'EF-HC one-way': '（多存 46 MB）', 'EF-HC two-way': '（傳輸 2.7–3.8 倍、多存 46 MB）', 'uniform-matched': '（慢設備多付約 2 倍）'}
    G = {lab: {'acc': [], 'z': [], 'worst': []} for lab in sty}
    for sc in SCS:
        Q, SRC, Tb = C[sc]['Q'], C[sc]['SRC'], C[sc]['Tb']
        for lab in sty:
            G[lab]['acc'].append(100 * at(Q[lab], Tb, 'closed_acc') - IA if lab in Q else np.nan)
            G[lab]['z'].append(at(Q[lab], Tb, 'det_auroc') - IZ if lab in Q else np.nan)
            G[lab]['worst'].append(min(SRC[lab][d] - IS[d] for d in DOM3) if lab in SRC else np.nan)
    pan = [('acc', '泛化（目標域 cartoon）', '離理想（分）', OV_NOISE['acc'], (-4.0, 2.2), '{:+.2f}'),
           ('z', '面檢測', '離理想（AUROC）', OV_NOISE['z'], (-0.050, 0.012), '{:+.4f}'),
           ('worst', '最差畫風的原圖準確率（三畫風中離理想最低者）', '離理想（分）', OV_NOISE['src'], (-14.5, 1.8), '{:+.1f}')]
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.2))
    xs = np.arange(3)
    print('三情境離理想多少（同傳輸時間點；理想＝全員同速）')
    for a, (key, ttl, yl, n_, yl_, fmt) in zip(ax, pan):
        a.axhspan(-n_, n_, color='k', alpha=0.08, lw=0); a.axhline(0, c='k', lw=1, ls='--')
        for lab, kw in sty.items():
            v = G[lab][key]
            a.plot(xs, v, label=OV_NAME[lab] + note.get(lab, ''), **kw)
            if lab in ('ours (group)', 'no-comp'):
                for x_, y_ in zip(xs, v):
                    a.annotate(fmt.format(y_), (x_, y_), textcoords='offset points', xytext=(9, -4 if lab == 'no-comp' else 6),
                               fontsize=9, color=kw['c'], fontweight='bold')
        o, n = np.array(G['ours (group)'][key]), np.array(G['no-comp'][key])
        bx = (0.40, 0.30) if key == 'worst' else (0.02, 0.03)
        a.text(*bx, f'按群校正：平均 {fmt.format(o.mean())}｜最差 {fmt.format(o.min())}\n照網速送：平均 {fmt.format(n.mean())}｜最差 {fmt.format(n.min())}',
               transform=a.transAxes, fontsize=9, va='bottom', bbox=dict(boxstyle='round', fc='white', ec='0.7'))
        a.set_xticks(xs); a.set_xticklabels([OV_SC[s] for s in SCS]); a.set_xlim(-0.3, 2.6); a.set_ylim(*yl_)
        a.set_title(ttl, loc='left'); a.set_ylabel(yl + '；0＝全員同速，灰帶＝雜訊寬度'); a.grid(alpha=.3)
        print(f'  [{ttl}]')
        for lab in sty:
            v = np.array(G[lab][key]); print(f'    {OV_NAME[lab]:14s} ' + '  '.join(fmt.format(x) for x in v) + f'｜平均 {fmt.format(np.nanmean(v))}｜最差 {fmt.format(np.nanmin(v))}｜擺幅 {fmt.format(np.nanmax(v) - np.nanmin(v))}')
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('三個集中慢情境：各方法離「全員同速（理想）」多少（cartoon 一折、單一隨機起點、同傳輸時間點）', x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96)); fig.savefig(ROOT + fig_path, dpi=130); print(f'\n圖：{fig_path}')


def pareto_section(fig_path='results/proto_trigger/1021_contrib2_cost_vs_outcome.png'):
    """[1021 成本對結果散點圖，dany 1021] 橫軸＝慢設備每台傳輸時間 ÷ 按群校正；縱軸＝離「全員同速（理想）」多少
    （最差畫風原圖、泛化、面檢測）。三個集中慢情境各一點。門檻同 plan §7.2：成本 1.3 倍、雜訊寬度。數字與 overview 同源。"""
    import io, contextlib
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'font.size': 10})
    C = {}
    with contextlib.redirect_stdout(io.StringIO()): fold_section(collect=C)
    at = lambda q, t, col: float(np.interp(t, q.cum_t, q[col])) if t <= q.cum_t.iloc[-1] else float(q[col].iloc[-1])
    DOM3 = ['art_painting', 'photo', 'sketch']
    _iq = C['normal']['Q']['uniform-full C0']; IA, IZ = 100 * _iq.closed_acc.loc[200], _iq.det_auroc.loc[200]
    IS = C['normal']['SRC']['uniform-full C0']
    SCS = ['sketch-slow', 'photo-slow', 'art-slow']; MK = {'sketch-slow': 'o', 'photo-slow': 's', 'art-slow': '^'}
    col = {'ours (group)': '#1f3b73', 'no-comp': '#7f7f7f', 'EF-HC one-way': '#d62728', 'EF-HC two-way': '#ff7f0e',
           'PushCen': '#2ca02c', 'uniform-matched': '#9467bd'}
    note = {'EF-HC one-way': '（多存 46 MB）', 'EF-HC two-way': '（多存 46 MB）'}
    P = []
    for sc in SCS:
        Q, SRC, COST, Tb = C[sc]['Q'], C[sc]['SRC'], C[sc]['COST'], C[sc]['Tb']; s0 = COST['ours (group)']['slow']
        for lab in col:
            if lab not in Q: continue
            P.append(dict(sc=sc, lab=lab, x=COST[lab]['slow'] / s0, worst=min(SRC[lab][d] - IS[d] for d in DOM3),
                          acc=100 * at(Q[lab], Tb, 'closed_acc') - IA, z=at(Q[lab], Tb, 'det_auroc') - IZ))
    pan = [('worst', '③ 最差畫風的原圖準確率', '離理想（分）', OV_NOISE['src'], (-14.5, 2.5)),
           ('acc', '④ 泛化（目標域 cartoon）', '離理想（分）', OV_NOISE['acc'], (-4.0, 2.2)),
           ('z', '④ 面檢測', '離理想（AUROC）', OV_NOISE['z'], (-0.050, 0.012))]
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.4))
    print('成本對結果（x＝慢設備每台傳輸時間 ÷ 按群校正；y＝離全員同速）')
    for a, (key, ttl, yl, n_, ylim) in zip(ax, pan):
        a.add_patch(plt.Rectangle((0.3, -n_), 1.3 - 0.3, 2 * n_, color='#2ca02c', alpha=0.10, lw=0))
        a.text(0.33, -n_, ' 條件全過的區域', fontsize=9, color='#1a7a1a', va='bottom')
        a.axhspan(-n_, n_, color='k', alpha=0.05, lw=0); a.axhline(0, c='k', lw=1, ls='--')
        a.axvline(1.3, c='k', lw=1, ls=':'); a.text(1.32, ylim[0] + 0.03 * (ylim[1] - ylim[0]), '成本門檻 1.3 倍', fontsize=8.5)
        for p in P:
            big = p['lab'] == 'ours (group)'; hollow = p['lab'] in note   # 空心＝要多存 46 MB（條件②不過）
            xj = p['x'] * {'no-comp': 0.955, 'PushCen': 1.045}.get(p['lab'], 1.0)   # 三者都恰為 1.00 倍，左右錯開以免重疊
            a.scatter(xj, p[key], marker=MK[p['sc']], s=150 if big else 70, facecolors='none' if hollow else col[p['lab']],
                      edgecolors='k' if big else col[p['lab']], linewidths=1.6 if hollow else 1.2, zorder=5 if big else 3)
        a.set_xscale('log'); a.set_xlim(0.3, 10); a.set_xticks([0.5, 1, 1.3, 2, 4, 8]); a.set_xticklabels(['0.5', '1', '1.3', '2', '4', '8'])
        a.set_ylim(*ylim); a.grid(alpha=.3, which='both')
        a.set_xlabel('慢設備每台傳輸時間（倍，按群校正＝1）'); a.set_ylabel(yl + '；0＝全員同速，灰帶＝雜訊寬度'); a.set_title(ttl, loc='left')
        print(f'  [{ttl}]')
        for p in P: print(f'    {OV_SC[p["sc"]]:9s} {OV_NAME[p["lab"]]:14s} x={p["x"]:.2f} y={p[key]:+.4f}')
    h = [Line2D([], [], ls='', marker='o', ms=9, mfc='none' if l in note else c, mec='k' if l == 'ours (group)' else c, mew=1.6,
                label=OV_NAME[l] + ('（空心＝多存 46 MB）' if l in note else '')) for l, c in col.items()]
    h += [Line2D([], [], ls='', marker=MK[s], ms=8, mfc='white', mec='k', label=OV_SC[s]) for s in SCS]
    fig.legend(handles=h, loc='lower center', ncol=9, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('花多少、得多少：慢設備的傳輸負擔 vs 離「全員同速（理想）」多少（cartoon 一折、單一隨機起點、同傳輸時間點；按群校正、照網速送、PushCen 都恰為 1 倍，左右錯開只為看清楚）',
                 x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96)); fig.savefig(ROOT + fig_path, dpi=130); print(f'\n圖：{fig_path}')


def final_table_section():
    """[1021 結果報告主表，dany 1021] 每個情境一張表：第一列＝全員同速（理想），其傳輸時間＝在該情境網速表下照理想送法
    （1007 C：九台同週期 6.2133）所需；其餘方法取跑完全程（ep200）。欄：單台最長傳輸時間（九台中累積傳輸時間最長那台，
    ×按群校正）、每台平均傳輸時間、最後泛化、最後面檢測、本地風格掉最多的那台（每台自己的模型＋自己的 BN 在自己畫風原圖上，
    減理想的同一台）。數字與 overview 同源（fold_section 的 collect）。"""
    import io, os, contextlib
    C = {}
    with contextlib.redirect_stdout(io.StringIO()): fold_section(collect=C)
    iq = C['normal']['Q']['uniform-full C0']; IA, IZ = 100 * iq.closed_acc.loc[200], iq.det_auroc.loc[200]

    def own(sf, sk):
        t = pd.read_csv(ROOT + sf); t = t[(t.arm == sk) & (t.bn == 'own') & (t.node_style == t.eval_domain)]
        t['i'] = t.node.str[5:].astype(int); return t.sort_values('i').closed_acc.values * 100
    IO = own(*FOLD_SCEN['normal'][1]['uniform-full C0'][5])
    c0 = pd.read_csv(ROOT + FOLD_SCEN['normal'][1]['uniform-full C0'][0] + '/async_diag/broadcast_log.csv')
    c0['i'] = c0.domain.str[5:].astype(int); c0s = c0.groupby('i').fired.sum().sort_index().values
    print(f'理想（全員同速＝1007 C）：泛化 {IA:.2f}｜面檢測 {IZ:.4f}｜逐台本地風格 {np.round(IO, 1).tolist()}｜逐台送出 {c0s.tolist()}')
    print('欄定義：單台最長傳輸時間＝九台中累積傳輸時間（送一次＝1÷自己網速；雙向含回傳，記帳同 EF-HC 原文）最長那台；理想列＝在該情境網速表下照理想送法所需')
    for sc in ['sketch-slow', 'photo-slow', 'art-slow', 'normal']:
        d = C.get(sc)
        if not d or 'ours (group)' not in d['Q']: continue
        b = d['b']; Q, COST = d['Q'], d['COST']; m0 = COST['ours (group)']['per_node'].max(); t0 = c0s / b
        sl = [f'n{i}' for i in range(9) if b[i] <= 0.1 + 1e-9]
        print(f'\n## {OV_SC[sc]}｜網速 node_0–8 = {np.round(b, 2).tolist()}｜慢設備 {"、".join(sl)}')
        SRC = d['SRC']; IS = C['normal']['SRC']['uniform-full C0']; ss = OV_SLOW[sc]; DOM3 = ['art_painting', 'photo', 'sketch']
        ca = f'{OV_DN[ss]} 原圖（慢畫風）' if ss else '慢畫風原圖'
        print(f'| 方法 | 單台最長傳輸時間 | 最後泛化 | 最後面檢測 | {ca} | 掉最多的畫風 |')   # dany 1021：數值（±相對 baseline）；時間用 %、其他用分數差
        print('|---|---|---|---|---|---|')
        T0 = t0.max()
        print(f'| **全員同速（baseline）** | {T0:,.0f}＊ | {IA:.2f} | {IZ:.4f} | {IS[ss]:.2f} | — |' if ss else f'| **全員同速（baseline）** | {T0:,.0f}＊ | {IA:.2f} | {IZ:.4f} | — | — |')
        nodes, avgs = [], []
        for lab in ['ours (group)', 'no-comp', 'EF-HC one-way', 'EF-HC two-way', 'PushCen', 'uniform-matched', 'base (ablation)']:
            if lab not in Q: continue
            pn = COST[lab]['per_node']; q = Q[lab]; xa = 100 * q.closed_acc.loc[200]; xz = q.det_auroc.loc[200]
            if lab in SRC:
                sr = SRC[lab]; g = {k_: sr[k_] - IS[k_] for k_ in DOM3}; kw = min(g, key=g.get)
                c4 = f'{sr[ss]:.2f}（{g[ss]:+.2f}）' if ss else '—'; c5 = f'{OV_DN[kw]} {sr[kw]:.2f}（{g[kw]:+.2f}）'
            else:
                c4 = c5 = '檔案不存在'
            sf, sk = FOLD_SCEN[sc][1][lab][5]
            if os.path.exists(ROOT + sf):
                v = own(sf, sk); dv = v - IO; k = int(np.argmin(dv)); nodes.append(f'{OV_NAME[lab]} n{k} {v[k]:.1f}（{dv[k]:+.1f}）')
            avgs.append(f'{OV_NAME[lab]} {pn.mean():,.0f}')
            nm = OV_NAME[lab] + ('（多存 46 MB）' if COST[lab]['store'] else '')
            print(f'| {nm} | {pn.max():,.0f}（{100 * (pn.max() - T0) / T0:+.0f}%） | {xa:.2f}（{xa - IA:+.2f}） | {xz:.4f}（{xz - IZ:+.4f}） | {c4} | {c5} |')
        print(f'　　＊ baseline 的傳輸時間＝在這個網路照理想送法（九台每 6.2 步送一次）所需，最忙的是網速 0.1 的設備；成績就是理想值（網速只記帳、不影響訓練）。')
        print(f'　　原圖＝ep200、BN 平均B、九台平均；baseline art {IS["art_painting"]:.2f}／photo {IS["photo"]:.2f}／sketch {IS["sketch"]:.2f}。掉最多的畫風＝三個畫風中比 baseline 掉最多（或升最少）的那個：是慢畫風自己＝被沖淡，是別的畫風＝被轉嫁。')
        print(f'　　補充（每台自己的模型＋自己的 BN，量自己畫風原圖，比 baseline 同一台掉最多的那台；baseline 逐台 {np.round(IO, 1).tolist()}）：' + '、'.join(nodes))
        print(f'　　每台平均傳輸時間：baseline {t0.mean():,.0f}、' + '、'.join(avgs))


def early_section(fig_path='results/proto_trigger/1027_contrib2_early_phase.png'):
    """[1027 dany] 前期收斂代價：四情境下各方法的泛化／面檢測逐 epoch 軌跡、與全員同速（理想）的差、分段平均、追平時點。
    數字與 overview_section 同源（fold_section 的 collect）；理想＝全員同速（1009 C0-dense）；九台平均、BN 平均B、每 5 epoch 一點。
    追平時點＝之後每個點都不再比理想低超過雜訊寬度（泛化 1.09、面 0.0144）的第一個 epoch。"""
    import io, contextlib
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'font.size': 9})
    C = {}
    with contextlib.redirect_stdout(io.StringIO()): fold_section(collect=C)
    I = C['normal']['Q']['uniform-full C0']; IA, IZ = 100 * I.closed_acc, I.det_auroc
    SCS = ['sketch-slow', 'photo-slow', 'art-slow', 'normal']
    SEG = [(5, 50), (55, 100), (105, 150), (155, 200)]
    LABS = ['ours (group)', 'no-comp', 'uniform-matched', 'EF-HC one-way', 'EF-HC two-way', 'PushCen']
    # dany 1027：圖只畫三個集中慢情境，比較對象＝主對手 EF-HC（單向／雙向）與 PushCen（表仍印四情境、全部方法）
    FIG_SCS = ['sketch-slow', 'photo-slow', 'art-slow']
    PLOT = {'ours (group)': dict(c='#2a78d6', lw=2.4, ls='-', zorder=5), 'EF-HC one-way': dict(c='#eb6834', lw=1.4, ls='-', zorder=4),
            'EF-HC two-way': dict(c='#1baf7a', lw=1.4, ls='-.', zorder=3), 'PushCen': dict(c='#eda100', lw=1.4, ls=(0, (4, 1.5)), zorder=3)}
    # dataviz 驗證：四格相鄰對全通過（最差 CVD ΔE 9.1）；湖綠 2.74、黃 2.11 對比不足 ⇒ 加線型＋圖例＋報告表格
    nA, nZ = OV_NOISE['acc'], OV_NOISE['z']

    def catch(g, n):
        bad = g.index[g < -n]
        return 5 if len(bad) == 0 else (int(bad.max()) + 5 if bad.max() < 200 else None)

    print('前期收斂：各方法 − 全員同速（理想）；泛化單位＝分（%），面檢測＝AUROC；九台平均、BN 平均B')
    print(f'理想分段：泛化 ' + '／'.join(f'ep{a}–{b} {IA.loc[a:b].mean():.2f}' for a, b in SEG) + f'／ep200 {IA.loc[200]:.2f}'
          + '｜面 ' + '／'.join(f'{IZ.loc[a:b].mean():.4f}' for a, b in SEG) + f'／{IZ.loc[200]:.4f}')
    for sc in SCS:
        Q = C[sc]['Q']
        print(f'\n## {OV_SC[sc]}')
        print('| 方法 | 泛化 ep5–50 | ep55–100 | ep105–150 | ep155–200 | ep200 | 泛化追平 | 面 ep5–50 | ep55–100 | ep105–150 | ep155–200 | ep200 | 面追平 |')
        print('|---|' + '---|' * 12)
        for lab in LABS:
            if lab not in Q: continue
            A, Z = 100 * Q[lab].closed_acc, Q[lab].det_auroc
            ca, cz = catch(A - IA, nA), catch(Z - IZ, nZ)
            sa = [f'{A.loc[a:b].mean():.2f}（{A.loc[a:b].mean() - IA.loc[a:b].mean():+.2f}）' for a, b in SEG] + [f'{A.loc[200]:.2f}（{A.loc[200] - IA.loc[200]:+.2f}）']
            sz = [f'{Z.loc[a:b].mean():.4f}（{Z.loc[a:b].mean() - IZ.loc[a:b].mean():+.4f}）' for a, b in SEG] + [f'{Z.loc[200]:.4f}（{Z.loc[200] - IZ.loc[200]:+.4f}）']
            print(f'| {OV_NAME[lab]} | ' + ' | '.join(sa) + f' | {"ep" + str(ca) if ca else "未追平"} | ' + ' | '.join(sz) + f' | {"ep" + str(cz) if cz else "未追平"} |')
    fig, ax = plt.subplots(3, len(FIG_SCS), figsize=(16, 12.5), sharex=True)
    for c_, sc in enumerate(FIG_SCS):
        Q = C[sc]['Q']
        a0, a1, a2 = ax[0, c_], ax[1, c_], ax[2, c_]
        a0.plot(IA.index, IA.values, c='#73726c', lw=1.6, ls='--', label='全員同速（理想）', zorder=2)
        for a in (a1, a2): a.axhline(0, c='#73726c', lw=1.2, ls='--', zorder=2)
        a1.axhspan(-nA, nA, color='k', alpha=0.07, lw=0); a2.axhspan(-nZ, nZ, color='k', alpha=0.07, lw=0)
        for lab, kw in PLOT.items():
            if lab not in Q: continue
            A, Z = 100 * Q[lab].closed_acc, Q[lab].det_auroc
            a0.plot(A.index, A.values, label=OV_NAME[lab] + {'EF-HC one-way': '（多存 46 MB）', 'EF-HC two-way': '（多傳 5–11 倍、多存 46 MB）'}.get(lab, ''), **kw)
            a1.plot(A.index, (A - IA).values, label=OV_NAME[lab], **kw)
            a2.plot(Z.index, (Z - IZ).values, label=OV_NAME[lab], **kw)
            if lab == 'ours (group)':
                ca = catch(A - IA, nA)
                if ca:
                    a1.axvline(ca, c=kw['c'], lw=1, ls=':')
                    a1.text(ca, 0.03, f' 追平 ep{ca}' if ca < 200 else 'ep200 才回到雜訊內 ', color=kw['c'], fontsize=8,
                            ha='left' if ca < 200 else 'right', transform=a1.get_xaxis_transform())
                m_ = (A - IA).loc[5:50].mean()
                a1.text(0.98, 0.12, f'我們 ep5–50 平均 {A.loc[5:50].mean():.2f}\n（理想 {IA.loc[5:50].mean():.2f}，差 {m_:+.2f}）',
                        transform=a1.transAxes, ha='right', va='bottom', fontsize=8, color='#2b2b29')
        for a in (a0, a1, a2): a.axvspan(5, 50, color='#2a78d6', alpha=0.05, lw=0); a.grid(alpha=.25)
        a0.set_title(f'{OV_SC[sc]}：泛化（cartoon，%）', loc='left'); a1.set_title(f'{OV_SC[sc]}：泛化 − 理想（分；灰帶＝雜訊 ±{nA}）', loc='left')
        a2.set_title(f'{OV_SC[sc]}：面檢測 − 理想（灰帶＝±{nZ}）', loc='left'); a2.set_xlabel('epoch（淺藍底＝前期 ep5–50）')
        a0.legend(fontsize=7.5, loc='lower right')
    # 縱軸依資料定、三欄共用（不切掉任何點）
    for r_, (key, pad) in enumerate([('raw', 1.0), ('ga', 0.5), ('gz', 0.004)]):
        v = []
        for sc in FIG_SCS:
            for lab in [l for l in PLOT if l in C[sc]['Q']]:
                A, Z = 100 * C[sc]['Q'][lab].closed_acc, C[sc]['Q'][lab].det_auroc
                v += list(A.values) if key == 'raw' else list((A - IA).values) if key == 'ga' else list((Z - IZ).values)
            if key == 'raw': v += list(IA.values)
        lo, hi = min(v) - pad, max(v) + pad
        if key == 'ga': lo, hi = min(lo, -nA - pad), max(hi, nA + pad)
        if key == 'gz': lo, hi = min(lo, -nZ - pad), max(hi, nZ + pad)
        for c_ in range(len(FIG_SCS)): ax[r_, c_].set_ylim(lo, hi)
        print(f'縱軸 {key}: {lo:.4g} ～ {hi:.4g}')
    fig.tight_layout(); fig.savefig(ROOT + fig_path, dpi=130); print(f'\n圖：{fig_path}')


EF31_TABLES = {   # [plan 1031] 照 EF-HC 原文亂數網速：表 → (網速表檔名, {組: (E 鍵, 估算規則, 雙向, 要多存 46 MB)})
    t: (f'bandwidth_{f}.csv', {'ours (group)': (f'E{k}', 'group', False, False), 'uniform-matched': (f'E{k + 1}', 'mh', False, False),
                               'PushCen': (f'E{k + 2}', 'pushcen', False, False), 'EF-HC two-way': (f'E{k + 3}', 'mh', True, True)})
    for t, f, k in [('u0', 'uni_s0', 1), ('u1', 'uni_s1', 5), ('b0', 'beta_s0', 9), ('b1', 'beta_s1', 13)]}
EF31_NAME = {'u0': '均勻 種子0', 'u1': '均勻 種子1', 'b0': '兩頭多 種子0', 'b1': '兩頭多 種子1'}
EF31_SKEWED = ('u0', 'b0', 'b1')   # plan §7.2：照網速送預測最吃虧份量 <0.25 的偏心表（事前）


def efhc1031_section():
    """[plan 1031 §7.2] 照 EF-HC 原文亂數網速 × 按群校正／同週期送／PushCen／EF-HC 雙向。邏輯沿用 fold_section：
    同傳輸時間點 T*＝按群校正全程的每台平均累積傳輸時間（送一次＝1÷送出方網速；雙向含回傳，記帳同 EF-HC 原文），各組在 T* 內插、
    總時間不到 T* 用最終值。成本較高＝單台最長傳輸時間 ≥1.3 倍或要多存 46 MB（plan §7.2）。公平＝各畫風原圖準確率（ep200、平均B、九台平均）
    相對全員同速（1007 C）的最大掉分，寬 1.54。主表另列 ep200 與全員同速（理想；其傳輸時間＝在該表網速下照理想送法所需）。"""
    import os, sys, re
    import networkx as nx
    sys.path.insert(0, ROOT + 'scripts/posthoc')
    from composition_inbox_sim import simulate
    src_py = open(ROOT + 'scripts/posthoc/source_domain_clean_accuracy.py', encoding='utf-8').read()
    ENAME = dict(re.findall(r"'(E\d+)': '([^']+)' \+ S", src_py))
    ev = pd.concat([pd.read_csv(ROOT + _FD + f'{c}_curve_cartoon_avgbnB.csv') for c in ('1009_bw', '1031_efhc')], ignore_index=True)
    ev = ev[ev.score_fn == 'zperp'].copy(); ev['ep'] = ev.ckpt.str.replace('epoch_', '').astype(int)
    ev = ev.drop_duplicates(['run', 'ep', 'node'], keep='last')
    G = nx.random_geometric_graph(9, 0.8, seed=1234); NBR = {i: sorted(G[i]) for i in range(9)}; DEG = {i: len(NBR[i]) for i in range(9)}
    wg, wf, wfair = 0.0109, 0.0144, 1.54
    rs = pd.read_csv(ROOT + _FD + '1012_artslow_source_clean_acc.csv'); rs = rs[(rs.arm == 'C0') & (rs.bn == 'avgB')]
    REF = rs.groupby('eval_domain').closed_acc.mean() * 100
    iq = ev[ev.run == BW_ARMS['C0'][1].replace('exp_result_', '', 1)].groupby('ep')[['closed_acc', 'det_auroc']].mean()
    IA, IZ = 100 * iq.closed_acc.loc[200], iq.det_auroc.loc[200]
    c0 = pd.read_csv(ROOT + BW_ARMS['C0'][1] + '/async_diag/broadcast_log.csv'); c0['i'] = c0.domain.str[5:].astype(int)
    c0s = c0.groupby('i').fired.sum().sort_index().values
    at = lambda q, t, col: float(np.interp(t, q.cum_t, q[col])) if t <= q.cum_t.iloc[-1] else float(q[col].iloc[-1])
    DOM3 = ['art_painting', 'photo', 'sketch']
    print(f'理想（全員同速＝1007 C）：泛化 {IA:.2f}｜面 {IZ:.4f}｜原圖 art {REF["art_painting"]:.2f} photo {REF["photo"]:.2f} sketch {REF["sketch"]:.2f}')
    OUT = {}
    for t, (bf, arms) in EF31_TABLES.items():
        bt = pd.read_csv(ROOT + _FD + '1031_efhc_bw/' + bf); bt['i'] = bt.node.str[5:].astype(int); b_ = bt.sort_values('i').b.values.astype(float)
        Q, COST, SH, SRC = {}, {}, {}, {}
        for lab, (ek, rule, bd, store) in arms.items():
            name = ENAME[ek] + _BWS; run = 'exp_result_' + name
            q = ev[ev.run == name].groupby('ep')[['closed_acc', 'det_auroc']].mean()
            if len(q) != 40: continue
            b = pd.read_csv(ROOT + run + '/async_diag/broadcast_log.csv'); b['i'] = b.domain.str[5:].astype(int)
            Fm = b.pivot(index='step', columns='i', values='fired').sort_index().values.astype(bool)
            if bd:
                V = np.zeros_like(Fm, dtype=float)
                for i in range(9): V[:, i] = sum(np.maximum(Fm[:, i], Fm[:, j]) for j in NBR[i]) / DEG[i]
            else:
                V = Fm.astype(float)
            cost = V / b_[None, :]
            ep_t = pd.Series(cost.sum(1) / 9).groupby(np.arange(len(cost)) // 20).sum(); ep_t.index += 1
            q['cum_t'] = ep_t.cumsum().reindex(q.index).values; Q[lab] = q
            COST[lab] = dict(total=ep_t.sum(), per_node=cost.sum(0), store=store, sends=Fm.sum(0), run=run, ek=ek)
            if rule == 'group':
                g = pd.read_csv(ROOT + run + '/async_diag/rate_group.csv').groupby('node').first()
                SH[lab] = simulate(Fm[:4000], 'rate_pair', False, u=np.array([g.loc[f'node_{i}', 'u'] for i in range(9)]))
            else:
                SH[lab] = simulate(Fm[:4000], rule, bd)
            sf = ROOT + _FD + f'1031_efhc_{ek}_source_clean_acc.csv'
            if os.path.exists(sf):
                d = pd.read_csv(sf); d = d[(d.arm == ek) & (d.bn == 'avgB')]
                if len(d): SRC[lab] = d.groupby('eval_domain').closed_acc.mean() * 100
        print('\n' + '#' * 110); print(f'# {EF31_NAME[t]}（{t}）｜網速 {np.round(b_, 4).tolist()}｜完成 {list(Q)}'); print('#' * 110)
        if 'ours (group)' not in Q:
            print('  按群校正尚未完成，略過'); continue
        ref = 'ours (group)'; Tb = COST[ref]['total']; M0 = COST[ref]['per_node'].max(); T0 = (c0s / b_).max()
        print(f'| 方法 | 單台最長傳輸時間（×按群） | 每台平均傳輸 | 泛化 ep200 | 面 ep200 | 泛化／面 @T*={Tb:.0f} | 原圖 art／photo／sketch（相對理想） | 最大掉分 | 實際份量 art／photo／sketch |')
        print('|---|---|---|---|---|---|---|---|---|')
        print(f'| 全員同速（理想） | {T0:,.0f}＊（{T0 / M0:.2f}×） | {(c0s / b_).mean():,.0f} | {IA:.2f} | {IZ:.4f} | — | {REF["art_painting"]:.2f}／{REF["photo"]:.2f}／{REF["sketch"]:.2f} | — | — |')
        md = {}
        for lab in Q:
            c = COST[lab]; q = Q[lab]; pn = c['per_node']; sh = SH[lab]['share']
            if lab in SRC:
                sr = SRC[lab]; dr = {k: REF[k] - sr[k] for k in DOM3}; md[lab] = max(dr.values())
                s_txt = '／'.join(f'{sr[k]:.2f}（{sr[k] - REF[k]:+.2f}）' for k in DOM3); m_txt = f'{md[lab]:+.2f}'
            else:
                s_txt = m_txt = '尚無'
            print(f'| {OV_NAME[lab]}{"（多存 46 MB）" if c["store"] else ""} | {pn.max():,.0f}（{pn.max() / M0:.2f}×） | {pn.mean():,.0f} | {100 * q.closed_acc.loc[200]:.2f} | {q.det_auroc.loc[200]:.4f}'
                  f' | {100 * at(q, Tb, "closed_acc"):.2f}／{at(q, Tb, "det_auroc"):.4f} | {s_txt} | {m_txt} | ' + '／'.join(f'{x:.3f}' for x in sh) + ' |')
        print(f'  逐台送出：' + '｜'.join(f'{OV_NAME[l]} {COST[l]["sends"].tolist()}' for l in Q))
        V_ = {}
        qa = Q[ref]
        for lab, q in Q.items():
            if lab == ref: continue
            da = at(qa, Tb, 'closed_acc') - at(q, Tb, 'closed_acc'); dz = at(qa, Tb, 'det_auroc') - at(q, Tb, 'det_auroc')
            c = COST[lab]; mx = c['per_node'].max() >= 1.3 * M0
            higher = mx or c['store']
            better = (da > wg) or (dz > wf); worse = (da < -wg) or (dz < -wf)
            if better and not worse: v = '按群勝'
            elif not better and not worse: v = '持平、按群成本較低' if higher else '持平'
            elif worse and not better: v = '按群輸' if not higher else '對手成績較好、但成本較高（取捨）'
            else: v = '取捨（一高一低）'
            fair = '—'
            if lab in md and ref in md:
                fd = md[lab] - md[ref]; fair = '按群較公平' if fd > wfair else ('對手較公平' if fd < -wfair else '相同')
                fair += f'（最大掉分 對手 {md[lab]:+.2f} vs 我們 {md[ref]:+.2f}）'
            why = [n for n, f in [('單台最長 ≥1.3×', mx), ('多存 46 MB', c['store'])] if f]
            V_[lab] = (v, fair)
            print(f'  [{OV_NAME[lab]}] @T* 泛化差 {da:+.4f} 面差 {dz:+.4f}｜對手成本較高：{"、".join(why) if why else "否"} ⇒ {v}｜公平：{fair}')
            if c['run'] and os.path.exists(ROOT + c['run'] + '/async_diag/reply_log.csv'):
                rp = pd.read_csv(ROOT + c['run'] + '/async_diag/reply_log.csv'); rp['y'] = rp.replier.str[5:].astype(int)
                nrep = rp.groupby('y').size().reindex(range(9)).fillna(0).astype(int).values; own_ = c['sends']; k_ = int(b_.argmin())
                print(f'      EF-HC 雙向每台上傳：自己送出 {own_.tolist()}＋回傳 {nrep.tolist()}；最慢 node_{k_}（b={b_[k_]:.4f}）回傳／自己送出 {nrep[k_] / max(own_[k_], 1):.1f}×')
        OUT[t] = V_
    print('\n' + '=' * 110); print('整份計畫四級判決（plan §7.2）'); print('=' * 110)
    full = all(t in OUT and len(OUT[t]) == 3 for t in EF31_TABLES)
    ok = [t for t in EF31_SKEWED if t in OUT and len(OUT[t]) == 3
          and OUT[t]['EF-HC two-way'][0] in ('持平、按群成本較低', '按群勝') and OUT[t]['uniform-matched'][0] in ('持平、按群成本較低', '按群勝')
          and (OUT[t]['PushCen'][0] == '按群勝' or OUT[t]['PushCen'][1].startswith('按群較公平'))]
    lose = any(v[0] == '按群輸' for t in OUT for v in OUT[t].values())
    if full:
        lv = '退化' if lose else ('強' if len(ok) == 3 else '中' if len(ok) == 2 else '邊際')
        print(f'  偏心表滿足的 {ok}｜任何「按群輸」：{lose} ⇒ {lv}')
    else:
        print(f'  尚未齊（已完成：{ {t: list(OUT[t]) for t in OUT} }）｜目前偏心表滿足 {ok}｜目前有「按群輸」：{lose}')


if '--efhc1031' in __import__('sys').argv:   # [plan 1031] 照 EF-HC 原文亂數網速 × 主對手
    efhc1031_section()
if '--early' in __import__('sys').argv:   # [1027] 前期收斂代價：四情境軌跡、與理想的差、追平時點
    early_section(); raise SystemExit(0)
if '--final_table' in __import__('sys').argv:   # [1021] 結果報告主表（單台最長傳輸時間）
    final_table_section(); raise SystemExit(0)
if '--pareto' in __import__('sys').argv:   # [1021] 成本對結果散點圖
    pareto_section(); raise SystemExit(0)
if '--swing' in __import__('sys').argv:   # [1021] 三情境擺盪圖
    swing_section(); raise SystemExit(0)
if '--overview' in __import__('sys').argv:   # [1019] 貢獻二目前跑完的組：表格＋曲線
    overview_section(); raise SystemExit(0)
if '--fold' in __import__('sys').argv:   # [plan 1019 一折全貌] 四情境 × 主對手
    fold_section(); raise SystemExit(0)
if '--r3b' in __import__('sys').argv:   # [plan 1019] 按群校正份量 vs 底座
    r3b_section(); raise SystemExit(0)
if '--comp' in __import__('sys').argv:   # [plan 1017] sketch 慢主表對手
    comp_section(); raise SystemExit(0)
if '--share_efhc' in __import__('sys').argv:   # [1017] 貢獻二的 EF-HC 圖 2 式呈現
    share_efhc_figure(); raise SystemExit(0)
if '--share' in __import__('sys').argv:   # [plan 1016] 份量平衡規則 A1–A3
    share_section(); raise SystemExit(0)
if '--curve' in __import__('sys').argv:
    curve_section(); raise SystemExit(0)
if '--bw' in __import__('sys').argv:
    bw_section(); raise SystemExit(0)
if '--efhc' in __import__('sys').argv:
    efhc_figure(); raise SystemExit(0)
if '--artslow' in __import__('sys').argv:
    artslow_section(); raise SystemExit(0)
if '--diag' in __import__('sys').argv:   # [plan 1014] sketch 慢拆解診斷
    diag_section(); raise SystemExit(0)
if '--efhc_scen' in __import__('sys').argv:   # [1014] art 慢／sketch 慢情境的 EF-HC 圖 2 式呈現
    efhc_scenario_figure(); raise SystemExit(0)
if '--sketchslow' in __import__('sys').argv:   # [plan 1013] 同 1012 判準；並列 ArtSlow
    artslow_section(curve_csv='results/proto_trigger/1013_sketchslow_curve_cartoon_avgbnB.csv',
                    fig_path='results/proto_trigger/1013_sketchslow_curve_cartoon.png',
                    X='SketchSlow', run_x=SKETCHSLOW_RUN, log_x='logs/proto_trigger/1013_sketchslow_train.log', style='sketch',
                    extra={'ArtSlow': (ARTSLOW_LOG, ARTSLOW_RUN, 'results/proto_trigger/1012_artslow_curve_cartoon_avgbnB.csv')})
    raise SystemExit(0)

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
