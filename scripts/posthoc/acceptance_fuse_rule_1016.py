"""[plan 1016] 融合規則的驗收：把短跑（約 140 步）的 async_diag 與參照 run 逐列比對。

  --mode neutral   ：fuse_rule=mh＋fusion_jump_log。broadcast_log 全部欄位須逐字相同（同時證明跳動記錄是唯讀的）。
  --mode pushsum   ：送出時機欄須逐字相同、step 0 delta 相同；依同一份 fired 逐步重建份量（送出 w/(d+1)、收件相加），
                     fusion_jump 的 w_in 須等於 Σ份額／(w_i＋Σ份額)；train log 的 [PUSHSUM] 份量總和須＝9、合併次數＝0。
  --mode pushcen   ：[plan 1017] 同 pushsum，另檢查 [PUSHSUM] 行 grad_scaled=False。
  --mode bidir     ：[plan 1017] EF-HC 雙向短跑的自我一致性（見 bidir_check），--ref 不使用。
  --mode rate_pair ：送出時機欄（fired／forced／n_pushed／n_received／n_style_nb）須逐字相同；
                     step 0 的 delta 須相同（第一次融合之前）；
                     fusion_jump.csv 每一列的「收進來的總比重 w_in」須等於解析值 Σ_j W_ij·min(r_i,r_j)/r_j
                     （送方集合照 train.py 事件順序由參照 run 的 fired 重建；r＝1/週期），且自己權重 1−w_in > 0。
  --mode rate_group：[plan 1019] 同 rate_pair，但解析值用 async_diag/rate_group.csv 的 u（Σ_j W_ij·min(u_i,u_j)/u_j）；
                     另查：組＝真實同畫風鄰居、u＝自己＋組員送出率的平均，並列出距離比（同組／跨組）。

用法：venv_matcha/bin/python scripts/posthoc/acceptance_fuse_rule_1016.py --ref <參照 run 目錄> --test <短跑目錄> --mode neutral|rate_pair
"""
import argparse, csv, sys
import numpy as np
import networkx as nx

COLS = ['step', 'domain', 'delta', 'thresh', 'gamma', 'fired', 'forced', 'n_pushed', 'n_received',
        'mean_staleness', 'n_style_nb', 'mean_style_age', 'mean_style_dist']
SCHED = ['fired', 'forced', 'n_pushed', 'n_received', 'n_style_nb']


def load(d):
    rows = list(csv.DictReader(open(f'{d}/async_diag/broadcast_log.csv')))
    by = {}
    for r in rows:
        by.setdefault(int(r['step']), {})[r['domain']] = r
    full = [k for k in sorted(by) if len(by[k]) == 9]
    return by, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True); ap.add_argument('--test', required=True)
    ap.add_argument('--mode', required=True, choices=['neutral', 'rate_pair', 'rate_group', 'pushsum', 'pushcen', 'bidir'])
    ap.add_argument('--train_log', default='')
    o = ap.parse_args()
    if o.mode == 'bidir':
        sys.exit(bidir_check(o.test))
    ra, fa = load(o.ref); rb, fb = load(o.test)
    steps = [k for k in fb if k in ra]
    # 只比「完整的步」，且去掉被中斷時可能寫一半的最後一步
    steps = steps[:-1] if len(steps) > 1 else steps
    doms = [f'node_{i}' for i in range(9)]
    print(f'比對步數 {len(steps)}（step {steps[0]}–{steps[-1]}）、列數 {len(steps) * 9}')
    cols = COLS if o.mode == 'neutral' else SCHED
    bad = 0
    for c in cols:
        eq = sum(ra[k][d][c] == rb[k][d][c] for k in steps for d in doms)
        print(f'  {c:<16} 逐字相同 {eq}/{len(steps) * 9}')
        bad += (eq != len(steps) * 9)
    if o.mode in ('rate_pair', 'rate_group', 'pushsum', 'pushcen'):
        d0 = sum(ra[0][d]['delta'] == rb[0][d]['delta'] for d in doms)
        print(f'  step 0 delta 逐字相同 {d0}/9'); bad += (d0 != 9)
        first = next((k for k in steps if any(ra[k][d]['delta'] != rb[k][d]['delta'] for d in doms)), None)
        print(f'  delta 第一次不同的步：{first}（應在第一次融合之後）')
        # 解析值
        bw = {r['node']: r for r in csv.DictReader(open(f'{o.test}/async_diag/bandwidth.csv'))}
        G = nx.random_geometric_graph(9, 0.8, seed=1234)
        assert [len(G[i]) for i in range(9)] == [int(bw[f'node_{i}']['degree']) for i in range(9)], '度數不符'
        rate = {i: 1.0 / float(bw[f'node_{i}']['period']) for i in range(9)}
        if o.mode == 'rate_group':  # [plan 1019] 解析值改用附的數 u
            rg = list(csv.DictReader(open(f'{o.test}/async_diag/rate_group.csv')))
            sty = {int(r['node'].split('_')[1]): r['style'] for r in rg}
            grp = {i: {int(x.split('_')[1]) for x in r['group'].split(';') if x} for i in range(9) for r in rg if int(r['node'].split('_')[1]) == i}
            truth = {i: {j for j in G[i] if sty[j] == sty[i]} for i in range(9)}
            u = {i: float(next(r['u'] for r in rg if int(r['node'].split('_')[1]) == i)) for i in range(9)}
            u_exp = {i: float(np.mean([rate[i]] + [rate[j] for j in sorted(grp[i])])) for i in range(9)}
            g_ok = all(grp[i] == truth[i] for i in range(9)); u_err = max(abs(u[i] - u_exp[i]) for i in range(9))
            print(f'  組＝真實同畫風：{g_ok}（' + ' '.join(f'{i}:{sorted(grp[i])}' for i in range(9)) + f'）｜u 重算最大誤差 {u_err:.2e}')
            rs = [float(r['ratio']) for r in rg if r['ratio'] and sty[int(r['neighbor'].split('_')[1])] == r['style']]
            rc = [float(r['ratio']) for r in rg if r['ratio'] and sty[int(r['neighbor'].split('_')[1])] != r['style']]
            if rs or rc:
                print(f'  距離比：同組 {len(rs)} 個 最大 {max(rs):.2f} 中位 {np.median(rs):.2f}｜跨組 {len(rc)} 個 最小 {min(rc):.2f} 中位 {np.median(rc):.2f}'
                      f'｜門檻 c＝{rg[0]["c"]}')
            bad += (not g_ok) or u_err > 1e-6
            rate = u
        fired = {k: {i: ra[k][f'node_{i}']['fired'] in ('1', 'True') for i in range(9)} for k in steps}
        jump = list(csv.DictReader(open(f'{o.test}/async_diag/fusion_jump.csv')))
        if o.mode in ('pushsum', 'pushcen'):
            # 逐步重建份量（照 train.py 事件順序：每台依序 訓練→送出→收件；編號小的同步送、編號大的上一步送）
            w = {i: 1.0 for i in range(9)}; share = {}; exp_rows = {}
            for k in steps:
                for i in range(9):
                    if fired[k][i]:
                        share[(k, i)] = w[i] / (len(G[i]) + 1); w[i] = share[(k, i)]
                    snd = [j for j in sorted(G[i]) if (fired[k][j] if j < i else (k > 0 and fired.get(k - 1, {}).get(j, False)))]
                    if snd:
                        sh = [share[(k if j < i else k - 1, j)] for j in snd]
                        exp_rows[(k, i)] = (sum(sh) / (w[i] + sum(sh)), len(snd)); w[i] += sum(sh)
            nchk = nbad = 0; worst = 0.0
            for r in jump:
                key = (int(r['step']), int(r['domain'].split('_')[1]))
                if key not in exp_rows: continue
                e, ns = exp_rows[key]; err = abs(float(r['w_in']) - e); worst = max(worst, err); nchk += 1
                nbad += (ns != int(r['n_senders'])) or err > 1e-5
            print(f'  推和法份量重建比對：{nchk} 次融合、不符 {nbad}、最大誤差 {worst:.2e}、重建後份量總和 {sum(w.values()):.6f}（未計在途）')
            bad += (nbad > 0) or nchk == 0
            if o.train_log:
                import re
                L = [l for l in open(o.train_log) if l.startswith('[PUSHSUM] epoch=')]
                tot = [float(re.search(r'total_incl_inflight=([0-9.]+)', l).group(1)) for l in L]
                mg = [int(re.search(r'merges=(\d+)', l).group(1)) for l in L]
                print(f'  [PUSHSUM] 行數 {len(L)}、份量總和 最小 {min(tot):.9f} 最大 {max(tot):.9f}、合併次數 {max(mg) if mg else "NA"}')
                bad += (not L) or abs(min(tot) - 9) > 1e-6 or abs(max(tot) - 9) > 1e-6 or (mg and max(mg) > 0)
                if o.mode == 'pushcen':  # [plan 1017] PushCen：梯度不縮放
                    gs = [re.search(r'grad_scaled=(\w+)', l) for l in L]
                    ok = all(g is not None and g.group(1) == 'False' for g in gs)
                    print(f'  grad_scaled 全為 False：{ok}（{len(gs)} 行）'); bad += (not ok)
            print('驗收：' + ('通過' if bad == 0 else f'未通過（{bad} 項）'))
            sys.exit(0 if bad == 0 else 1)
        jump = [r for r in jump if int(r['step']) in fired and int(r['step']) - 1 in fired or int(r['step']) == 0]
        nchk = nbad = 0; worst = 0.0; minself = 1.0
        for r in jump:
            k, i = int(r['step']), int(r['domain'].split('_')[1])
            snd = [j for j in sorted(G[i]) if (fired[k][j] if j < i else (k > 0 and fired.get(k - 1, {}).get(j, False)))]
            exp = sum((1.0 / (1 + max(len(G[i]), len(G[j])))) * min(rate[i], rate[j]) / rate[j] for j in snd)
            err = abs(float(r['w_in']) - exp); worst = max(worst, err); nchk += 1
            nbad += (len(snd) != int(r['n_senders'])) or err > 1e-5
            minself = min(minself, 1.0 - float(r['w_in']))
        print(f'  fusion_jump 解析值比對：{nchk} 次融合、不符 {nbad}、最大誤差 {worst:.2e}、自己權重最小值 {minself:.4f}')
        bad += (nbad > 0) or (minself <= 0) or nchk == 0
    print('驗收：' + ('通過' if bad == 0 else f'未通過（{bad} 項）'))
    sys.exit(0 if bad == 0 else 1)


def bidir_check(test):
    """[plan 1017] EF-HC 雙向短跑的自我一致性（不需參照 run）：
    ① 回傳紀錄：(k, y→x) 存在 ⇔ x 在第 k 步觸發，且不是「y 編號較小且也觸發」（那種情況 y 已推送給 x）
    ② 推送數：觸發的 y 在第 k 步推送數＝度數 − #{x∈鄰居：x<y 且 x 第 k 步觸發}（已回傳給 x 的連線不再推送）
    ③ 融合：每列 fusion_jump 的送方集合／收進總比重＝依交換規則重建（觸發的台＝所有鄰居；
       沒觸發的 x＝第 k 步觸發的 y<x，加上第 k−1 步觸發、且 x 當步沒觸發的 y>x），比重 MH＝1/(1+max(度數))
    ④ 只記錄：回傳後與自己送出後的 delta 比值（不當判準）"""
    import pandas as pd
    b = pd.read_csv(f'{test}/async_diag/broadcast_log.csv'); b['i'] = b.domain.str[5:].astype(int)
    full = b.groupby('step').size(); steps = sorted(full[full == 9].index)[:-1]
    b = b[b.step.isin(steps)]
    F = {(int(r.step), int(r.i)): bool(r.fired) for r in b.itertuples()}
    NP = {(int(r.step), int(r.i)): int(r.n_pushed) for r in b.itertuples()}
    DL = {(int(r.step), int(r.i)): float(r.delta) for r in b.itertuples()}
    rp = pd.read_csv(f'{test}/async_diag/reply_log.csv')
    rp['y'] = rp.replier.str[5:].astype(int); rp['x'] = rp.target.str[5:].astype(int)
    rp = rp[rp.step.isin(steps)]
    R = set(zip(rp.step, rp.y, rp.x))
    G = nx.random_geometric_graph(9, 0.8, seed=1234); deg = {i: len(G[i]) for i in range(9)}
    bad = 0
    exp_R = {(k, y, x) for k in steps for x in range(9) if F[(k, x)] for y in G[x] if not (y < x and F[(k, y)])}
    print(f'雙向短跑：比對步數 {len(steps)}（step {steps[0]}–{steps[-1]}）、觸發次數 {sum(F.values())}、回傳 {len(R)} 筆')
    print(f'  ① 回傳紀錄 實際 {len(R)}、預期 {len(exp_R)}、多出 {len(R - exp_R)}、缺少 {len(exp_R - R)}'); bad += (R != exp_R)
    np_bad = sum(1 for k in steps for y in range(9) if F[(k, y)]
                 and NP[(k, y)] != deg[y] - sum(1 for x in G[y] if x < y and F[(k, x)]))
    print(f'  ② 推送數不符 {np_bad} 次'); bad += (np_bad > 0)
    jump = pd.read_csv(f'{test}/async_diag/fusion_jump.csv'); jump = jump[jump.step.isin(steps)]
    nchk = nbad = 0; worst = 0.0
    for r in jump.itertuples():
        k, x = int(r.step), int(r.domain.split('_')[1])
        if F[(k, x)]:
            snd = list(G[x])
        else:
            snd = [y for y in G[x] if (y < x and F[(k, y)]) or (y > x and k > 0 and F.get((k - 1, y), False) and not F.get((k - 1, x), False))]
        e = sum(1.0 / (1 + max(deg[x], deg[y])) for y in snd); err = abs(float(r.w_in) - e); worst = max(worst, err)
        nchk += 1; nbad += (len(snd) != int(r.n_senders)) or err > 1e-5
    print(f'  ③ 融合重建：{nchk} 次、不符 {nbad}、最大誤差 {worst:.2e}'); bad += (nbad > 0) or nchk == 0
    rr = [DL[(k + 1, y)] / DL[(k, y)] for (k, y, x) in R if (k + 1, y) in DL and not F[(k, y)] and DL[(k, y)] > 0]
    ro = [DL[(k + 1, y)] / DL[(k, y)] for (k, y) in F if F[(k, y)] and (k + 1, y) in DL and DL[(k, y)] > 0]
    mr, mo = (float(np.median(rr)) if rr else float('nan')), (float(np.median(ro)) if ro else float('nan'))
    print(f'  ④（只記錄，不判）下一步 delta／當步 delta 中位數：回傳後 {mr:.3f}（{len(rr)} 筆）vs 自己送出後 {mo:.3f}（{len(ro)} 筆）'
          '；雙向下自己送出後會吸收所有鄰居，兩者都可能不小，「回傳不更新 ŵ」以讀碼為準（bidir_reply 不碰 last_broadcast_params）')
    print('驗收：' + ('通過' if bad == 0 else f'未通過（{bad} 項）'))
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    main()
