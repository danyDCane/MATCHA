# -*- coding: utf-8 -*-
"""D 臂（參數觸發＋門檻隨學習率衰減，EF-HC/PersonalizedET 原版 γ(k)=α(k)）的係數 r 離線推估（2026-10-07，零 GPU）

目標：找 r，使 `fired ⟺ delta ≥ r·lr(k)/lr(0)  或  距上次送出 ≥50 步` 的總傳輸份數 ≈ 29,614（＝A、C）。
難點：D 的送出時機不同 ⇒ 「距上次送出 g 步的參數距離」要用模型推估：delta(k, g) ≈ v_i(k)·g^β。
  - v_i(k)、β：用 C 的 broadcast_log 擬合（C 每 6–7 步重設參照、全程都有 g＝1…7 的觀測）。
  - β 外推到長間隔：用 A 後段（g 可到 50）檢查。
lr(k)：CosineAnnealingLR(T_max＝4000, eta_min＝0)，scheduler 在決策前已 step ⇒ γ(k)＝½(1+cos(π(k+1)/4000))。
⚠️ 這是近似：D 的融合時序與 C 不同、v 用 C 的軌跡代替 D 的軌跡。發車後用前 50 epoch 的實際份數核對（見輸出的逐段預期）。
"""
import numpy as np, pandas as pd, networkx as nx, sys, os
ROOT = '/home/server5090/Desktop/M11307320/MATCHA/'
BW_MODE = '--bw' in sys.argv     # [plan 1009] 頻寬不平均模式：見檔尾 bw_mode()
C = ROOT + 'exp_result_v1_stage2_leave_cartoon_p1a_async_periodic6.2133_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234'
A = ROOT + 'exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234_fix'
T, N, MAXI, TARGET = 4000, 9, 50, 29614
G = nx.random_geometric_graph(9, 0.8, seed=1234); deg = np.array([G.degree(i) for i in range(N)])
gam = 0.5 * (1 + np.cos(np.pi * (np.arange(T) + 1) / T))


def with_gap(path):
    b = pd.read_csv(path + '/async_diag/broadcast_log.csv'); b['i'] = b.domain.str[5:].astype(int)
    out = []
    for i, g in b.sort_values('step').groupby('i'):
        last, gaps = 0, []
        for k, f in zip(g.step.values, g.fired.values):
            gaps.append(k - last)
            if f: last = k
        g = g.copy(); g['gap'] = gaps; out.append(g)
    return pd.concat(out)


c = with_gap(C); c = c[(c.gap >= 1) & (c.delta > 0)]
# β：逐 epoch 段做 log delta ~ log gap 迴歸
print('β 擬合（C，g＝1…7）：')
for lo, hi in [(1, 10), (11, 50), (51, 100), (101, 150), (151, 200)]:
    q = c[(c.epoch >= lo) & (c.epoch <= hi)]
    beta = np.polyfit(np.log(q.gap), np.log(q.delta), 1)[0]
    print(f'  ep{lo}-{hi}: β＝{beta:.3f}（n＝{len(q)}）')
a = with_gap(A); a = a[(a.gap >= 1) & (a.delta > 0)]
qa = a[a.epoch > 150]
print(f'  A 後段 ep151-200（g 1…50）：β＝{np.polyfit(np.log(qa.gap), np.log(qa.delta), 1)[0]:.3f}（n＝{len(qa)}）')
BETA = float(np.polyfit(np.log(c.gap), np.log(c.delta), 1)[0])
print(f'  全程（C）β＝{BETA:.3f} ⇒ 模擬用此值；另跑 β＝0.5／1.0 看敏感度')

# v_i(epoch)：每節點每 epoch 的 median(delta / g^β)
def v_table(beta):
    cc = c.copy(); cc['v'] = cc.delta / cc.gap ** beta
    V = cc.groupby(['i', 'epoch']).v.median().unstack(0).reindex(range(1, 201)).ffill().bfill().values  # [200, 9]
    return V


def simulate(r, beta, V, rho=None, maxi=MAXI, res=None, seed=0):
    """rho：每節點門檻係數（plan 1009 的 EF-HC ρ_i）；None＝全 1（1007 行為）。maxi：強制送出間隔。
    res：對數殘差樣本（plan 1009）；給了就每步乘上 exp(抽樣殘差)＝把逐步波動放進來。None＝1007 行為（不加波動）。"""
    fires = np.zeros(N, int); seg = np.zeros((4, N), int)
    rng = np.random.default_rng(seed)
    for i in range(N):
        s = 0
        ri = 1.0 if rho is None else rho[i]
        eps = np.exp(rng.choice(res, T)) if res is not None else None
        for k in range(T):
            g = k - s
            if g == 0:
                continue
            d = V[k // 20, i] * g ** beta * (eps[k] if eps is not None else 1.0)
            if d >= r * gam[k] * ri or g >= maxi:
                fires[i] += 1; seg[min(k // 1000, 3), i] += 1; s = k
    return (fires * deg).sum(), fires, seg


for beta in ([] if BW_MODE else [BETA, 0.5, 1.0]):
    V = v_table(beta)
    lo, hi = 1e-8, 1e-3
    for _ in range(40):                       # 對數二分：總份數隨 r 單調遞減
        mid = np.sqrt(lo * hi); tot = simulate(mid, beta, V)[0]
        lo, hi = (mid, hi) if tot > TARGET else (lo, mid)
    r = np.sqrt(lo * hi); tot, fires, seg = simulate(r, beta, V)
    print(f'\nβ＝{beta:.3f}：r ≈ {r:.3e} ⇒ 預估總傳輸 {tot}、送出 {fires.sum()}、逐節點 {fires.tolist()}')
    print('  逐段送出占比（ep1-50/51-100/101-150/151-200）：' + ' / '.join(f'{100*seg[q].sum()/fires.sum():.1f}%' for q in range(4))
          + f'；逐段預估傳輸份數 ' + ' / '.join(str(int((seg[q]*deg).sum())) for q in range(4)))


# =====================================================================================================
# [plan 1009] 頻寬不平均四臂：D-bw 的 τ 推估、C-bw 週期表、C-match 週期表（零 GPU）
#   python lr_threshold_calibration_estimate.py --bw [--bw_seed 0] [--target 29613]
#   python lr_threshold_calibration_estimate.py --bw --match_from <D-bw run 目錄>   ← D-bw 跑完後產生 C-match 週期表
# 頻寬表與 train.py 共用 communicator.stratified_bandwidth_table（單一來源）。
# 推估模型同 1007：delta(k, g) ≈ v_i(k)·g^β；改用 **D（1007，同為參數觸發＋門檻隨 lr）** 的紀錄擬合，
# 並先驗證模擬器能否重現 D 的實際總份數 33,467（1007 用 C 擬合時少估 13%）。
# =====================================================================================================
D1007 = ROOT + 'exp_result_v1_stage2_leave_cartoon_p1a_async_lr_tau1.765e-05_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234'
STYLES = ['art_painting'] * 3 + ['photo'] * 3 + ['sketch'] * 3     # node_0..8（util.assign_nodes_to_domains 區塊式）


def write_periods(path, periods):
    with open(path, 'w') as f:
        f.write('node,period\n')
        for i, p in enumerate(periods):
            f.write(f'node_{i},{p:.6f}\n')


def periodic_fires(periods, phase_seed=0):
    """與 communicator.periodic_configure 相同的相位抽法，算出全程逐節點送出次數（精確，非估計）。"""
    u = np.random.default_rng(phase_seed).random(N)
    return np.array([int(np.floor(T / p + u[i]) - np.floor(u[i])) for i, p in enumerate(periods)])


def bw_mode():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--bw', action='store_true')
    ap.add_argument('--bw_seed', type=int, default=0)
    ap.add_argument('--bw_floor', type=float, default=0.1)
    ap.add_argument('--target', type=int, default=29613, help='總傳輸份數目標（＝1007 C 實際值）')
    ap.add_argument('--out_dir', default=ROOT + 'results/proto_trigger/1009_bw')
    ap.add_argument('--match_from', default='', help='D-bw run 目錄 ⇒ 產生 C-match 週期表')
    o = ap.parse_args()
    sys.path.insert(0, ROOT)
    from communicator import stratified_bandwidth_table
    bw = np.array(stratified_bandwidth_table(STYLES, seed=o.bw_seed, floor=o.bw_floor))
    os.makedirs(o.out_dir, exist_ok=True)
    print(f'頻寬表（seed {o.bw_seed}、floor {o.bw_floor}）：' + ', '.join(f'node_{i}({STYLES[i][:5]})={b:.4f}' for i, b in enumerate(bw)))
    print('  各畫風平均：' + ' / '.join(f'{s[:5]} {bw[[i for i in range(N) if STYLES[i] == s]].mean():.3f}' for s in sorted(set(STYLES))))

    if o.match_from:
        b = pd.read_csv(o.match_from.rstrip('/') + '/async_diag/broadcast_log.csv'); b['i'] = b.domain.str[5:].astype(int)
        sends = b.groupby('i').fired.sum().reindex(range(N)).fillna(0).astype(int).values
        forced = int(b.forced.sum())
        per = np.array([T / s if s > 0 else T + 1.0 for s in sends])
        out = o.out_dir + f'/period_Cmatch_seed{o.bw_seed}.csv'; write_periods(out, per)
        fz = periodic_fires(per)
        print(f'C-match：D-bw 實際逐節點送出 {sends.tolist()}（forced {forced}）⇒ 週期 {np.round(per, 4).tolist()}')
        print(f'  依相位精確推算 C-match 送出 {fz.tolist()}、總份數 {int((fz * deg).sum())}（D-bw 實際 {int((sends * deg).sum())}）')
        print(f'  寫出 {out}')
        return

    # C0：均勻週期的成本
    s0 = periodic_fires([6.2133] * N)
    t0 = (s0 / bw).sum()
    print(f'\nC0：每節點送出 {s0.tolist()}、總份數 {int((s0 * deg).sum())}、總傳輸時間 {t0:.0f} 單位；'
          f'最慢節點 node_{int(bw.argmin())}（b={bw.min():.3f}）佔 {s0[bw.argmin()] / bw.min() / t0:.1%}')

    # C-bw：週期 ∝ 1/b_i（每台傳輸時間相等），係數對齊總份數
    c = o.target / (bw * deg).sum() / T          # sends_i ≈ c·T·b_i
    per = 1.0 / (c * bw)
    fz = periodic_fires(per)
    out = o.out_dir + f'/period_Cbw_seed{o.bw_seed}.csv'; write_periods(out, per)
    print(f'\nC-bw：週期 {np.round(per, 4).tolist()}')
    print(f'  依相位精確推算送出 {fz.tolist()}、總份數 {int((fz * deg).sum())}、總傳輸時間 {(fz / bw).sum():.0f}、'
          f'每台時間 {np.round(fz / bw).astype(int).tolist()}；寫出 {out}')

    # D-bw：用 D（1007）的紀錄擬合 β、v_i(k)（逐節點逐 epoch 中位數）與逐步對數殘差
    # 只用中位數會少估（觸發常發生在波動偏高的那幾步）：D 上實測中位數版比值 0.861、含殘差抽樣版 0.999（1009）
    d = with_gap(D1007); d = d[(d.gap >= 1) & (d.delta > 0)]
    beta_d = float(np.polyfit(np.log(d.gap), np.log(d.delta), 1)[0])
    dd = d.copy(); dd['v'] = dd.delta / dd.gap ** beta_d
    med = dd.groupby(['i', 'epoch']).v.transform('median'); res = np.log(dd.v / med).values
    Vd = dd.groupby(['i', 'epoch']).v.median().unstack(0).reindex(range(1, 201)).ffill().bfill().values
    act = pd.read_csv(D1007 + '/async_diag/broadcast_log.csv')
    act_tot = int(act.n_pushed.sum())
    SEEDS = (0, 1, 2)

    def sim_avg(r, rho, maxi):
        outs = [simulate(r, beta_d, Vd, rho=rho, maxi=maxi, res=res, seed=sd) for sd in SEEDS]
        return (np.mean([o_[0] for o_ in outs]), np.mean([o_[1] for o_ in outs], 0), np.mean([o_[2] for o_ in outs], 0))
    t_chk, f_chk, _ = sim_avg(1.765e-5, None, 50)
    print(f'\nD（1007）擬合：β＝{beta_d:.3f}、對數殘差標準差 {res.std():.3f}；'
          f'模擬器重現 D（r＝1.765e-5、強制 50、含波動、{len(SEEDS)} 次平均）⇒ 預估 {t_chk:.0f} vs 實際 {act_tot}（比值 {t_chk / act_tot:.3f}）')
    print(f'  逐節點預估 {np.round(f_chk).astype(int).tolist()} vs 實際 '
          f'{act.groupby(act.domain.str[5:].astype(int)).fired.sum().astype(int).tolist()}')
    rho = 1.0 / bw
    lo, hi = 1e-7, 1e-3
    for _ in range(30):
        mid = np.sqrt(lo * hi); t_ = sim_avg(mid, rho, 10 ** 9)[0]
        lo, hi = (mid, hi) if t_ > o.target else (lo, mid)
    r = np.sqrt(lo * hi); tot, fires, seg = sim_avg(r, rho, 10 ** 9)
    print(f'D-bw：τ ≈ {r:.4e} ⇒ 預估總份數 {tot:.0f}、逐節點送出 {np.round(fires).astype(int).tolist()}、'
          f'總傳輸時間 {(fires / bw).sum():.0f}')
    print('  逐段送出占比（ep1-50/51-100/101-150/151-200）：' + ' / '.join(f'{100 * seg[q].sum() / fires.sum():.1f}%' for q in range(4))
          + '；逐段份數 ' + ' / '.join(str(int((seg[q] * deg).sum())) for q in range(4)))
    for f_ in (0.85, 1.15):
        print(f'  敏感度：τ×{f_} ⇒ 預估總份數 {sim_avg(r * f_, rho, 10 ** 9)[0]:.0f}')

if BW_MODE:
    bw_mode()
