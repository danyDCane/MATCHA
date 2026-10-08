# -*- coding: utf-8 -*-
"""
階段一影子訊號分析 R2–R5
=========================
輸入：exp_result_.../async_diag/shadow_signals.csv（41 欄、36000 列 ＝ 4000 step × 9 節點）
      同目錄 broadcast_log.csv（規則 0 的 ground truth，欄位 delta/thresh/fired/forced）

R2 原型表填滿曲線（cells_filled 上限 54 ＝ 6 類別 × 9 節點；盯 node_8 的 art）
R3 staleness 分布（age_mean / age_max）
R4 ★ 三訊號決策重疊（門檻校準到同觸發率 → Cohen's kappa；事前判準 κ ≥ 0.90 ⇒ 換刻度）
R5 安全閥占比（離線重建 forced 占觸發的幾成；對照規則 0 的實測值）

⚠️ 所有分析排除 is_warm == 1（前 10 epoch 暖機）。
⚠️ 度數不足的邊欄位是空字串、不是 0。
"""
import sys
import numpy as np
import pandas as pd

CSV = ('/home/server5090/Desktop/M11307320/MATCHA/exp_result_v1_stage2_leave_cartoon_p1a_'
       'async_const_tau1e-5_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234_fix/'
       'async_diag/shadow_signals.csv')
BCL = CSV.replace('shadow_signals.csv', 'broadcast_log.csv')
MAX_INTERVAL = 50
N_CELLS = 54

def rule(s):
    print('=' * 96)
    print(s)
    print('=' * 96)

df = pd.read_csv(CSV)
bl = pd.read_csv(BCL)
nodes = sorted(df['domain'].unique())
E = [f'e{i}' for i in range(8)]

print(f'載入 {CSV.split("/")[-2]}/shadow_signals.csv：{len(df)} 列、{len(df.columns)} 欄')
print(f'節點 {len(nodes)} 個｜step 範圍 {df["step"].min()}–{df["step"].max()}｜epoch 範圍 {df["epoch"].min()}–{df["epoch"].max()}')

# ---- 一致性檢查：shadow 的 fired/forced 必須與 broadcast_log 完全一致 ----
m = df[['step','domain','fired','forced']].merge(
        bl[['step','domain','fired','forced','delta','thresh']],
        on=['step','domain'], suffixes=('_s','_b'))
assert len(m) == len(df), '兩檔 step×domain 對不上'
print(f'一致性：fired 相同 {int((m.fired_s==m.fired_b).sum())}/{len(m)}｜'
      f'forced 相同 {int((m.forced_s==m.forced_b).sum())}/{len(m)}｜'
      f'sig_r0 與 delta 最大差 {np.abs(df["sig_r0"].values-m["delta"].values).max():.3e}')
THRESH0 = bl['thresh'].iloc[-1]
print(f'規則 0 門檻 thresh（末列）＝ {THRESH0:.3e}｜gamma 唯一值 {sorted(bl["gamma"].unique())}')

warm = df[df.is_warm == 1]
d = df[df.is_warm == 0].copy().reset_index(drop=True)
print(f'\n排除暖機：is_warm==1 共 {len(warm)} 列（epoch {warm.epoch.min()}–{warm.epoch.max()}）'
      f'⇒ 留下 {len(d)} 列（epoch {d.epoch.min()}–{d.epoch.max()}）')

# ===================== R2 =====================
rule('R2 原型表填滿曲線（cells_filled，上限 54）')
full_all = df.groupby('domain')['cells_filled'].agg(['min','max'])
print('全程（含暖機）逐節點 cells_filled：')
for n in nodes:
    sub = df[df.domain == n]
    first_full = sub.loc[sub.cells_filled >= N_CELLS, 'step']
    fs = int(first_full.min()) if len(first_full) else None
    ep = int(sub.loc[sub.step == fs, 'epoch'].iloc[0]) if fs is not None else None
    n_below = int((sub.cells_filled < N_CELLS).sum())
    print(f'  {n}  min {int(sub.cells_filled.min()):2d}  max {int(sub.cells_filled.max()):2d}  '
          f'首次填滿 step {fs}（epoch {ep}）  之後仍 <54 的列數 {n_below - (fs if fs is not None else 0)}')
print(f'\n非暖機區 cells_filled < 54 的列數：{int((d.cells_filled < N_CELLS).sum())} / {len(d)}')
print('逐節點「每畫風聽到幾個 owner」（ns_0 art / ns_1 photo / ns_2 sketch，非暖機區的最小值）：')
for n in nodes:
    sub = d[d.domain == n]
    print(f'  {n}  ns_min = ({int(sub.ns_0.min())}, {int(sub.ns_1.min())}, {int(sub.ns_2.min())})   '
          f'ns 常見值 = ({int(sub.ns_0.mode()[0])}, {int(sub.ns_1.mode()[0])}, {int(sub.ns_2.mode()[0])})   '
          f'度數（非空邊欄位數）= {int(sub[[f"r2c_{e}" for e in E]].notna().iloc[0].sum())}')
zero_ns = d[(d.ns_0 == 0) | (d.ns_1 == 0) | (d.ns_2 == 0)]
print(f'\n🚨 ns_s == 0（某畫風一個都沒聽到）的列數：{len(zero_ns)} / {len(d)}')

# ===================== R3 =====================
rule('R3 staleness 分布（age_mean / age_max，單位 ＝ step；非暖機區）')
print(f'{"節點":<8}{"age_mean 平均":>14}{"中位":>8}{"p90":>8}{"age_max 平均":>14}{"p90":>8}{"全程最大":>10}')
for n in nodes:
    sub = d[d.domain == n]
    print(f'{n:<8}{sub.age_mean.mean():>14.2f}{sub.age_mean.median():>8.2f}{sub.age_mean.quantile(.9):>8.2f}'
          f'{sub.age_max.mean():>14.2f}{sub.age_max.quantile(.9):>8.2f}{sub.age_max.max():>10.0f}')
print(f'{"全體":<8}{d.age_mean.mean():>14.2f}{d.age_mean.median():>8.2f}{d.age_mean.quantile(.9):>8.2f}'
      f'{d.age_max.mean():>14.2f}{d.age_max.quantile(.9):>8.2f}{d.age_max.max():>10.0f}')
print('\n分 epoch 區段（age_mean 全體平均）：')
for lo, hi in [(11,50),(51,100),(101,150),(151,200)]:
    sub = d[(d.epoch >= lo) & (d.epoch <= hi)]
    print(f'  epoch {lo:>3}–{hi:<3}  age_mean {sub.age_mean.mean():6.2f}   age_max {sub.age_max.mean():6.2f}')

# ===================== R4 =====================
rule('R4 ★ 三訊號的決策重疊（Cohen\'s kappa）')

fired0 = d['fired'].values.astype(bool)
forced0 = d['forced'].values.astype(bool)
nat0 = (d['sig_r0'].values >= THRESH0)
print(f'規則 0（非暖機區）：fired {fired0.mean()*100:.2f}%｜自然觸發 {nat0.mean()*100:.2f}%｜'
      f'forced {forced0.mean()*100:.2f}%（占觸發 {forced0.sum()/fired0.sum()*100:.2f}%）')
print(f'  ⇒ 校準目標觸發率 p0 = {nat0.mean()*100:.4f}%（用【自然觸發】對齊，安全閥另外在 R5 處理）')
p0 = nat0.mean()
print(f'  chance level（基礎率 {p0*100:.2f}%）＝ {p0**2 + (1-p0)**2:.4f} ⇒ 原始一致度不可用')

def kappa(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    n = len(a)
    po = (a == b).mean()
    pe = a.mean()*b.mean() + (1-a.mean())*(1-b.mean())
    return (po - pe) / (1 - pe), po, pe

def calib_global(sig, target):
    """全域單一門檻，校準到 target 觸發率（取 1-target 分位）"""
    s = np.asarray(sig, float)
    th = np.nanquantile(s, 1 - target)
    return s >= th, th

def calib_pernode(sig, dom, nat_ref):
    """逐節點門檻：每個節點校準到【該節點自己】的規則 0 自然觸發率"""
    s = np.asarray(sig, float); out = np.zeros(len(s), bool); ths = {}
    for n in np.unique(dom):
        m = (dom == n)
        t = nat_ref[m].mean()
        th = np.nanquantile(s[m], 1 - t)
        out[m] = s[m] >= th; ths[n] = (th, t)
    return out, ths

dom = d['domain'].values
sig_r1 = d['sig_r1'].values.astype(float)
print(f'\nsig_r1 非空 {np.isfinite(sig_r1).sum()}/{len(sig_r1)}')

# 規則 2c 的逐邊合併
raw = d[[f'r2c_{e}' for e in E]].to_numpy(dtype=float)
com = d[[f'r2cC_{e}' for e in E]].to_numpy(dtype=float)
deg = np.isfinite(raw).sum(1)
print(f'逐節點度數：' + '  '.join(f'{n}={int(deg[dom==n][0])}' for n in nodes))

def comb(mat, how):
    with np.errstate(all='ignore'):
        if how == 'mean':  return np.nanmean(mat, 1)
        if how == 'max':   return np.nanmax(mat, 1)
        if how == 'min':   return np.nanmin(mat, 1)
        if how == 'median':return np.nanmedian(mat, 1)
    raise ValueError(how)

cands = {'規則1（原型 × 自己的過去）': sig_r1}
for how in ['mean','max','median','min']:
    cands[f'規則2c-{how}（原始口徑）'] = comb(raw, how)
    cands[f'規則2c-{how}（共同可見）'] = comb(com, how)

print(f'\n{"訊號":<28}{"門檻":>12}{"實際觸發率":>11}{"κ vs 規則0":>12}{"原始一致":>10}{"chance":>9}')
print('-' * 96)
dec = {}
for name, s in cands.items():
    dd, th = calib_global(s, p0)
    k, po, pe = kappa(dd, nat0)
    dec[name] = dd
    print(f'{name:<28}{th:>12.4e}{dd.mean()*100:>10.2f}%{k:>12.4f}{po:>10.4f}{pe:>9.4f}')

print(f'\n逐節點校準（每節點對齊自己的規則 0 觸發率）：')
print(f'{"訊號":<28}{"κ vs 規則0":>12}{"原始一致":>10}')
print('-' * 96)
dec_pn = {}
for name, s in cands.items():
    dd, ths = calib_pernode(s, dom, nat0)
    k, po, pe = kappa(dd, nat0)
    dec_pn[name] = dd
    print(f'{name:<28}{k:>12.4f}{po:>10.4f}')

print('\n主效應拆解（全域校準）：')
best2c = max([n for n in dec if n.startswith('規則2c')], key=lambda n: kappa(dec[n], nat0)[0])
k01, _, _ = kappa(dec['規則1（原型 × 自己的過去）'], nat0)
k0c, _, _ = kappa(dec[best2c], nat0)
k1c, _, _ = kappa(dec['規則1（原型 × 自己的過去）'], dec[best2c])
print(f'  規則0 vs 規則1   κ = {k01:.4f}   ＝「用原型取代參數」的效應（參照物固定）')
print(f'  規則0 vs {best2c}  κ = {k0c:.4f}')
print(f'  規則1 vs {best2c}  κ = {k1c:.4f}   ＝「比共識取代比自己過去」的效應（訊號固定）')
print('  ⛔ 交互項無法識別 —— 報告必須明說沒查。')

print('\n事前判準（κ ≥ 0.90 ⇒ 判為換刻度、該訊號收掉）：')
for name in ['規則1（原型 × 自己的過去）', best2c]:
    k, _, _ = kappa(dec[name], nat0)
    print(f'  {name:<28} κ = {k:.4f}  ⇒ {"🔴 判為換刻度" if k >= 0.90 else "✅ 不是換刻度（保留）"}')

# 度數相關
print('\n🚨 度數偏誤：各合併方式的觸發率 vs 度數（全域單一門檻下，逐節點觸發率）')
print(f'{"合併方式":<28}' + ''.join(f'{n.replace("node_","n"):>7}' for n in nodes) + f'{"與度數 r":>10}')
for name in [n for n in dec if n.startswith('規則2c')] + ['規則1（原型 × 自己的過去）']:
    rates = np.array([dec[name][dom == n].mean() for n in nodes])
    degs = np.array([deg[dom == n][0] for n in nodes], float)
    r = np.corrcoef(rates, degs)[0, 1]
    print(f'{name:<28}' + ''.join(f'{x*100:>6.1f}%' for x in rates) + f'{r:>10.3f}')
rates0 = np.array([nat0[dom == n].mean() for n in nodes])
degs = np.array([deg[dom == n][0] for n in nodes], float)
print(f'{"規則0（參照）":<28}' + ''.join(f'{x*100:>6.1f}%' for x in rates0)
      + f'{np.corrcoef(rates0, degs)[0,1]:>10.3f}')

# 汙染量
rule('R4b 汙染量：r2c（原始）vs r2cC（共同可見）')
ncom = d[[f'ncom_{e}' for e in E]].to_numpy(dtype=float)
fin = np.isfinite(raw) & np.isfinite(com)
rel = np.abs(raw - com)[fin] / np.maximum(np.abs(raw)[fin], 1e-12)
print(f'逐邊相對差 |r2c − r2cC| / |r2c|：平均 {rel.mean():.4e}｜中位 {np.median(rel):.4e}｜'
      f'p99 {np.quantile(rel,.99):.4e}｜最大 {rel.max():.4e}')
print(f'完全相同（差 ＝ 0）的邊占比：{(rel == 0).mean()*100:.2f}%')
print(f'共同可見 owner 數 ncom：最小 {np.nanmin(ncom):.0f}｜中位 {np.nanmedian(ncom):.0f}｜最大 {np.nanmax(ncom):.0f}')
for how in ['mean','max']:
    a, _ = calib_global(comb(raw, how), p0)
    b, _ = calib_global(comb(com, how), p0)
    k, po, _ = kappa(a, b)
    print(f'  合併 {how}：兩口徑彼此 κ = {k:.4f}（原始一致 {po:.4f}）⇒ 改口徑{"幾乎不改決策" if k>=0.9 else "會改決策"}')

# ===================== R5 =====================
rule('R5 安全閥占比（離線重建；對照規則 0 實測 2.5%）')
def replay(sig, th, dom, steps):
    """逐節點重放：sig ≥ th 自然觸發；否則連續 MAX_INTERVAL 步未廣播則強制觸發"""
    nat = np.zeros(len(sig), bool); frc = np.zeros(len(sig), bool)
    for n in np.unique(dom):
        m = np.where(dom == n)[0]
        last = steps[m[0]] - 1
        for i in m:
            s = sig[i]
            isnat = np.isfinite(s) and s >= th
            isfrc = (steps[i] - last) >= MAX_INTERVAL
            if isnat or isfrc:
                nat[i] = isnat; frc[i] = isfrc and not isnat
                last = steps[i]
    return nat, frc

steps = d['step'].values
print(f'{"訊號":<28}{"總觸發率":>10}{"自然":>9}{"強制":>9}{"強制占觸發":>12}  判讀')
print('-' * 96)
nat_r0, frc_r0 = replay(d['sig_r0'].values.astype(float), THRESH0, dom, steps)
tot = (nat_r0 | frc_r0)
print(f'{"規則0（重放自檢）":<28}{tot.mean()*100:>9.2f}%{nat_r0.mean()*100:>8.2f}%{frc_r0.mean()*100:>8.2f}%'
      f'{frc_r0.sum()/tot.sum()*100:>11.2f}%  ← 應貼近實測 {forced0.sum()/fired0.sum()*100:.2f}%')
for name, s in cands.items():
    _, th = calib_global(s, p0)
    nat, frc = replay(np.asarray(s, float), th, dom, steps)
    tot = (nat | frc)
    share = frc.sum() / max(tot.sum(), 1) * 100
    verdict = '🔴 退化成固定週期排程' if share > 30 else '✅ 未退化'
    print(f'{name:<28}{tot.mean()*100:>9.2f}%{nat.mean()*100:>8.2f}%{frc.mean()*100:>8.2f}%{share:>11.2f}%  {verdict}')

rule('⚠️ 本分析的射程限制（寫進報告）')
print("""1. 影子訊號是在【規則 0 驅動的廣播歷史】下記錄的。sig_r1 的參照 proto_ref 在 push 時重置、
   規則 2c 看到的鄰居原型也由鄰居的規則 0 排程決定 ⇒ 若真的改用規則 1／2c 當觸發器，
   軌跡會不同。κ 測的是「同一段歷史下兩個訊號會不會在同一時刻發火」，這正是要的受控比較；
   但 R5 的離線重放是【近似】（重置時點會變），只能當量級參考、不可當定論。
2. 交互項（換訊號 × 換參照物）無法識別，要補得重跑階段一。
3. 全部結果排除 is_warm == 1。""")
