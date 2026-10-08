# -*- coding: utf-8 -*-
"""規則 2c 量的是「信箱新舊」還是「學習」？（2026-10-02，影子訊號 CSV）
∠(G_i, G_j) 兩邊都是同樣九個 owner 的平均，唯一差別是兩人上次同步的時點 ⇒ 預期它跟原型表新舊（age）走。
🚨 混淆：訓練後期 age 變大、所有訊號都變小 ⇒ 全體相關會被 epoch 撐大 ⇒ 必須在同 epoch 內、同(節點,epoch)內重算。
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
CSV=('exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_'
     'osdg_excl_person_seed2026_topo1234_fix/async_diag/shadow_signals.csv')
E=[f'e{i}' for i in range(8)]
d=pd.read_csv(CSV); d=d[d.is_warm==0].reset_index(drop=True)
dom=d['domain'].values; ep=d['epoch'].values
r0=d['sig_r0'].values.astype(float); r1=d['sig_r1'].values.astype(float)
with np.errstate(all='ignore'): r2c=np.nanmean(d[[f'r2c_{e}' for e in E]].to_numpy(float),1)
am=d['age_mean'].values.astype(float)
print('訊號 vs age_mean 的等級相關（Spearman）')
print(f'{"":<10}{"全體(含 epoch 混淆)":>20}{"同 epoch 內平均":>16}{"同(節點,epoch)內平均":>22}')
for nm,x in [('規則0',r0),('規則1',r1),('規則2c',r2c)]:
    a=spearmanr(x,am).statistic
    b=np.nanmean([spearmanr(x[ep==e],am[ep==e]).statistic for e in np.unique(ep)])
    c=np.nanmean([spearmanr(x[(ep==e)&(dom==n)],am[(ep==e)&(dom==n)]).statistic
                  for e in np.unique(ep) for n in np.unique(dom)])
    print(f'{nm:<10}{a:>20.3f}{b:>16.3f}{c:>22.3f}')
print(f'\n規則2c 與規則1 的等級相關 {spearmanr(r2c,r1).statistic:.3f}（⇒ 不是完全不含學習訊號）')
print('⇒ 只有規則2c 在控制 epoch 後仍跟著原型表新舊走；規則0／1 ≈ 0。')
