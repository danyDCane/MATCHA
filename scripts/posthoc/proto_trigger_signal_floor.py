# -*- coding: utf-8 -*-
"""原型觸發訊號 sig_r1 會不會隨「距上次廣播的步數」累積？（2026-10-05，_fix run 的 shadow_signals.csv，零 GPU）
問：①後段原型是不是不動了（只剩保底）②訊號是累積變化還是抖動地板。gap＝該節點距上次 fired 的步數（proto_ref 在 fired 時重設）。"""
import pandas as pd, numpy as np
R='exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234_fix'
d=pd.read_csv(f'{R}/async_diag/shadow_signals.csv',usecols=['step','epoch','domain','is_warm','sig_r1','fired','forced','n_received'])
print('rows',len(d),'steps',d.step.nunique(),'epochs',d.epoch.min(),d.epoch.max(),'steps/epoch',d.step.nunique()/d.epoch.nunique())
d=d.sort_values(['domain','step'])
# gap = steps since the reference was last reset (reset happens at a fired step, after signal is logged? check)
out=[]
for dom,g in d.groupby('domain'):
    last=None; gaps=[]
    for st,f in zip(g.step.values,g.fired.values):
        gaps.append(np.nan if last is None else st-last)
        if f==1: last=st
    g=g.copy(); g['gap']=gaps; out.append(g)
d=pd.concat(out)
d=d[np.isfinite(d.sig_r1)]
seg=pd.cut(d.epoch,[0,50,100,150,200],labels=['ep1-50','ep51-100','ep101-150','ep151-200'])
d['seg']=seg
bins=[0,1,2,3,6,10,20,50,1e9]; lab=['1','2','3','4-6','7-10','11-20','21-50','>50']
d['gb']=pd.cut(d.gap,bins,labels=lab)
print('\n觸發量 sig_r1（度）中位數：列＝訓練階段，欄＝距上次廣播幾步')
print(d.pivot_table(index='seg',columns='gb',values='sig_r1',aggfunc='median',observed=False).round(2).to_string())
print('\n樣本數')
print(d.pivot_table(index='seg',columns='gb',values='sig_r1',aggfunc='size',observed=False).to_string())
print('\n各階段 sig_r1 中位數、forced 占 fired 比例（現行參數觸發）')
for s,g in d.groupby('seg',observed=False):
    print(s, round(g.sig_r1.median(),2), 'fired率',round(g.fired.mean()*100,2),'% forced/fired',round(g.forced.sum()/max(g.fired.sum(),1)*100,1),'%')
print('\n後段(ep151-200)：同一間隔下，這一步有沒有收到鄰居模型')
g=d[d.seg=='ep151-200']
print(g.pivot_table(index=g.n_received>0,columns='gb',values='sig_r1',aggfunc='median',observed=False).round(2).to_string())
print('\n逐節點：後段 gap 21-50 中位數 vs gap 4-6 中位數')
for dom,h in g.groupby('domain'):
    a=h[h.gb=='4-6'].sig_r1.median(); b=h[h.gb=='21-50'].sig_r1.median()
    print(dom, round(a,2), round(b,2), 'ratio', round(b/a,3))
