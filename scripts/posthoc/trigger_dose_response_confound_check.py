# -*- coding: utf-8 -*-
"""「觸發被鄰居聚合帶動」那條劑量反應的混淆檢查（2026-10-03，零 GPU）

起因：dany 問「A 融合 B 之後還會先訓練才廣播，內容怎麼分得開」，追查時發現三件事要檢查：
 一、事件順序：每步流程是 訓練 → 判斷觸發 → 若發火則推送並重設參照點 → 收信融合。
     舊算法（階段一報告 §3.2 的 16.74×）把「第 t 步判斷之後才收到的信」也算進第 t 步，順序錯一格。
 二、訓練階段混淆：前期大家常發火也常收信，後期都很少 ⇒ 全程混在一起可能放大倍數。
 三、距上次廣播的時間混淆：越久沒廣播 ⇒ 收到越多封，也 ⇒ 自己累積的變化越大。
 四、融合那一跳 vs 一步訓練：各讓觸發量增加多少（分階段）。
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
RUN=('exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_'
     'osdg_excl_person_seed2026_topo1234_fix')
d=pd.read_csv(f'{RUN}/async_diag/shadow_signals.csv')
dom=d['domain'].values; step=d['step'].values; ep_all=d['epoch'].values
fired=d['fired'].values.astype(bool); nat0_all=(d['sig_r0'].values>=1e-5); nrec=d['n_received'].values.astype(int)
old=np.zeros(len(d),int); new=np.zeros(len(d),int); clk=np.zeros(len(d),int)
for n in sorted(np.unique(dom)):
    m=np.where(dom==n)[0]; a_old=0; a_new=0; last=-1
    for i in m:
        a_old+=nrec[i]; old[i]=a_old
        if fired[i]: a_old=0
        new[i]=a_new; clk[i]=step[i]-last
        if fired[i]: a_new=0; last=step[i]
        a_new+=nrec[i]
keep=d['is_warm'].values==0
ep=ep_all[keep]; nat0=nat0_all[keep]; cum=new[keep]; ck=clk[keep]; cold=old[keep]

print('='*90); print('一、事件順序修正前後的劑量反應（排除暖機）'); print('='*90)
print(f'兩種算法逐列相同：{(cold==cum).mean()*100:.1f}%')
print(f'{"累積收到":>8}{"舊算法":>10}{"n":>7}{"正確順序":>11}{"n":>7}')
rn=[]
for v in range(0,13):
    mo=cold==v; mn=cum==v
    a=f'{nat0[mo].mean()*100:>9.2f}%' if mo.sum()>=50 else f'{"—":>10}'
    b=f'{nat0[mn].mean()*100:>10.2f}%' if mn.sum()>=50 else f'{"—":>11}'
    print(f'{v:>8}{a}{mo.sum():>7}{b}{mn.sum():>7}')
    if mn.sum()>=100: rn.append((v,nat0[mn].mean()))
print(f'正確順序（n≥100）：{rn[0][1]*100:.2f}% → {rn[-1][1]*100:.2f}%（{rn[-1][1]/rn[0][1]:.2f}×）；'
      f'單調：{all(np.diff([r for _,r in rn])>0)}')

print('\n'+'='*90); print('二、分訓練階段（正確順序；各格 n≥80）'); print('='*90)
PH=[(11,50),(51,100),(101,150),(151,200)]
for a,b in PH:
    m=(ep>=a)&(ep<=b); vs=[v for v in range(16) if ((cum==v)&m).sum()>=80]
    r0=nat0[m&(cum==0)].mean(); rk=nat0[m&(cum==vs[-1])].mean()
    print(f'  ep{a:>3}-{b:<4} 0 封 {r0*100:5.1f}% → {vs[-1]} 封 {rk*100:5.1f}%（{rk/max(r0,1e-9):5.1f}×） 等級相關 {spearmanr(cum[m],nat0[m]).statistic:+.3f}')

print('\n'+'='*90); print('三、同階段、同「距上次廣播幾步」的格子內：收到多 vs 少（多＝高於格內中位數）'); print('='*90)
CB=[(1,1),(2,2),(3,3),(4,5),(6,8),(9,15)]; W=[];S=[];DIF=[];REL=[]
print(f'{"階段":<12}{"距上次廣播":<10}{"n":>6}{"收到少":>9}{"收到多":>9}{"差":>8}{"等級相關":>10}')
for a,b in PH:
    for lo,hi in CB:
        m=(ep>=a)&(ep<=b)&(ck>=lo)&(ck<=hi)
        if m.sum()<150: continue
        c=cum[m]; f=nat0[m]; med=np.median(c); L=c<=med; H=c>med
        if H.sum()<30 or L.sum()<30 or f.sum()<10: continue
        r=spearmanr(c,f).statistic; dl=f[H].mean()-f[L].mean()
        W.append(m.sum()); S.append(r); DIF.append(dl)
        print(f'ep{a:>3}-{b:<6}{f"{lo}-{hi} 步":<10}{m.sum():>6}{f[L].mean()*100:>8.1f}%{f[H].mean()*100:>8.1f}%{dl*100:>+7.1f}{r:>+10.3f}')
W=np.array(W)
print(f'加權平均：差 {np.average(DIF,weights=W)*100:+.1f} 個百分點、等級相關 {np.average(S,weights=W):+.3f}；差為正 {int((np.array(DIF)>0).sum())}/{len(DIF)} 格')
print('⚠️ ep151-200 的格子因發火太少（每格 <10 次）全被排除。')

print('\n'+'='*90); print('四、融合那一跳 vs 一步訓練：觸發量 delta 的單步增量（第 t-1 步沒發火、參照點未變）'); print('='*90)
b=pd.read_csv(f'{RUN}/async_diag/broadcast_log.csv'); b['i']=b['domain'].str.replace('node_','').astype(int)
D=b.pivot(index='step',columns='i',values='delta').sort_index().values
Fb=b.pivot(index='step',columns='i',values='fired').sort_index().values.astype(bool)
Rb=b.pivot(index='step',columns='i',values='n_received').sort_index().values
for lo,hi in [(201,1000),(1001,2000),(2001,3000),(3001,3999)]:
    rr=[(Rb[t-1,i],D[t,i]-D[t-1,i]) for i in range(9) for t in range(lo,hi+1) if not Fb[t-1,i]]
    y=pd.DataFrame(rr,columns=['recv','inc']); a=y[y.recv==0].inc.median(); c=y[y.recv>=2].inc.median()
    print(f'  epoch {lo//20+1:>3}–{hi//20+1:<3} 上一步沒收到（純一步訓練）{a*1e6:7.3f}e-6   上一步收到 ≥2 封（融合＋一步訓練）{c*1e6:7.3f}e-6   比值 {c/a:4.1f}×')
print('⚠️ delta 是範數，增量不可相加；融合可能把模型往參照點拉回（抵銷自己的漂移），所以比值 <1 是可能的。')
