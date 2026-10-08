# -*- coding: utf-8 -*-
"""觸發門檻怎麼正規化：每節點門檻（空間）與時間排程（時間）（2026-10-02，影子訊號 CSV ＋ checkpoint）

第一部分 每節點門檻：dany 提案「以『我到全域共識的距離』當每個節點自己的門檻」＝ 規則1 ÷ 3a；
                     反向 規則1 × 3a；控制組＝每節點除以自己的歷史中位數（純本地）。
第二部分 時間排程：固定門檻／γ 綁 cosine 學習率（我們的改編）／γ＝1/√(1+k)（論文實際用的）／
                   除以自己的移動平均（純本地）。
第三部分 控制「距上次廣播幾步」後的每步漂移（檢驗「原型角度無量綱、不隨訓練衰減」這條論證）。
全部是【離線重放】：影子訊號在規則 0 驅動的廣播歷史下記錄，真部署時軌跡會不同。
"""
import numpy as np, pandas as pd, glob, torch, torch.nn.functional as F
RUN=('exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_'
     'osdg_excl_person_seed2026_topo1234_fix')
dall=pd.read_csv(f'{RUN}/async_diag/shadow_signals.csv'); K=int(dall['step'].max())+1
d=dall[dall.is_warm==0].reset_index(drop=True)
dom=d['domain'].values; ep=d['epoch'].values; step=d['step'].values; nodes=sorted(np.unique(dom))
fired0=d['fired'].values.astype(bool); nat0=(d['sig_r0'].values>=1e-5); p0=nat0.mean()
r0=d['sig_r0'].values.astype(float); r1=d['sig_r1'].values.astype(float)
BANDS=[(11,50),(51,100),(101,150),(151,200)]

# ---------------- 第一部分 ----------------
def own(j,e):
    f=glob.glob(f'{RUN}/*_node_{j}_epoch_{e}.pth')[0]
    bs=torch.load(f,map_location='cpu',weights_only=False)['backbone_state']
    return F.normalize(bs['prototypes'][:,j//3,:].float(),dim=1)
EPS=[50,100,150,200]; C={e:torch.stack([own(j,e) for j in range(9)]) for e in EPS}
ang=lambda A,B: torch.rad2deg(torch.arccos((F.normalize(A,dim=-1)*F.normalize(B,dim=-1)).sum(-1).clamp(-1+1e-7,1-1e-7)))
tl=lambda X: F.normalize(torch.stack([F.normalize(X[[k for k in range(9) if k//3==s]].mean(0),dim=1) for s in range(3)]).mean(0),dim=1)
A3={e:np.array([float(ang(C[e][j],tl(C[e])).mean()) for j in range(9)]) for e in EPS}
a3=np.zeros(len(d))
for j,n in enumerate(nodes):
    m=dom==n; a3[m]=np.interp(ep[m],EPS,[A3[e][j] for e in EPS])
loc=np.zeros(len(d))
for n in nodes:
    m=np.where(dom==n)[0]; buf=[]
    for i in m:
        buf.append(r1[i]); loc[i]=np.median(buf[-400:])
def pernode(nm,x):
    dd=x>=np.nanquantile(x,1-p0); rates=np.array([dd[dom==n].mean() for n in nodes])
    per=[rates[[j for j in range(9) if j//3==s]].mean() for s in range(3)]
    print(f'{nm:<34}'+''.join(f'{r*100:>6.1f}' for r in rates)
          +f'{rates.max()/rates.min():>8.2f}{np.std(rates)/np.mean(rates):>9.3f}{(max(per)-min(per))/np.mean(per):>10.3f}')
print('='*104); print('第一部分 每節點觸發率（全域單一 α 校準到 15.25%；節點 0-2 art｜3-5 photo｜6-8 sketch）'); print('='*104)
print(f'{"方案":<34}'+''.join(f'{f"n{j}":>6}' for j in range(9))+f'{"最大/最小":>8}{"變異係數":>9}{"畫風間極差":>10}')
pernode('規則0 參數位移（現行）', r0)
pernode('規則1 原型（全域單一門檻）', r1)
pernode('規則1 ÷ 我到全域（dany 提案）', r1/a3)
pernode('規則1 × 我到全域（反向）', r1*a3)
pernode('〔控制〕規則1 ÷ 自己的歷史中位數', r1/np.maximum(loc,1e-9))
print('逐節點「我到全域」（ep200）：'+' '.join(f'{v:.2f}' for v in A3[200])
      +f'  ⇒ 節點間只差 {A3[200].max()/A3[200].min():.2f} 倍，修不動 4.39 倍的落差')

# ---------------- 第二部分 ----------------
g_cos=(1+np.cos(np.pi*step/K))/2
g_pap=1/np.sqrt(1+step)
def ema_norm(x,hl=50):
    a=1-0.5**(1/hl); o=np.zeros(len(x))
    for n in nodes:
        m=np.where(dom==n)[0]; s=None
        for i in m:
            s=x[i] if s is None else (1-a)*s+a*x[i]; o[i]=x[i]/max(s,1e-12)
    return o
def band(nm,x):
    dd=x>=np.nanquantile(x,1-p0); r=[dd[(ep>=a)&(ep<=b)].mean() for a,b in BANDS]
    print(f'{nm:<40}'+''.join(f'{v*100:>11.2f}%' for v in r)+f'{np.std(r)/np.mean(r):>10.3f}')
print('\n'+'='*104); print('第二部分 時間排程（校準到全程 15.25%，理想＝四段都 15.25%；最後一欄＝四段變異係數）'); print('='*104)
print(f'{"門檻形式":<40}'+''.join(f'{f"ep{a}-{b}":>12}' for a,b in BANDS)+f'{"變異係數":>10}')
band('規則0 固定門檻（本 run 跑的 const）', r0)
band('規則0 γ 綁 cosine 學習率（我們的改編）', r0/np.maximum(g_cos,1e-9))
band('規則0 γ＝1/√(1+k)（論文 §IV 實際用的）', r0/g_pap)
band('規則0 ÷ 自己的移動平均（半衰期 50）', ema_norm(r0))
band('規則1 固定門檻', r1)
band('規則1 ÷ 自己的移動平均（半衰期 50）', ema_norm(r1))
print(f'γ 綁 cosine 在四段的平均：'+' '.join(f'{g_cos[(ep>=a)&(ep<=b)].mean():.4f}' for a,b in BANDS))

# ---------------- 第三部分 ----------------
clk=np.zeros(len(d))
for n in nodes:
    m=np.where(dom==n)[0]; last=step[m[0]]-1
    for i in m:
        clk[i]=max(step[i]-last,1)
        if fired0[i]: last=step[i]
print('\n'+'='*104); print('第三部分 控制「距上次廣播幾步」後的每步漂移（中位數；相對首段）'); print('='*104)
b0=b1=None
for a,b in BANDS:
    m=(ep>=a)&(ep<=b); x0=np.median(r0[m]/clk[m]); x1=np.median(r1[m]/clk[m])
    if b0 is None: b0,b1=x0,x1
    print(f'  ep{a:>3}–{b:<4} 參數 {x0:.3e}（{x0/b0:.2f}×）  原型 {x1:.3e}（{x1/b1:.2f}×）  距上次廣播中位 {np.median(clk[m]):.0f} 步')
print('⇒ 兩者都崩，原型只好約 2 倍 ⇒ 「原型角度無量綱、不隨訓練衰減」這條論證不成立。')
