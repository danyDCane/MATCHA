# -*- coding: utf-8 -*-
"""觸發候選：逐類別加權 ＋ 子空間旋轉 ＋ 類別中心降秩（2026-10-02，九節點 ep50/100/150/200 checkpoint）"""
print("#"*30, "第一部分：逐類別加權（perclass）", "#"*30)
# 問三件事（全部離線、用 9 個節點的 ep50/100/150/200 checkpoint）：
#  (a) 九個節點對「哪一類」的分歧程度，逐類別差很多嗎？（差很多 ⇒ 可以當權重）
#  (b) 同一個節點，六個類別的漂移量差很多嗎？（差很多 ⇒ mean 丟掉了資訊）
#  (c) 用 (a) 當權重去加權 (b)，跟直接平均差多少？（差很多 ⇒ 不是換刻度）
#  (d) 「六個中心張成的子空間」轉了多少，跟「六個中心各自轉了多少的平均」是不是同一件事？
import os,sys,glob
sys.path.insert(0,'/home/server5090/Desktop/M11307320/MATCHA')
import torch, torch.nn.functional as F, numpy as np
RUN=('exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_'
     'osdg_excl_person_seed2026_topo1234_fix')
NC,NS,PER=6,3,3; EPS=[50,100,150,200]
CLS=['dog','elephant','giraffe','guitar','horse','house']   # PACS 去 person 後字典序
def own(j,e):
    f=glob.glob(os.path.join(RUN,f'*_node_{j}_epoch_{e}.pth'))[0]
    bs=torch.load(f,map_location='cpu',weights_only=False)['backbone_state']
    return F.normalize(bs['prototypes'][:,j//PER,:].float(),dim=1)   # [6,P]
C={e:torch.stack([own(j,e) for j in range(9)]) for e in EPS}          # [9,6,P]
ang=lambda A,B: torch.rad2deg(torch.arccos((F.normalize(A,dim=-1)*F.normalize(B,dim=-1)).sum(-1).clamp(-1+1e-7,1-1e-7)))

print('='*88); print('(a) 九個節點對每個類別的分歧程度（ep200；到該類九節點平均的夾角）'); print('='*88)
X=C[200]
sp=[]
for c in range(NC):
    m=F.normalize(X[:,c,:].mean(0),dim=0)
    d=[float(ang(X[j,c,:],m)) for j in range(9)]
    within=np.mean([np.mean([float(ang(X[j,c,:],F.normalize(X[[k for k in range(9) if k//PER==s],c,:].mean(0),dim=0)))
                             for j in range(9) if j//PER==s]) for s in range(NS)])
    sp.append(np.mean(d))
    print(f'  {CLS[c]:<10} 全體分歧 {np.mean(d):6.2f}°   同畫風內分歧 {within:6.2f}°')
sp=np.array(sp); print(f'  ⇒ 逐類別全體分歧：平均 {sp.mean():.2f}°、範圍 {sp.min():.2f}–{sp.max():.2f}°、'
      f'最大/最小 {sp.max()/sp.min():.2f}×、變異係數 {sp.std()/sp.mean():.3f}')

print('\n'+'='*88); print('(b) 同一節點、六個類別的漂移量差多少（相鄰 checkpoint）'); print('='*88)
print(f'{"區間":<14}{"逐類別漂移（度）":<46}{"最大/最小":>10}{"變異係數":>10}')
drift={}
for a,b in zip(EPS[:-1],EPS[1:]):
    per=np.array([[float(ang(C[b][j,c,:],C[a][j,c,:])) for c in range(NC)] for j in range(9)])  # [9,6]
    drift[(a,b)]=per
    m=per.mean(0)
    print(f'ep{a}→{b:<8}'+' '.join(f'{v:5.2f}' for v in m)+f'{m.max()/m.min():>14.2f}{m.std()/m.mean():>10.3f}')
r=np.concatenate([ (p/p.mean(1,keepdims=True)) for p in drift.values()])
print(f'  ⇒ 逐(節點,區間) 內部：六類漂移的 最大/最小 中位 {np.median([p.max()/p.min() for p in np.concatenate(list(drift.values())).reshape(-1,NC)]):.2f}×')

print('\n'+'='*88); print('(c) 用「全體分歧」當權重 vs 直接平均：是不是換刻度？'); print('='*88)
w=sp/sp.sum()
rows=[]
for (a,b),per in drift.items():
    plain=per.mean(1); wt=(per*w).sum(1)*NC
    for j in range(9): rows.append((plain[j],wt[j]))
rows=np.array(rows)
from scipy.stats import spearmanr,pearsonr
print(f'  n={len(rows)}（9 節點 × 3 區間）  Pearson {pearsonr(rows[:,0],rows[:,1]).statistic:.4f}'
      f'   Spearman {spearmanr(rows[:,0],rows[:,1]).statistic:.4f}')
print(f'  加權/未加權 比值：中位 {np.median(rows[:,1]/rows[:,0]):.4f}、範圍 {(rows[:,1]/rows[:,0]).min():.4f}–{(rows[:,1]/rows[:,0]).max():.4f}')
print('  ⇒ 相關 ≈1 且比值幾乎是常數 ⇒ 只是換刻度；相關明顯 <1 ⇒ 真的排序不同')

print('\n'+'='*88); print('(d) 子空間旋轉 vs 六個中心各自旋轉的平均'); print('='*88)
print(f'{"區間":<14}{"主角度平均(子空間)":>20}{"六中心夾角平均":>18}{"比值":>8}')
for a,b in zip(EPS[:-1],EPS[1:]):
    s1=[];s2=[]
    for j in range(9):
        Q1=torch.linalg.qr(C[a][j].T)[0]; Q2=torch.linalg.qr(C[b][j].T)[0]
        sv=torch.linalg.svdvals(Q1.T@Q2).clamp(-1,1)
        s1.append(float(torch.rad2deg(torch.arccos(sv)).mean()))
        s2.append(float(ang(C[b][j],C[a][j]).mean()))
    print(f'ep{a}→{b:<8}{np.mean(s1):>20.3f}{np.mean(s2):>18.3f}{np.mean(s1)/np.mean(s2):>8.3f}')
print('  ⇒ 比值遠離 1 ⇒ 兩者不是同一個量（子空間旋轉忽略「在面內的移動」）')

print("\n"+"#"*30, "第二部分：子空間旋轉與零點（subspace）", "#"*30)
# 問三件事：① 六個主角度的分布（是不是被某一個尾巴撐大）
#          ② 零點：六個向量【各自隨機】動同樣幅度時，子空間會轉多少？
#          ③ 它跟規則1 是不是換刻度（像 (c) 那樣查相關與比值）
import os,sys,glob
sys.path.insert(0,'/home/server5090/Desktop/M11307320/MATCHA')
import torch, torch.nn.functional as F, numpy as np
from scipy.stats import pearsonr, spearmanr
RUN=('exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_'
     'osdg_excl_person_seed2026_topo1234_fix')
NC,PER=6,3; EPS=[50,100,150,200]
def own(j,e):
    f=glob.glob(os.path.join(RUN,f'*_node_{j}_epoch_{e}.pth'))[0]
    bs=torch.load(f,map_location='cpu',weights_only=False)['backbone_state']
    return F.normalize(bs['prototypes'][:,j//PER,:].float(),dim=1)
C={e:torch.stack([own(j,e) for j in range(9)]) for e in EPS}
ang=lambda A,B: torch.rad2deg(torch.arccos((F.normalize(A,dim=-1)*F.normalize(B,dim=-1)).sum(-1).clamp(-1+1e-7,1-1e-7)))
def pang(A,B):
    Q1=torch.linalg.qr(A.T)[0]; Q2=torch.linalg.qr(B.T)[0]
    return torch.rad2deg(torch.arccos(torch.linalg.svdvals(Q1.T@Q2).clamp(-1,1)))

print('='*88); print('① 六個主角度的分布（node_0，看是不是被尾巴撐大）'); print('='*88)
for a,b in zip(EPS[:-1],EPS[1:]):
    v=pang(C[a][0],C[b][0]).numpy()
    print(f'ep{a}→{b:<6} 主角度 '+' '.join(f'{x:6.2f}' for x in sorted(v))+f'   平均 {v.mean():6.2f}  中心夾角平均 {float(ang(C[b][0],C[a][0]).mean()):6.2f}')

print('\n'+'='*88); print('② 零點：六個中心【各自隨機】轉同樣幅度 ⇒ 子空間會轉多少？'); print('='*88)
g=torch.Generator().manual_seed(2026)
def rot_null(X, degs, n=200):
    """把每個中心沿隨機正交方向轉 degs[c] 度，重算主角度"""
    out=[]
    for _ in range(n):
        Y=[]
        for c in range(NC):
            x=X[c]; r=torch.randn(x.shape,generator=g); r=r-(r@x)*x; r=F.normalize(r,dim=0)
            th=np.radians(degs[c]); Y.append(F.normalize(x*np.cos(th)+r*np.sin(th),dim=0))
        out.append(float(pang(X,torch.stack(Y)).mean()))
    return np.mean(out), np.std(out)
print(f'{"區間":<12}{"實測子空間":>12}{"零點(隨機同幅)":>16}{"實測/零點":>11}   判讀')
for a,b in zip(EPS[:-1],EPS[1:]):
    real=np.mean([float(pang(C[a][j],C[b][j]).mean()) for j in range(9)])
    nulls=[]
    for j in range(9):
        degs=[float(ang(C[b][j,c,:],C[a][j,c,:])) for c in range(NC)]
        m,s=rot_null(C[a][j],degs,n=60); nulls.append(m)
    nm=np.mean(nulls)
    print(f'ep{a}→{b:<6}{real:>12.2f}{nm:>16.2f}{real/nm:>11.3f}   '
          f'{"✅ 高於零點、有結構" if real/nm>1.15 else ("🚨 落在零點上 ⇒ 只是「六個向量各自動了」的換算" if real/nm<1.15 and real/nm>0.85 else "低於零點")}')

print('\n'+'='*88); print('③ 子空間旋轉 vs 規則1（六中心夾角平均）：是不是換刻度？'); print('='*88)
rows=[]
for a,b in zip(EPS[:-1],EPS[1:]):
    for j in range(9):
        rows.append((float(ang(C[b][j],C[a][j]).mean()), float(pang(C[a][j],C[b][j]).mean())))
rows=np.array(rows)
print(f'  n={len(rows)}  Pearson {pearsonr(rows[:,0],rows[:,1]).statistic:.4f}  Spearman {spearmanr(rows[:,0],rows[:,1]).statistic:.4f}')
print(f'  比值 中位 {np.median(rows[:,1]/rows[:,0]):.3f}  範圍 {(rows[:,1]/rows[:,0]).min():.3f}–{(rows[:,1]/rows[:,0]).max():.3f}')
# 同區間內（去掉「時間」這個共同趨勢）
for a,b in zip(EPS[:-1],EPS[1:]):
    sub=np.array([(float(ang(C[b][j],C[a][j]).mean()), float(pang(C[a][j],C[b][j]).mean())) for j in range(9)])
    print(f'  ep{a}→{b}（9 節點內，去掉時間趨勢）Spearman {spearmanr(sub[:,0],sub[:,1]).statistic:+.3f}')

print("\n"+"#"*30, "第三部分：六個類別中心的奇異值（降秩檢查）", "#"*30)
print(f'{"節點":<9}{"σ1":>7}{"σ2":>7}{"σ3":>7}{"σ4":>7}{"σ5":>7}{"σ6":>8}{"σ6/σ1":>9}{"‖Σc‖":>8}')
rr=[]
for j in range(9):
    X=C[200][j]; sv=torch.linalg.svdvals(X).numpy(); rr.append(sv[5]/sv[0])
    print(f'node_{j:<4}'+''.join(f'{v:>7.3f}' for v in sv[:5])+f'{sv[5]:>8.4f}{sv[5]/sv[0]:>9.4f}{float(X.sum(0).norm()):>8.3f}')
g2=torch.Generator().manual_seed(0)
rnd=[float((lambda s: s[5]/s[0])(torch.linalg.svdvals(F.normalize(torch.randn(6,128,generator=g2),dim=1)))) for _ in range(50)]
print(f'⇒ σ6/σ1 實測平均 {np.mean(rr):.4f}；六個隨機單位向量 {np.mean(rnd):.4f} ⇒ 實質只張 5 維（單純形構型）')
print('⇒ 對面讀出的影響另見 scripts/posthoc/plane_readout_rank5_check.py（結果：無影響）')
