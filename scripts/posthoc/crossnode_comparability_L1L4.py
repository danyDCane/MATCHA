"""L1-L4：跨節點特徵空間到底可不可比？（dany 2026-08-17 設計的四層驗證）
關鍵：同一張圖必須餵進 9 個節點各自的完整模型（含各自的 BN），端到端比較。"""
import os,sys,numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
import torch,util
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers
PACS=["art_painting","cartoon","photo","sketch"]
DESC="v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"
CK="exp_result_"+DESC; leave="cartoon"; UNK=6; N=9; NB=3   # 每個 loader 取前 NB 個 batch
avail=[d for d in PACS if d!=leave]; per=N//len(avail)
CLS=["dog","elephant","giraffe","guitar","horse","house"]

def grab(ld,nb):
    xs,ys=[],[]
    for i,b in enumerate(ld):
        if i>=nb: break
        d,y,_=util.unpack_batch(b); xs.append(d); ys.append(np.asarray(y).flatten())
    return torch.cat(xs), np.concatenate(ys)

tgt_ld=TD.load_pacs_test_data("../datasets/",leave,64,4)[0]
src_ld=TD.load_pacs_test_data("../datasets/","art_painting",64,4)[0]
X_t,Y_t=grab(tgt_ld,NB); X_s,Y_s=grab(src_ld,NB)
print(f"取樣：來源域(art) {len(X_s)} 張、cartoon {len(X_t)} 張\n")

Zs=[];Zt=[];P=[]
for i in range(N):
    bb,_=load_backbone_diffusion(os.path.join(CK,f"{DESC}_node_{i}_final.pth"),6,"cuda")
    P.append(class_centers(bb.prototypes,bb.proto_count).cpu().numpy())
    with torch.no_grad():
        for X,acc in [(X_s,Zs),(X_t,Zt)]:
            out=[]
            for k in range(0,len(X),64):
                d=X[k:k+64].to("cuda")
                z3=bb.forward_to_layer3_style(d,communicator=None)
                _,vec=bb.forward_from_layer3(z3)
                out.append(bb.project(vec).cpu().numpy())
            acc.append(np.concatenate(out))
    del bb
Zs=np.stack(Zs); Zt=np.stack(Zt); P=np.stack(P)   # [9,Ns,128] [9,Nt,128] [9,6,128]
ang=lambda a,b: np.degrees(np.arccos(np.clip((a*b).sum(-1),-1,1)))

print("="*70)
print("★L1 同一張【來源域】圖片，在 9 個節點的特徵夾角")
d=[ang(Zs[i],Zs[j]) for i in range(N) for j in range(i+1,N)]
same_img=np.concatenate(d)
# 基準：同一個節點內，不同圖片之間的夾角
base=[]
for i in range(N):
    z=Zs[i]; c=np.clip(z@z.T,-1,1); iu=np.triu_indices(len(z),1)
    base.append(np.degrees(np.arccos(c[iu])))
base=np.concatenate(base)
print(f"   同一張圖 跨節點      : mean={same_img.mean():6.2f}°  p95={np.quantile(same_img,.95):6.2f}°")
print(f"   不同張圖 同節點(基準): mean={base.mean():6.2f}°  p05={np.quantile(base,.05):6.2f}°")
print(f"   ⇒ 比值 {same_img.mean()/base.mean():.3f}（越小越代表跨節點可比）")

print("\n★L2 同一張【cartoon】圖片，在 9 個節點的特徵夾角")
d=[ang(Zt[i],Zt[j]) for i in range(N) for j in range(i+1,N)]
same_t=np.concatenate(d)
baset=[]
for i in range(N):
    z=Zt[i]; c=np.clip(z@z.T,-1,1); iu=np.triu_indices(len(z),1)
    baset.append(np.degrees(np.arccos(c[iu])))
baset=np.concatenate(baset)
print(f"   同一張圖 跨節點      : mean={same_t.mean():6.2f}°  p95={np.quantile(same_t,.95):6.2f}°")
print(f"   不同張圖 同節點(基準): mean={baset.mean():6.2f}°")
print(f"   ⇒ 比值 {same_t.mean()/baset.mean():.3f}   （對照來源域 {same_img.mean()/base.mean():.3f}）")

print("\n★L3 54 個原型（9節點×6類）的兩兩夾角結構")
flat=P.reshape(-1,128); lab=np.tile(np.arange(6),N); nod=np.repeat(np.arange(N),6)
A=np.degrees(np.arccos(np.clip(flat@flat.T,-1,1)))
iu=np.triu_indices(len(flat),1)
sc=(lab[iu[0]]==lab[iu[1]]); sn=(nod[iu[0]]==nod[iu[1]])
v=A[iu]
print(f"   同類別 跨節點 : n={int((sc&~sn).sum()):4d}  mean={v[sc&~sn].mean():6.2f}°  max={v[sc&~sn].max():6.2f}°")
print(f"   不同類別 跨節點: n={int((~sc&~sn).sum()):4d}  mean={v[~sc&~sn].mean():6.2f}°  min={v[~sc&~sn].min():6.2f}°")
print(f"   不同類別 同節點: n={int((~sc&sn).sum()):4d}  mean={v[~sc&sn].mean():6.2f}°  min={v[~sc&sn].min():6.2f}°")
gap=v[~sc&~sn].min()-v[sc&~sn].max()
print(f"   ⇒ 【最壞情況】同類別最遠 {v[sc&~sn].max():.2f}° vs 不同類別最近 {v[~sc&~sn].min():.2f}°  間隔 {gap:+.2f}°")
print(f"   ⇒ {'✅ 完全不重疊，類別結構強於節點結構' if gap>0 else '❌ 有重疊，存在錯配風險'}")

print("\n★L4 cartoon 樣本到 54 個原型：最近的那個是不是『自己類別』？")
m=Y_t!=UNK
for tag,Zq,lq in [("cartoon已知",Zt,Y_t)]:
    hit=[];hit6=[]
    for i in range(N):
        a=np.degrees(np.arccos(np.clip(Zq[i][m]@flat.T,-1,1)))   # [n,54]
        nearest=lab[a.argmin(1)]
        hit.append((nearest==lq[m]).mean())
        a6=np.degrees(np.arccos(np.clip(Zq[i][m]@P[i].T,-1,1)))  # 只用自己的 6 個
        hit6.append((a6.argmin(1)==lq[m]).mean())
    print(f"   用【54 個原型】: 最近＝真實類別 {np.mean(hit):.4f}")
    print(f"   用【自己 6 個】 : 最近＝真實類別 {np.mean(hit6):.4f}")
    print(f"   ⇒ 差 {np.mean(hit)-np.mean(hit6):+.4f}")
