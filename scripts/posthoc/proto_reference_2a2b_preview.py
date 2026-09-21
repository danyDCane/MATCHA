"""2b 的 post-hoc 預演：把讀出從『自己的 6 個原型』換成『18 格(6類×3畫風)』或『54 個』，
   直接算四軸。這是機制鏈假設 A1/A2/A3 的最終檢驗——不用重訓。"""
import os,sys,numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
import torch,util
from sklearn.metrics import roc_auc_score
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers
PACS=["art_painting","cartoon","photo","sketch"]
DESC="v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"
CK="exp_result_"+DESC; leave="cartoon"; UNK=6; N=9
avail=[d for d in PACS if d!=leave]; per=N//len(avail)
tgt=TD.load_pacs_test_data("../datasets/",leave,128,4)[0]
src={d:TD.load_pacs_test_data("../datasets/",d,128,4)[0] for d in avail}

P=[]                                   # 先收集 9 節點原型
for i in range(N):
    bb,_=load_backbone_diffusion(os.path.join(CK,f"{DESC}_node_{i}_final.pth"),6,"cuda")
    P.append(class_centers(bb.prototypes,bb.proto_count).cpu().numpy()); del bb
P=np.stack(P)                          # [9,6,128]
dom_of=np.array([min(i//per,2) for i in range(N)])
P18=np.stack([P[dom_of==d].mean(0) for d in range(3)])           # [3,6,128] 同畫風三節點先平均
P18=P18/np.linalg.norm(P18,axis=-1,keepdims=True)
P54=P.reshape(-1,128)                                            # [54,128]
P6avg=P.mean(0); P6avg/=np.linalg.norm(P6avg,axis=-1,keepdims=True)  # 2a：6 個原型跨節點平均

def scores(Z, ref):                    # ref: [K,128] → 每樣本到最近參照點的角距離
    return np.degrees(np.arccos(np.clip(Z@ref.T,-1,1))).min(1)

rows={k:[[],[],[]] for k in ["自己6個(現況)","2a:6個跨節點平均","2b:18格","54個全部"]}
for i in range(N):
    bb,_=load_backbone_diffusion(os.path.join(CK,f"{DESC}_node_{i}_final.pth"),6,"cuda")
    own=avail[min(i//per,len(avail)-1)]; Z={}
    with torch.no_grad():
        for tag,ld in [('s',src[own]),('t',tgt)]:
            zs,ys=[],[]
            for b in ld:
                d,y,_=util.unpack_batch(b); d=d.to("cuda")
                z3=bb.forward_to_layer3_style(d,communicator=None)
                _,vec=bb.forward_from_layer3(z3)
                zs.append(bb.project(vec).cpu().numpy()); ys.append(np.asarray(y).flatten())
            Z[tag]=(np.concatenate(zs),np.concatenate(ys))
    del bb
    (z1,y1),(z2,y2)=Z['s'],Z['t']
    REF={"自己6個(現況)":P[i], "2a:6個跨節點平均":P6avg,
         "2b:18格":P18.reshape(-1,128), "54個全部":P54}
    for k,r in REF.items():
        rows[k][0].append(scores(z1[y1!=UNK],r))
        rows[k][1].append(scores(z2[y2!=UNK],r))
        rows[k][2].append(scores(z2[y2==UNK],r))

print(f"{'讀出用的參照點':<20}{'①來源域':>9}{'②cartoon':>10}{'③person':>9}{'畫風→.5':>9}{'部署↑':>8}{'誤拒↓':>8}")
print("="*74)
for k,(A,B,C) in rows.items():
    au=lambda p,n: np.mean([roc_auc_score(np.r_[np.ones(len(x)),np.zeros(len(y))],np.r_[x,y]) for x,y in zip(p,n)])
    fpr=np.mean([(b>np.quantile(a,.95)).mean() for a,b in zip(A,B)])
    print(f"{k:<18}{np.mean([x.mean() for x in A]):>9.2f}{np.mean([x.mean() for x in B]):>10.2f}"
          f"{np.mean([x.mean() for x in C]):>9.2f}{au(B,A):>9.4f}{au(C,B):>8.4f}{fpr:>8.4f}")
print("\n對照：同 checkpoint 的 energy 畫風=0.7514 部署=0.8242 誤拒=0.3970")
