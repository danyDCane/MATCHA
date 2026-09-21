"""角距離丟掉了範數(強度)。範數本身有判別力嗎？＝原型讀出輸給 energy 的結構性原因？"""
import os,sys,numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
import torch, util
from sklearn.metrics import roc_auc_score
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers
PACS=["art_painting","cartoon","photo","sketch"]
def collect(desc,ckdir,leave="cartoon",nodes=9,unk=6):
    avail=[d for d in PACS if d!=leave]; per=nodes//len(avail)
    tgt=TD.load_pacs_test_data("../datasets/",leave,128,4)[0]
    src={d:TD.load_pacs_test_data("../datasets/",d,128,4)[0] for d in avail}
    R=[]
    for i in range(nodes):
        bb,_=load_backbone_diffusion(os.path.join(ckdir,f"{desc}_node_{i}_final.pth"),6,"cuda")
        C=class_centers(bb.prototypes,bb.proto_count)
        own=avail[min(i//per,len(avail)-1)]; D={}
        for tag,ld in [('s',src[own]),('t',tgt)]:
            ang,pn,vn,en,ys=[],[],[],[],[]
            with torch.no_grad():
                for b in ld:
                    d,y,_=util.unpack_batch(b); d=d.to("cuda")
                    z3=bb.forward_to_layer3_style(d,communicator=None)
                    lg,vec=bb.forward_from_layer3(z3)
                    p=bb.proj_head(vec) if bb.use_proj_head else vec     # 未正規化的投影特徵
                    z=torch.nn.functional.normalize(p,dim=1)
                    a=torch.arccos((z@C.t()).clamp(-1+1e-7,1-1e-7))
                    ang.append(a.min(1).values.cpu().numpy())
                    pn.append(p.norm(dim=1).cpu().numpy())               # ‖proj_head(vec)‖
                    vn.append(vec.norm(dim=1).cpu().numpy())             # ‖512維 penultimate‖
                    en.append(torch.logsumexp(lg,1).cpu().numpy())
                    ys.append(np.asarray(y).flatten())
            D[tag]=[np.concatenate(x) for x in (ang,pn,vn,en,ys)]
        R.append(D); del bb
    return R
def evaluate(tag,R,unk=6):
    print(f"\n--- {tag} ---")
    SC={"角距離 min_c (現行)":lambda a,p,v,e: a,
        "‖proj‖ 取負 (只有長度)":lambda a,p,v,e: -p,
        "‖512d penult‖ 取負":lambda a,p,v,e: -v,
        "角距離 − log‖proj‖":lambda a,p,v,e: a-np.log(p),
        "energy 取負 (對照)":lambda a,p,v,e: -e}
    for name,fn in SC.items():
        sty,sem,dep,fpr=[],[],[],[]
        for D in R:
            (a1,p1,v1,e1,y1),(a2,p2,v2,e2,y2)=D['s'],D['t']
            A=fn(a1,p1,v1,e1)[y1!=unk]; B=fn(a2,p2,v2,e2)[y2!=unk]; Cc=fn(a2,p2,v2,e2)[y2==unk]
            au=lambda p,n: roc_auc_score(np.r_[np.ones(len(p)),np.zeros(len(n))],np.r_[p,n])
            sty.append(au(B,A)); sem.append(au(Cc,A)); dep.append(au(Cc,B))
            fpr.append((B>np.quantile(A,.95)).mean())
        print(f"  {name:<24} 畫風={np.mean(sty):.4f} 語意={np.mean(sem):.4f} "
              f"部署={np.mean(dep):.4f} 誤拒={np.mean(fpr):.4f}")
FIX="v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"
AP ="v1_stage2_leave_cartoon_p1ap_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234"
evaluate("1a-fix",collect(FIX,"exp_result_"+FIX))
evaluate("1ap",  collect(AP,"exp_result_"+AP))
