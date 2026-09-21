"""雜訊基準：四臂 × ep50/100/150/200 的方向與長度讀出軌跡。
同一臂內相鄰 epoch 的波動＝『訓練後期時間變異』，是判斷跨臂效應是否真實的最低門檻。"""
import os,sys,numpy as np,json
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
import torch,util
from sklearn.metrics import roc_auc_score
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers
PACS=["art_painting","cartoon","photo","sketch"]
ARMS=[("λ=0","v1_stage2_leave_cartoon_proto_lam0_m0.95w10_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234"),
      ("1a","v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234"),
      ("1a-fix","v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"),
      ("1ap","v1_stage2_leave_cartoon_p1ap_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234")]
UNK=6; leave="cartoon"; nodes=9
avail=[d for d in PACS if d!=leave]; per=nodes//len(avail)
tgt=TD.load_pacs_test_data("../datasets/",leave,128,4)[0]
src={d:TD.load_pacs_test_data("../datasets/",d,128,4)[0] for d in avail}
def one(desc,ep):
    ckdir="exp_result_"+desc; per_node=[]
    for i in range(nodes):
        p=os.path.join(ckdir,f"{desc}_node_{i}_epoch_{ep}.pth")
        if not os.path.exists(p): return None
        bb,_=load_backbone_diffusion(p,6,"cuda"); C=class_centers(bb.prototypes,bb.proto_count); D={}
        for tag,ld in [('s',src[avail[min(i//per,len(avail)-1)]]),('t',tgt)]:
            ang,pn,vn,en,ys=[],[],[],[],[]
            with torch.no_grad():
                for b in ld:
                    d,y,_=util.unpack_batch(b); d=d.to("cuda")
                    z3=bb.forward_to_layer3_style(d,communicator=None)
                    lg,vec=bb.forward_from_layer3(z3)
                    pr=bb.proj_head(vec) if bb.use_proj_head else vec
                    z=torch.nn.functional.normalize(pr,dim=1)
                    a=torch.arccos((z@C.t()).clamp(-1+1e-7,1-1e-7))
                    ang.append(a.min(1).values.cpu().numpy()); pn.append(pr.norm(dim=1).cpu().numpy())
                    vn.append(vec.norm(dim=1).cpu().numpy()); en.append(torch.logsumexp(lg,1).cpu().numpy())
                    ys.append(np.asarray(y).flatten())
            D[tag]={k:np.concatenate(v) for k,v in dict(ang=ang,pn=pn,vn=vn,en=en,y=ys).items()}
        per_node.append(D); del bb
    SC={"dir":lambda d:d['ang'],"len_proj":lambda d:-d['pn'],
        "len_512":lambda d:-d['vn'],"combo":lambda d:d['ang']-np.log(d['pn']),"energy":lambda d:-d['en']}
    out={}
    for name,fn in SC.items():
        v=[]
        for D in per_node:
            s,t=D['s'],D['t']; A=fn(s)[s['y']!=UNK]; B=fn(t)[t['y']!=UNK]; Cc=fn(t)[t['y']==UNK]
            au=lambda p,n: roc_auc_score(np.r_[np.ones(len(p)),np.zeros(len(n))],np.r_[p,n])
            v.append([au(B,A),au(Cc,A),au(Cc,B),(B>np.quantile(A,.95)).mean()])
        v=np.array(v); out[name]={"mean":v.mean(0).tolist(),"node_std":v.std(0).tolist()}
    return out
res={}
for tag,desc in ARMS:
    res[tag]={}
    for ep in [50,100,150,200]:
        r=one(desc,ep)
        if r: res[tag][ep]=r; print(f"[done] {tag} ep{ep} dir_deploy={r['dir']['mean'][2]:.4f} "
                                    f"len_deploy={r['len_proj']['mean'][2]:.4f}",flush=True)
json.dump(res,open(os.path.join(os.path.dirname(__file__),"..","..","logs","prototype_probe","0815_traj_four_arms.json"),"w"))
print("SAVED")
