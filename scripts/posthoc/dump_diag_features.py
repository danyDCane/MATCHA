"""補完 0815 的五個洞：①四臂方向vs長度 ②person 距離分布 ③512d 長度誤拒率是否假象
④person 被拉向哪幾類 ⑤‖proj‖ 誤拒惡化但部署變好。全 post-hoc、final ckpt、cartoon fold。"""
import os,sys,numpy as np,pickle
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
import torch,util
from sklearn.metrics import roc_auc_score
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers
PACS=["art_painting","cartoon","photo","sketch"]
CLS=["dog","elephant","giraffe","guitar","horse","house"]   # ImageFolder 字母序，person=6 已排除
ARMS=[("λ=0","v1_stage2_leave_cartoon_proto_lam0_m0.95w10_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234"),
      ("1a","v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234"),
      ("1a-fix","v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"),
      ("1ap","v1_stage2_leave_cartoon_p1ap_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234")]
def collect(desc,leave="cartoon",nodes=9,unk=6):
    ckdir="exp_result_"+desc
    avail=[d for d in PACS if d!=leave]; per=nodes//len(avail)
    tgt=TD.load_pacs_test_data("../datasets/",leave,128,4)[0]
    src={d:TD.load_pacs_test_data("../datasets/",d,128,4)[0] for d in avail}
    out=[]
    for i in range(nodes):
        bb,_=load_backbone_diffusion(os.path.join(ckdir,f"{desc}_node_{i}_final.pth"),6,"cuda")
        C=class_centers(bb.prototypes,bb.proto_count); D={}
        for tag,ld in [('s',src[avail[min(i//per,len(avail)-1)]]),('t',tgt)]:
            ang,nc,pn,vn,en,ys=[],[],[],[],[],[]
            with torch.no_grad():
                for b in ld:
                    d,y,_=util.unpack_batch(b); d=d.to("cuda")
                    z3=bb.forward_to_layer3_style(d,communicator=None)
                    lg,vec=bb.forward_from_layer3(z3)
                    p=bb.proj_head(vec) if bb.use_proj_head else vec
                    z=torch.nn.functional.normalize(p,dim=1)
                    a=torch.arccos((z@C.t()).clamp(-1+1e-7,1-1e-7))
                    ang.append(a.min(1).values.cpu().numpy()); nc.append(a.argmin(1).cpu().numpy())
                    pn.append(p.norm(dim=1).cpu().numpy()); vn.append(vec.norm(dim=1).cpu().numpy())
                    en.append(torch.logsumexp(lg,1).cpu().numpy()); ys.append(np.asarray(y).flatten())
            D[tag]={k:np.concatenate(v) for k,v in
                    dict(ang=ang,nc=nc,pn=pn,vn=vn,en=en,y=ys).items()}
        out.append(D); del bb
    return out
R={}
for tag,desc in ARMS:
    print(f"[collect] {tag}",flush=True); R[tag]=collect(desc)
pickle.dump(R,open(os.path.join(os.path.dirname(__file__),"..","..","logs","prototype_probe","0815_diag_features.pkl"),"wb"))
print("saved logs/prototype_probe/0815_diag_features.pkl")
