"""L_comp 為何讓誤拒率變差？M1(①變窄→門檻下移) vs M2(①②差距擴大)"""
import os,sys,numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
import torch,util
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers
PACS=["art_painting","cartoon","photo","sketch"]
M={"λ=0":"v1_stage2_leave_cartoon_proto_lam0_m0.95w10_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234",
   "1a-fix":"v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"}
leave="cartoon";UNK=6;N=9;avail=[d for d in PACS if d!=leave];per=N//len(avail)
tgt=TD.load_pacs_test_data("../datasets/",leave,64,4)[0]
src={d:TD.load_pacs_test_data("../datasets/",d,64,4)[0] for d in avail}
@torch.no_grad()
def go(bb,ld,C):
    A,Y=[],[]
    for b in ld:
        d,y,_=util.unpack_batch(b); d=d.to("cuda")
        z3=bb.forward_to_layer3_style(d,communicator=None); _,v=bb.forward_from_layer3(z3)
        z=bb.project(v)
        A.append(torch.arccos((z@C.t()).clamp(-1+1e-7,1-1e-7)).min(1).values.cpu().numpy()*57.2958)
        Y.append(np.asarray(y).flatten())
    return np.concatenate(A),np.concatenate(Y)
print(f"{'模型':<9}{'①mean':>8}{'①std':>7}{'①p95(門檻)':>11}{'②mean':>8}{'②std':>7}{'③mean':>8}"
      f"{'①②差':>8}{'②③差':>8}{'誤拒':>8}")
print("="*84)
for mn,desc in M.items():
    S1,S2,S3,TAU=[],[],[],[]
    for i in range(N):
        bb,_=load_backbone_diffusion(os.path.join("exp_result_"+desc,f"{desc}_node_{i}_final.pth"),6,"cuda")
        C=class_centers(bb.prototypes,bb.proto_count)
        a1,y1=go(bb,src[avail[min(i//per,2)]],C); a2,y2=go(bb,tgt,C)
        S1.append(a1[y1!=UNK]); S2.append(a2[y2!=UNK]); S3.append(a2[y2==UNK]); del bb
    f=lambda X,g: np.mean([g(x) for x in X])
    tau=[np.quantile(x,.95) for x in S1]
    fr=np.mean([(b>t).mean() for b,t in zip(S2,tau)])
    print(f"{mn:<9}{f(S1,np.mean):>8.2f}{f(S1,np.std):>7.2f}{np.mean(tau):>11.2f}"
          f"{f(S2,np.mean):>8.2f}{f(S2,np.std):>7.2f}{f(S3,np.mean):>8.2f}"
          f"{f(S2,np.mean)-f(S1,np.mean):>8.2f}{f(S3,np.mean)-f(S2,np.mean):>8.2f}{fr:>8.4f}")
