"""energy 與原型讀出的逐節點誤拒率離散度——決定「跨節點門檻不一致」是不是共通問題。"""
import os,sys,numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
import torch,util
from sklearn.metrics import roc_auc_score
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers
PACS=["art_painting","cartoon","photo","sketch"]
DESC="v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"
CK="exp_result_"+DESC; leave="cartoon";UNK=6;N=9;DEG=57.29577951308232
avail=[d for d in PACS if d!=leave];per=3;OWN=[avail[min(i//per,2)] for i in range(N)]
SH={"art_painting":"art","photo":"photo","sketch":"sketch"}
ld={d:TD.load_pacs_test_data("../datasets/",d,64,4)[0] for d in PACS}
S=[torch.load(os.path.join(CK,f"{DESC}_node_{i}_final.pth"),map_location="cpu",weights_only=False)["backbone_state"] for i in range(N)]
BK=[k for k in S[0] if k.endswith("running_mean") or k.endswith("running_var")]
AVG={}
for k in BK:
    if k.endswith("running_mean"): AVG[k]=torch.stack([S[i][k].float() for i in range(N)]).mean(0)
for k in BK:
    if k.endswith("running_var"):
        mk=k.replace("running_var","running_mean")
        mi=torch.stack([S[i][mk].float() for i in range(N)]);vi=torch.stack([S[i][k].float() for i in range(N)])
        AVG[k]=(vi+mi**2).mean(0)-mi.mean(0)**2
del S
@torch.no_grad()
def get(bb,C,loader):
    A,E,Y=[],[],[]
    for b in loader:
        x,y,_=util.unpack_batch(b)
        h=bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(x.to("cuda")))))
        h=bb.backbone.layer1(h);h=bb.backbone.layer2(h);h=bb.backbone.layer3(h)
        lo,v=bb.forward_from_layer3(h)
        z=bb.project(v).cpu().numpy();z=z/np.linalg.norm(z,axis=1,keepdims=True)
        A.append(np.arccos(np.clip(z@C.T,-1+1e-7,1-1e-7)).min(1)*DEG)
        E.append((-torch.logsumexp(lo,1)).cpu().numpy());Y.append(np.asarray(y).flatten())
    return np.concatenate(A),np.concatenate(E),np.concatenate(Y)
for bn in ["原樣","BN平均"]:
    rows=[]
    for i in range(N):
        bb,_=load_backbone_diffusion(os.path.join(CK,f"{DESC}_node_{i}_final.pth"),6,"cuda")
        if bn=="BN平均":
            sd=bb.state_dict()
            for k,v in AVG.items(): sd[k].copy_(v.to(sd[k].device).to(sd[k].dtype))
        C=class_centers(bb.prototypes,bb.proto_count).cpu().numpy();C=C/np.linalg.norm(C,axis=1,keepdims=True)
        a1,e1,y1=get(bb,C,ld[OWN[i]]); a2,e2,y2=get(bb,C,ld[leave])
        m1,m2=y1!=UNK,y2!=UNK
        r={}
        for nm,(s1,s2) in [("proto",(a1[m1],a2)),("energy",(e1[m1],e2))]:
            t=np.quantile(s1,0.95); r[nm+"_fpr"]=float((s2[m2]>t).mean())
            r[nm+"_dep"]=roc_auc_score([0]*m2.sum()+[1]*(~m2).sum(),np.r_[s2[m2],s2[~m2]])
        rows.append((SH[OWN[i]],r)); del bb
    print("="*80); print(f"【{bn}】逐節點誤拒率（各節點用自己的來源域校準門檻）"); print("="*80)
    print(f"{'node':<6}{'群':<8}{'原型誤拒':>11}{'energy誤拒':>12}{'原型部署':>11}{'energy部署':>12}")
    for i,(g,r) in enumerate(rows):
        print(f"{i:<6}{g:<8}{r['proto_fpr']:>11.4f}{r['energy_fpr']:>12.4f}{r['proto_dep']:>11.4f}{r['energy_dep']:>12.4f}")
    for nm,lab in [("proto","原型"),("energy","energy")]:
        v=[r[nm+"_fpr"] for _,r in rows]
        print(f"  {lab:<8} 誤拒率 平均 {np.mean(v):.4f}  全距 {max(v)-min(v):.4f}  最低 {min(v):.4f}  最高 {max(v):.4f}  倍數 {max(v)/max(min(v),1e-9):.1f}x")
    print()
