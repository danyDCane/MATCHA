"""一致性核對：A 組（cartoon＋來源域統計量）是否重現 0818 §3.4 的 48.48°"""
import os,sys,numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
import torch,util
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers
from style_transforms import adain
PACS=["art_painting","cartoon","photo","sketch"]
DESC="v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"
CK="exp_result_"+DESC; UNK=6;N=9;DEG=57.29577951308232
avail=[d for d in PACS if d!="cartoon"]; per=3
tgt=TD.load_pacs_test_data("../datasets/","cartoon",64,4)[0]
src={d:TD.load_pacs_test_data("../datasets/",d,64,4)[0] for d in avail}
@torch.no_grad()
def dom_stats(bb,ld):
    acc={k:[[],[]] for k in ("layer1","layer2","layer3")}
    for b in ld:
        d,_,_=util.unpack_batch(b); F=bb.extract_features_to_layer3(d.to("cuda"))
        for k in acc:
            f=F[k];B,C_,H,W=f.shape;fl=f.view(B,C_,-1)
            acc[k][0].append(fl.mean(2).cpu()); acc[k][1].append(fl.std(2).cpu())
    return {k:(torch.cat(v[0]).mean(0).cuda(),torch.cat(v[1]).mean(0).cuda()) for k,v in acc.items()}
@torch.no_grad()
def ang(bb,ld,C,st=None):
    out=[]
    for b in ld:
        d,y,_=util.unpack_batch(b); y=np.asarray(y).flatten()
        h=bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(d.to("cuda")))))
        h=bb.backbone.layer1(h)
        if st: h=adain(h,*st["layer1"])
        h=bb.backbone.layer2(h)
        if st: h=adain(h,*st["layer2"])
        h=bb.backbone.layer3(h)
        if st: h=adain(h,*st["layer3"])
        _,vec=bb.forward_from_layer3(h); z=bb.project(vec).cpu().numpy()
        a=np.arccos(np.clip(z@C.T,-1+1e-7,1-1e-7))*DEG
        m=y!=UNK
        if m.sum(): out.append(a[m][np.arange(m.sum()),y[m].astype(int)])
    return np.concatenate(out).mean()
R=[]
for i in range(N):
    bb,_=load_backbone_diffusion(os.path.join(CK,f"{DESC}_node_{i}_final.pth"),6,"cuda")
    C=class_centers(bb.prototypes,bb.proto_count).cpu().numpy(); C=C/np.linalg.norm(C,axis=1,keepdims=True)
    od=avail[min(i//per,len(avail)-1)]
    R.append(ang(bb,tgt,C,dom_stats(bb,src[od]))); del bb
print(f"A組（cartoon＋來源域統計量）= {np.mean(R):.2f}°   0818 §3.4 報 48.48°   差 {np.mean(R)-48.48:+.2f}°")
