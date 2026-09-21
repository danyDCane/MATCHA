"""推論時把【所有】測試樣本的 channel 統計量正規化到來源域，算完整四軸。
守門員：部署 AUROC 不得下降（0815 §4.3 定案）。"""
import os,sys,numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
import torch,util
from sklearn.metrics import roc_auc_score
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers
from style_transforms import adain
PACS=["art_painting","cartoon","photo","sketch"]
DESC="v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"
CK="exp_result_"+DESC; leave="cartoon"; UNK=6; N=9
avail=[d for d in PACS if d!=leave]; per=N//len(avail)
tgt=TD.load_pacs_test_data("../datasets/",leave,64,4)[0]
src={d:TD.load_pacs_test_data("../datasets/",d,64,4)[0] for d in avail}

@torch.no_grad()
def dom_stats(bb,ld):
    acc={k:[[],[]] for k in ("layer1","layer2","layer3")}
    for b in ld:
        d,_,_=util.unpack_batch(b); d=d.to("cuda")
        F=bb.extract_features_to_layer3(d)
        for k in acc:
            f=F[k]; B,C,H,W=f.shape; fl=f.view(B,C,-1)
            acc[k][0].append(fl.mean(2).cpu()); acc[k][1].append(fl.std(2).cpu())
    return {k:(torch.cat(v[0]).mean(0).cuda(), torch.cat(v[1]).mean(0).cuda()) for k,v in acc.items()}

@torch.no_grad()
def collect(bb,ld,C,st):
    ang,en,ys=[],[],[]
    for b in ld:
        d,y,_=util.unpack_batch(b); d=d.to("cuda")
        h=bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(d))))
        h=bb.backbone.layer1(h)
        if st: h=adain(h,*st["layer1"])
        h=bb.backbone.layer2(h)
        if st: h=adain(h,*st["layer2"])
        h=bb.backbone.layer3(h)
        if st: h=adain(h,*st["layer3"])
        lg,vec=bb.forward_from_layer3(h)
        z=bb.project(vec)
        ang.append(torch.arccos((z@C.t()).clamp(-1+1e-7,1-1e-7)).min(1).values.cpu().numpy())
        en.append(-torch.logsumexp(lg,1).cpu().numpy()); ys.append(np.asarray(y).flatten())
    return np.concatenate(ang),np.concatenate(en),np.concatenate(ys)

R={("原樣",s):[[],[],[]] for s in ["proto","energy"]}
R.update({("正規化到來源域",s):[[],[],[]] for s in ["proto","energy"]})
for i in range(N):
    bb,_=load_backbone_diffusion(os.path.join(CK,f"{DESC}_node_{i}_final.pth"),6,"cuda")
    C=class_centers(bb.prototypes,bb.proto_count)
    own=avail[min(i//per,len(avail)-1)]; st=dom_stats(bb,src[own])
    for tag,S in [("原樣",None),("正規化到來源域",st)]:
        a1,e1,y1=collect(bb,src[own],C,S)      # ① 也做同樣處理（推論時不知道誰是誰）
        a2,e2,y2=collect(bb,tgt,C,S)
        for sc,(s1,s2) in [("proto",(a1,a2)),("energy",(e1,e2))]:
            R[(tag,sc)][0].append(s1[y1!=UNK]); R[(tag,sc)][1].append(s2[y2!=UNK]); R[(tag,sc)][2].append(s2[y2==UNK])
    del bb
    print(f"  node_{i}",flush=True)
au=lambda p,n: roc_auc_score(np.r_[np.ones(len(p)),np.zeros(len(n))],np.r_[p,n])
print("\n"+"="*80)
print(f"{'處理':<18}{'分數':<9}{'畫風→.5':>9}{'語意↑':>8}{'部署↑(守門員)':>15}{'誤拒↓':>9}")
print("="*80)
for (tag,sc),(A,B,Cc) in R.items():
    st_=np.mean([au(b,a) for a,b in zip(A,B)]); se=np.mean([au(c,a) for a,c in zip(A,Cc)])
    dp=np.mean([au(c,b) for b,c in zip(B,Cc)]); fr=np.mean([(b>np.quantile(a,.95)).mean() for a,b in zip(A,B)])
    print(f"{tag:<16}{sc:<9}{st_:>9.4f}{se:>8.4f}{dp:>15.4f}{fr:>9.4f}")
print("="*80)
for sc in ["proto","energy"]:
    a=R[("原樣",sc)]; b=R[("正規化到來源域",sc)]
    d=lambda X,f: np.mean([f(x) for x in zip(*X)])
    dp0=np.mean([au(c,bb_) for bb_,c in zip(a[1],a[2])]); dp1=np.mean([au(c,bb_) for bb_,c in zip(b[1],b[2])])
    fr0=np.mean([(x>np.quantile(y,.95)).mean() for y,x in zip(a[0],a[1])])
    fr1=np.mean([(x>np.quantile(y,.95)).mean() for y,x in zip(b[0],b[1])])
    print(f"  {sc:<8} Δ部署={dp1-dp0:+.4f}（守門員，須≥0） Δ誤拒={fr1-fr0:+.4f}")
