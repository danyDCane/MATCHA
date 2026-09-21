"""疊加測試：角距離 + 512維範數 + 推論時風格正規化，對上「什麼都沒有」的 baseline。
靶：λ=0 的 msp 誤拒率 0.3534，目標降 6pp ⇒ ≤0.2934，且部署 AUROC 不得輸 λ=0 energy 的 0.8234。"""
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
        d,_,_=util.unpack_batch(b); d=d.to("cuda"); F=bb.extract_features_to_layer3(d)
        for k in acc:
            f=F[k];B,C,H,W=f.shape;fl=f.view(B,C,-1)
            acc[k][0].append(fl.mean(2).cpu()); acc[k][1].append(fl.std(2).cpu())
    return {k:(torch.cat(v[0]).mean(0).cuda(),torch.cat(v[1]).mean(0).cuda()) for k,v in acc.items()}
@torch.no_grad()
def collect(bb,ld,C,st):
    A,V,E,Y=[],[],[],[]
    for b in ld:
        d,y,_=util.unpack_batch(b); d=d.to("cuda")
        h=bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(d))))
        h=bb.backbone.layer1(h)
        if st: h=adain(h,*st["layer1"])
        h=bb.backbone.layer2(h)
        if st: h=adain(h,*st["layer2"])
        h=bb.backbone.layer3(h)
        if st: h=adain(h,*st["layer3"])
        lg,vec=bb.forward_from_layer3(h); z=bb.project(vec)
        A.append(torch.arccos((z@C.t()).clamp(-1+1e-7,1-1e-7)).min(1).values.cpu().numpy())
        V.append(vec.norm(dim=1).cpu().numpy()); E.append(-torch.logsumexp(lg,1).cpu().numpy())
        Y.append(np.asarray(y).flatten())
    return [np.concatenate(x) for x in (A,V,E,Y)]
zs=lambda x,r:(x-r.mean())/r.std()
SC={"角距離":lambda a,v,e,r:a, "energy":lambda a,v,e,r:e,
    "角距離+範數":lambda a,v,e,r:0.5*zs(a,r[0])+0.5*zs(-v,-r[1])}
R={}
for i in range(N):
    bb,_=load_backbone_diffusion(os.path.join(CK,f"{DESC}_node_{i}_final.pth"),6,"cuda")
    C=class_centers(bb.prototypes,bb.proto_count); own=avail[min(i//per,len(avail)-1)]
    st=dom_stats(bb,src[own])
    for tag,S in [("原樣",None),("＋推論正規化",st)]:
        a1,v1,e1,y1=collect(bb,src[own],C,S); a2,v2,e2,y2=collect(bb,tgt,C,S)
        m1=y1!=UNK; m2=y2!=UNK; m3=y2==UNK
        ref=(a1[m1],v1[m1])
        for sn,f in SC.items():
            k=(tag,sn); R.setdefault(k,[[],[],[]])
            R[k][0].append(f(a1[m1],v1[m1],e1[m1],ref))
            R[k][1].append(f(a2[m2],v2[m2],e2[m2],ref))
            R[k][2].append(f(a2[m3],v2[m3],e2[m3],ref))
    del bb
au=lambda p,n: roc_auc_score(np.r_[np.ones(len(p)),np.zeros(len(n))],np.r_[p,n])
print("="*78)
print(f"{'處理':<16}{'分數':<14}{'畫風→.5':>9}{'部署↑':>9}{'誤拒↓':>9}   vs 靶")
print("="*78)
print(f"{'【靶】什麼都沒有':<14}{'λ=0 msp':<14}{0.7423:>9.4f}{0.7532:>9.4f}{0.3534:>9.4f}")
print(f"{'':<14}{'λ=0 energy':<14}{0.7459:>9.4f}{0.8234:>9.4f}{0.3656:>9.4f}")
print("-"*78)
for (tag,sn),(A,B,Cc) in R.items():
    st_=np.mean([au(b,a) for a,b in zip(A,B)]); dp=np.mean([au(c,b) for b,c in zip(B,Cc)])
    fr=np.mean([(b>np.quantile(a,.95)).mean() for a,b in zip(A,B)])
    w=("✅贏誤拒 %+.4f"%(0.3534-fr)) if fr<0.3534 else ("❌輸誤拒 %+.4f"%(0.3534-fr))
    g="✅達標" if fr<=0.2934 else ""
    print(f"{tag:<14}{sn:<14}{st_:>9.4f}{dp:>9.4f}{fr:>9.4f}   {w} {g}")
print("="*78)
