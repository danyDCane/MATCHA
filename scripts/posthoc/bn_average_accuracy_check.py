"""BN 平均對【分類準確率】與【誤拒率的來源】的影響——這是判斷它是不是假象的關鍵。"""
import os,sys,numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
import torch,util
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers
PACS=["art_painting","cartoon","photo","sketch"]
RUNS={"1a-fix":"v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix",
      "λ=0":"v1_stage2_leave_cartoon_proto_lam0_m0.95w10_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234"}
leave="cartoon";UNK=6;N=9;DEG=57.29577951308232
avail=[d for d in PACS if d!=leave];per=3
OWN=[avail[min(i//per,2)] for i in range(N)]
ld={d:TD.load_pacs_test_data("../datasets/",d,64,4)[0] for d in PACS}
@torch.no_grad()
def run(bb,C,loader):
    acc=[];ang=[]
    for b in loader:
        x,y,_=util.unpack_batch(b)
        h=bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(x.to("cuda")))))
        h=bb.backbone.layer1(h);h=bb.backbone.layer2(h);h=bb.backbone.layer3(h)
        logits,vec=bb.forward_from_layer3(h)
        yy=np.asarray(y).flatten();m=yy!=UNK
        pred=logits.argmax(1).cpu().numpy()
        if m.sum(): acc.append((pred[m]==yy[m]).astype(float))
        z=bb.project(vec).cpu().numpy();z=z/np.linalg.norm(z,axis=1,keepdims=True)
        ang.append(np.arccos(np.clip(z@C.T,-1+1e-7,1-1e-7)).min(1)*DEG)
    return float(np.concatenate(acc).mean()*100), np.concatenate(ang), np.concatenate([np.asarray(y).flatten() for _,y,_ in []]) if False else None
for tag,DESC in RUNS.items():
    CK="exp_result_"+DESC
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
    out={}
    for vn,av in [("原樣",None),("BN平均B",AVG)]:
        cart=[];src=[];q95=[];tgt_ang=[]
        for i in range(N):
            bb,_=load_backbone_diffusion(os.path.join(CK,f"{DESC}_node_{i}_final.pth"),6,"cuda")
            if av is not None:
                sd=bb.state_dict()
                for k,v in av.items(): sd[k].copy_(v.to(sd[k].device).to(sd[k].dtype))
            C=class_centers(bb.prototypes,bb.proto_count).cpu().numpy();C=C/np.linalg.norm(C,axis=1,keepdims=True)
            a_s,ang_s,_=run(bb,C,ld[OWN[i]]); a_c,ang_c,_=run(bb,C,ld[leave])
            src.append(a_s);cart.append(a_c);q95.append(np.quantile(ang_s,0.95))
            tgt_ang.append(ang_c.mean())
            del bb
        out[vn]=(np.mean(src),np.mean(cart),np.mean(q95),np.mean(tgt_ang))
    print("="*76); print(f"【{tag}】"); print("="*76)
    print(f"{'':<12}{'來源域 acc':>12}{'cartoon acc(DG)':>18}{'①門檻(95分位)':>16}{'②平均角度':>13}")
    for vn in out:
        s,c,q,t=out[vn]; print(f"{vn:<12}{s:>11.2f}%{c:>16.2f}%{q:>15.2f}°{t:>12.2f}°")
    s0,c0,q0,t0=out["原樣"];s1,c1,q1,t1=out["BN平均B"]
    print(f"{'Δ':<12}{s1-s0:>+11.2f}%{c1-c0:>+16.2f}%{q1-q0:>+15.2f}°{t1-t0:>+12.2f}°")
    print(f"  ★ 誤拒率下降來源：門檻上移 {q1-q0:+.2f}°  vs  ②本身變近 {t1-t0:+.2f}°")
