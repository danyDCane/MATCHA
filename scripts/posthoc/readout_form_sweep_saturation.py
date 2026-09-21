"""補：α 掃描沒到飽和就下判決是錯的。掃到極限（含純 std 的漸近值）。
   並更正 V5 熵的符號（原式取了負號，方向反了；AUROC(-s)=1-AUROC(s)）。"""
import os,sys,numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
import torch,util
from sklearn.metrics import roc_auc_score
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers
PACS=["art_painting","cartoon","photo","sketch"]
RUNS={"1a-fix":"v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix",
      "λ=0":"v1_stage2_leave_cartoon_proto_lam0_m0.95w10_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234"}
leave="cartoon";UNK=6;N=9;DEG=57.29577951308232;TM=0.3572
avail=[d for d in PACS if d!=leave];per=3;OWN=[avail[min(i//per,2)] for i in range(N)]
ld={d:TD.load_pacs_test_data("../datasets/",d,64,4)[0] for d in PACS}
nrm=lambda v: v/np.linalg.norm(v,axis=-1,keepdims=True)
def bnavg(CK,D):
    S=[torch.load(os.path.join(CK,f"{D}_node_{i}_final.pth"),map_location="cpu",weights_only=False)["backbone_state"] for i in range(N)]
    BK=[k for k in S[0] if k.endswith("running_mean") or k.endswith("running_var")];A={}
    for k in BK:
        if k.endswith("running_mean"): A[k]=torch.stack([S[i][k].float() for i in range(N)]).mean(0)
    for k in BK:
        if k.endswith("running_var"):
            mk=k.replace("running_var","running_mean")
            mi=torch.stack([S[i][mk].float() for i in range(N)]);vi=torch.stack([S[i][k].float() for i in range(N)])
            A[k]=(vi+mi**2).mean(0)-mi.mean(0)**2
    del S;return A
@torch.no_grad()
def coll(bb,loader,C):
    A,L,Y=[],[],[]
    for b in loader:
        x,y,_=util.unpack_batch(b)
        h=bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(x.to("cuda")))))
        h=bb.backbone.layer1(h);h=bb.backbone.layer2(h);h=bb.backbone.layer3(h)
        lo,v=bb.forward_from_layer3(h);z=nrm(bb.project(v).cpu().numpy())
        A.append(np.arccos(np.clip(z@C.T,-1+1e-7,1-1e-7))*DEG);L.append(lo.cpu().numpy());Y.append(np.asarray(y).flatten())
    return np.concatenate(A),np.concatenate(L),np.concatenate(Y)
ACC={}
for tag,D in RUNS.items():
    CK="exp_result_"+D;AV=bnavg(CK,D);rec=[]
    for i in range(N):
        bb,_=load_backbone_diffusion(os.path.join(CK,f"{D}_node_{i}_final.pth"),6,"cuda")
        sd=bb.state_dict()
        for k,v in AV.items(): sd[k].copy_(v.to(sd[k].device).to(sd[k].dtype))
        C=nrm(class_centers(bb.prototypes,bb.proto_count).cpu().numpy())
        D1,L1,y1=coll(bb,ld[OWN[i]],C);D2,L2,y2=coll(bb,ld[leave],C);del bb
        k1,k2,u2=y1!=UNK,y2!=UNK,y2==UNK
        rec.append(dict(D=(D1[k1],D2[k2],D2[u2]),L=(L1[k1],L2[k2],L2[u2])))
    ACC[tag]=rec
ALPHA=[0,0.5,1,2,3,5,8,12,20,50,1e6]
def run(rec,kind,tag):
    print("="*84);print(f"【{tag} / {'原型 min−α·std' if kind=='D' else 'energy−α·std(logit)'}】掃到飽和");print("="*84)
    print(f"  {'α':>10}{'部署 AUROC':>14}{'誤拒@放行.3572':>18}")
    best=(-1,None)
    for a in ALPHA:
        au,fp=[],[]
        for d in rec:
            def sc(X,Lg):
                if kind=="D":
                    m=X.min(1);s=X.std(1)
                    return -s if a>1e5 else m-a*s
                t=torch.from_numpy(Lg);e=(-torch.logsumexp(t,1)).numpy();s=Lg.std(1)
                return -s if a>1e5 else e-a*s
            s2=sc(d[kind][1],d["L"][1]);s3=sc(d[kind][2],d["L"][2])
            au.append(roc_auc_score([0]*len(s2)+[1]*len(s3),np.r_[s2,s3]))
            fp.append(float((s2>float(np.quantile(s3,TM))).mean()))
        A_,F_=float(np.mean(au)),float(np.mean(fp))
        if A_>best[0]: best=(A_,a)
        lab="∞(純 −std)" if a>1e5 else f"{a:g}"
        print(f"  {lab:>10}{A_:>14.4f}{F_:>18.4f}")
    print(f"  ⇒ 最佳 α={best[1]}　部署 {best[0]:.4f}\n")
    return best
b_p=run(ACC["1a-fix"],"D","1a-fix");b_e=run(ACC["1a-fix"],"L","1a-fix");b_e0=run(ACC["λ=0"],"L","λ=0")
print("="*84);print("★ 更正：V5 熵的符號原本取反（AUROC(−s)=1−AUROC(s)）");print("="*84)
for T in [2.,5.,10.,20.,50.]:
    au=[]
    for d in ACC["1a-fix"]:
        f=lambda X:(lambda p: (p*np.log(p+1e-12)).sum(1)*-1)(torch.softmax(torch.from_numpy(-X/T),1).numpy())
        s2,s3=f(d["D"][1]),f(d["D"][2])
        au.append(roc_auc_score([0]*len(s2)+[1]*len(s3),np.r_[s2,s3]))
    print(f"  熵(T={T:g}) 正確符號  部署 {np.mean(au):.4f}")
print("\n"+"="*84);print("★ 最終判決");print("="*84)
print(f"  原型最佳（掃到飽和） {b_p[0]:.4f} @α={b_p[1]}")
print(f"  energy 同等待遇最佳  {b_e[0]:.4f} @α={b_e[1]}")
print(f"  λ=0 energy 同等待遇  {b_e0[0]:.4f} @α={b_e0[1]}")
print(f"  ⇒ 原型 vs 目標 0.8380：{b_p[0]-0.8380:+.4f}　vs energy 同等待遇：{b_p[0]-b_e[0]:+.4f}")
