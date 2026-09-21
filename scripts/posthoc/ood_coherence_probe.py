"""person(OOD) 的表徵一致性：std 膨脹是「散到各類別去」還是「整體變鬆但方向仍一致」？"""
import os,sys,numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
import torch
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from joint_eval_mixed_stream import score_and_predict
PACS=["art_painting","cartoon","photo","sketch"]
def feats(desc,ckdir,leave="cartoon",nodes=9,unk=6):
    avail=[d for d in PACS if d!=leave]; per=nodes//len(avail)
    tgt=TD.load_pacs_test_data("../datasets/",leave,128,4)[0]
    src={d:TD.load_pacs_test_data("../datasets/",d,128,4)[0] for d in avail}
    out=[]
    for i in range(nodes):
        bb,df=load_backbone_diffusion(os.path.join(ckdir,f"{desc}_node_{i}_final.pth"),6,"cuda")
        own=avail[min(i//per,len(avail)-1)]
        Z={}
        for tag,ld in [('1',src[own]),('t',tgt)]:
            zs,ys=[],[]
            with torch.no_grad():
                for b in ld:
                    import util
                    d,y,_=util.unpack_batch(b); d=d.to("cuda")
                    z3=bb.forward_to_layer3_style(d,communicator=None)
                    _,vec=bb.forward_from_layer3(z3)
                    zs.append(bb.project(vec).cpu().numpy()); ys.append(np.asarray(y).flatten())
            Z[tag]=(np.concatenate(zs),np.concatenate(ys))
        z1,y1=Z['1']; zt,yt=Z['t']
        from dood.prototype import class_centers
        C=class_centers(bb.prototypes,bb.proto_count).cpu().numpy()
        out.append((z1[y1!=unk], zt[yt!=unk], zt[yt==unk], C))
        del bb,df
    return out
def stats(tag,F):
    print(f"\n--- {tag} ---")
    rows={}
    for name,idx in [('①來源域已知',0),('②cartoon已知',1),('③person',2)]:
        pair,ent,rad=[],[],[]
        for z1,z2,z3,C in F:
            Z=[z1,z2,z3][idx]
            rng=np.random.default_rng(0); s=Z[rng.choice(len(Z),min(600,len(Z)),replace=False)]
            cos=np.clip(s@s.T,-1,1); iu=np.triu_indices(len(s),1)
            pair.append(np.degrees(np.arccos(cos[iu])).mean())          # 樣本彼此的平均夾角
            nc=np.argmin(np.degrees(np.arccos(np.clip(Z@C.T,-1,1))),1)  # 最近中心是哪一類
            p=np.bincount(nc,minlength=6)/len(nc); p=p[p>0]
            ent.append(-(p*np.log(p)).sum()/np.log(6))                  # 正規化熵 0~1
            rad.append(np.linalg.norm(Z.mean(0)))                       # 平均向量長度＝方向一致性
        rows[name]=(np.mean(pair),np.mean(ent),np.mean(rad))
        print(f"  {name:<14} 樣本兩兩夾角={np.mean(pair):6.2f}°  最近中心分布熵={np.mean(ent):.4f}  "
              f"平均向量長={np.mean(rad):.4f}")
    return rows
FIX="v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"
AP ="v1_stage2_leave_cartoon_p1ap_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234"
a=stats("1a-fix",feats(FIX,"exp_result_"+FIX)); b=stats("1ap",feats(AP,"exp_result_"+AP))
print(f"\n=== Δ (1ap − 1a-fix) ===")
print(f"{'堆':<16}{'Δ兩兩夾角':>12}{'Δ分布熵':>11}{'Δ向量長':>11}")
for k in a: print(f"{k:<14}{b[k][0]-a[k][0]:>+12.2f}{b[k][1]-a[k][1]:>+11.4f}{b[k][2]-a[k][2]:>+11.4f}")
