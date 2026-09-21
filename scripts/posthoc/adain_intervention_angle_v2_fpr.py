"""cartoon 到自己類別原型的 53.57°，有多少是 channel 統計量造成的？
確定性 AdaIN 介入（layer1/2/3，同 P3 位置），只用 adain()，不用 StyleExplore/MixStyle。"""
import os,sys,numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
import torch,util
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
    """域平均的 channel 統計量（確定性，不取樣）"""
    acc={k:[[],[]] for k in ("layer1","layer2","layer3")}
    for b in ld:
        d,_,_=util.unpack_batch(b); d=d.to("cuda")
        F=bb.extract_features_to_layer3(d)
        for k in acc:
            f=F[k]; B,C,H,W=f.shape; fl=f.view(B,C,-1)
            acc[k][0].append(fl.mean(2).cpu()); acc[k][1].append(fl.std(2).cpu())
    return {k:(torch.cat(v[0]).mean(0).cuda(), torch.cat(v[1]).mean(0).cuda()) for k,v in acc.items()}

@torch.no_grad()
def fwd(bb,x,st=None):
    """前向；st 不為 None 時在 layer1/2/3 之後做確定性 AdaIN"""
    h=bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(x))))
    h=bb.backbone.layer1(h)
    if st: h=adain(h,*st["layer1"])
    h=bb.backbone.layer2(h)
    if st: h=adain(h,*st["layer2"])
    h=bb.backbone.layer3(h)
    if st: h=adain(h,*st["layer3"])
    _,vec=bb.forward_from_layer3(h)
    return bb.project(vec)

@torch.no_grad()
def run(bb,ld,C,st=None):
    own,near,lab=[],[],[]
    for b in ld:
        d,y,_=util.unpack_batch(b); d=d.to("cuda"); y=np.asarray(y).flatten()
        z=fwd(bb,d,st)
        a=torch.arccos((z@C.t()).clamp(-1+1e-7,1-1e-7)).cpu().numpy()*57.2958
        m=y!=UNK
        if m.sum():
            own.append(a[m][np.arange(m.sum()),y[m].astype(int)]); near.append(a[m].min(1))
        lab.append(y)
    return np.concatenate(own),np.concatenate(near),np.concatenate(near)

RES={k:[[],[],[]] for k in ["①來源域(下界)","②cartoon 原樣(基準)","②+來源域統計量(介入)","②+另半cartoon統計量(控制)"]}
for i in range(N):
    bb,_=load_backbone_diffusion(os.path.join(CK,f"{DESC}_node_{i}_final.pth"),6,"cuda")
    C=class_centers(bb.prototypes,bb.proto_count)
    own_d=avail[min(i//per,len(avail)-1)]
    st_src=dom_stats(bb,src[own_d])          # 該節點自己來源域的統計量
    st_ctl=dom_stats(bb,tgt)                 # cartoon 自己的統計量（同域控制）
    for k,(ld,st) in {"①來源域(下界)":(src[own_d],None),
                      "②cartoon 原樣(基準)":(tgt,None),
                      "②+來源域統計量(介入)":(tgt,st_src),
                      "②+另半cartoon統計量(控制)":(tgt,st_ctl)}.items():
        o,n,raw=run(bb,ld,C,st); RES[k][0].append(o.mean()); RES[k][1].append(n.mean()); RES[k][2].append(raw)
    del bb
    print(f"  node_{i} 完成",flush=True)

base=np.mean(RES["②cartoon 原樣(基準)"][0]); low=np.mean(RES["①來源域(下界)"][0])
total=base-low
print("\n"+"="*72)
print(f"{'組別':<28}{'到自己類別':>11}{'到最近中心':>11}{'解釋比例':>10}")
print("="*72)
for k,(o,n,_r) in RES.items():
    om=np.mean(o)
    ex="" if "下界" in k or "基準" in k else f"{(base-om)/total*100:>9.1f}%"
    print(f"{k:<26}{om:>11.2f}°{np.mean(n):>10.2f}°{ex:>10}")
print("="*72)
print(f"可解釋總量 = 基準 {base:.2f}° − 下界 {low:.2f}° = {total:.2f}°")
ex_i=(base-np.mean(RES["②+來源域統計量(介入)"][0]))/total*100
ex_c=(base-np.mean(RES["②+另半cartoon統計量(控制)"][0]))/total*100
print(f"\n★ 介入組解釋 {ex_i:.1f}% ／ 控制組解釋 {ex_c:.1f}% ／ 淨效果 {ex_i-ex_c:.1f}%")
print()
tau=[np.quantile(x,.95) for x in RES["①來源域(下界)"][2]]
print("★ 誤拒率（門檻＝各節點來源域分數的95%分位，來源域未被介入）")
for k in ["②cartoon 原樣(基準)","②+來源域統計量(介入)","②+另半cartoon統計量(控制)"]:
    fr=np.mean([(b>t).mean() for b,t in zip(RES[k][2],tau)])
    print(f"   {k:<28} 誤拒率={fr:.4f}")
print("   對照 同checkpoint energy=0.3970  msp=0.3855")
print(f"★ 判準：控制組 >15% ⇒ 實驗無效 ｜ 淨 >50% ⇒ channel-stat 是主成分 ｜ 淨 <20% ⇒ 不是主因")
