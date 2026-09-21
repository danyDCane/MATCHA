"""為什麼原型差 7.08°，換上去卻只變 0.00°？逐樣本看，而不是只看平均。"""
import os,sys,numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
import torch,util
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers
DESC="v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"
CK="exp_result_"+DESC; UNK=6; DEG=57.29577951308232
ld=TD.load_pacs_test_data("../datasets/","art_painting",64,4)[0]
P=[]
for i in [0,1,3]:   # node0=art, node1=art(同風格他節點), node3=photo(不同風格)
    bb,_=load_backbone_diffusion(os.path.join(CK,f"{DESC}_node_{i}_final.pth"),6,"cuda")
    C=class_centers(bb.prototypes,bb.proto_count).cpu().numpy(); P.append(C/np.linalg.norm(C,axis=1,keepdims=True))
    if i==0:
        Z,Y=[],[]
        with torch.no_grad():
            for b in ld:
                d,y,_=util.unpack_batch(b)
                h=bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(d.to("cuda")))))
                h=bb.backbone.layer1(h);h=bb.backbone.layer2(h);h=bb.backbone.layer3(h)
                _,v=bb.forward_from_layer3(h); Z.append(bb.project(v).cpu().numpy()); Y.append(np.asarray(y).flatten())
        Z=np.concatenate(Z);Y=np.concatenate(Y);m=Y!=UNK;Z,Y=Z[m],Y[m].astype(int)
        Z=Z/np.linalg.norm(Z,axis=1,keepdims=True)
    del bb
def ang(C): return np.arccos(np.clip((Z@C.T)[np.arange(len(Y)),Y],-1+1e-7,1-1e-7))*DEG
a0,a1,a3=ang(P[0]),ang(P[1]),ang(P[2])
pd01=float(np.arccos(np.clip((P[0]*P[1]).sum(1),-1,1)).mean()*DEG)
pd03=float(np.arccos(np.clip((P[0]*P[2]).sum(1),-1,1)).mean()*DEG)
print("="*74)
print("node0(art) 的 art 圖片，換不同的原型當尺（n=%d 樣本）"%len(Y))
print("="*74)
for nm,a,pd in [("自己的原型 (node0)",a0,0.0),("同風格他節點 (node1)",a1,pd01),("不同風格節點 (node3=photo)",a3,pd03)]:
    d=a-a0
    print(f"  {nm:<26} 平均角度 {a.mean():6.2f}°   原型距自己 {pd:5.2f}°")
    if pd>0:
        print(f"  {'':<26} 逐樣本變化：平均 {d.mean():+.2f}°  標準差 {d.std():.2f}°  "
              f"變近 {(d<0).mean()*100:.0f}% / 變遠 {(d>0).mean()*100:.0f}%")
        print(f"  {'':<26} 最多變近 {d.min():+.2f}°  最多變遠 {d.max():+.2f}°  |變化|平均 {np.abs(d).mean():.2f}°")
print()
print("★ 讀法：原型確實被挪了幾度，逐樣本也確實有人變近有人變遠，")
print("        但變近與變遠的人數幾乎各半、幅度相抵 ⇒ 平均下來的淨效果 ≈ 0")
