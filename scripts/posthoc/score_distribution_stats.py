"""三堆的分數分布完整統計 + 標準化分離度。回答：AUROC 變差是哪一種不均勻造成的。"""
import os,sys,numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
import torch
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from joint_eval_mixed_stream import score_and_predict
PACS=["art_painting","cartoon","photo","sketch"]
def run(desc, ckdir, leave="cartoon", nodes=9, unk=6):
    avail=[d for d in PACS if d!=leave]; per=nodes//len(avail)
    tgt=TD.load_pacs_test_data("../datasets/",leave,128,4)[0]
    src={d:TD.load_pacs_test_data("../datasets/",d,128,4)[0] for d in avail}
    acc={k:{'p':[], 'e':[]} for k in ('1','2','3')}
    for i in range(nodes):
        ck=os.path.join(ckdir,f"{desc}_node_{i}_final.pth")
        bb,df=load_backbone_diffusion(ck,6,"cuda")
        own=avail[min(i//per,len(avail)-1)]
        o=score_and_predict(bb,df,src[own],[],"eps_mse","cuda",return_proto=True)
        _,_,en_s,_,lab_s,pr_s=o[:6]; m=lab_s!=unk
        acc['1']['p'].append(pr_s[m]); acc['1']['e'].append(-en_s[m])
        o=score_and_predict(bb,df,tgt,[],"eps_mse","cuda",return_proto=True)
        _,_,en_t,_,lab_t,pr_t=o[:6]; mk=lab_t!=unk
        acc['2']['p'].append(pr_t[mk]); acc['2']['e'].append(-en_t[mk])
        acc['3']['p'].append(pr_t[~mk]); acc['3']['e'].append(-en_t[~mk])
        del bb,df
    return {k:{s:np.concatenate(v) for s,v in d.items()} for k,d in acc.items()}
def report(tag,R,score='p'):
    S={k:R[k][score] for k in ('1','2','3')}
    print(f"\n--- {tag} / {'proto_angle(弧度)' if score=='p' else 'energy(取負)'} ---")
    for k,lbl in [('1','①來源域已知'),('2','②cartoon已知'),('3','③person')]:
        v=S[k]; print(f"  {lbl:<14} n={len(v):5d} mean={v.mean():8.4f} std={v.std():7.4f} "
                      f"p05={np.quantile(v,.05):7.4f} p50={np.quantile(v,.5):7.4f} p95={np.quantile(v,.95):7.4f}")
    def dprime(a,b): return (b.mean()-a.mean())/np.sqrt((a.var()+b.var())/2)
    print(f"  ★標準化分離度 d'（決定 AUROC 的量，非絕對差）")
    print(f"     畫風 d'(①→②) = {dprime(S['1'],S['2']):+.4f}   ← 理想 0")
    print(f"     語意 d'(①→③) = {dprime(S['1'],S['3']):+.4f}")
    print(f"     部署 d'(②→③) = {dprime(S['2'],S['3']):+.4f}   ← 越大越好")
    return S
FIX="v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"
AP ="v1_stage2_leave_cartoon_p1ap_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234"
A=run(FIX,"exp_result_"+FIX); B=run(AP,"exp_result_"+AP)
Sa=report("1a-fix",A); Sb=report("1ap",B)
report("1a-fix",A,'e'); report("1ap",B,'e')
print("\n\n=== ★ 壓縮率分解（proto_angle）：均勻壓縮不影響 AUROC，只有不均勻才會 ===")
print(f"{'堆':<16}{'1a-fix mean':>13}{'1ap mean':>11}{'壓縮率':>9}{'1a-fix std':>12}{'1ap std':>10}{'std壓縮率':>11}")
for k,lbl in [('1','①來源域已知'),('2','②cartoon已知'),('3','③person')]:
    a,b=Sa[k],Sb[k]
    print(f"{lbl:<14}{a.mean():>13.4f}{b.mean():>11.4f}{b.mean()/a.mean():>9.4f}"
          f"{a.std():>12.4f}{b.std():>10.4f}{b.std()/a.std():>11.4f}")
