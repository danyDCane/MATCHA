"""診斷：部署 AUROC 掉，是「特徵學壞」還是「分數讀的是絕對半徑」？
post-hoc 比較四種由同一組角距離矩陣導出的分數。零重訓、同一次前向。"""
import os,sys,numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
from sklearn.metrics import roc_auc_score
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from joint_eval_mixed_stream import score_and_predict
PACS=["art_painting","cartoon","photo","sketch"]
VAR={
 "min_c (現行)":            lambda A: A.min(1),
 "min_c − mean_c (相對)":   lambda A: A.min(1)-A.mean(1),
 "min_c / mean_c (比例)":   lambda A: A.min(1)/A.mean(1),
 "min2_c − min_c (margin)": lambda A: -(np.sort(A,1)[:,1]-A.min(1)),
}
def run(desc,ckdir,leave="cartoon",nodes=9,unk=6):
    avail=[d for d in PACS if d!=leave]; per=nodes//len(avail)
    tgt=TD.load_pacs_test_data("../datasets/",leave,128,4)[0]
    src={d:TD.load_pacs_test_data("../datasets/",d,128,4)[0] for d in avail}
    res={k:{'sty':[],'sem':[],'dep':[],'fpr':[]} for k in VAR}
    for i in range(nodes):
        bb,df=load_backbone_diffusion(os.path.join(ckdir,f"{desc}_node_{i}_final.pth"),6,"cuda")
        own=avail[min(i//per,len(avail)-1)]
        o=score_and_predict(bb,df,src[own],[],"eps_mse","cuda",return_proto=True,return_proto_full=True)
        lab_s,pf_s=o[4],o[6]; ms=lab_s!=unk
        o=score_and_predict(bb,df,tgt,[],"eps_mse","cuda",return_proto=True,return_proto_full=True)
        lab_t,pf_t=o[4],o[6]; mk=lab_t!=unk
        for name,fn in VAR.items():
            A=fn(pf_s)[ms]; B=fn(pf_t)[mk]; C=fn(pf_t)[~mk]
            au=lambda p,n: roc_auc_score(np.r_[np.ones(len(p)),np.zeros(len(n))],np.r_[p,n])
            tau=np.quantile(A,.95)
            res[name]['sty'].append(au(B,A)); res[name]['sem'].append(au(C,A))
            res[name]['dep'].append(au(C,B)); res[name]['fpr'].append((B>tau).mean())
        del bb,df
    return {k:{m:float(np.mean(v)) for m,v in d.items()} for k,d in res.items()}
FIX="v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"
AP ="v1_stage2_leave_cartoon_p1ap_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234"
A=run(FIX,"exp_result_"+FIX); B=run(AP,"exp_result_"+AP)
print(f"\n{'分數形式':<26}{'臂':<9}{'畫風→.5':>9}{'語意↑':>8}{'部署↑':>8}{'誤拒↓':>8}")
print("="*70)
for k in VAR:
    for tag,R in [("1a-fix",A),("1ap",B)]:
        r=R[k]; print(f"{k:<24}{tag:<9}{r['sty']:>9.4f}{r['sem']:>8.4f}{r['dep']:>8.4f}{r['fpr']:>8.4f}")
    d=B[k]['dep']-A[k]['dep']
    print(f"{'':<24}{'Δ部署':<9}{'':>9}{'':>8}{d:>+8.4f}")
    print("-"*70)
print("\n★ 對照：energy 部署 1a-fix .8242 / 1ap .8214（靶＝λ=0 energy .8234）")
