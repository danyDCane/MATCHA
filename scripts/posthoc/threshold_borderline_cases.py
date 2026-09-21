"""門檻上的真實個案：logit 與 softmax 長什麼樣（dany 2026-09-11 要求看真資料）

用途：把「門檻附近的樣本」從統計量還原成看得見的個案，供口試回答用。
     每一組印 2 張：來源域切線上的、目標域已知類貼著門檻的、目標域 person 貼著門檻的，
     再各印 1 張兩端極值（最有把握的正常樣本／最明顯的未知）當對照。

⚠️ 個案是**舉例不是證據**，統計量在 §5.2／§5.3；本檔的用途是讓人看懂那些統計量在講什麼。

用法：./venv_matcha/bin/python scripts/posthoc/threshold_borderline_cases.py [fold] [node]
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/posthoc"); sys.path.insert(0, ".")
from osdg_eval import load_backbone_diffusion
from bn_common import bn_avg, apply_bn
from dood.prototype import class_centers, residual_projector, residual_score
import util
import test_domain_ood_scores as TD

np.set_printoptions(precision=2, suppress=True)
CLS = ["dog", "elephant", "giraffe", "guitar", "horse", "house"]
FOLD = sys.argv[1] if len(sys.argv) > 1 else "sketch"
NODE = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ARMS = [("StyleDDG+energy／原樣（讀出＝energy）",
         f"v1_stage2_leave_{FOLD}_nodiff_osdg_excl_person_seed2026_topo1234", False, False),
        ("我方+‖z⊥‖面／平均B（讀出＝殘差）",
         f"v1_stage2_leave_{FOLD}_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234"
         + ("_fix" if FOLD == "cartoon" else ""), True, True)]


def scan(bb, proj, dom):
    L, Y, S = [], [], []
    for b in TD.load_pacs_test_data("../datasets/", dom, 128, 4)[0]:
        d, y, _ = util.unpack_batch(b)
        with torch.no_grad():
            lg, vec = bb.forward_from_layer3(bb.forward_to_layer3_style(d.cuda(), communicator=None))
            s = residual_score(bb.project(vec), proj) if proj is not None else -torch.logsumexp(lg, 1)
        L.append(lg.cpu()); Y.append(np.asarray(y).flatten()); S.append(s.cpu())
    return torch.cat(L), np.concatenate(Y), torch.cat(S).numpy()


def main():
    own = list(np.load(f"results/osa/{FOLD}.npz", allow_pickle=True)["own"])[NODE]
    for arm, desc, ours, avg in ARMS:
        bb, _ = load_backbone_diffusion(
            os.path.join(f"exp_result_{desc}", f"{desc}_node_{NODE}_final.pth"), 6, "cuda")
        if avg:
            apply_bn(bb, bn_avg(f"exp_result_{desc}", desc, N=9, ckpt_tag="final"))
        bb.eval()
        proj = residual_projector(class_centers(bb.prototypes, bb.proto_count).to("cuda")) if ours else None
        Ls, Ys, Ss = scan(bb, proj, own)
        Lt, Yt, St = scan(bb, proj, FOLD)
        del bb
        m = Ys != 6
        tau = float(np.quantile(Ss[m], 0.95))
        print("=" * 104)
        print(f"★★ {arm}   fold={FOLD} node_{NODE} 來源畫風={own}")
        print(f"   門檻 τ = {tau:.4f}（來源域已知類 95 分位）")
        print("=" * 104)

        def show(title, idx, S, L, Y):
            print(f"\n  ── {title} ──")
            for i in idx:
                lg = L[i]; p = torch.softmax(lg, 0)
                true = CLS[Y[i]] if Y[i] != 6 else "person(未知)"
                print(f"    分數 {S[i]:+8.4f}（{'拒絕' if S[i] > tau else '放行'}，"
                      f"離門檻 {S[i]-tau:+.4f}）  真實={true:<13} 預測={CLS[lg.argmax()]}")
                print(f"      logit   = {lg.numpy()}")
                print(f"      softmax = {p.numpy()}   最大={p.max():.3f}")

        ik = np.where(Yt != 6)[0]; iu = np.where(Yt == 6)[0]
        near = lambda I: I[np.argsort(np.abs(St[I] - tau))][:2]
        isk = np.where(Ys != 6)[0]
        show("來源域・門檻切線上（被算成那 5% 的邊界）",
             isk[np.argsort(np.abs(Ss[isk] - tau))][:2], Ss, Ls, Ys)
        show("目標域已知類・貼著門檻", near(ik), St, Lt, Yt)
        show("目標域 person・貼著門檻", near(iu), St, Lt, Yt)
        lo = ik[St[ik] < np.quantile(St[ik], 0.05)]
        show("（對照）目標域已知類・最有把握", lo[np.argsort(St[lo])][:1], St, Lt, Yt)
        hi = iu[St[iu] > np.quantile(St[iu], 0.95)]
        show("（對照）目標域 person・最明顯的未知", hi[np.argsort(-St[hi])][:1], St, Lt, Yt)
        print()


if __name__ == "__main__":
    main()
