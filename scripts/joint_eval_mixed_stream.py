"""
Joint "classification-under-domain-shift WITH rejection" evaluation on a MIXED test
stream, for the decentralized P2P MATCHA setting (post-hoc, no retraining).

Mixed stream per LOO fold:  P_wild = (1-pi_s)*[unseen-domain PACS test, labeled] + pi_s*[far-OOD]
  - unseen-domain ID (covariate shift) -> must ACCEPT and classify correctly
  - far-OOD (SVHN/DTD/noise, semantic)  -> must REJECT (accepting it is always an error)

Reject score g(x): ON = diffusion score (eps_mse; HIGHER = more OOD) | baselines = MSP, energy.
Metric: risk-coverage curve -> AURC (primary), AUGRC, risk@cov=0.9; plus closed-set ID Acc
(must be held constant across score functions) and detection AUROC (sanity).

Reuses: test_domain_ood_scores (loaders + load_checkpoint), dood get_diffusion_scores,
collapse_analysis (backbone/diffusion build). No training, no checkpoint writes.
"""
import os
import sys
import csv
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn.functional as F

import util
import collapse_analysis as CA
from dood.utils.diffusion import get_diffusion_scores
from sklearn.metrics import roc_auc_score
import test_domain_ood_scores as TD

PACS = CA.PACS


def load_backbone_diffusion(ckpt_path, device):
    backbone = util.select_model(7, CA.build_backbone_args()).to(device)
    diffusion = CA.get_diffusion_model(
        ft_size=512, denoiser_type="unet0d",
        diffusion_denoiser_channels=512, num_diffusion_steps=1000,
    ).to(device)
    CA.load_checkpoint(ckpt_path, backbone, diffusion, device)
    backbone.eval(); diffusion.eval()
    return backbone, diffusion


@torch.no_grad()
def score_and_predict(backbone, diffusion, loader, diff_steps, score_type, device,
                      return_proto=False, return_proto_full=False):
    """Returns per-sample: diffusion score, MSP, energy-confidence(logsumexp), pred, label.

    return_proto=True 時**額外**回傳第 6 個元素：原型角距離 S(x)=min_c arccos(z·c_c)
    （階段 1 的檢測讀出，0803 §2.2）。預設 False ⇒ 既有 7 個呼叫點的回傳數量完全不變。

    return_proto_full=True 時再**額外**回傳一個 [N, C] 的**完整角距離矩陣**（到每個類別中心）。
    ⚠️ 為什麼需要完整矩陣：檢測分數取的是 `min_c`，它分不出「特徵散開」與「特徵跑到別的
       類別去」。一個 cartoon 的狗若被推到馬的群附近，`min_c` 量到的是「到馬中心」的距離
       （看起來很近、很正常），只有「到**狗**中心」的距離才會揭露它跑掉了。
       兩者相減＝該樣本坐得對不對（0811 dany）。

    ⚠️ 量到的是 **6 個類別中心**、不是 18 個原型取 min：後者會讓坐在三個小群「中間」的
       covariate-shifted ID 到每個小群都有 r_見過 的距離，而坐在小群上的來源樣本距離≈0
       ⇒ 分數本身就在製造畫風 AUROC 的落差（0803 §2.2 的框）。
    """
    centers = None
    _need_proto = return_proto or return_proto_full
    if _need_proto:
        if not hasattr(backbone, 'prototypes'):
            raise RuntimeError(
                "return_proto/return_proto_full=True 但 checkpoint 沒有 prototypes buffer——"
                "該 run 不是用 --use_proto_reg 訓練的。")
        from dood.prototype import class_centers
        centers = class_centers(backbone.prototypes, backbone.proto_count)
    d_sc, msp, en, preds, labels, proto_sc = [], [], [], [], [], []
    proto_all = []
    for batch in loader:
        data, y, _ = util.unpack_batch(batch)
        data = data.to(device)
        if diffusion is not None:
            latents = backbone.intermediate_forward(data)             # latent fed to diffusion
            s, _ = get_diffusion_scores(latents, diffusion, diff_steps, score_type,
                                        normalize=True, dtype=torch.float32)
        else:
            # 階段 1 起 checkpoint 可能沒有 diffusion（USE_OOD=0）；填 NaN，呼叫端須忽略該欄
            s = torch.full((data.size(0),), float('nan'))
        z3 = backbone.forward_to_layer3_style(data, communicator=None)  # clean forward
        logits, vec = backbone.forward_from_layer3(z3)
        d_sc.append(np.asarray(s.detach().cpu()).flatten())
        msp.append(F.softmax(logits, 1).max(1).values.cpu().numpy())
        en.append(torch.logsumexp(logits, 1).cpu().numpy())            # higher = more ID
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(np.asarray(y).flatten() if y is not None else np.full(len(data), -1))
        if _need_proto:
            # 與 dood.prototype.detection_score 同一條式子（含相同 clamp），確保
            # `ang.min(1)` 與 detection_score 的輸出逐位一致。
            _cos = (backbone.project(vec) @ centers.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            _ang = torch.arccos(_cos)                                   # [B, C]
            if return_proto:
                proto_sc.append(_ang.min(dim=1).values.cpu().numpy())
            if return_proto_full:
                proto_all.append(_ang.cpu().numpy())
    out = (np.concatenate(d_sc), np.concatenate(msp), np.concatenate(en),
           np.concatenate(preds), np.concatenate(labels))
    if return_proto:
        out = out + (np.concatenate(proto_sc),)
    if return_proto_full:
        out = out + (np.concatenate(proto_all, axis=0),)
    return out


def risk_coverage(conf, err):
    """conf: higher = more likely ACCEPT. err: 1 if accepting the sample is an error.
    Returns AURC, AUGRC, risk@cov0.9. Defensively orient conf so ID(err-low) ranks first."""
    order = np.argsort(-conf, kind="stable")
    e = err[order].astype(float)
    n = len(e)
    cov = np.arange(1, n + 1) / n
    sel_risk = np.cumsum(e) / np.arange(1, n + 1)   # error among accepted
    gen_risk = np.cumsum(e) / n                      # joint P(error & accept)
    aurc = float(np.trapz(sel_risk, cov))
    augrc = float(np.trapz(gen_risk, cov))
    i90 = int(np.ceil(0.9 * n)) - 1
    return aurc, augrc, float(sel_risk[i90])


def oriented_conf(score, is_ood_mask, higher_is_ood):
    """Make a confidence where higher = more ID. higher_is_ood: True if higher score => OOD."""
    conf = -score if higher_is_ood else score.copy()
    # defensive: ID should have higher conf than OOD; flip if violated
    if is_ood_mask.any() and (~is_ood_mask).any():
        if conf[~is_ood_mask].mean() < conf[is_ood_mask].mean():
            conf = -conf
    return conf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--leave_out", required=True, choices=PACS)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--lambda_tag", default="0")
    p.add_argument("--datasetRoot", default="../datasets/")
    p.add_argument("--ood_sources", default="svhn,textures,noise")
    p.add_argument("--pi_s", default="0.05,0.1,0.3")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_eval_steps", type=int, default=25)
    p.add_argument("--ood_eval_scores_type", default="eps_mse")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_csv", required=True)
    p.add_argument("--id_domain", default="target", choices=["target", "train"],
                   help="'target'=unseen/leave_out domain as ID (deployment-honest); "
                        "'train'=node's own training domain as ID (old separate-eval setup)")
    p.add_argument("--num_nodes", type=int, default=0,
                   help="0=centralized (one model per source domain, domain-named ckpt); "
                        ">0=P2P virtual-node (nodes=node_0..node_{N-1}, ckpt {desc}_node_{i}). "
                        "Mirrors osdg_eval.py. id_domain=target uses leave_out as ID (P2P-safe).")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    diff_steps = list(range(args.num_eval_steps))
    pis = [float(x) for x in args.pi_s.split(",") if x.strip()]
    ood_srcs = [x.strip() for x in args.ood_sources.split(",") if x.strip()]
    if args.num_nodes and args.num_nodes > 0:
        nodes = [f"node_{i}" for i in range(args.num_nodes)]   # P2P virtual nodes
    else:
        nodes = [d for d in PACS if d != args.leave_out]       # centralized: per source domain
    rng = np.random.default_rng(args.seed)
    root = args.datasetRoot

    rows = []
    for node in nodes:
        ckpt = os.path.join(args.checkpoint_dir, f"{args.description}_{node}_final.pth")
        if not os.path.exists(ckpt):
            print(f"[skip] missing {ckpt}"); continue
        print(f"\n=== leave_out={args.leave_out} node={node} ===")
        backbone, diffusion = load_backbone_diffusion(ckpt, device)

        # ID domain: target=unseen/leave_out (deployment-honest) | train=node's own domain (old setup)
        id_dom = node if args.id_domain == "train" else args.leave_out
        id_loader = TD.load_pacs_test_data(root, id_dom, args.batch_size, args.num_workers)[0]
        id_sc, id_msp, id_en, id_pred, id_lab = score_and_predict(
            backbone, diffusion, id_loader, diff_steps, args.ood_eval_scores_type, device)
        id_err = (id_pred != id_lab).astype(int)
        closed_acc = float(1.0 - id_err.mean())
        print(f"  closed-set ID acc (unseen domain, no reject) = {closed_acc:.4f}  N_id={len(id_sc)}")

        for src in ood_srcs:
            if src == "svhn":
                ood_loader = TD.load_svhn_ood_loader(root, None, args.batch_size, args.num_workers)[0]
            elif src == "textures":
                ood_loader = TD.load_textures_ood_loader(root, None, args.batch_size, args.num_workers)[0]
            elif src == "noise":
                ood_loader = TD.get_noise_loader(len(id_sc), args.batch_size, args.num_workers)
            else:
                print(f"  [skip ood] unknown {src}"); continue
            o_sc, o_msp, o_en, _, _ = score_and_predict(
                backbone, diffusion, ood_loader, diff_steps, args.ood_eval_scores_type, device)
            lab = np.concatenate([np.zeros(len(id_sc)), np.ones(len(o_sc))])  # ID=0/OOD=1
            # pure OOD-detection AUROC per score (full ID vs full OOD, pi-independent).
            # OOD-ness: diffusion higher=OOD; MSP/energy(logsumexp) higher=ID so negate.
            auroc = float(roc_auc_score(lab, np.concatenate([id_sc, o_sc])))
            auroc_msp = float(roc_auc_score(lab, np.concatenate([-id_msp, -o_msp])))
            auroc_en = float(roc_auc_score(lab, np.concatenate([-id_en, -o_en])))

            for pi in pis:
                n_ood = min(len(o_sc), int(round(pi / (1 - pi) * len(id_sc))))
                sel = rng.choice(len(o_sc), size=n_ood, replace=False)
                # build mixed arrays: [ID ; sampled OOD]
                d = np.concatenate([id_sc, o_sc[sel]])
                m = np.concatenate([id_msp, o_msp[sel]])
                e = np.concatenate([id_en, o_en[sel]])
                err = np.concatenate([id_err, np.ones(n_ood, dtype=int)])  # accepted OOD always error
                is_ood = np.concatenate([np.zeros(len(id_sc), bool), np.ones(n_ood, bool)])
                risk_cov1 = float(err.mean())

                for name, conf in [
                    ("diffusion", oriented_conf(d, is_ood, higher_is_ood=True)),
                    ("msp",       oriented_conf(m, is_ood, higher_is_ood=False)),
                    ("energy",    oriented_conf(e, is_ood, higher_is_ood=False)),
                ]:
                    aurc, augrc, r90 = risk_coverage(conf, err)
                    rows.append(dict(
                        leave_out=args.leave_out, node=node, ood_src=src, pi_s=pi,
                        score_fn=name, aurc=round(aurc, 4), augrc=round(augrc, 4),
                        risk_at_cov0p9=round(r90, 4), risk_at_cov1=round(risk_cov1, 4),
                        closed_acc=round(closed_acc, 4),
                        det_auroc_diff=round(auroc, 4), det_auroc_msp=round(auroc_msp, 4),
                        det_auroc_energy=round(auroc_en, 4),
                        n_id=len(id_sc), n_ood=n_ood))
                print(f"  {src} pi={pi}: AURC diff/msp/energy = "
                      f"{rows[-3]['aurc']}/{rows[-2]['aurc']}/{rows[-1]['aurc']} "
                      f"| risk@cov1={risk_cov1:.3f} | det_auroc={auroc:.3f}")

    if not rows:
        print("No rows."); return
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    write_header = not os.path.exists(args.output_csv)
    with open(args.output_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"\nAppended {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
