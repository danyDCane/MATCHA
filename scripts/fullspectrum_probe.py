"""Full-spectrum 診斷探針：檢測分數是在抓「畫風陌生」還是「類別陌生」？

問題（0730 dany）：SEM(arXiv 2204.05306) 批評 output-based 方法「largely depend on the
covariate shift to detect OOD samples」——這在 MATCHA 的 PACS/OSDG 設定下成不成立？

把測試樣本分成三堆（同一批資料、同一組 checkpoint、零重訓）：
  ① src_known : 該節點**自己來源域**的 6 個已知類別   → 訓練分布，該判「正常」
  ② tgt_known : **target 域**的 6 個已知類別          → 畫風陌生、類別正常，**該判「正常」**
  ③ tgt_unk   : **target 域**的 person（全程未訓練）  → 類別陌生，該判「異常」

判準（reject score 已定向為「高＝越像 OOD」）：
  AUROC(①vs③) = 純語意分離度            → 越高越好（檢測器的本職）
  AUROC(①vs②) = **純畫風分離度**         → **理想 = 0.5**。越高代表分數越是在抓畫風
  AUROC(②vs③) = 實際部署面對的分離度      → 越高越好
  FPR@src95    = 用 ① 校準到 TPR95 的門檻，套到 ② 的**誤拒率** → 越低越好

⇒ 若 AUROC(①vs②) 顯著 > 0.5 且 FPR@src95 高 ⇒ **SEM 的批評在本設定成立**：
   分數把「畫風陌生」誤當「類別陌生」，而這正是 DG 要免疫的東西。

純 post-hoc inference、零訓練、不改任何 checkpoint。
重用：osdg_eval.load_backbone_diffusion / joint_eval_mixed_stream.score_and_predict /
      test_domain_ood_scores.load_pacs_test_data

用法：
  venv_matcha/bin/python scripts/fullspectrum_probe.py \
    --leave_out cartoon --checkpoint_dir exp_result_<desc> --description <desc> \
    --num_nodes 9 --output_csv research/bn_fusion/0730_fullspectrum.csv
"""
import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import roc_auc_score

import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from joint_eval_mixed_stream import score_and_predict

PACS = ["art_painting", "cartoon", "photo", "sketch"]


def fpr_at_tpr(id_scores, ood_scores, tpr=0.95):
    """用 ID 分數校準到指定 TPR 的門檻，回傳 OOD 側被判為 OOD 的比率。
    reject score: 高 = 越像 OOD ⇒ 門檻取 ID 的 tpr 分位數（接受 tpr 比例的 ID）。"""
    tau = np.quantile(id_scores, tpr)
    return float((ood_scores > tau).mean()), float(tau)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--leave_out", required=True, choices=PACS)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--datasetRoot", default="../datasets/")
    p.add_argument("--num_classes", type=int, default=6)
    p.add_argument("--unknown_idx", type=int, default=6, help="ImageFolder label of person")
    p.add_argument("--num_nodes", type=int, default=9)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_eval_steps", type=int, default=25)
    p.add_argument("--ood_eval_scores_type", default="eps_mse")
    p.add_argument("--device", default="cuda")
    p.add_argument("--ckpt_epoch", default="final")
    p.add_argument("--output_csv", required=True)
    args = p.parse_args()

    import torch
    device = args.device if torch.cuda.is_available() else "cpu"
    diff_steps = list(range(args.num_eval_steps))
    root = args.datasetRoot
    suffix = "final" if str(args.ckpt_epoch) == "final" else f"epoch_{args.ckpt_epoch}"

    # node -> 自己的來源域。9 節點 / 3 域均分、contiguous（與 util.assign_nodes_to_domains 等價，
    # 已由訓練 log 的 "Node assignment" 逐項驗證）。
    available = [d for d in PACS if d != args.leave_out]
    per = args.num_nodes // len(available)
    node_src = {f"node_{i}": available[min(i // per, len(available) - 1)]
                for i in range(args.num_nodes)}

    # target 域只需載入一次（所有節點共用同一批測試資料）
    tgt_loader = TD.load_pacs_test_data(root, args.leave_out, args.batch_size, args.num_workers)[0]
    src_loaders = {d: TD.load_pacs_test_data(root, d, args.batch_size, args.num_workers)[0]
                   for d in available}

    rows = []
    for i in range(args.num_nodes):
        node = f"node_{i}"
        ckpt = os.path.join(args.checkpoint_dir, f"{args.description}_{node}_{suffix}.pth")
        if not os.path.exists(ckpt):
            print(f"[skip] missing {ckpt}")
            continue
        own = node_src[node]
        print(f"\n=== {node} (own_src={own}, target={args.leave_out}) ===")
        backbone, diffusion = load_backbone_diffusion(ckpt, args.num_classes, device)

        # 階段 1 起：checkpoint 若含 prototypes buffer，就多算一個角距離分數（0803 §2.2）。
        # ⚠️ 同一個模型上同時算 energy 與角距離 ⇒「換讀出有沒有用」是**同模型、單一變因**的
        #    比較，完全不受「拿掉 diffusion 造成的亂數位移」影響（0810 plan §4.2）。
        _has_proto = hasattr(backbone, 'prototypes')

        # ① 自己來源域的已知類別
        _o = score_and_predict(
            backbone, diffusion, src_loaders[own], diff_steps, args.ood_eval_scores_type, device,
            return_proto=_has_proto, return_proto_full=_has_proto)
        d_s, msp_s, en_s, _, lab_s = _o[:5]
        pr_s = _o[5] if _has_proto else None
        pf_s = _o[6] if _has_proto else None          # [N, C] 到每個類別中心的角距離
        m_src = lab_s != args.unknown_idx

        # ②③ target 域
        _o = score_and_predict(
            backbone, diffusion, tgt_loader, diff_steps, args.ood_eval_scores_type, device,
            return_proto=_has_proto, return_proto_full=_has_proto)
        d_t, msp_t, en_t, pred_t, lab_t = _o[:5]
        pr_t = _o[5] if _has_proto else None
        pf_t = _o[6] if _has_proto else None
        m_tk = lab_t != args.unknown_idx
        m_tu = ~m_tk

        # ---- 原型落位分析（0811 dany）：min_c 分不出「散開」與「跑到別人家」 ----------
        # 對每個已知類樣本量三件事：到**自己**類別中心的角距離、到**最近的錯誤**類別中心的
        # 角距離、以及最近的中心是不是自己的類別。前兩者相減＝該樣本坐得對不對的餘裕。
        # ⚠️ OOD（person）沒有對應類別 ⇒ 只有「到最近中心」有意義（＝既有的 mean_tgt_unk）。
        placement = {}
        if _has_proto:
            def _place(ang, lab, mask):
                A = ang[mask]                                   # [n, C]
                y = lab[mask].astype(int)
                own_a = A[np.arange(len(y)), y]                 # 到自己類別中心
                W = A.copy()
                W[np.arange(len(y)), y] = np.inf                # 屏蔽自己 ⇒ 最近的錯誤類別
                wrong_a = W.min(axis=1)
                nearest_true = (A.argmin(axis=1) == y)
                return (float(np.degrees(own_a.mean())),
                        float(np.degrees(wrong_a.mean())),
                        float(np.degrees((wrong_a - own_a).mean())),
                        float(nearest_true.mean()))
            placement['src'] = _place(pf_s, lab_s, m_src)
            placement['tgt'] = _place(pf_t, lab_t, m_tk)
            # OOD：沒有真實類別，只報「到最近中心」與「最近的是哪一類的分布熵」
            _ood_min = np.degrees(pf_t[m_tu].min(axis=1))
            placement['unk'] = (float('nan'), float(_ood_min.mean()), float('nan'), float('nan'))
            print(f"  [原型落位] {'堆':<10}{'到自己類別':>11}{'到最近錯類':>11}{'餘裕':>9}{'最近=真實':>11}")
            for _tag, _lbl in [('src', '①來源域'), ('tgt', '②cartoon已知'), ('unk', '③person')]:
                _o_, _w_, _m_, _n_ = placement[_tag]
                print(f"             {_lbl:<12}"
                      f"{('—' if np.isnan(_o_) else f'{_o_:.2f}°'):>11}"
                      f"{_w_:>10.2f}°"
                      f"{('—' if np.isnan(_m_) else f'{_m_:+.2f}°'):>10}"
                      f"{('—' if np.isnan(_n_) else f'{_n_:.4f}'):>11}")

        # reject score: 高 = 越像 OOD。diffusion 與角距離本來就是；msp/energy 取負號。
        _scores = [("msp", -msp_s, -msp_t), ("energy", -en_s, -en_t)]
        if diffusion is not None:
            _scores.insert(0, ("diffusion", d_s, d_t))
        if _has_proto:
            _scores.append(("proto_angle", pr_s, pr_t))
        for name, s_src, s_tgt in _scores:
            A = s_src[m_src]      # ① src_known
            B = s_tgt[m_tk]       # ② tgt_known（畫風陌生、類別正常）
            C = s_tgt[m_tu]       # ③ tgt_unknown（person）

            def auroc(pos, neg):  # pos = 應判 OOD 的那側
                y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
                return float(roc_auc_score(y, np.concatenate([pos, neg])))

            au_style = auroc(B, A)   # ①vs② 純畫風，理想 0.5
            au_sem = auroc(C, A)     # ①vs③ 純語意，越高越好
            au_deploy = auroc(C, B)  # ②vs③ 實際部署，越高越好
            fpr_style, tau = fpr_at_tpr(A, B, 0.95)   # ② 的誤拒率
            tpr_sem, _ = fpr_at_tpr(A, C, 0.95)       # ③ 的正確拒絕率（同一門檻）

            rows.append(dict(
                run=args.description, leave_out=args.leave_out, node=node, own_src=own,
                score_fn=name,
                auroc_style_1v2=round(au_style, 4),
                auroc_semantic_1v3=round(au_sem, 4),
                auroc_deploy_2v3=round(au_deploy, 4),
                fpr_style_at_src95=round(fpr_style, 4),
                tpr_semantic_at_src95=round(tpr_sem, 4),
                mean_src_known=round(float(A.mean()), 4),
                mean_tgt_known=round(float(B.mean()), 4),
                mean_tgt_unk=round(float(C.mean()), 4),
                std_src_known=round(float(A.std()), 4),
                n_src=len(A), n_tgt_known=len(B), n_tgt_unk=len(C),
                # 原型落位（單位：度）。只有 proto_angle 那列有值，其餘為空——這些量描述的是
                # 特徵在原型空間的位置，與用哪個分數讀出無關，放同一列只是為了不另開檔案。
                **({f"place_{k}_{n}": round(v, 4)
                    for k in ('src', 'tgt', 'unk')
                    for n, v in zip(('own', 'wrong', 'margin', 'nearest_true'), placement[k])}
                   if (name == "proto_angle" and placement) else
                   {f"place_{k}_{n}": ""
                    for k in ('src', 'tgt', 'unk')
                    for n in ('own', 'wrong', 'margin', 'nearest_true')})))
            print(f"  {name:9s}: AUROC 畫風(1v2)={au_style:.4f} 語意(1v3)={au_sem:.4f} "
                  f"部署(2v3)={au_deploy:.4f} | 誤拒率@src95={fpr_style:.4f} 正確拒絕={tpr_sem:.4f}")

        del backbone, diffusion

    if not rows:
        print("No rows."); return

    print(f"\n=== node-mean ({args.description}) ===")
    for name in ["energy", "msp", "diffusion"]:
        sub = [r for r in rows if r["score_fn"] == name]
        if not sub:
            continue
        mean = lambda k: float(np.mean([r[k] for r in sub]))
        print(f"  {name:9s}: AUROC 畫風={mean('auroc_style_1v2'):.4f}  語意={mean('auroc_semantic_1v3'):.4f}  "
              f"部署={mean('auroc_deploy_2v3'):.4f} | 誤拒率@src95={mean('fpr_style_at_src95'):.4f}  "
              f"正確拒絕={mean('tpr_semantic_at_src95'):.4f}")

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
