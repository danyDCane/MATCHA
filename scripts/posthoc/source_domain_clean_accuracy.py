"""[1012 追加] 來源畫風「乾淨」準確率：已訓練 checkpoint 在 art／photo／sketch 原圖（無風格增強、測試用前處理）上的
六類閉集準確率（person 不計，與訓練一致）。用途：檢查 1012 art 慢時「art 三台訓練準確率 −6 分」是不是增強造成的。
⚠️ PACS 載入把來源畫風整份拿去訓練、沒有保留集 ⇒ 這仍是「訓練資料上」的準確率（擬合），不是來源域泛化。
沿用 scripts/osdg_eval.py 的 load_backbone_diffusion／compute_avg_bn_buffers（BN 平均B）與
test_domain_ood_scores.load_pacs_test_data（與目標域評估同一套前處理）、joint_eval_mixed_stream.score_and_predict。
  python source_domain_clean_accuracy.py --epochs 200 [--arms C0,Cbw,ArtSlow] --out results/proto_trigger/1012_artslow_source_clean_acc.csv
"""
import os, sys, csv, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import numpy as np, torch
from osdg_eval import load_backbone_diffusion, compute_avg_bn_buffers
import test_domain_ood_scores as TD
from joint_eval_mixed_stream import score_and_predict

S = '_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234'
ARMS = {'C0': 'v1_stage2_leave_cartoon_p1a_async_periodic6.2133' + S + '_d5',
        'Dbw': 'v1_stage2_leave_cartoon_p1a_async_lr_tau1.368e-05_bwb0r' + S,
        'Cbw': 'v1_stage2_leave_cartoon_p1a_async_pbw_bwb0' + S,
        'Cmatch': 'v1_stage2_leave_cartoon_p1a_async_pmatch_bwb0' + S,
        'ArtSlow': 'v1_stage2_leave_cartoon_p1a_async_pbwart_bwfart0.1' + S,
        'SketchSlow': 'v1_stage2_leave_cartoon_p1a_async_pbwsk_bwfsketch0.1' + S,
        'DiagA': 'v1_stage2_leave_cartoon_p1a_async_pbw_bwb0_lrsk0.13' + S,       # [plan 1014] (a)
        'DiagC': 'v1_stage2_leave_cartoon_p1a_async_pbw_bwb0_fw0.22' + S,         # [plan 1014] (c)
        'ShareA1': 'v1_stage2_leave_cartoon_p1a_async_pbwsk_bwfsketch0.1_frp' + S,  # [plan 1016] 底座 × sketch 慢
        'ShareA2': 'v1_stage2_leave_cartoon_p1a_async_pbw_bwb0_frp' + S,            # [plan 1016] 底座 × 正常網速
        'ShareA3': 'v1_stage2_leave_cartoon_p1a_async_pbwsk_bwfsketch0.1_fps' + S,  # [plan 1016] 推和法 × sketch 慢
        'CompU': 'v1_stage2_leave_cartoon_p1a_async_puni_bwfsketch0.1' + S,  # [plan 1017] 等預算均勻送 × sketch 慢
        'CompP': 'v1_stage2_leave_cartoon_p1a_async_pbwsk_bwfsketch0.1_fpc' + S,       # [plan 1017] PushCen × sketch 慢
        'CompD': 'v1_stage2_leave_cartoon_p1a_async_lr_tau8.7e-06_bwfskr' + S,          # [plan 1017] D-bw × sketch 慢
        'CompE': 'v1_stage2_leave_cartoon_p1a_async_lr_tau8.7e-06_bwfskr_bd' + S,       # [plan 1017] EF-HC 雙向 × sketch 慢
        'R3bN': 'v1_stage2_leave_cartoon_p1a_async_pbw_bwb0_frg' + S,                    # [plan 1019] 按群 × 正常網速
        'R3bS': 'v1_stage2_leave_cartoon_p1a_async_pbwsk_bwfsketch0.1_frg' + S,          # [plan 1019] 按群 × sketch 慢
        # [plan 1019 一折全貌] F1–F14
        'FoldF1': 'v1_stage2_leave_cartoon_p1a_async_pbwph_bwfphoto0.1' + S,          # photo 慢 不補償
        'FoldF2': 'v1_stage2_leave_cartoon_p1a_async_pbwph_bwfphoto0.1_frg' + S,      # photo 慢 按群
        'FoldF3': 'v1_stage2_leave_cartoon_p1a_async_lr_tau1.11e-05_bwfphr' + S,      # photo 慢 EF-HC 單向
        'FoldF4': 'v1_stage2_leave_cartoon_p1a_async_lr_tau1.11e-05_bwfphr_bd' + S,   # photo 慢 EF-HC 雙向
        'FoldF5': 'v1_stage2_leave_cartoon_p1a_async_pbwph_bwfphoto0.1_fpc' + S,      # photo 慢 PushCen
        'FoldF6': 'v1_stage2_leave_cartoon_p1a_async_puni_bwfphoto0.1' + S,           # photo 慢 等預算均勻送
        'FoldF7': 'v1_stage2_leave_cartoon_p1a_async_pbwart_bwfart0.1_frg' + S,       # art 慢 按群
        'FoldF8': 'v1_stage2_leave_cartoon_p1a_async_lr_tau1.1e-05_bwfartr' + S,      # art 慢 EF-HC 單向
        'FoldF9': 'v1_stage2_leave_cartoon_p1a_async_lr_tau1.1e-05_bwfartr_bd' + S,   # art 慢 EF-HC 雙向
        'FoldF10': 'v1_stage2_leave_cartoon_p1a_async_pbwart_bwfart0.1_fpc' + S,      # art 慢 PushCen
        'FoldF11': 'v1_stage2_leave_cartoon_p1a_async_puni_bwfart0.1' + S,            # art 慢 等預算均勻送
        'FoldF12': 'v1_stage2_leave_cartoon_p1a_async_pbw_bwb0_fpc' + S,              # 正常 PushCen
        'FoldF13': 'v1_stage2_leave_cartoon_p1a_async_lr_tau1.368e-05_bwb0r_bd' + S,  # 正常 EF-HC 雙向
        'FoldF14': 'v1_stage2_leave_cartoon_p1a_async_puni_bwb0' + S}                 # 正常 等預算均勻送
DOMS = ['art_painting', 'photo', 'sketch', 'cartoon']   # cartoon＝錨點：avgB 須逐位重現正式評估 CSV 的 closed_acc
UNK = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--epochs', default='200')
    ap.add_argument('--out', default='results/proto_trigger/1012_artslow_source_clean_acc.csv')
    ap.add_argument('--datasetRoot', default='../datasets/')
    o = ap.parse_args()
    dev = 'cuda'
    loaders = {d: TD.load_pacs_test_data(o.datasetRoot, d, 128, 4)[0] for d in DOMS}
    rows = []
    for arm in o.arms.split(','):
        name = ARMS[arm]; cdir = os.path.join(ROOT, 'exp_result_' + name)
        for ep in o.epochs.split(','):
            ck = [os.path.join(cdir, f'{name}_node_{i}_epoch_{ep}.pth') for i in range(9)]
            avg = compute_avg_bn_buffers(ck, 6, dev, var_mode='merged')
            for i, p in enumerate(ck):
                bb, diff = load_backbone_diffusion(p, 6, dev)
                own = {k: v.detach().clone() for k, v in bb.named_buffers() if k.endswith(('running_mean', 'running_var'))}
                for bn in ('own', 'avgB'):
                    bd = dict(bb.named_buffers())
                    for k, v in (own if bn == 'own' else avg).items():
                        bd[k].copy_(v)
                    bb.eval()
                    for d in DOMS:
                        _o = score_and_predict(bb, diff, loaders[d], list(range(1)), 'eps_mse', dev)
                        pred, lab = _o[3], _o[4]
                        known = lab != UNK
                        acc = float(((pred == lab) & known).sum()) / max(int(known.sum()), 1)
                        rows.append(dict(arm=arm, epoch=ep, node=f'node_{i}', node_style=['art_painting', 'photo', 'sketch'][i // 3],
                                         bn=bn, eval_domain=d, closed_acc=round(acc, 4), n_known=int(known.sum())))
                del bb
                print(f'[{arm} ep{ep}] node_{i} done', flush=True)
    os.makedirs(os.path.dirname(os.path.join(ROOT, o.out)), exist_ok=True)
    with open(os.path.join(ROOT, o.out), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import pandas as pd
    df = pd.DataFrame(rows)
    print('\n== 九台平均閉集準確率（%）：列＝臂×epoch×BN，欄＝評估畫風 ==')
    t = df.groupby(['arm', 'epoch', 'bn', 'eval_domain']).closed_acc.mean().unstack('eval_domain') * 100
    print(t.round(2).to_string())
    print('\n== 九台之間最大−最小（%，看是否已收斂成同一模型）==')
    print((df.groupby(['arm', 'epoch', 'bn', 'eval_domain']).closed_acc.agg(lambda s: s.max() - s.min()).unstack('eval_domain') * 100).round(2).to_string())
    print(f'\n寫出 {o.out}（{len(df)} 列）')


if __name__ == '__main__':
    main()
