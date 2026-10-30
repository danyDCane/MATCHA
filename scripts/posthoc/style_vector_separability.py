"""1016（dany）：候選一「用風格向量分群推斷鄰居畫風」的可部署前提——PACS 三個來源畫風的風格向量分不分得開。

【量的是哪一份向量】＝訓練時每步夾帶在模型訊息裡送出的那份（train.py「第一階段」）：
  - 每台用**自己的模型**、**自己的資料分割**、**訓練前處理**（隨機縮放裁切＋翻轉）抽一批 bs 張；
  - model.train()（BN 用批次統計，與訓練時相同）、no_grad，extract_features_to_layer3；
  - style_stats.compute_multi_layer_style_stats → flatten_style_stats（layer1–3 各四塊 [mu, sigma, Σmu², Σsigma²]）。
  資料與節點切分：直接用 checkpoint 內存的 args 呼叫 util.load_dataset_single_process（＝訓練當時的節點資料）。

【另附 BN 訊號】每台 checkpoint 的 BN running_mean／running_var（也隨模型訊息送出；本設定 BN 有融合）。

【量法】（零點與上界都報；9 台分 3×3 的相異分割只有 280 個 ⇒ 置換檢定用窮舉）
  ① 兩兩「同畫風？」：不同節點的單批向量兩兩距離 → AUROC（距離越小越像同畫風）。分全體節點對／只看圖上的邊。
  ② 本地視角：每台只看「自己一批＋每個鄰居一批」，用「跟我的距離」判鄰居是否同畫風 → 逐台 AUROC。
  ③ 分群：每次抽「每台一批」共 9 個向量做 k-means(k=3)／階層式分群 → 與真實畫風的 ARI；完全分對的比例。
  ④ 拆塊：layer1/2/3 × 四種統計量各自的 ①。
  ⑤ 雜訊尺度：同台不同批、同畫風不同台、不同畫風的距離中位數。

⚠️ 射程：cartoon 折、seed2026、topo1234；1009 C-bw（正常網速）與 1013 sketch 慢兩支 run；ep5／50／200。
   與 research/0916_edge_diag_review.md §8（sketch 折、ep≤30、逐邊聚合前偏離 AUROC 0.88）口徑不同，不可並排比大小。

用法：venv_matcha/bin/python scripts/posthoc/style_vector_separability.py [--batches 30] [--epochs 5,50,200] [--runs cbw,sketchslow]
     [1019 R3b] --pretrained_ref：改用「訓練起點的預訓練模型」（util.select_model＋checkpoint args，九台起點相同、凍結）
     算九台的向量（train／eval 兩種 BN 模式），只存檔不分析，供 r3b_grouping_estimate.py 用；預設存 results/proto_trigger/1019_style_vectors_pretrained.npz
"""
import os, sys, argparse, itertools, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)
import numpy as np, torch
import networkx as nx
from sklearn.metrics import roc_auc_score, adjusted_rand_score
from sklearn.cluster import KMeans, AgglomerativeClustering
import util
from osdg_eval import load_backbone_diffusion
from style_stats import compute_multi_layer_style_stats, flatten_style_stats

S = '_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234'
RUNS = {'cbw': 'v1_stage2_leave_cartoon_p1a_async_pbw_bwb0' + S,
        'sketchslow': 'v1_stage2_leave_cartoon_p1a_async_pbwsk_bwfsketch0.1' + S}
N = 9; STY = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])  # node_0–2 art、3–5 photo、6–8 sketch（區塊式配置）
G = nx.random_geometric_graph(N, 0.8, seed=1234)
assert [len(G[i]) for i in range(N)] == [5, 2, 5, 6, 6, 3, 8, 6, 5], 'RGG 度數與訓練不符'
EDGES = sorted((min(a, b), max(a, b)) for a, b in G.edges())
# 280 個相異的 3×3 分割（零點用）
PARTS = []
for a in itertools.combinations(range(1, N), 2):
    A = (0,) + a; rest = [x for x in range(N) if x not in A]
    for b in itertools.combinations(rest[1:], 2):
        B = (rest[0],) + b; C = tuple(x for x in rest if x not in B)
        lab = np.zeros(N, int); lab[list(B)] = 1; lab[list(C)] = 2; PARTS.append(lab)
assert len(PARTS) == 280


def style_vectors(run, ep, nb, loaders, dev, common=None, ref=None, ref_mode='train'):
    """common＝None：每台用自己的模型（＝訓練時）；common＝k：所有台都用 node_k 的模型（對照：拿掉『模型不同』只留『資料不同』）。
    ref＝模型物件：九台都用它（1019 預訓練起點），ref_mode＝BN 用批次統計（train）或預訓練存的統計（eval）；不讀 checkpoint、不回 BN。"""
    V = np.zeros((N, nb, 1792), np.float32); BN = []
    for i in range(N):
        if ref is None:
            name = RUNS[run]
            ck = os.path.join(ROOT, 'exp_result_' + name, f'{name}_node_{i if common is None else common}_epoch_{ep}.pth')
            bb, _ = load_backbone_diffusion(ck, 6, dev)
            BN.append(torch.cat([b.detach().flatten().cpu() for k, b in bb.named_buffers()
                                 if k.endswith(('running_mean', 'running_var'))]).numpy())
            bb.train()
        else:
            bb = ref; bb.train() if ref_mode == 'train' else bb.eval()
        torch.manual_seed(1000 + i)  # 批次抽樣可重現；各 run／epoch 用同一組批次
        it = iter(loaders[f'node_{i}'][0])
        with torch.no_grad():
            for b in range(nb):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loaders[f'node_{i}'][0]); batch = next(it)
                data, _, _ = util.unpack_batch(batch)
                feats = bb.extract_features_to_layer3(data.to(dev, non_blocking=True))
                sv = flatten_style_stats(compute_multi_layer_style_stats(feats, eta=1e-5), ['layer1', 'layer2', 'layer3'])
                V[i, b] = sv.float().cpu().numpy()
        if ref is None:
            del bb; torch.cuda.empty_cache()
    return V, (np.stack(BN) if BN else None)


def pair_auroc(X, lab, pairs):
    """X: [N, d] 一次抽樣；回傳同畫風（正類）以 −距離 為分數的 AUROC。"""
    y = [int(lab[a] == lab[b]) for a, b in pairs]
    if len(set(y)) < 2: return np.nan
    d = [np.linalg.norm(X[a] - X[b]) for a, b in pairs]
    return roc_auc_score(y, -np.array(d))


def analyse(V, BNv, nb, log):
    allpairs = [(a, b) for a in range(N) for b in range(a + 1, N)]
    mu = V.reshape(-1, V.shape[-1]).mean(0); sd = V.reshape(-1, V.shape[-1]).std(0) + 1e-12
    Z = (V - mu) / sd  # 全域標準化（分析用；部署時拿不到，另報原始距離）
    rng = np.random.default_rng(0)
    out = {}
    for tag, W in [('原始 L2', V), ('逐維標準化 L2', Z)]:
        # ① 每次抽「每台一批」，算全體節點對與圖上邊的 AUROC；重複 nb 次取平均
        aa = [pair_auroc(W[np.arange(N), rng.integers(0, nb, N)], STY, allpairs) for _ in range(200)]
        ae = [pair_auroc(W[np.arange(N), rng.integers(0, nb, N)], STY, EDGES) for _ in range(200)]
        # 零點：280 個分割當標籤（用同 200 次抽樣的平均向量近似：取節點平均以降噪，只為零點分佈）
        Wm = W.mean(1)
        null = np.array([pair_auroc(Wm, p, allpairs) for p in PARTS])
        real = pair_auroc(Wm, STY, allpairs)
        rank = int((null >= real - 1e-12).sum())
        out[tag] = (np.mean(aa), np.mean(ae))
        log(f'  ① {tag}：單批向量兩兩 AUROC 全體節點對 {np.mean(aa):.3f}（抽 200 次、最低 {np.min(aa):.3f}）'
            f'｜只看 23 條邊 {np.nanmean(ae):.3f}（最低 {np.nanmin(ae):.3f}）'
            f'｜節點平均向量 AUROC {real:.3f}，280 分割零點平均 {null.mean():.3f}、第 95 百分位 {np.percentile(null, 95):.3f}、真實分割名次 {rank}/280')
    # ② 本地視角：每台用「跟我的距離」判鄰居是否同畫風（原始 L2，部署可算）
    per = []
    for i in range(N):
        nbr = sorted(G[i]); y = [int(STY[j] == STY[i]) for j in nbr]
        if len(set(y)) < 2:
            per.append((i, np.nan, sum(y), len(y))); continue
        sc = []
        for _ in range(200):
            pick = rng.integers(0, nb, N)
            sc.append(roc_auc_score(y, [-np.linalg.norm(V[i, pick[i]] - V[j, pick[j]]) for j in nbr]))
        per.append((i, np.mean(sc), sum(y), len(y)))
    log('  ② 本地視角（原始 L2，每台只看自己一批＋每個鄰居一批）逐台 AUROC：'
        + '、'.join(f'node_{i} {a:.2f}（同畫風鄰居 {s}/{n}）' if a == a else f'node_{i} 無法算（同畫風鄰居 {s}/{n}）' for i, a, s, n in per))
    # ③ 分群：每次每台一批 → 9 向量分 3 群
    for tag, W in [('原始', V), ('逐維標準化', Z)]:
        ari_k, ari_h, exact = [], [], 0
        for _ in range(200):
            X = W[np.arange(N), rng.integers(0, nb, N)]
            lk = KMeans(3, n_init=10, random_state=0).fit_predict(X)
            lh = AgglomerativeClustering(3, linkage='average').fit_predict(X)
            ari_k.append(adjusted_rand_score(STY, lk)); ari_h.append(adjusted_rand_score(STY, lh)); exact += adjusted_rand_score(STY, lh) > 0.999
        null = np.array([adjusted_rand_score(STY, p) for p in PARTS])
        log(f'  ③ 分群（{tag}，每次每台一批共 9 向量，抽 200 次）：k-means ARI {np.mean(ari_k):.3f}、階層式 ARI {np.mean(ari_h):.3f}、'
            f'階層式完全分對 {exact}/200｜零點（隨機 3×3 分割）ARI 平均 {null.mean():.3f}、上界 1')
    # ④ 拆塊（原始 L2，節點平均以外：用單批）
    C = {'layer1': 64, 'layer2': 128, 'layer3': 256}; off = 0; rows = []
    for L, c in C.items():
        for k, nm in enumerate(['mu', 'sigma', 'Σmu²', 'Σsigma²']):
            sl = slice(off + k * c, off + (k + 1) * c)
            a = np.mean([pair_auroc(V[np.arange(N), rng.integers(0, nb, N)][:, sl], STY, allpairs) for _ in range(100)])
            rows.append(f'{L}.{nm} {a:.2f}')
        off += 4 * c
    assert off == 1792
    log('  ④ 拆塊 AUROC（原始 L2、單批）：' + '｜'.join(rows))
    # ⑤ 雜訊尺度
    same_node = [np.linalg.norm(V[i, b1] - V[i, b2]) for i in range(N) for b1, b2 in [(0, 1), (2, 3), (4, 5)] if b2 < nb]
    same_sty = [np.linalg.norm(V[a, 0] - V[b, 1]) for a, b in allpairs if STY[a] == STY[b]]
    diff_sty = [np.linalg.norm(V[a, 0] - V[b, 1]) for a, b in allpairs if STY[a] != STY[b]]
    log(f'  ⑤ 距離中位數（原始 L2）：同台不同批 {np.median(same_node):.4f}｜同畫風不同台 {np.median(same_sty):.4f}｜不同畫風 {np.median(diff_sty):.4f}'
        f'（不同畫風 ÷ 同畫風 ＝ {np.median(diff_sty) / np.median(same_sty):.2f}）')
    # BN 訊號（每台一個向量、無批次雜訊）
    bn_all = pair_auroc(BNv, STY, allpairs); bn_e = pair_auroc(BNv, STY, EDGES)
    nullb = np.array([pair_auroc(BNv, p, allpairs) for p in PARTS])
    log(f'  ⑥ BN running 統計（checkpoint、每台一個向量）兩兩 AUROC：全體 {bn_all:.3f}｜只看邊 {bn_e:.3f}｜280 分割零點第 95 百分位 {np.percentile(nullb, 95):.3f}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', type=int, default=30)
    ap.add_argument('--epochs', default='5,50,200')
    ap.add_argument('--runs', default='cbw,sketchslow')
    ap.add_argument('--save', default='results/proto_trigger/1016_style_vectors.npz')
    ap.add_argument('--pretrained_ref', action='store_true')
    o = ap.parse_args()
    dev = 'cuda'
    ck0 = torch.load(os.path.join(ROOT, 'exp_result_' + RUNS['cbw'], f"{RUNS['cbw']}_node_0_epoch_5.pth"), map_location='cpu', weights_only=False)
    args = ck0['args']
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        loaders, node_to_domain = util.load_dataset_single_process(args)
    assert [node_to_domain[f'node_{i}'] for i in range(N)] == ['art_painting'] * 3 + ['photo'] * 3 + ['sketch'] * 3, node_to_domain
    print(f'[自檢] 節點↔畫風 {[node_to_domain[f"node_{i}"][:5] for i in range(N)]}｜bs={args.bs}｜各台樣本數 '
          + str([len(loaders[f"node_{i}"][0].dataset) for i in range(N)]) + f'｜圖上邊 {len(EDGES)} 條', flush=True)
    log = lambda s: print(s, flush=True)
    save = {}
    if o.pretrained_ref:
        assert args.pretrained, '訓練起點不是預訓練模型'
        ref = util.select_model(6, args).to(dev)  # 與 train.py 相同的建法；九台起點＝這份的複製
        sd0 = {k: v.clone() for k, v in ref.state_dict().items()}
        for mode in ['train', 'eval']:
            ref.load_state_dict(sd0)  # train 模式前向會更新 BN 累計統計（不影響 train 模式輸出），換模式前還原
            V, _ = style_vectors(None, None, o.batches, loaders, dev, ref=ref, ref_mode=mode)
            assert V.shape[-1] == 1792 and np.isfinite(V).all()
            save[f'pretrained_{mode}'] = V; log(f'  預訓練起點 BN={mode}：向量 {V.shape}')
        out = o.save if o.save != ap.get_default('save') else 'results/proto_trigger/1019_style_vectors_pretrained.npz'
        np.savez_compressed(out, **save); log(f'寫出 {out}'); return
    for run in o.runs.split(','):
        for ep in o.epochs.split(','):
            t0 = time.time()
            V, BNv = style_vectors(run, ep, o.batches, loaders, dev)
            assert V.shape[-1] == 1792 and np.isfinite(V).all()
            log(f'\n== {run} ep{ep}（向量 {V.shape}、BN 向量 {BNv.shape}，{time.time() - t0:.0f}s）==')
            analyse(V, BNv, o.batches, log)
            save[f'{run}_ep{ep}'] = V; save[f'{run}_ep{ep}_bn'] = BNv
            # ⑦ 共用模型對照（node_0 的模型算全部九台的資料）：分得開若主要來自資料畫風，這裡應仍分得開
            Vc, _ = style_vectors(run, ep, o.batches, loaders, dev, common=0)
            allpairs = [(a, b) for a in range(N) for b in range(a + 1, N)]
            rng = np.random.default_rng(1)
            ac = [pair_auroc(Vc[np.arange(N), rng.integers(0, o.batches, N)], STY, allpairs) for _ in range(200)]
            lh = [adjusted_rand_score(STY, AgglomerativeClustering(3, linkage='average').fit_predict(Vc[np.arange(N), rng.integers(0, o.batches, N)])) for _ in range(200)]
            log(f'  ⑦ 共用模型對照（全部用 node_0 的模型）：兩兩 AUROC {np.mean(ac):.3f}（最低 {np.min(ac):.3f}）｜階層式 ARI {np.mean(lh):.3f}')
            save[f'{run}_ep{ep}_common0'] = Vc
    os.makedirs(os.path.dirname(o.save), exist_ok=True)
    np.savez_compressed(o.save, **save)
    log(f'\n寫出 {o.save}')


if __name__ == '__main__':
    main()
