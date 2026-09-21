"""風格向量四塊的能量占比與各自的畫風分離度（0916 逐邊偏離複核 §6 的待驗項）。

問題：實際交換的風格向量是每層四塊串接 `[mu_bar, sigma_bar, Sigma_mu_sq, Sigma_sigma_sq]`
      （`style_stats.py:flatten_style_stats`），整條向量的分離倍率（跨畫風 ÷ 同畫風）實測只有 **2.26×**
      （`0916_edge_diag_review.md` §2），但只用 `[mu, sigma]` 兩塊外推卻是 **6–9×**（同檔 §5）。
      落差是不是因為兩個 `Σ` 塊佔掉了大部分能量、而它們幾乎沒有畫風結構？

做法：用既有 dump 的**逐樣本**通道統計量重算四塊（不需重跑訓練、不需新 dump）：
      `logs/prototype_probe/0826_features_full.npz` 的 `n{i}_src_s{1,2,3}`
      ＝ `torch.cat([f.mean((2,3)), f.std((2,3))], 1)`（`probe_information_content.py:collect`）
      ⇒ 前半是逐樣本通道平均、後半是逐樣本通道標準差 ⇒ 四塊全部可重建。
      對每個節點抽 R 個大小 n 的批次算風格向量，再比「同畫風節點對」與「跨畫風節點對」的距離。

⚠️ 三個口徑限制（引用時必帶）：
  1. **dump 來自 cartoon 折、ep200 final、BN 平均B、eval 模式**；被解釋的 2.26× 來自 **sketch 折、ep≤30、訓練期、各節點本地 BN**。
     ⇒ 本檔回答的是「四塊的能量結構長什麼樣」，**不是** ep30 那個數字的逐位重現。
  2. dump 內同畫風的三個節點看的是**同一份來源域測試集**（只有模型不同、批次抽樣不同）；
     訓練時三個節點各有**自己的資料分割** ⇒ 同畫風距離的雜訊結構不完全相同。
  3. dump 的標準差用 `torch.std`（unbiased=True），訓練用 `var(unbiased=False)`；n≥64 時差 <1%，但不是逐位相同。

用法：./venv_matcha/bin/python scripts/posthoc/style_vector_block_energy.py [--n 64] [--reps 20]
"""
import argparse

import numpy as np

DUMP = "logs/prototype_probe/0826_features_full.npz"
LAYERS = ["s1", "s2", "s3"]
BLOCKS = ["mu_bar", "sigma_bar", "Sigma_mu_sq", "Sigma_sigma_sq"]
ETA = 1e-5
N_NODES = 9
PER = 3          # 每個來源畫風 3 個節點（util.assign_nodes_to_domains 的區塊式配置）


def blocks_of(batch):
    """batch: [n, 2C] 逐樣本 [通道平均 | 通道標準差] → 四塊各 [C]，與 compute_layer_style_stats 同式。"""
    c = batch.shape[1] // 2
    mu, sg = batch[:, :c], batch[:, c:]
    return {"mu_bar": mu.mean(0), "sigma_bar": sg.mean(0),
            "Sigma_mu_sq": mu.var(0) + ETA, "Sigma_sigma_sq": sg.var(0) + ETA}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="每個風格向量用幾張圖（訓練時＝batch size 64）")
    ap.add_argument("--reps", type=int, default=20, help="每個節點抽幾個批次")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    z = np.load(DUMP, allow_pickle=True)

    # 每個節點 × 每次重複 → {block: 串接三層的向量}
    V = [[{} for _ in range(a.reps)] for _ in range(N_NODES)]
    for i in range(N_NODES):
        per_layer = [z[f"n{i}_src_{s}"].astype(np.float64) for s in LAYERS]
        n_avail = per_layer[0].shape[0]
        for r in range(a.reps):
            idx = rng.choice(n_avail, size=a.n, replace=False)
            bl = [blocks_of(f[idx]) for f in per_layer]
            V[i][r] = {b: np.concatenate([x[b] for x in bl]) for b in BLOCKS}

    same_pairs = [(i, j) for i in range(N_NODES) for j in range(i + 1, N_NODES) if i // PER == j // PER]
    cross_pairs = [(i, j) for i in range(N_NODES) for j in range(i + 1, N_NODES) if i // PER != j // PER]

    def sq(pairs, block):
        """該塊在這些節點對上的平方距離（每對取 reps 次配對的平均）。"""
        out = []
        for i, j in pairs:
            out.append(np.mean([np.sum((V[i][r][block] - V[j][r][block]) ** 2) for r in range(a.reps)]))
        return np.array(out)

    S = {b: sq(same_pairs, b) for b in BLOCKS}
    C = {b: sq(cross_pairs, b) for b in BLOCKS}
    S_tot, C_tot = sum(S.values()), sum(C.values())

    print(f"風格向量四塊分解｜dump={DUMP}｜每向量 {a.n} 張圖 × {a.reps} 次重複｜"
          f"同畫風 {len(same_pairs)} 對、跨畫風 {len(cross_pairs)} 對")
    print(f"{'塊':<16}{'同畫風 d':>12}{'跨畫風 d':>12}{'倍率':>8}{'同畫風能量占比':>16}{'跨畫風能量占比':>16}")
    for b in BLOCKS:
        ds, dc = np.sqrt(S[b].mean()), np.sqrt(C[b].mean())
        print(f"{b:<16}{ds:>12.4f}{dc:>12.4f}{dc / ds:>8.2f}"
              f"{S[b].mean() / S_tot.mean():>15.1%}{C[b].mean() / C_tot.mean():>15.1%}")
    ds, dc = np.sqrt(S_tot.mean()), np.sqrt(C_tot.mean())
    print(f"{'整條向量':<16}{ds:>12.4f}{dc:>12.4f}{dc / ds:>8.2f}{1:>15.1%}{1:>15.1%}")

    # 只用 [mu_bar, sigma_bar] 兩塊（＝複核 §5 外推用的舊基底）
    s2 = sum(S[b] for b in BLOCKS[:2]).mean()
    c2 = sum(C[b] for b in BLOCKS[:2]).mean()
    print(f"\n只取 [mu_bar, sigma_bar] 兩塊：倍率 {np.sqrt(c2 / s2):.2f}"
          f"（複核 §5 的舊基底；四塊整條是 {dc / ds:.2f}）")
    print(f"兩個 Σ 塊佔同畫風平方能量 {(S['Sigma_mu_sq'] + S['Sigma_sigma_sq']).mean() / S_tot.mean():.1%}"
          f"、跨畫風 {(C['Sigma_mu_sq'] + C['Sigma_sigma_sq']).mean() / C_tot.mean():.1%}")


if __name__ == "__main__":
    main()
