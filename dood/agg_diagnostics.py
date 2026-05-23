"""
聚合健康度診斷 logger（D1–D4）。

設計原則（V1 驗證計畫）：
  * 純記錄，**零 sys.exit**。是否中斷訓練由人/agent 讀 log 判讀後手動 kill 進程。
  * 每通訊輪 append 一行 CSV 並 flush，讓外部能即時讀到最新值。
  * 異常數值（NaN/Inf）只標紅警告 + CSV 記 has_nan，不中斷。

**對訓練零副作用（關鍵）**：所有診斷計算都包在 `_frozen()` context 內：
  1. 把涉及的 model 設 eval（連帶 diffusion_model submodule）——擋掉
     - backbone BatchNorm running stats 更新（intermediate_forward）
     - FeatureNormalization 的 mins/maxs/means/stds queue in-place 更新
       （diffusion_model.py:147 `if self.training: _update_running_stats` —— @torch.no_grad 擋不住 buffer 寫入）
  2. 進出時保存/還原 torch + cuda RNG state——p_losses 在 noise=None 時 `torch.randn`
     會消耗全域 RNG（gaussian_diffusion.py:202），不還原會破壞訓練可重現性。
  跑完一律還原各 model 原本的 training 旗標。

掛載點（單進程 SingleProcessCommunicator._aggregate_models）：
  * before_agg(): 聚合前快照後呼叫 —— 抓 D4 的 L_pre（用聚合前 denoiser，固定 probe 特徵）。
  * after_agg() : diffusion 參數聚合後呼叫 —— 算 D1（denoiser 參數發散度）與 D4（ΔL，用同一 feat_pre 隔離 denoiser 效應）。
D2/D3 為 epoch 級，由 train.py 主迴圈在 epoch 邊界呼叫。

節點 vs domain：graphid=6 的 9 個虛擬節點中，每 3 節點共享 1 個 source domain。
透過 node_to_domain 把節點映回 domain，D1/D3 才能分層 intra-domain vs inter-domain。
"""

import os
import csv
import math
import contextlib
import torch


def _has_diffusion(model):
    return hasattr(model, "diffusion_model") and model.diffusion_model is not None


def _is_bad(x):
    return (x is None) or math.isnan(x) or math.isinf(x)


class AggDiagnostics:
    def __init__(self, save_dir, node_to_domain, t_probe,
                 d4_every=1, verbose=True):
        """
        Args:
            save_dir: 診斷 CSV 輸出目錄（建議 = {savePath}/agg_diag）。
            node_to_domain: {node_name: source_domain}；單一拓樸下若無映射可傳 {node: node}。
            t_probe: D2/D4 固定 diffusion timestep。
            d4_every: D4（聚合前後 DSM loss）每幾通訊輪算一次（D1 每輪都算，便宜）。
            verbose: 是否在 console 印每輪摘要。
        """
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.node_to_domain = dict(node_to_domain) if node_to_domain else {}
        self.t_probe = int(t_probe)
        self.d4_every = max(1, int(d4_every))
        self.verbose = verbose

        self.round = 0
        self.node_probe_images = {}   # {node: (data, target)} 固定影像批（不隨訓練改變）
        self._d4_pre = {}             # {node: (feat_pre_tensor, loss_pre)}
        self._param_order = None      # denoiser named_parameters 的固定 key 順序

        # ---- 開三個 CSV，寫 header ----
        self._f_d14 = open(os.path.join(save_dir, "diag_D1_D4.csv"), "a", newline="")
        self._w_d14 = csv.writer(self._f_d14)
        if self._f_d14.tell() == 0:
            self._w_d14.writerow([
                "round", "epoch", "n_nodes",
                "d1_global", "d1_max", "d1_intra", "d1_inter",
                "d4_dL_mean", "d4_dL_max", "has_nan",
            ])
            self._f_d14.flush()

        self._f_d2 = open(os.path.join(save_dir, "diag_D2.csv"), "a", newline="")
        self._w_d2 = csv.writer(self._f_d2)
        if self._f_d2.tell() == 0:
            self._w_d2.writerow(["epoch", "scope", "name", "domain", "nll", "has_nan"])
            self._f_d2.flush()

        self._f_d3 = open(os.path.join(save_dir, "diag_D3.csv"), "a", newline="")
        self._w_d3 = csv.writer(self._f_d3)
        if self._f_d3.tell() == 0:
            self._w_d3.writerow(["epoch", "i_node", "j_node", "i_domain", "j_domain", "score", "has_nan"])
            self._f_d3.flush()

    # ------------------------------------------------------------------
    # probe 影像快取（由 train.py 在訓練開始前對每節點抽一批固定影像）
    # ------------------------------------------------------------------
    def set_probe_images(self, node, data, target):
        self.node_probe_images[node] = (data.detach().clone(), target.detach().clone())

    # ------------------------------------------------------------------
    # 零副作用 guard：eval 隔離 + RNG 保存還原
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _frozen(self, models):
        prev = [(m, m.training) for m in models]
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            for m, _ in prev:
                m.eval()            # 連帶 diffusion_model submodule -> 擋 BN 與 normalize queue 更新
            yield
        finally:
            for m, was in prev:
                m.train(was)        # 還原各自原本的 training 旗標
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    # ------------------------------------------------------------------
    # 低階工具（呼叫者須已在 _frozen() 內）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _node_feat(self, model, data):
        """用當前 backbone 抽 512-d 特徵。eval 隔離由外層 _frozen 保證。"""
        return model.intermediate_forward(data)

    @torch.no_grad()
    def _dsm_loss(self, model, feat, t):
        """固定 timestep 的 DSM loss（normalize 用該節點自己的統計量；eval 下不更新 queue）。"""
        dm = model.diffusion_model
        x = dm.normalize(feat)
        return float(dm.get_loss_at_timestep(x, t).item())

    def _denoiser_vec(self, model):
        """把 denoiser 參數攤平成單一向量（key 順序固定，跨節點一致）。"""
        named = dict(model.diffusion_model.named_parameters())
        if self._param_order is None:
            self._param_order = sorted(named.keys())
        return torch.cat([named[k].data.reshape(-1) for k in self._param_order])

    def _warn(self, msg):
        print(f"\033[91m[AGG-DIAG WARN] {msg}\033[0m", flush=True)

    def _diff_models(self, models_dict):
        return [m for m in models_dict.values() if _has_diffusion(m)]

    # ------------------------------------------------------------------
    # D4 pre：聚合前快照（在 _aggregate_models snapshot 之後、copy_ 之前呼叫）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def before_agg(self, models_dict, epoch):
        if (self.round + 1) % self.d4_every != 0:
            return  # 這一輪不算 D4
        self._d4_pre = {}
        with self._frozen(self._diff_models(models_dict)):
            for node, model in models_dict.items():
                if not _has_diffusion(model) or node not in self.node_probe_images:
                    continue
                data, _ = self.node_probe_images[node]
                feat = self._node_feat(model, data)            # 聚合前 backbone 特徵
                lpre = self._dsm_loss(model, feat, self.t_probe)
                self._d4_pre[node] = (feat, lpre)

    # ------------------------------------------------------------------
    # D1 + D4 post：聚合後（在 diffusion 參數 copy_ 之後呼叫）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def after_agg(self, models_dict, epoch):
        self.round += 1
        has_nan = 0

        with self._frozen(self._diff_models(models_dict)):
            # ---- D1：denoiser 參數發散度（純參數，不需 RNG/forward，但一併包進 guard 無害）----
            vecs, node_list = {}, []
            for node, model in models_dict.items():
                if not _has_diffusion(model):
                    continue
                vecs[node] = self._denoiser_vec(model)
                node_list.append(node)

            d1_global = d1_max = d1_intra = d1_inter = float("nan")
            if len(node_list) >= 1:
                stacked = torch.stack([vecs[n] for n in node_list])     # [K, P]
                gmean = stacked.mean(dim=0)
                gnorm = gmean.norm().item() + 1e-12
                rel = [(vecs[n] - gmean).norm().item() / gnorm for n in node_list]
                d1_global = sum(rel) / len(rel)
                d1_max = max(rel)

                # 分層：intra-domain（組內對組質心）vs inter-domain（組質心對全域質心）
                groups = {}
                for n in node_list:
                    groups.setdefault(self.node_to_domain.get(n, n), []).append(n)
                intra_vals, inter_vals = [], []
                for members in groups.values():
                    cmean = torch.stack([vecs[n] for n in members]).mean(dim=0)
                    cnorm = cmean.norm().item() + 1e-12
                    for n in members:
                        intra_vals.append((vecs[n] - cmean).norm().item() / cnorm)
                    inter_vals.append((cmean - gmean).norm().item() / gnorm)
                d1_intra = sum(intra_vals) / len(intra_vals) if intra_vals else float("nan")
                d1_inter = sum(inter_vals) / len(inter_vals) if inter_vals else float("nan")

            # ---- D4：聚合前後 ΔL（用聚合前快取的 feat，隔離 denoiser 效應）----
            d4_mean = d4_max = float("nan")
            if self._d4_pre:
                dls = []
                for node, (feat_pre, lpre) in self._d4_pre.items():
                    model = models_dict.get(node)
                    if model is None or not _has_diffusion(model):
                        continue
                    lpost = self._dsm_loss(model, feat_pre, self.t_probe)  # 聚合後 denoiser
                    dls.append(lpost - lpre)
                if dls:
                    d4_mean = sum(dls) / len(dls)
                    d4_max = max(dls, key=abs)
                self._d4_pre = {}

        # ---- NaN/Inf 守望（只警告，不中斷）----
        for label, v in [("D1_global", d1_global), ("D4_mean", d4_mean)]:
            if _is_bad(v):
                has_nan = 1
                self._warn(f"round={self.round} epoch={epoch}: {label} is NaN/Inf — 模型可能已崩，建議檢查並考慮 kill")

        self._w_d14.writerow([
            self.round, epoch, len(node_list),
            f"{d1_global:.6e}", f"{d1_max:.6e}", f"{d1_intra:.6e}", f"{d1_inter:.6e}",
            f"{d4_mean:.6e}", f"{d4_max:.6e}", has_nan,
        ])
        self._f_d14.flush()

        if self.verbose:
            print(f"[AGG-DIAG] r={self.round} ep={epoch} | "
                  f"D1 glob={d1_global:.3e} max={d1_max:.3e} intra={d1_intra:.3e} inter={d1_inter:.3e} | "
                  f"D4 dL_mean={d4_mean:.3e}", flush=True)

    # ------------------------------------------------------------------
    # D2：聚合模型在 ID 特徵上的 DSM loss（epoch 級）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def log_d2(self, models_dict, epoch):
        per_domain = {}
        with self._frozen(self._diff_models(models_dict)):
            for node, model in models_dict.items():
                if not _has_diffusion(model) or node not in self.node_probe_images:
                    continue
                data, _ = self.node_probe_images[node]
                feat = self._node_feat(model, data)
                nll = self._dsm_loss(model, feat, self.t_probe)
                dom = self.node_to_domain.get(node, node)
                bad = 1 if _is_bad(nll) else 0
                if bad:
                    self._warn(f"D2 epoch={epoch} node={node}: NLL NaN/Inf")
                self._w_d2.writerow([epoch, "node", node, dom, f"{nll:.6e}", bad])
                per_domain.setdefault(dom, []).append(nll)
        # per-domain 平均
        for dom, vals in per_domain.items():
            good = [v for v in vals if not _is_bad(v)]
            avg = sum(good) / len(good) if good else float("nan")
            self._w_d2.writerow([epoch, "domain", dom, dom, f"{avg:.6e}", 0 if good else 1])
        self._f_d2.flush()

    # ------------------------------------------------------------------
    # D3：跨節點異質性矩陣 S[i,j] = node_i denoiser 在 node_j 特徵上的 loss（每 M epoch）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def log_d3(self, models_dict, epoch):
        with self._frozen(self._diff_models(models_dict)):
            # 先用各節點自己的 backbone 抽其 probe 特徵
            feats = {}
            for node, model in models_dict.items():
                if not _has_diffusion(model) or node not in self.node_probe_images:
                    continue
                data, _ = self.node_probe_images[node]
                feats[node] = self._node_feat(model, data)
            nodes = list(feats.keys())
            for i in nodes:
                for j in nodes:
                    s = self._dsm_loss(models_dict[i], feats[j], self.t_probe)
                    bad = 1 if _is_bad(s) else 0
                    self._w_d3.writerow([
                        epoch, i, j,
                        self.node_to_domain.get(i, i), self.node_to_domain.get(j, j),
                        f"{s:.6e}", bad,
                    ])
        self._f_d3.flush()

    def close(self):
        for f in (self._f_d14, self._f_d2, self._f_d3):
            try:
                f.close()
            except Exception:
                pass
