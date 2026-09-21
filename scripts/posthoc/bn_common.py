"""BN 平均（合併變異數版 B）——TaskBoard §A 評估協定的共用實作。

協定（2026-08-25 定案）：所有指標與幾何診斷一律先把 9 節點的 BN running 統計量平均再算。
理由：跨節點落差的 87% 是 BN 統計量（conv 權重分歧 7.7e-5 vs BN running 0.131），
不平均就是在被污染的基底上診斷。
⚠️ 例外：目的就是要看「節點之間不一致」的分析不可先平均。

原始實作出處：compactness_transfer_four_arms.py:48（2026-08-25 抽出共用）。
"""
import os
import torch


def bn_avg(CK, DESC, N=9, ckpt_tag="final"):
    """回傳 {BN key: 平均後的張量}。合併變異數：mean(var_i + mean_i²) − mean(mean_i)²"""
    S = [torch.load(os.path.join(CK, f"{DESC}_node_{i}_{ckpt_tag}.pth"), map_location="cpu",
                    weights_only=False)["backbone_state"] for i in range(N)]
    BKs = [k for k in S[0] if k.endswith("running_mean") or k.endswith("running_var")]
    A = {}
    for k in BKs:
        if k.endswith("running_mean"):
            A[k] = torch.stack([S[i][k].float() for i in range(N)]).mean(0)
    for k in BKs:
        if k.endswith("running_var"):
            mk = k.replace("running_var", "running_mean")
            mi = torch.stack([S[i][mk].float() for i in range(N)])
            vi = torch.stack([S[i][k].float() for i in range(N)])
            A[k] = (vi + mi ** 2).mean(0) - mi.mean(0) ** 2
    del S
    return A


def apply_bn(bb, AVG):
    """把平均後的統計量塞回模型（原地）"""
    sd = bb.state_dict()
    for k, v in AVG.items():
        sd[k].copy_(v.to(sd[k].device).to(sd[k].dtype))
    return bb
