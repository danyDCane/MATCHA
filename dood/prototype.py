"""類別原型的檢測讀出：buffer、EMA 更新、三個損失、角距離分數。

設計的單一真相源＝`research/decentralized_ood_transplant_review/0803_method_design_and_stage_plan.md`
（§2.2 資料流、§2.6 兩層結構、§2.7 模組角色）。本檔只實作，不重複理由。

【要長成的結構（0803 §2.6）】
    六個「大群」＝六個類別，彼此角度拉開
    每個大群內部＝三個「小群」＝三個來源畫風，彼此靠近但不重合
    未見畫風的已知類別 → 落在大群內部、往類別中心靠
    未知類別           → 落在大群之外

    層級              負責的項            作用域
    小群自己要緊      L_comp (λ_c)        樣本 → 自己的 (類別,畫風) 原型
    小群互相靠近      L_style (λ_s)       樣本 → 自己的類別中心      ★ 階段 1 恆為 0
    大群彼此分開      L_disp (λ_d)        6 個類別中心兩兩            ⚠️ 不是 18 個原型兩兩

【損失形式沿用 CIDER（`CIDER-main/utils/losses.py`）】
  - L_comp 對應 CompLoss（:162-185）：對原型集合做 softmax、正樣本取自身類別
    ★ 我們把正樣本遮罩放寬為「**同類別、任一畫風**」——
      階段 1（每節點單域、只有 6 格有值）⇒ 每個樣本剛好一個正樣本 ⇒ **退化成 CIDER 原形**；
      階段 2b（18 格有值）⇒ 三個正樣本 ⇒ **不會把同類別的三個畫風原型推開**，正是 0803 §2.6.2 要的。
  - L_disp 對應 DisLoss（:266-285）：但**作用在 6 個類別中心**，不是 18 個原型。
  - EMA 更新對應 :263 的**逐樣本**迴圈（不是逐 batch 平均）——窗口約 1/(1-m) 個「樣本」。

【與 CIDER 的已知偏離（引用時必須標明）】
  1. CIDER 訓練時**沒有交叉熵**（其 `get_criterion` 是 palm/CE 互斥二選一），我們是聯合訓練
     ⇒ **CIDER 的「緊緻不傷分類」結論不可搬**（0803 §3.5）。連帶 `λ_c=2` 也不可照抄。
  2. 原型用 `register_buffer`，CIDER 用 `nn.Parameter`（:242）——後者在本架構會被優化器與
     weight decay 碰到（0803 §2.7.4）。
  3. EMA 只吃**未擾動**特徵，緊緻項拉的是**擾動後**特徵（0803 §4.1「乾淨 vs 擾動」）。
"""

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# buffer
# --------------------------------------------------------------------------- #
def init_prototype_buffers(model, n_cls, n_dom, dim, device=None):
    """在 model 上掛原型 buffer。

    形狀從一開始就是 [類別, 畫風, 維度]，階段 1 只填自己那一格 ⇒ 階段 2b 不必改形狀。
    ⚠️ 必須是 buffer 不是 Parameter：否則會被優化器與 weight decay 碰到。
    """
    if hasattr(model, "prototypes"):
        return
    model.register_buffer("prototypes", torch.zeros(n_cls, n_dom, dim, device=device))
    model.register_buffer("proto_count", torch.zeros(n_cls, n_dom, device=device))


def class_centers(prototypes, proto_count, eps=1e-8):
    """由 [C, D, P] 的原型算出 [C, P] 的類別中心。

    用 proto_count 當遮罩，避免把「還沒有值的格子」（全零）平均進去——
    階段 1 每個節點只有一個畫風有值，若不遮罩，中心會被零向量拉掉 2/3。
    """
    mask = (proto_count > 0).float().unsqueeze(-1)          # [C, D, 1]
    summed = (prototypes * mask).sum(dim=1)                 # [C, P]
    denom = mask.sum(dim=1).clamp_min(1.0)                  # [C, 1]
    return F.normalize(summed / denom, dim=1, eps=eps)


@torch.no_grad()
def update_prototypes_(model, z_clean, labels, dom_idx, m):
    """逐樣本 EMA 更新（對齊 CIDER losses.py:263）。

    Args:
        z_clean: [B, P] **未擾動**特徵經投影並 L2 正規化後的結果
        labels:  [B]
        dom_idx: int，本節點的來源畫風索引（每節點固定）
        m:       記憶強度（CIDER 的 proto_m）。越大越看重舊值。

    ⚠️ 必須傳未擾動特徵。若傳擾動後的，(類別,畫風) 語意會被鄰居畫風污染，
       而兩層結構與跨節點聚合全都建立在這個語意上（0803 §4.1）。
    """
    protos = model.prototypes
    counts = model.proto_count
    for j in range(z_clean.size(0)):
        c = int(labels[j].item())
        if counts[c, dom_idx] == 0:
            protos[c, dom_idx] = F.normalize(z_clean[j], dim=0)      # 第一次直接放進去
        else:
            protos[c, dom_idx] = F.normalize(
                protos[c, dom_idx] * m + z_clean[j] * (1.0 - m), dim=0)
        counts[c, dom_idx] += 1


# --------------------------------------------------------------------------- #
# 三個損失
# --------------------------------------------------------------------------- #
def comp_loss(z, labels, prototypes, proto_count, temperature=0.1):
    """L_comp：把樣本拉向「自己類別」的原型（沿用 CIDER CompLoss 的 softmax 形式）。

    ★ 正樣本遮罩＝「同類別、任一畫風」：
      階段 1（只有 6 格有值）⇒ 每樣本一個正樣本 ⇒ 與 CIDER 原形等價
      階段 2b（18 格有值）  ⇒ 每樣本三個正樣本 ⇒ **不會把同類別的三個畫風原型推開**
    """
    C, D, _ = prototypes.shape
    filled = (proto_count > 0).view(-1)                              # [C*D]
    if filled.sum() < 2:
        return z.new_zeros(())                                       # 還沒暖機完，沒有可比的原型

    flat = F.normalize(prototypes.view(C * D, -1)[filled], dim=1)    # [F, P]
    proxy_cls = torch.arange(C, device=z.device).repeat_interleave(D)[filled]   # [F]

    logits = (z @ flat.t()) / temperature                            # [B, F]
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()  # 數值穩定
    log_prob = logits - torch.log(torch.exp(logits).sum(dim=1, keepdim=True))

    pos = (labels.view(-1, 1) == proxy_cls.view(1, -1)).float()      # [B, F] 同類別即正樣本
    n_pos = pos.sum(dim=1).clamp_min(1.0)
    return -((pos * log_prob).sum(dim=1) / n_pos).mean()


def style_loss(z, labels, centers):
    """L_style：把樣本拉向自己的**類別中心**，讓同類別的三個畫風靠近。

    ⚠️ 階段 1 恆為 0 的原因：每節點只有一個畫風有值 ⇒ 類別中心 ≡ 該畫風的原型
       ⇒ 本項與 L_comp 是同一個量。故階段 1 應把 λ_s 設 0，避免重複計算。
       （0803 §2.6.5）
    """
    return (1.0 - (z * centers[labels]).sum(dim=1)).mean()


def ema_prototypes_live(prototypes, proto_count, z_clean, labels, dom_idx, m):
    """逐樣本 EMA，回傳「**帶梯度**的原型」與更新後的計數（存檔用 `.detach()`）。

    ★ 這是 `differentiable_prototypes` 的正確版本（2026-08-12 取代之）。差別在哪：

        舊（影子原型）  存檔×0.95 + **z_aug 批平均**×0.05
                        ⇒ L_disp 的梯度只作用在 z_aug 批平均上，**存檔原型從未被直接推開**；
                        ⇒ 一次只混 5%，梯度通道比 CIDER 窄約 8 倍（CIDER 是 1-m^k，k≈10 時 40%）。
        新（本函式）    逐樣本 EMA、餵 **z_clean**（帶梯度）
                        ⇒ 回傳值與存檔原型**數值完全相同**，只差有沒有帶梯度
                        ⇒ 與 CIDER `DisLoss.forward`（losses.py:261-264）同構。

    ★ 為什麼餵 z_clean 而不是 z_aug：存檔原型必須只由**未擾動**特徵決定，(類別,畫風) 語意
      才守得住，階段 2 的跨節點聚合才有意義（0803 §4.1）。CIDER 沒這個約束（它只有一組原型、
      也不做跨節點聚合），所以它餵的是兩個 augmented view。

    ⚠️ 呼叫端必須讓 z_clean **帶計算圖**（乾淨前向不可包在 no_grad 裡），否則本函式退化成
       舊行為：梯度恆為 0、L_disp 純裝飾（= V2B1 失效模式）。
    ⚠️ 梯度沿 EMA 鏈往前反傳（第 j 個樣本的更新依賴第 j-1 個），與 CIDER 相同。

    Returns:
        (protos, counts)：protos 帶梯度、counts 不帶。存檔請用 `protos.detach()`。
    """
    protos = prototypes.clone()
    counts = proto_count.clone()
    for j in range(z_clean.size(0)):
        c = int(labels[j].item())
        if counts[c, dom_idx] == 0:
            protos[c, dom_idx] = F.normalize(z_clean[j], dim=0)
        else:
            protos[c, dom_idx] = F.normalize(
                protos[c, dom_idx] * m + z_clean[j] * (1.0 - m), dim=0)
        counts[c, dom_idx] += 1
    return protos, counts


def differentiable_prototypes(prototypes, proto_count, z, labels, dom_idx, m):
    """⛔ 已停用（2026-08-12）——改用 `ema_prototypes_live`。僅保留供重現 1a 的結果。

    停用原因（兩個都是概念錯誤，不是效能問題）：
      1. `upd = prototypes.detach()` ⇒ 存檔那部分是常數，梯度只能從 `batch_mean(z_aug)×0.05` 流
         ⇒ **L_disp 推的是「本批 z_aug 的類別平均」，存檔原型從未被直接推開**。
         1a 的存檔原型仍達到理論最優的 99.8%，是因為 z_aug/z_clean 共享 backbone 而間接被帶動。
      2. 「批平均混一次 5%」≠ CIDER 的「逐樣本 EMA」。以每類每批 ~10 個樣本計，CIDER 讓新特徵
         佔 1-0.95^10 = 40%，本函式只佔 5% ⇒ **梯度通道窄約 8 倍**（實測 ‖g_disp‖=1.3 vs ‖g_cls‖=5.7），
         再由梯度平衡器把 λ_d 放大到 0.46 硬補回 10% 的份額。

    ---- 以下為原始說明（保留供對照）----

    回傳「若用本批 z 更新過」的可微原型副本（存起來的那組不受影響）。

    ★ 為什麼需要這個（2026-08-10 smoke test 抓到）：
      存起來的原型是 buffer、在 no_grad 下用**乾淨**特徵更新 ⇒ 對參數是常數
      ⇒ 直接對它算 L_disp **梯度恆為 0**（實測 ||g||=0、lam=0），該項純裝飾。
      這正是 V2B1 的失效模式（‖g_reg‖≈0 ⇒ 根本沒動 backbone）。

    ★ CIDER 為什麼沒這問題：其 DisLoss.forward **把 EMA 更新寫在損失裡面**、用當前 batch 的
      features（有梯度），損失用的區域變數仍帶著計算圖，**之後**才 detach 存起來
      （`CIDER-main/utils/losses.py:262-264`）⇒ 推開的梯度經由 EMA 流回特徵。

    ★ 本函式＝同一條公式，只是餵進去的新貢獻取自 z_aug（擾動後、有梯度），與 L_comp 拉的
      對象一致；而**存起來、拿去算檢測分數的那組原型仍只吃乾淨特徵**，(類別,畫風) 語意不變。
      梯度會被 (1-m) 衰減——這與 CIDER 相同，靠梯度範數平衡補回來。

    ⚠️ 不可拿掉 L_disp 改靠 L_comp 的 softmax 分母：CIDER Table 3 消融顯示「只有 compactness」
       AUROC 54.06、加上 dispersion 87.67（差 33 分）；Table 4 量出 SupCon 式的隱含散布只有
       75.50° vs CIDER 87.53°。⚠️ 但該消融是在**無交叉熵**的設定下量的，54.06 不是我們的數字。
    """
    C, D, P = prototypes.shape
    upd = prototypes.detach().clone()
    # 本批各類別的平均（只動本節點自己那一格畫風）
    onehot = F.one_hot(labels, C).to(z.dtype)                        # [B, C]
    cnt = onehot.sum(dim=0)                                          # [C]
    summed = onehot.t() @ z                                          # [C, P]
    present = cnt > 0
    if not present.any():
        return upd
    batch_mean = F.normalize(summed[present] / cnt[present].unsqueeze(1), dim=1)

    old = upd[present, dom_idx]                                      # [n, P]
    seen = (proto_count[present, dom_idx] > 0).unsqueeze(1).to(z.dtype)
    # 首次出現的格子直接用本批平均，其餘照 EMA 混合（與 update_prototypes_ 的規則一致）
    mixed = F.normalize(seen * (old * m + batch_mean * (1.0 - m))
                        + (1.0 - seen) * batch_mean, dim=1)
    out = upd.clone()
    idx = torch.nonzero(present, as_tuple=True)[0]
    out[idx, dom_idx] = mixed                                        # 可微
    return out


def disp_loss(centers, proto_count, temperature=0.1):
    """L_disp：把 6 個**類別中心**互相推開（沿用 CIDER DisLoss 的形式）。

    ⚠️ 作用在類別中心、**不是 18 個原型兩兩推**——後者會把同類別的三個畫風原型也推開，
       主動撐大大群、與目標結構相反（0803 §2.6.2）。
    ⚠️ 傳進來的 centers 必須由 `differentiable_prototypes()` 算出，否則**梯度恆為 0**。
    """
    alive = (proto_count > 0).any(dim=1)                             # [C] 有值的類別
    if alive.sum() < 2:
        return centers.new_zeros(())
    cen = centers[alive]
    logits = (cen @ cen.t()) / temperature
    off = 1.0 - torch.eye(cen.size(0), device=cen.device)
    mean_prob_neg = torch.log((off * torch.exp(logits)).sum(dim=1) / off.sum(dim=1))
    mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]
    if mean_prob_neg.numel() == 0:
        return centers.new_zeros(())
    return mean_prob_neg.mean()


def rel_loss(score_aug, score_clean, margin=0.0):
    """L_rel：風格擾動後的檢測分數，不得比原圖高出 margin 以上（單邊，變低不罰）。

    來源＝Generalize or Detect?（NeurIPS 2024, arXiv:2411.03829）Eq.4 第三項。
    ⚠️ **採官方原始碼版，與論文式子符號相反**——官方 `lib/loss.py`：
         F.relu(anomaly_score[aug] - anomaly_score[orig] - margin)
       論文 Eq.4 展開後要求擾動版比原版**低** margin。照論文修正反而會實作出不同的約束。
    """
    return F.relu(score_aug - score_clean - margin).mean()


# --------------------------------------------------------------------------- #
# 檢測分數
# --------------------------------------------------------------------------- #
def detection_score(z, centers):
    """S(x) = min_c arccos(z · c_c)：到最近**類別中心**的角距離。值域 [0, π]，越大越像 OOD。

    ⚠️ 量到 6 個類別中心、**不是 18 個原型取 min**：後者會讓坐在三個小群「中間」的
       covariate-shifted ID 到每個小群都有 r_見過 的距離，而坐在小群上的來源樣本距離≈0
       ⇒ 分數本身就在製造畫風 AUROC 的落差（0803 §2.2）。
    """
    cos = (z @ centers.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.arccos(cos).min(dim=1).values
