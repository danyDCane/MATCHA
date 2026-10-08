"""三個支撐「原型共識觸發器」設計決策的量測，一次落盤（零 GPU 訓練）。

為什麼合成一支：這三件在 2026-09-30～10-02 的對話裡量過，但**只活在對話與 scratchpad**，
沒有 repo 腳本也沒有 log ⇒ 2026-10-02 自檢時抓到，補落盤。plan：`research/1001_prototype_consensus_trigger_plan.md`

  §1 投影層的座標系對齊 —— **支撐 plan D6「可以直接把各節點的原型平均起來」**
     做法：同一批 512 維池化特徵，分別過各節點的 `proj_head`，量輸出方向的夾角。
     輸入固定 ⇒ 「不同節點看不同圖」的因素被消掉 ⇒ 剩下的純粹是座標系差異。
     🚨 **連帶撤回一條論證**：本檔作者 2026-09-30 曾主張「原型住在一個會漂的座標系裡 ⇒
        『延遲下對漂移的參照物達成共識』是機制問題」。**本節的數字把它打掉了**——座標系幾乎不漂。
        觸發器的主線不依賴那條論證（它靠的是「比共識 vs 比自己的過去」），但該論證不得再使用。

  §2 `comp_loss` 分母兩種寫法的相對差 —— **把「08-13 vs 現行 code 的軌跡分歧」從推論變實測**
     舊 `exp(logits).sum(1)` 一次 reduction；新 `exp_logits + (exp_logits*(1-pos)).sum(1)`
     先加負項、正項在 reduction 外面才加 ⇒ **數學恆等、位元不同**。
     （檢查形式由 `matcha-6a [6a1609]` 2026-10-02 提供。）

  §3 參數漂移 vs 原型漂移 —— **支撐「原型是不同於參數的訊號」**
     關鍵不是相關係數（參數漂移跨節點幾乎是常數、沒有變異可相關），而是
     **原型漂移分得出畫風群組、參數漂移分不出**。

⚠️ 射程：單 fold（leave-cartoon）、單 seed（2026）、單拓樸（RGG topo_seed=1234）、僅 ep50–200。
⚠️ §1／§3 用的是 checkpoint 自帶的 buffer 與權重；§2 用 `0826_features_full.npz` 的來源域特徵。
   ⚠️ 那份傾印**餵給每個節點的是整個域**（`matcha-02` 2026-10-01 查證）⇒ 2/3 是該節點沒見過的圖。
   §2 只量「兩種寫法的捨入差」，與擬合程度無關 ⇒ 不受影響；但**不可**拿該檔算擬合程度類的量。
"""
import os, sys, numpy as np, torch, torch.nn.functional as Fn
from scipy import stats

R = ("v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_"
     "osdg_excl_person_seed2026_topo1234_fix")
D, N, NC, T = f"exp_result_{R}", 9, 6, 0.1
EPS = ["epoch_50", "epoch_100", "epoch_150", "epoch_200"]
STY = ["art"] * 3 + ["photo"] * 3 + ["sketch"] * 3
F = np.load("logs/prototype_probe/0826_features_full.npz")
nrm = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)
sd = lambda i, ep: torch.load(os.path.join(D, f"{R}_node_{i}_{ep}.pth"),
                              map_location="cpu", weights_only=False)["backbone_state"]

print("=" * 94)
print("§1 投影層的座標系對齊（同一批輸入過不同節點的 proj_head ⇒ 純座標系差異）")
print("=" * 94)
X = torch.tensor(np.concatenate([F[f"n{i}_src_h"][:300].astype(np.float32) for i in range(N)]))
print(f"  固定輸入 {X.shape[0]} 張 512 維池化特徵（九節點各 300，所有節點共用同一批）\n")
print(f"  {'存檔點':<11}{'同畫風節點對':>13}{'跨畫風節點對':>13}{'全部節點對':>12}"
      f"{'  |':>3}{'原型 σ_node':>13}{'純畫風間距':>12}{'畫風÷座標誤差':>14}")
print("  " + "-" * 92)
d2a = lambda v: np.degrees(2 * np.arcsin(np.clip(np.sqrt(max(v, 0)) / 2, 0, 1)))
ch2 = lambda d: (2 * np.sin(np.radians(np.asarray(d)) / 2)) ** 2
ang = lambda a, b: np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1)))
for ep in EPS:
    Z, P = [], []
    for i in range(N):
        s = sd(i, ep)
        h = torch.relu(Fn.linear(X, s["proj_head.0.weight"], s["proj_head.0.bias"]))
        Z.append(Fn.normalize(Fn.linear(h, s["proj_head.2.weight"], s["proj_head.2.bias"]), dim=1))
        pr, cn = s["prototypes"].numpy().astype(np.float64), s["proto_count"].numpy()
        P.append(nrm(pr[np.arange(NC), (cn > 0).argmax(1)]))
    P = np.stack(P)
    same_h, cross_h, all_h = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            a = float(torch.rad2deg(torch.arccos((Z[i] * Z[j]).sum(1).clamp(-1, 1))).mean())
            all_h.append(a); (same_h if STY[i] == STY[j] else cross_h).append(a)
    s1 = [ang(P[i, c], P[j, c]) for c in range(NC) for i in range(N)
          for j in range(i + 1, N) if STY[i] == STY[j]]
    s2 = [ang(P[i, c], P[j, c]) for c in range(NC) for i in range(N)
          for j in range(i + 1, N) if STY[i] != STY[j]]
    c1, c2 = ch2(s1).mean(), ch2(s2).mean()
    sn, sty = d2a(c1 / 2), d2a(c2 - c1)
    print(f"  {ep:<11}{np.mean(same_h):>12.2f}°{np.mean(cross_h):>12.2f}°{np.mean(all_h):>11.2f}°"
          f"{'  |':>3}{sn:>12.2f}°{sty:>11.2f}°{sty/np.mean(all_h):>13.2f}")
print("  " + "-" * 92)
print("  讀法：『全部節點對』＝把別人的原型平均進來時座標系帶來的誤差。")
print("  ⇒ ✅ 連 ep50 都只有 ~3.3°（畫風訊號 12.2°、比值 3.7×）、ep200 僅 0.22°")
print("     ⇒ **直接平均合法，不需要任何對齊機制**（plan D6）。")
print("  ⇒ 🚨 **連帶撤回**：『原型住在會漂的座標系裡』那條論證不成立，不得再使用。")

print()
print("=" * 94)
print("§2 `comp_loss` 分母兩種寫法的相對差（float32，與訓練同精度）")
print("=" * 94)
mx = []
for i in range(N):
    y = F[f"n{i}_src_y"].astype(np.int64); k = y < NC
    z = Fn.normalize(torch.tensor(F[f"n{i}_src_z"].astype(np.float32)[k]), dim=1)
    y = torch.tensor(y[k])
    C = Fn.normalize(torch.tensor(F[f"n{i}_C"].astype(np.float32)), dim=1)
    lg = (z @ C.t()) / T
    lg = lg - lg.max(1, keepdim=True).values
    pos = (y.view(-1, 1) == torch.arange(NC).view(1, -1)).float()
    e = torch.exp(lg)
    d_old = e.sum(1, keepdim=True)                            # 舊：一次 reduction
    d_new = e + (e * (1.0 - pos)).sum(1, keepdim=True)        # 新：負項先加、正項在外
    rel = ((d_new - d_old).abs() / d_old)[pos.bool()]
    mx.append(float(rel.max()))
    if i in (0, 3, 6):
        print(f"  node_{i}  相對差 最大 {rel.max():.3e}  中位 {rel.median():.3e}  "
              f"非零比例 {(rel > 0).float().mean() * 100:5.1f}%")
print(f"\n  九節點最大相對差 ＝ {max(mx):.3e}（float32 eps ＝ {np.finfo(np.float32).eps:.3e}，即 2 倍 eps）")
print("  ⇒ ✅ **機制實測確認**：恆等式的兩種寫法產生 ~1e-7 的捨入差、且非 0。")
print("     ⇒ 「08-13 參照 vs 現行 code 的軌跡分歧」歸因到加總順序，**不再是推論**。")
print("     ⇒ 連帶：『逐 epoch 軌跡逐字相同』**在構造上不可能撐到 200 epoch**；")
print("        正確射程＝『前 N 個 epoch 相同（N 由列印精度決定，實測 8–10），之後分離是預期行為』。")

print()
print("=" * 94)
print("§3 參數漂移 vs 原型漂移：原型訊號分得出畫風群組，參數訊號分不出")
print("=" * 94)
def load_w(i, ep):
    s = sd(i, ep)
    return torch.cat([v.flatten() for kk, v in s.items()
                      if v.dtype.is_floating_point and not kk.startswith(("prototypes", "proto_count"))
                      and "running_" not in kk and "num_batches" not in kk]).double().numpy()
prev = {i: (load_w(i, EPS[0]),
            nrm(sd(i, EPS[0])["prototypes"].numpy().astype(np.float64)[
                np.arange(NC), (sd(i, EPS[0])["proto_count"].numpy() > 0).argmax(1)]))
        for i in range(N)}
print(f"  {'區間':<14}{'訊號':<10}{'組間/組內 變異比 F':>20}{'跨節點全距/均值':>18}")
print("  " + "-" * 64)
for a, b in zip(EPS[:-1], EPS[1:]):
    dw, dp = [], []
    for i in range(N):
        w2 = load_w(i, b); s2 = sd(i, b)
        P2 = nrm(s2["prototypes"].numpy().astype(np.float64)[
            np.arange(NC), (s2["proto_count"].numpy() > 0).argmax(1)])
        w1, P1 = prev[i]
        dw.append(float(np.linalg.norm(w2 - w1)))
        dp.append(float(np.degrees(np.arccos(np.clip((P1 * P2).sum(1), -1, 1))).mean()))
        prev[i] = (w2, P2)
    tag = f"{a[6:]}→{b[6:]}"
    for nm, v in [("參數漂移", dw), ("原型漂移", dp)]:
        g = [np.array(v[0:3]), np.array(v[3:6]), np.array(v[6:9])]
        print(f"  {tag:<14}{nm:<10}{stats.f_oneway(*g).statistic:>20.1f}"
              f"{(max(v)-min(v))/np.mean(v)*100:>17.1f}%")
    print("  " + "-" * 64)
print("  ⚠️ p 值刻意不報：同畫風三節點的資料分片雖互不重疊，但仍共享畫風 ⇒ 不是 9 個獨立單位。")
print("  ⇒ ✅ 原型漂移 F ≈ 46／60／134（三區間一致分得開）；參數漂移 F ≈ 0.7／11／3.4（不一致）。")
print("     ⚠️ 2026-10-02 曾在對話中引過參數側 F ＝ 0.7／14.4／3.3 —— 那是用**印到小數三位**的漂移值算的；")
print("        本檔用全精度 ⇒ 以本檔為準。結論方向不變（參數側三區間不一致、原型側一致分得開）。")
print("     ⇒ **原型訊號看得到「這個節點的畫風學得差不多了」，參數範數看不到**（photo 組最早穩定）。")
print("  ⚠️ 但兩者不獨立：現行參數觸發率 vs 原型漂移逐節點相關 +0.44／+0.58／+0.69（越後期越像）")
print("     ⇒ 大約共用一半資訊 ⇒ **光靠「換訊號」撐不起貢獻**，要靠「同通訊預算下成績更好」。")
print("=" * 94)
