"""混合速度 vs 學習速度——0921 連結度軸報告 §4.6／§4.7 的數字來源。

要回答的問題：拓樸變稀疏（甚至改成非同步），為什麼**泛化成績的收斂速度**幾乎不變？

論證骨架（本檔把三組數字放到同一個尺度上）：
  1. 成績曲線是被**每一步的梯度更新**推上去的，而每個節點每一步都照做一次更新，
     **與拓樸無關**；拓樸只決定「別人做的那一步，多久會反映到我身上」。
  2. ⇒ 只要「鄰居的貢獻傳到我」比「成績看得出變化」快夠多，曲線就畫不出差別。
  3. 本檔算：①各臂的混合時間（幾步把九個節點拉成一樣）
               ②學習的時間尺度（成績每進步 1pp 要幾步）
               ③非同步的實際通訊密度（每步幾條邊在動）

⚠️ 三個口徑限制（引用時必帶）：
  1. 混合時間是**圖論計算**（用訓練時實際的混合矩陣 W = I − α·L），不是量訓練中的共識誤差
     ——同步臂沒有逐 epoch 的共識記錄（`consensus.csv` 只有非同步路徑才寫）。
  2. 學習時間尺度用 `avg_test_acc`（**分母含 person**）算，只能當「曲線變化的快慢」，
     不可當最終成績（A2 曲線最低但 closed_acc 最高）。
  3. 非同步的「有效混合時間」是用**每步活躍邊數比例**換算的**粗估**（mixing ≈ 同步值 ÷ 活躍比例），
     不是直接量的。

用法：./venv_matcha/bin/python scripts/posthoc/topology_mixing_vs_learning.py
"""
import csv
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import util  # noqa: E402
from graph_manager import FixedProcessor  # noqa: E402

N_NODES = 9
TOPO_SEED = 1234
STEPS_PER_EPOCH = 20     # = max(各節點 batch 數)；sketch 節點 20 batch/epoch（訓練 log 第 49-65 行）
EPOCHS = 200
LOGS = {"A1 完全圖 36邊": "logs/topology_axis/0918_A1_fc_sync_cartoon.log",
        "A2 RGG 23邊": "logs/topology_axis/0918_A2_rgg_sync_cartoon.log",
        "A3 環狀 9邊": "logs/topology_axis/0918_A3_ring_sync_cartoon.log"}
ARMS = [("A1 完全圖", dict(radius=1.5, topology="rgg")),
        ("A2 RGG", dict(radius=0.8, topology="rgg")),
        ("A3 環狀", dict(radius=0.8, topology="ring"))]


def mixing(n, **kw):
    """回傳 (α, |λ2|, 混合時間, 完全混勻步數)。W = I − α·ΣL_i，取第二大特徵值。"""
    GP = FixedProcessor(util.select_graph(6, num_nodes=n, seed=TOPO_SEED, **kw),
                        1.0, 0, n, 10, True, comm=None)
    a = GP.getAlpha()
    W = np.eye(n) - a * sum(np.array(m) for m in GP.L_matrices)
    rho = np.sort(np.abs(np.linalg.eigvals(W)))[::-1][1]
    return a, rho, 1 / (1 - rho), np.log(0.01) / np.log(rho)


def main():
    total = STEPS_PER_EPOCH * EPOCHS
    print(f"一個 epoch = {STEPS_PER_EPOCH} 步；全程 {EPOCHS} epoch = {total} 步\n")

    print("【1】混合速度：幾步把九個節點的模型拉成一樣")
    print(f"  {'臂':<12}{'α':>10}{'|λ2|':>10}{'混合時間':>12}{'完全混勻':>12}{'佔訓練':>10}")
    for tag, kw in ARMS:
        a, rho, tmix, t99 = mixing(N_NODES, **kw)
        print(f"  {tag:<12}{a:>10.4f}{rho:>10.4f}{tmix:>11.1f}步{t99:>11.0f}步{t99/total:>10.1%}")

    print("\n【2】學習速度：成績每進步 1 個百分點要幾步（A1，10-epoch 移動平均）")
    txt = open(LOGS["A1 完全圖 36邊"], encoding="utf-8", errors="ignore").read()
    c = np.array([float(x) for x in re.findall(r"avg_test_acc=([0-9.]+)%", txt)])
    sm = np.convolve(c, np.ones(10) / 10, mode="valid")
    for a_, b_, tag in [(0, 20, "ep 1–20 起步"), (20, 50, "ep 21–50"),
                        (50, 100, "ep 51–100"), (100, 150, "ep 101–150")]:
        d = sm[b_ - 1] - sm[a_]
        rate = d / (b_ - a_)
        s = STEPS_PER_EPOCH / rate if rate > 1e-3 else float("inf")
        print(f"  {tag:<14}{sm[a_]:>7.2f} → {sm[b_-1]:<7.2f}({d:+.2f}pp)"
              + (f"  ⇒ 每 1pp 需要 {s:,.0f} 步" if np.isfinite(s) else "  ⇒ 已持平"))

    print("\n【3】非同步的實際通訊密度（各折 async_diag/broadcast_log.csv）")
    _, _, tmix_rgg, _ = mixing(N_NODES, radius=0.8, topology="rgg")
    pat = "exp_result_v1_stage2_leave_*_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234*/async_diag/broadcast_log.csv"
    for d in sorted(glob.glob(pat)):
        fold = d.split("leave_")[1].split("_p1a")[0]
        rows = list(csv.DictReader(open(d)))
        fired = np.array([int(r["fired"]) for r in rows])
        push = np.array([int(r["n_pushed"]) for r in rows])
        step = np.array([int(r["step"]) for r in rows])
        n = step.max() + 1
        tag = f"{fold}{'(A0)' if d.count('_fix') else ''}"
        print(f"  {tag:<20} 步數 {n:>5}  觸發率 {fired.mean():>6.2%}"
              f"（每 {1/fired.mean():>4.1f} 步）  每步推播 {push.sum()/n:>5.2f} 條")
        if "_fix" in d:                       # A0：再拆前後半段
            for lo, hi, t in [(0, n // 2, "ep 1–100"), (n // 2, n, "ep 101–200")]:
                m = (step >= lo) & (step < hi)
                edges = push[m].sum() / (hi - lo)
                frac = edges / 23.0           # 同步 RGG 每步 23 條
                print(f"      {t}: 觸發 {fired[m].mean():>6.2%}（每 {1/fired[m].mean():>4.1f} 步）"
                      f" 每步 {edges:>5.2f} 條 = 同步的 {frac:>5.1%}"
                      f" ⇒ 有效混合 ≈ {tmix_rgg/frac:>4.1f} 步[粗估]")

    print("\n【4】規模外推：環狀節點數變多時的混合成本")
    print(f"  {'節點數':>8}{'完全混勻':>12}{'≈epoch':>10}{'佔訓練':>10}")
    for n in [9, 15, 21, 27, 51, 99]:
        _, _, _, t99 = mixing(n, radius=0.8, topology="ring")
        print(f"  {n:>8}{t99:>11.0f}步{t99/STEPS_PER_EPOCH:>10.1f}{t99/total:>10.1%}")


if __name__ == "__main__":
    main()
