"""三個同步臂的「鄰居畫風覆蓋」——0921 連結度軸報告 §4.5 的數字來源。

問題：為什麼連結度從 36 條邊砍到 23 條邊泛化不動、砍到環狀（9 條邊）才掉？

關鍵前提（讀 code 確認，非臆測）：**畫風統計量只傳一跳、不轉發、每輪清空**
  - `communicator.py:decenCommunicator.averaging` 的風格分支：每輪先 `neighbor_style_vecs.clear()`，
    再把 `self.style_send_buffer` 送給鄰居；該 buffer ＝ `self.local_style_vec`（`prepare_style_buffer`）
    ＝ 本節點自己那批資料算出來的統計量（`train.py:654` 的 `flatten_style_stats`）。
  - ⇒ 沒有任何「把收到的鄰居風格再轉發出去」的路徑 ⇒ **看不到的畫風不是晚點會到，是永遠不會到**。
  - 對照：模型參數走的是多跳、會累積的通道（同函式的參數分支）⇒ 只要圖連通，晚到 ≠ 不到。

因此「每個節點見得到幾種別人的畫風」＝ 直接鄰居的畫風組成，與邊的總數沒有直接關係。
本檔重建三臂的實際鄰接圖並逐節點算這個量。

用法：./venv_matcha/bin/python scripts/posthoc/topology_style_coverage.py
      （純 CPU、不需 checkpoint；圖的建構與訓練時走同一個 `util.select_graph`）

⚠️ 射程：cartoon 折（來源畫風＝art_painting／photo／sketch）、9 節點、topo_seed 1234。
   換折只改畫風名稱、不改節點分組（區塊式配置）⇒ 覆蓋數的結構相同。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import util  # noqa: E402

DOMS = ["art_painting", "photo", "sketch"]   # cartoon 折的三個來源畫風
N_NODES = 9
TOPO_SEED = 1234

ARMS = [
    ("A1 完全圖 (r=1.5)", dict(radius=1.5, topology="rgg")),
    ("A2 RGG (r=0.8)", dict(radius=0.8, topology="rgg")),
    ("A3 環狀", dict(radius=0.8, topology="ring")),
]


def adjacency(subgraphs):
    """子圖（matching）清單 → 無向鄰接表。"""
    adj = {i: set() for i in range(N_NODES)}
    for matching in subgraphs:
        for i, j in matching:
            adj[i].add(j)
            adj[j].add(i)
    return adj


def main():
    n2d, _ = util.assign_nodes_to_domains(DOMS, N_NODES)
    print("節點→來源畫風（util.assign_nodes_to_domains，區塊式）：")
    print("  " + "  ".join(f"node_{i}={n2d[f'node_{i}'][:4]}" for i in range(N_NODES)))

    print(f"\n{'臂':<20}{'邊數':>6}{'平均他人畫風數':>16}{'覆蓋 0 種的節點':>20}")
    rows = []
    for tag, kw in ARMS:
        adj = adjacency(util.select_graph(6, num_nodes=N_NODES, seed=TOPO_SEED, **kw))
        n_edges = sum(len(v) for v in adj.values()) // 2
        others, starved = [], []
        for i in range(N_NODES):
            mine = n2d[f"node_{i}"]
            k = len({n2d[f"node_{j}"] for j in adj[i]} - {mine})
            others.append(k)
            if k == 0:
                starved.append(i)
        rows.append((tag, n_edges, adj, others, starved))
        print(f"{tag:<20}{n_edges:>6}{sum(others) / N_NODES:>16.2f}"
              f"{(str(starved) if starved else '無'):>20}")

    for tag, n_edges, adj, others, starved in rows:
        print(f"\n--- {tag}｜{n_edges} 條邊 ---")
        for i in range(N_NODES):
            mine = n2d[f"node_{i}"]
            cov = sorted({n2d[f"node_{j}"][:4] for j in adj[i]})
            flag = "   ⚠️ 只看得到自己的畫風" if others[i] == 0 else ""
            print(f"  node_{i}({mine[:4]})  鄰居={sorted(adj[i])}  鄰居畫風={cov}  "
                  f"他人畫風數={others[i]}（滿分 2）{flag}")


if __name__ == "__main__":
    main()
