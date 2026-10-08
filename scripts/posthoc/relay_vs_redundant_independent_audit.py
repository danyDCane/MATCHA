# -*- coding: utf-8 -*-
"""獨立複核 matcha-b6 稽核報告 §10「轉傳還是重複」（2026-10-03，matcha-02 寫）

不 import b6 的腳本。三個攻擊點：
  §1  I0 檢查：時間軸對齊有沒有錯一格（含**同一步內跨節點的順序傳遞**這個地雷）
      —— 用自己的事件模擬重建 `n_received` 與 `mean_staleness`，對 36000 列逐列比。
  §2  分類與零點：「多出來的 96% 是重複」vs「每次廣播的有用筆數」哪個講法對
"""
import numpy as np, pandas as pd, networkx as nx, sys
sys.path.insert(0, '/home/server5090/Desktop/M11307320/MATCHA')
from util import select_graph

B = ('/home/server5090/Desktop/M11307320/MATCHA/exp_result_v1_stage2_leave_cartoon_p1a_async_const_'
     'tau1e-5_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234_fix/async_diag/broadcast_log.csv')
bl = pd.read_csv(B)
N, MAXSTEP = 9, int(bl.step.max()) + 1

# ── 拓撲：用 production 的同一支函式，不自己重建 ────────────────────────────
# select_graph 回傳的是 matchings（不相交邊集的清單）⇒ 全部邊的聯集就是拓撲
matchings = select_graph(6, num_nodes=N, radius=0.8, seed=1234, topology='rgg')
G = nx.Graph(); G.add_nodes_from(range(N))
for mt in matchings:
    for e in mt:
        G.add_edge(int(e[0]), int(e[1]))
adj = {i: sorted(G.neighbors(i)) for i in range(N)}
deg = [len(adj[i]) for i in range(N)]
print('=' * 100); print('§0 拓撲自檢'); print('=' * 100)
print(f'  度數 {deg}   ← 影子 CSV 的非空 r2c 欄位數應為 [5,2,5,6,6,3,8,6,5]')
print(f'  node_8 的鄰居 {adj[8]}（art＝0,1,2 ⇒ 應無 art 鄰居）')
print(f'  三角形數 {sum(nx.triangles(G).values())//3}   平均群聚係數 {nx.average_clustering(G):.3f}  ← b6 報 28／0.834')

dom2i = {f'node_{i}': i for i in range(N)}
fired = np.zeros((MAXSTEP, N), bool); nrec = np.full((MAXSTEP, N), -1); mstal = np.full((MAXSTEP, N), np.nan)
for r in bl.itertuples():
    k, i = int(r.step), dom2i[r.domain]
    fired[k, i] = bool(r.fired); nrec[k, i] = int(r.n_received); mstal[k, i] = float(r.mean_staleness)

print()
print('=' * 100)
print('§1 ★ I0 檢查：依序處理模型能不能逐列重建 n_received 與 mean_staleness')
print('=' * 100)
print('  模型：for k: for i in 0..8: ①若 fired 則寫入每個鄰居 inbox[j][i]=k（dedup 覆蓋）')
print('        ②讀自己 inbox、算 age=k−push_step、清空')
print('  🚨 地雷：**同一步內**，編號比我小的鄰居已經推了（我本步就收到、age=0）；')
print('     編號比我大的還沒推（我要到下一步才收到）⇒ 不對稱，模型必須照抄這個順序。')


def simulate(fire):
    inbox = [dict() for _ in range(N)]
    nr = np.zeros((MAXSTEP, N), int); ms = np.zeros((MAXSTEP, N)); deliv = []
    for k in range(MAXSTEP):
        for i in range(N):
            if fire[k, i]:
                for j in adj[i]:
                    inbox[j][i] = k                       # dedup 覆蓋
            box = inbox[i]
            nr[k, i] = len(box)
            if box:
                ages = [k - s for s in box.values()]
                ms[k, i] = sum(ages) / len(ages)
                for s, ps in box.items():
                    deliv.append((k, s, i, ps))
                inbox[i] = {}
    return nr, ms, deliv


nr, ms, deliv = simulate(fired)
ok_n = int((nr == nrec).sum()); ok_m = int((np.abs(ms - mstal) < 5e-3).sum())
print(f'\n  n_received      逐列吻合 {ok_n}/{MAXSTEP*N}  ({ok_n/(MAXSTEP*N)*100:.2f}%)')
print(f'  mean_staleness  逐列吻合 {ok_m}/{MAXSTEP*N}  ({ok_m/(MAXSTEP*N)*100:.2f}%)（容差 5e-3；CSV 存兩位小數）')
a0 = sum(1 for (k, s, i, ps) in deliv if k == ps)
print(f'  🚨 同一步送達（age=0）的筆數：{a0}／{len(deliv)} ＝ {a0/len(deliv)*100:.1f}%')
print(f'     ⇒ 模型若不照抄節點順序，這 {a0/len(deliv)*100:.0f}% 全部錯一格 ⇒ 100% 吻合本身就是對齊正確的證明。')


def simulate_wrong(fire):
    """反證：一律「下一步才送達」。若這個也 100% 吻合，上面的檢查就沒有鑑別力。"""
    inbox = [dict() for _ in range(N)]; pend = [dict() for _ in range(N)]
    nr = np.zeros((MAXSTEP, N), int)
    for k in range(MAXSTEP):
        for i in range(N):
            inbox[i].update(pend[i]); pend[i] = {}
        for i in range(N):
            if fire[k, i]:
                for j in adj[i]:
                    pend[j][i] = k
            nr[k, i] = len(inbox[i]); inbox[i] = {}
    return nr


nw = simulate_wrong(fired)
print(f'  對照（錯誤模型：一律下一步才送達）n_received 吻合 {int((nw == nrec).sum())/(MAXSTEP*N)*100:.2f}%'
      f'  ⇒ 這個檢查**有鑑別力**，不是恆真。')

print()
print('=' * 100)
print('§2 分類（新鮮度模型）與零點')
print('=' * 100)
WARM_STEPS = 200          # epoch 1–10，每 epoch 20 步


def classify(fire, deliv):
    know = [dict() for _ in range(N)]
    for i in range(N):
        know[i][i] = -1
    absorbed = [set() for _ in range(N)]
    by_k = {}
    for rec in deliv:
        by_k.setdefault(rec[0], []).append(rec)
    cnt = dict(useful=0, redundant=0, echo=0)
    rel = {'self': [0, 0, 0], 'adj': [0, 0, 0], 'far': [0, 0, 0]}
    per_bc = []
    for k in range(MAXSTEP):
        for i in range(N):
            if fire[k, i]:
                if k >= WARM_STEPS:
                    u = r_ = e = 0
                    for src in sorted(absorbed[i]):
                        for j in adj[i]:
                            if src == j:
                                cnt['echo'] += 1; e += 1; rel['self'][2] += 1
                            elif know[i][src] > know[j].get(src, -1):
                                cnt['useful'] += 1; u += 1
                                rel['adj' if src in adj[j] else 'far'][0] += 1
                            else:
                                cnt['redundant'] += 1; r_ += 1
                                rel['adj' if src in adj[j] else 'far'][1] += 1
                    per_bc.append((u, r_, e))
                absorbed[i] = set()
                for j in adj[i]:
                    know[j][i] = k
                    for src, st in list(know[i].items()):
                        if src != j and st > know[j].get(src, -1):
                            know[j][src] = st
        for (kk, s, i, ps) in by_k.get(k, []):
            if ps > know[i].get(s, -1):
                know[i][s] = ps
            absorbed[i].add(s)
    return cnt, rel, np.array(per_bc)


cnt, rel, per_bc = classify(fired, deliv)
tot = sum(cnt.values())
print(f'  實際：總筆數 {tot}   有用 {cnt["useful"]/tot*100:.1f}%   重複 {cnt["redundant"]/tot*100:.1f}%'
      f'   回聲 {cnt["echo"]/tot*100:.1f}%     ← b6 報 173,954／8.2%／79.1%／12.7%')
for key, lab in [('self', '來源就是收件人'), ('adj', '來源與收件人直接相連'), ('far', '來源與收件人不相連')]:
    s = sum(rel[key])
    if s:
        print(f'    {lab:<22} 占 {s/tot*100:5.1f}%   有用 {rel[key][0]/s*100:5.1f}%   重複 {rel[key][1]/s*100:5.1f}%')

rng = np.random.default_rng(20261003)
rows = []
for _ in range(20):
    f2 = np.zeros_like(fired)
    for i in range(N):
        sh = int(rng.integers(200, MAXSTEP - 200))
        f2[:, i] = np.roll(fired[:, i], sh)
    _, _, dv = simulate(f2)
    c2, _, pb2 = classify(f2, dv)
    t2 = sum(c2.values())
    rows.append((c2['useful']/t2*100, c2['redundant']/t2*100, c2['echo']/t2*100, t2,
                 pb2[:, 0].mean(), len(pb2)))
R = np.array(rows)
print(f'\n  零點（各節點 fired 環狀位移、20 次）：'
      f'有用 {R[:,0].mean():.1f}±{R[:,0].std():.1f}%  重複 {R[:,1].mean():.1f}±{R[:,1].std():.1f}%'
      f'  回聲 {R[:,2].mean():.1f}±{R[:,2].std():.1f}%  總筆數 {R[:,3].mean():.0f}±{R[:,3].std():.0f}')
print('                                  ← b6 報 10.2±0.7／76.8±0.6／13.0±0.4／118,531')

print()
print('  ★ 兩種講法的對照（b6 問的）：')
print(f'    {"":<30}{"實際":>12}{"零點":>14}{"倍率":>9}')
print(f'    {"總吸收筆數":<30}{tot:>12}{R[:,3].mean():>14.0f}{tot/R[:,3].mean():>8.2f}x')
print(f'    {"廣播次數":<30}{len(per_bc):>12}{R[:,5].mean():>14.0f}{len(per_bc)/R[:,5].mean():>8.2f}x')
print(f'    {"**每次廣播的有用筆數**":<30}{per_bc[:,0].mean():>12.3f}{R[:,4].mean():>14.3f}{per_bc[:,0].mean()/R[:,4].mean():>8.2f}x')
print(f'    {"每次廣播的總吸收筆數":<30}{per_bc.sum(1).mean():>12.3f}{(R[:,3]/R[:,5]).mean():>14.3f}'
      f'{per_bc.sum(1).mean()/(R[:,3]/R[:,5]).mean():>8.2f}x')
d_tot = tot - R[:, 3].mean(); d_use = cnt['useful'] - (R[:, 0]/100*R[:, 3]).mean()
print(f'\n    b6 的差分講法：多出 {d_tot:.0f} 筆、其中有用 {d_use:.0f} 筆'
      f' ⇒ 多出來的 {100-d_use/d_tot*100:.1f}% 是重複（b6 報 96%）')
print('=' * 100)

# ── §3 我自己的零點有瑕疵：環狀位移會把發火移進/移出被排除的暖機窗 ──────────
print()
print('=' * 100)
print('§3 我自己零點的瑕疵檢查，以及修好之後的「每次廣播有用筆數」')
print('=' * 100)
print(f'  全程發火總數：實際 {int(fired.sum())}（環狀位移必然保持不變）')
print(f'  非暖機區發火數：實際 {int(fired[WARM_STEPS:].sum())}')
f2 = np.zeros_like(fired); rng2 = np.random.default_rng(7)
cs = [int(np.roll(fired[:, i], int(rng2.integers(200, MAXSTEP-200))).reshape(-1)[WARM_STEPS:].sum())
      for i in range(N)]
print(f'  ⇒ 全序列位移後非暖機區發火數會變（單次抽樣 {sum(cs)}）⇒ **分母不同、不可直接比總筆數**')

# 修好的零點：只在非暖機區內環狀位移 ⇒ 非暖機發火數逐節點完全守恆
rows2 = []
rng3 = np.random.default_rng(20261003)
for _ in range(20):
    f3 = fired.copy()
    for i in range(N):
        seg = f3[WARM_STEPS:, i]
        f3[WARM_STEPS:, i] = np.roll(seg, int(rng3.integers(100, len(seg) - 100)))
    _, _, dv3 = simulate(f3)
    c3, _, pb3 = classify(f3, dv3)
    t3 = sum(c3.values())
    rows2.append((c3['useful']/t3*100, c3['redundant']/t3*100, t3, pb3[:, 0].mean(), len(pb3)))
R2 = np.array(rows2)
print(f'\n  修好的零點（只在非暖機區內位移，發火數守恆）：')
print(f'    {"":<30}{"實際":>12}{"零點":>16}{"倍率":>9}')
print(f'    {"廣播次數":<30}{len(per_bc):>12}{R2[:,4].mean():>16.0f}{len(per_bc)/R2[:,4].mean():>8.3f}x')
print(f'    {"有用占比":<30}{cnt["useful"]/tot*100:>11.1f}%{R2[:,0].mean():>15.1f}%'
      f'   （差 {cnt["useful"]/tot*100 - R2[:,0].mean():+.1f}pp、{abs(cnt["useful"]/tot*100-R2[:,0].mean())/R2[:,0].std():.1f}σ）')
print(f'    {"總吸收筆數":<30}{tot:>12}{R2[:,2].mean():>16.0f}{tot/R2[:,2].mean():>8.2f}x')
print(f'    {"🚨 每次廣播的有用筆數":<30}{per_bc[:,0].mean():>12.3f}{R2[:,3].mean():>16.3f}'
      f'{per_bc[:,0].mean()/R2[:,3].mean():>8.2f}x   （零點 std {R2[:,3].std():.3f} ⇒ '
      f'{(per_bc[:,0].mean()-R2[:,3].mean())/R2[:,3].std():+.1f}σ）')
print()
print('  ⇒ 成本的單位是**廣播次數**（每次 46.1 MB，與夾帶幾個來源無關），不是「筆數」。')
print('    同樣次數的廣播下，現行時機送出的**有用轉傳比隨機時機多**。')
print('=' * 100)
