# -*- coding: utf-8 -*-
"""受鄰居影響的廣播：有多少是「有用的轉傳」、多少是「重複」？（2026-10-03，零 GPU）

背景：規則 0 的觸發率隨「自上次廣播以來收到幾次推播」4.44% → 74.24%（16.74×）。
      問題是那些被鄰居帶動的廣播，送出去的「吸收來的資訊」對收件人是新的（轉傳）還是舊的（重複）。

資料：broadcast_log.csv 的 fired ＋ 重建的拓撲（RGG 9 節點、半徑 0.8、seed 1234、23 邊）。
      已驗證：每步九節點【依序】處理；節點 i 在自己那一輪「先推、後收」；推＝寫進所有鄰居信箱
      （同寄件人覆蓋），收＝一次清空信箱。⇒ 編號比 i 小的鄰居同一步推的、比 i 大的鄰居上一步推的，
      會在 i 收信時進信箱。用這個模型重建 n_received 與 mean_staleness：36000 列 100% 吻合。

模型（資訊新鮮度）：每個節點 r 對每個來源 o 記一個「我手上關於 o 最新的資訊是第幾步」known[r][o]。
  - 節點自己永遠知道自己的最新狀態：known[i][i]＝當下步數。
  - 推送時訊息帶著寄件人當下整個 known 向量的快照。
  - 收件人消費訊息時，對每個來源 o：訊息裡的比自己的新 ⇒ 更新。
對每一次廣播（寄件人 s、第 t 步），「這段期間吸收進來的資訊」D_s ＝ 自上次廣播以來 known[s][o] 有前進的來源 o（o≠s）。
對每個收件人 r、每個 o∈D_s：
  - o＝r                           ⇒ 回聲（把 r 自己的東西送回去）
  - 訊息裡的 o 比 r 消費前手上的新   ⇒ 有用的轉傳
  - 否則                            ⇒ 重複（r 已經從別的路徑拿到同樣新或更新的）
⚠️ 限制：①把「資訊」當成有或沒有；實際 MH 平均是加權混合，兩跳轉傳的權重約為直接的 1/7（見末段）。
         ②「有用」只代表當下更新鮮，不代表那份資訊對收件人的模型影響大。
"""
import numpy as np, pandas as pd, networkx as nx
RUN = ('exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_'
       'osdg_excl_person_seed2026_topo1234_fix')
G = nx.random_geometric_graph(9, 0.8, seed=1234); assert nx.is_connected(G)
NB = {i: sorted(G[i]) for i in range(9)}; N = 9
b = pd.read_csv(f'{RUN}/async_diag/broadcast_log.csv'); b['i'] = b['domain'].str.replace('node_', '').astype(int)
F = b.pivot(index='step', columns='i', values='fired').sort_index().values.astype(bool)
R = b.pivot(index='step', columns='i', values='n_received').sort_index().values
T = F.shape[0]; WARM = 200   # epoch ≤ 10 ⇔ step < 200（每 epoch 20 步）

# ---------- 自檢：重建接收 ----------
pred = np.zeros_like(R)
for i in range(N):
    for j in NB[i]:
        if j < i: pred[:, i] += F[:, j]
        else:     pred[1:, i] += F[:-1, j]
print(f'自檢 1：重建 n_received 與紀錄逐列吻合 {(pred == R).mean()*100:.3f}%')

# ---------- 模擬 ----------
NEG = -10**9
known = np.full((N, N), NEG, dtype=np.int64)
prev_push = np.full((N, N), NEG, dtype=np.int64)     # 上次推送時的 known 快照
inbox = {i: {} for i in range(N)}
deliv = []        # 每筆：t, s, r, o, kind, gain, s_recv_since, warm
bcast = []        # 每次廣播：t, s, |D_s|, 自上次廣播以來收了幾封
recv_since = np.zeros(N, dtype=int)
own_new = own_tot = 0
for t in range(T):
    for i in range(N):
        known[i, i] = t
        if F[t, i]:
            snap = known[i].copy()
            D = [o for o in range(N) if o != i and snap[o] > prev_push[i, o]]
            bcast.append((t, i, len(D), recv_since[i]))
            for r in NB[i]:
                inbox[r][i] = (snap, D, t, recv_since[i])
            prev_push[i] = snap; recv_since[i] = 0
        for s in sorted(inbox[i]):
            snap, D, ts, rs = inbox[i][s]
            before = known[i].copy()
            own_tot += 1; own_new += int(snap[s] > before[s])
            for o in D:
                if o == i: kind = 'echo'
                elif snap[o] > before[o]: kind = 'relay'
                else: kind = 'dup'
                deliv.append((ts, s, i, o, kind, int(snap[o] - before[o]) if kind == 'relay' else 0,
                              rs, ts < WARM, int(o in NB[i])))
            known[i] = np.maximum(known[i], snap)
            recv_since[i] += 1
        inbox[i] = {}
print(f'自檢 2：寄件人自己的最新狀態對收件人一定是新的 ⇒ {own_new}/{own_tot}（{own_new/own_tot*100:.2f}%）')

d = pd.DataFrame(deliv, columns=['t', 's', 'r', 'o', 'kind', 'gain', 'rs', 'warm', 'o_adj_r'])
bc = pd.DataFrame(bcast, columns=['t', 's', 'nD', 'rs'])
print(f'廣播 {len(bc)} 次；吸收來的資訊被送到收件人共 {len(d)} 筆（來源×收件人）')

def share(x):
    n = len(x); c = x['kind'].value_counts()
    return n, c.get('relay', 0)/n*100, c.get('dup', 0)/n*100, c.get('echo', 0)/n*100

print('\n' + '=' * 92)
print('一、吸收來的資訊送出去之後：有用的轉傳 vs 重複 vs 回聲')
print('=' * 92)
print(f'{"範圍":<28}{"筆數":>8}{"有用的轉傳":>12}{"重複":>10}{"回聲":>10}')
for nm, x in [('全程', d), ('排除暖機（epoch ≥ 11）', d[~d.warm])]:
    n, a, c, e = share(x)
    print(f'{nm:<28}{n:>8}{a:>11.1f}%{c:>9.1f}%{e:>9.1f}%')

dn = d[~d.warm]
print('\n依「來源 o 跟收件人 r 的關係」拆（排除暖機）：')
for nm, x in [('o＝r（送回原主＝回聲）', dn[dn.o == dn.r]),
              ('o 與 r 直接相連（r 本來就收得到 o）', dn[(dn.o != dn.r) & (dn.o_adj_r == 1)]),
              ('o 與 r 不相連（r 只能靠轉傳）', dn[(dn.o != dn.r) & (dn.o_adj_r == 0)])]:
    n, a, c, e = share(x)
    print(f'  {nm:<34}{n:>7} 筆（占 {n/len(dn)*100:4.1f}%）   有用 {a:5.1f}%  重複 {c:5.1f}%  回聲 {e:5.1f}%')

print('\n' + '=' * 92)
print('二、跟 16.74 倍那條劑量反應接起來：廣播前收得越多，送出去的吸收資訊有多少是有用的？（排除暖機）')
print('=' * 92)
print(f'{"自上次廣播以來收到幾封":<22}{"廣播次數":>9}{"每次平均吸收來源數":>18}{"送出筆數":>10}{"有用":>8}{"重複":>8}{"回聲":>8}')
bn = bc[bc.t >= WARM]
for lo, hi, lab in [(0, 0, '0'), (1, 2, '1–2'), (3, 5, '3–5'), (6, 8, '6–8'), (9, 99, '9 以上')]:
    bb = bn[(bn.rs >= lo) & (bn.rs <= hi)]
    x = dn[(dn.rs >= lo) & (dn.rs <= hi)]
    if len(x) == 0:
        print(f'{lab:<22}{len(bb):>9}{bb.nD.mean() if len(bb) else 0:>18.2f}{0:>10}{"—":>8}{"—":>8}{"—":>8}'); continue
    n, a, c, e = share(x)
    print(f'{lab:<22}{len(bb):>9}{bb.nD.mean():>18.2f}{n:>10}{a:>7.1f}%{c:>7.1f}%{e:>7.1f}%')

print('\n' + '=' * 92)
print('三、有用的轉傳「有多有用」：讓收件人的資訊新了幾步（排除暖機）')
print('=' * 92)
g = dn[dn.kind == 'relay'].gain
print(f'  新鮮度提升（步）：中位 {g.median():.0f}、平均 {g.mean():.1f}、p25 {g.quantile(.25):.0f}、p75 {g.quantile(.75):.0f}；'
      f'只新 1 步的占 {(g == 1).mean()*100:.1f}%')
# 「反正很快也會從別處拿到」：同一收件人、同一來源，H 步內另一個寄件人送來同樣新或更新的資訊
print('\n  ⚠️ 有用的轉傳裡，有多少「反正很快也會從別人那裡拿到」？（同收件人、同來源、H 步內別的寄件人送到同樣新的）')
rel = dn[dn.kind == 'relay'][['t', 's', 'r', 'o']].values
# 以 (r, o) 建立「誰在何時送來關於 o、新鮮度多少」的清單（含重複與有用，凡帶到 o 的都算）
allmsg = {}
for (t, s, r, o, kind, gain, rs, warm, adj) in deliv:
    allmsg.setdefault((r, o), []).append((t, s))
for H in (1, 3, 10):
    soon = 0
    for t, s, r, o in rel:
        if any(s2 != s and t < t2 <= t + H for (t2, s2) in allmsg.get((r, o), [])): soon += 1
    print(f'    H＝{H:>2} 步：{soon/len(rel)*100:5.1f}%')

print('\n' + '=' * 92)
print('四、拓撲本身的上限：每條邊上，寄件人的其他鄰居有幾成不是收件人的鄰居（＝結構上可能轉傳的比例）')
print('=' * 92)
br = []
for s in range(N):
    for r in NB[s]:
        oth = [o for o in NB[s] if o != r]
        if oth: br.append(np.mean([o not in NB[r] for o in oth]))
print(f'  有向邊 {len(br)} 條；平均 {np.mean(br)*100:.1f}%（範圍 {min(br)*100:.0f}–{max(br)*100:.0f}%）')
print(f'  三角形數 {sum(nx.triangles(G).values())//3}、平均群聚係數 {nx.average_clustering(G):.3f}')

print('\n' + '=' * 92)
print('五、權重提醒：MH 邊權 w＝1/(1+max(度數))；直接 vs 兩跳轉傳的權重量級')
print('=' * 92)
w = [1/(1+max(len(NB[a]), len(NB[c]))) for a in range(N) for c in NB[a]]
print(f'  直接一跳的邊權：平均 {np.mean(w):.3f}（範圍 {min(w):.3f}–{max(w):.3f}）⇒ 兩跳約 {np.mean(w)**2:.4f}（約為一跳的 {np.mean(w):.2f} 倍）')

print('\n' + '=' * 92)
print('六、逐節點（排除暖機）')
print('=' * 92)
print(f'{"寄件人":<8}{"度數":>5}{"送出筆數":>9}{"有用":>8}{"重複":>8}{"回聲":>8}')
for s in range(N):
    x = dn[dn.s == s]; n, a, c, e = share(x)
    print(f'node_{s:<3}{len(NB[s]):>5}{n:>9}{a:>7.1f}%{c:>7.1f}%{e:>7.1f}%')

# ============================================================================
# 七、零點：發火時機若跟「有沒有收到推播」無關，結果會怎樣？
#     2026-10-03 依 matcha-02 複核修正：原本把整條 4000 步序列環狀位移，發火會被移進／移出暖機窗
#     ⇒ 非暖機廣播次數不守恆（5544–5807 vs 實際 5355），「廣播次數相同」不成立。
#     改為【只在非暖機區內】各節點各自環狀位移 ⇒ 逐節點非暖機發火數完全守恆。
#     🚨 成本單位：一次廣播送一個完整模型（46.1 MB），與它夾帶幾個來源的資訊無關
#        ⇒ 成本相關的效率指標是「每次廣播送出幾筆有用轉傳」，不是「有用占比」。
# ============================================================================
def simulate(Fm):
    known = np.full((N, N), NEG, dtype=np.int64); prev = np.full((N, N), NEG, dtype=np.int64)
    box = {i: {} for i in range(N)}; cnt = {'relay': 0, 'dup': 0, 'echo': 0}; nb = 0
    for t in range(T):
        for i in range(N):
            known[i, i] = t
            if Fm[t, i]:
                snap = known[i].copy()
                D = [o for o in range(N) if o != i and snap[o] > prev[i, o]]
                for r in NB[i]: box[r][i] = (snap, D, t)
                prev[i] = snap
                if t >= WARM: nb += 1
            for s in sorted(box[i]):
                snap, D, ts = box[i][s]
                if ts >= WARM:
                    for o in D:
                        cnt['echo' if o == i else ('relay' if snap[o] > known[i, o] else 'dup')] += 1
                known[i] = np.maximum(known[i], snap)
            box[i] = {}
    n = sum(cnt.values())
    return nb, cnt['relay']/n*100, cnt['dup']/n*100, cnt['echo']/n*100, n, cnt['relay']/nb

print('\n' + '=' * 92)
print('七、零點：發火時機與收到推播無關時（只在非暖機區內各節點環狀位移，20 次）')
print('=' * 92)
real = simulate(F)
rng = np.random.default_rng(2026); nulls = []
for _ in range(20):
    Fs = F.copy()
    for i in range(N): Fs[WARM:, i] = np.roll(F[WARM:, i], int(rng.integers(100, T - WARM - 100)))
    nulls.append(simulate(Fs))
nu = np.array(nulls)
print(f'{"":<24}{"廣播次數":>9}{"有用":>12}{"重複":>12}{"回聲":>12}{"送出筆數":>10}{"每次廣播有用筆數":>16}')
print(f'{"實際（規則 0 的時機）":<24}{real[0]:>9}{real[1]:>11.1f}%{real[2]:>11.1f}%{real[3]:>11.1f}%{real[4]:>10}{real[5]:>16.3f}')
print(f'{"零點（時機打亂）":<24}{nu[:,0].mean():>9.0f}{nu[:,1].mean():>7.1f}±{nu[:,1].std():.1f}%{nu[:,2].mean():>7.1f}±{nu[:,2].std():.1f}%'
      f'{nu[:,3].mean():>7.1f}±{nu[:,3].std():.1f}%{nu[:,4].mean():>10.0f}{nu[:,5].mean():>10.3f}±{nu[:,5].std():.3f}')
print(f'  廣播次數守恆：{int(nu[:,0].min())}–{int(nu[:,0].max())}（實際 {real[0]}）')
print(f'  有用占比：實際比零點 {real[1]-nu[:,1].mean():+.2f} 個百分點（{(real[1]-nu[:,1].mean())/nu[:,1].std():+.1f}σ）')
print(f'  ★ 每次廣播有用筆數：實際/零點 ＝ {real[5]/nu[:,5].mean():.2f}×（{(real[5]-nu[:,5].mean())/nu[:,5].std():+.1f}σ）')
print('  ⇒ 讀法 1：重複率本身主要由拓撲造成（實際與零點都約九成）。')
print('  ⇒ 讀法 2（修正）：現行時機的有用占比較低，但每次廣播夾帶較多吸收資訊，每次廣播送出的有用轉傳反而較多；')
print('     成本按廣播次數計 ⇒ 在成本相關的指標上，現行時機優於隨機時機。')
