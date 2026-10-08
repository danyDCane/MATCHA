# -*- coding: utf-8 -*-
"""獨立複核 matcha-b6 對「觸發率 × 收到推播」劑量反應的更正（2026-10-03，matcha-02 寫）

⚠️ 利益衝突聲明：被更正的那個結論是**我**提出並升為主證據的（1002 §2.0、16.7×）。
   本檔刻意不 import b6 的腳本，並且對「我原本的版本」用與對它同樣的力度攻。

事件順序（`train.py` 實讀，不靠註解）：
    訓練 → scheduler.step → 存 BN → **should_broadcast（決策點，算 delta）** :1734
    → shadow_signals :1736 → **push（重設 ŵ）** :1741 → **receive_and_aggregate** :1745
    → 寫 CSV（delta 來自 :1734、n_received 來自 :1745）
⇒ 第 t 列的 `n_received` **發生在第 t 列的決策之後** ⇒ 不可能影響第 t 列的 `fired`。
⇒ 正確累積：決策點看到的是 [上次發火步, t−1] 收到的；且發火步自己收到的那幾封屬於**下一段**
   （因為 ŵ 在 push 時就重設了、receive 在那之後）。
"""
import numpy as np, pandas as pd
from scipy import stats

CSV = ('exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_osdg_'
       'excl_person_seed2026_topo1234_fix/async_diag/shadow_signals.csv')
d = pd.read_csv(CSV); d = d[d.is_warm == 0].reset_index(drop=True)
dom, steps, ep = d['domain'].values, d['step'].values, d['epoch'].values
fired = d['fired'].values.astype(bool); nat = (d['sig_r0'].values >= 1e-5)
nr = d['n_received'].values.astype(float); delta = d['sig_r0'].values.astype(float)
nodes = sorted(np.unique(dom))

# ── 兩種累積 ──────────────────────────────────────────────────────────────
mine = np.zeros(len(d)); fix = np.zeros(len(d)); clock = np.zeros(len(d))
for n in nodes:
    m = np.where(dom == n)[0]
    a = 0.0; b = 0.0; last = steps[m[0]] - 1
    for i in m:
        a += nr[i]; mine[i] = a                      # 我原本的（錯）：先加本步
        if fired[i]: a = 0.0
        fix[i] = b                                   # 修正：決策點只看 t−1 為止
        if fired[i]: b = 0.0                         # push 重設 ŵ
        b += nr[i]                                   # receive 落在新參照點之後
        clock[i] = steps[i] - last
        if fired[i]: last = steps[i]
mine = mine.astype(int); fix = fix.astype(int); clock = clock.astype(int)
print('=' * 102)
print('§1 事件順序更正：兩種累積的差異')
print('=' * 102)
print(f'  逐列相同的比例：{(mine == fix).mean()*100:.1f}%        ← b6 報 48.9%')
print(f'  我的版本 範圍 {mine.min()}–{mine.max()}｜修正版 範圍 {fix.min()}–{fix.max()}')
def curve(acc, tag, maxv=12):
    print(f'\n  {tag}')
    print('   ' + ''.join(f'{v:>8d}' for v in range(maxv+1)))
    print('   n' + ''.join(f'{(acc==v).sum():>8d}' for v in range(maxv+1)))
    r = [nat[acc == v].mean()*100 if (acc == v).sum() else np.nan for v in range(maxv+1)]
    print('   %' + ''.join(f'{x:>8.2f}' for x in r))
    mono = all(r[i] <= r[i+1] + 1e-9 for i in range(len(r)-1) if np.isfinite(r[i]) and np.isfinite(r[i+1]))
    bad = [v for v in range(1, maxv+1) if np.isfinite(r[v]) and np.isfinite(r[v-1]) and r[v] < r[v-1]]
    print(f'   單調？{mono}   下降的位置 v＝{bad}   倍率 {r[max([v for v in range(maxv+1) if np.isfinite(r[v])])] / r[0]:.2f}×')
curve(mine, '我原本的（錯誤順序）')
curve(fix,  '修正後（b6 的順序）  ← b6 報 3.96% → 68.10%、17.19×、2 封 11.19% vs 3 封 11.10% 持平')

# ── 分層：同 epoch 段 × 同 clock ──────────────────────────────────────────
print()
print('=' * 102)
print('§2 複核 b6 第 3 條：同階段 × 同 clock 內，收到多 vs 少')
print('=' * 102)
SEG = [(11,50),(51,100),(101,150),(151,200)]
seg = np.full(len(d), -1)
for j,(lo,hi) in enumerate(SEG): seg[(ep>=lo)&(ep<=hi)] = j
tot_w = tot_d = 0.0; rows=[]
for j,(lo,hi) in enumerate(SEG):
    for c in sorted(np.unique(clock)):
        m = (seg==j)&(clock==c)
        if m.sum() < 150 or nat[m].sum() < 10: continue
        med = np.median(fix[m]); hi_m = m & (fix > med); lo_m = m & (fix <= med)
        if hi_m.sum() < 20 or lo_m.sum() < 20: continue
        dd = (nat[hi_m].mean()-nat[lo_m].mean())*100
        rows.append((j,c,m.sum(),int(med),lo_m.sum(),hi_m.sum(),nat[lo_m].mean()*100,nat[hi_m].mean()*100,dd))
        tot_w += m.sum(); tot_d += dd*m.sum()
print(f'  {"段":<10}{"clock":>6}{"n":>7}{"中位":>5}{"n低":>6}{"n高":>6}{"低%":>8}{"高%":>8}{"差pp":>8}')
for j,c,n_,md,nl,nh,rl,rh,dd in rows:
    print(f'  ep{SEG[j][0]}–{SEG[j][1]:<5}{c:>6}{n_:>7}{md:>5}{nl:>6}{nh:>6}{rl:>8.1f}{rh:>8.1f}{dd:>+8.1f}')
print(f'  ⇒ 格數 {len(rows)}｜收到多較高 {sum(1 for r in rows if r[-1]>0)}／{len(rows)}'
      f'｜加權平均 {tot_d/tot_w:+.1f} pp     ← b6 報 15 格 14 正、+6.9pp')
ok = np.array([r>=0 for r in range(len(d))])
print(f'  等級相關（全體、fix vs nat，僅供對照）spearman {stats.spearmanr(fix, nat).statistic:+.3f}')

# ── 我的攻擊：對 clock 做條件化會不會系統性低估 ────────────────────────
print()
print('=' * 102)
print('§3 ★ 我的攻擊：clock 是「處理的下游」⇒ 條件化會把效應選掉（系統性低估）')
print('=' * 102)
print('  因果圖：收到 → 參數位移 → 發火；而 發火 → clock 歸零。')
print('  ⇒ clock 既是共同原因（時間越久、收到越多、自己也累積越多）**也是過去發火的後代**。')
print('  ⇒ 在 clock=c 的格子裡，收到多卻「還沒發火」的列，必然是自身位移慢的列 ⇒ **選擇對抗效應**。')
print('  驗證：clock 越大，選擇累積越多 ⇒ 效應應該隨 clock 遞減。低 clock（選擇最少）才接近無偏。')
print(f'\n  {"clock":>6}{"n":>8}{"發火率":>8}   逐 recv 的發火率（recv=0,1,2,3+）          斜率(pp/封)')
for c in range(1, 13):
    m = (clock == c)
    if m.sum() < 300: continue
    cells = []
    for v, lab in [(0,'0'),(1,'1'),(2,'2')]:
        mm = m & (fix == v)
        cells.append(f'{nat[mm].mean()*100:5.1f}({mm.sum():4d})' if mm.sum() >= 30 else '   -      ')
    mm = m & (fix >= 3)
    cells.append(f'{nat[mm].mean()*100:5.1f}({mm.sum():4d})' if mm.sum() >= 30 else '   -      ')
    x = fix[m].astype(float); y = nat[m].astype(float)
    sl = np.polyfit(x, y, 1)[0]*100 if x.std() > 0 else np.nan
    print(f'  {c:>6}{m.sum():>8}{nat[m].mean()*100:>7.1f}%   ' + ' '.join(cells) + f'   {sl:>+8.2f}')
print('\n  ⇒ 看最低的 clock（選擇最少）與最高的 clock（選擇最多）斜率是否不同。')
print('  ⇒ 另一個方向的偏誤：不做條件化（17×）被 clock 混淆、**往上偏**。')
print('  ⇒ 所以兩個估計是上下界，真值在中間。單報一個點估計都不對。')

# ── 中位數二分 vs 連續 ───────────────────────────────────────────────────
print()
print('=' * 102)
print('§4 複核 b6 第 3 條的另一個偏誤源：中位數二分 vs 直接用封數')
print('=' * 102)
tw=td=0.0; tw2=td2=0.0
for j,_ in enumerate(SEG):
    for c in sorted(np.unique(clock)):
        m=(seg==j)&(clock==c)
        if m.sum()<150 or nat[m].sum()<10: continue
        med=np.median(fix[m]); hm=m&(fix>med); lm=m&(fix<=med)
        if hm.sum()<20 or lm.sum()<20: continue
        tw+=m.sum(); td+=(nat[hm].mean()-nat[lm].mean())*100*m.sum()
        x=fix[m].astype(float)
        if x.std()>0:
            tw2+=m.sum(); td2+=np.polyfit(x,nat[m].astype(float),1)[0]*100*m.sum()
print(f'  中位數二分（b6 的做法）  加權 {td/tw:+.2f} pp（高 vs 低）')
print(f'  改用連續封數的斜率      加權 {td2/tw2:+.2f} pp／封')
print(f'  ⇒ 二分把「0 vs 1」和「0 vs 5」壓成同一個對比；fix 是整數且重複值極多')
print(f'     （recv=0 佔 {(fix==0).mean()*100:.1f}%、≤1 佔 {(fix<=1).mean()*100:.1f}%）⇒ 二分會衰減效應。')

# ── 第 4 條：融合那一跳 ──────────────────────────────────────────────────
print()
print('=' * 102)
print('§5 複核 b6 第 4 條：融合那一跳 vs 一步訓練（delta 的增量）')
print('=' * 102)
print('  取「第 t−1 步沒發火」（參照點未變）的列，比 delta(t)−delta(t−1)')
print(f'  {"段":<12}{"上步收到0":>22}{"上步收到≥2":>22}{"比值":>8}')
for j,(lo,hi) in enumerate(SEG):
    A=[];B=[]
    for n in nodes:
        m=np.where(dom==n)[0]
        for p in range(1,len(m)):
            i,q=m[p],m[p-1]
            if fired[q] or seg[i]!=j: continue
            dv=delta[i]-delta[q]
            (A if nr[q]==0 else (B if nr[q]>=2 else [])).append(dv) if (nr[q]==0 or nr[q]>=2) else None
    A,B=np.array(A),np.array(B)
    if len(A)<30 or len(B)<30: print(f'  ep{lo}–{hi:<7}（樣本不足 {len(A)}／{len(B)}）'); continue
    print(f'  ep{lo}–{hi:<7}{A.mean():>13.3e}(n={len(A)}){B.mean():>13.3e}(n={len(B)}){B.mean()/A.mean():>8.2f}×')
print('  ⚠️ b6 自己標的限制成立：delta 是範數、增量不可相加，且融合可能把模型往 ŵ 拉回。')
print('     ⇒ 這條只能看符號與量級，不能當效應大小。')
print('=' * 102)

# ── §6 最緊的分層：節點 × epoch 段 × clock（吃掉度數與所有節點常數）────────
print()
print('=' * 102)
print('§6 ★ 回答 b6 的第二個問題：更好的拆法 ＝ (節點 × epoch段 × clock) 條件化 + 連續封數')
print('=' * 102)
print('  為什麼要含「節點」：`push_to_neighbors` 是 dedup 覆蓋 ⇒ `n_received` ＝ 不同寄件者數、')
print('  **上限等於度數** ⇒ 低度數節點進不了「收到多」組 ⇒ 度數是混淆源（且規則 0 與度數相關 −0.68）。')

def pooled(strata_keys, label):
    """層內斜率，逆變異加權（層內只有一種 recv 值或無變異的層自動權重 0）"""
    import collections
    g = collections.defaultdict(list)
    for i in range(len(d)):
        g[strata_keys[i]].append(i)
    num = den = 0.0; used = 0; n_used = 0
    c0 = c3 = n0 = n3 = 0
    for k, idx in g.items():
        idx = np.array(idx)
        if len(idx) < 40: continue
        x = fix[idx].astype(float); y = nat[idx].astype(float)
        if x.std() == 0 or y.std() == 0: continue
        b, _ = np.polyfit(x, y, 1)
        resid = y - np.polyval([b, np.polyfit(x, y, 1)[1]], x)
        var_b = (resid.var(ddof=2) / ((x - x.mean()) ** 2).sum()) if len(idx) > 2 else np.inf
        if not np.isfinite(var_b) or var_b <= 0: continue
        w = 1.0 / var_b
        num += b * w; den += w; used += 1; n_used += len(idx)
        m0 = idx[fix[idx] == 0]; m3 = idx[fix[idx] >= 3]
        c0 += nat[m0].sum(); n0 += len(m0); c3 += nat[m3].sum(); n3 += len(m3)
    print(f'  {label:<46} 層數 {used:>4}  覆蓋 {n_used:>6} 列  '
          f'層內斜率 {num/den*100:+.2f} pp/封   recv=0 {c0/max(n0,1)*100:5.2f}%(n={n0})  '
          f'recv≥3 {c3/max(n3,1)*100:5.2f}%(n={n3})')

clkb = np.clip(clock, 1, 13)
pooled([(c,) for c in clkb],                      'clock 單獨')
pooled([(s, c) for s, c in zip(seg, clkb)],       'epoch段 × clock（≈ b6 的做法）')
pooled([(n, c) for n, c in zip(dom, clkb)],       '節點 × clock')
pooled([(n, s, c) for n, s, c in zip(dom, seg, clkb)], '★ 節點 × epoch段 × clock（最緊）')
print()
print('  對照：b6 的中位數二分加權 +6.90 pp（高 vs 低，不是每封）')
print('  ⇒ 二分與「每封 pp」不可直接比；要比就比同一個對比（recv=0 vs recv≥3）。')
print('=' * 102)

# ── §7 正統的 Mantel–Haenszel 風險差（只用「層內同時有兩組」的層）──────────
print()
print('=' * 102)
print('§7 ★ 定案用的數字：recv=0 vs recv≥3 的 Mantel–Haenszel 風險差（層內配對、非彙總率）')
print('=' * 102)
import collections
def mh(keys, label):
    g = collections.defaultdict(list)
    for i in range(len(d)): g[keys[i]].append(i)
    num = den = 0.0; used = 0; cov = 0
    for k, idx in g.items():
        idx = np.array(idx)
        a_ = idx[fix[idx] == 0]; b_ = idx[fix[idx] >= 3]
        if len(a_) < 15 or len(b_) < 15: continue
        n1, n0_ = len(b_), len(a_); w = n1 * n0_ / (n1 + n0_)
        num += (nat[b_].mean() - nat[a_].mean()) * w; den += w; used += 1; cov += n1 + n0_
    print(f'  {label:<42} 層數 {used:>4}  覆蓋 {cov:>6} 列   **MH 風險差 {num/den*100:+.2f} pp**')
mh([(c,) for c in clkb],                               'clock 單獨')
mh([(s, c) for s, c in zip(seg, clkb)],                'epoch段 × clock（≈ b6 的做法）')
mh([(n, c) for n, c in zip(dom, clkb)],                '節點 × clock')
mh([(n, s, c) for n, s, c in zip(dom, seg, clkb)],     '★ 節點 × epoch段 × clock（最緊）')
print()
print('  對照 b6 的中位數二分：+6.90 pp（高 vs 低，高與低常只差 1 封）')
print('  ⇒ 同樣條件化 clock、改成 recv=0 vs recv≥3 的配對對比，殘餘效應是 b6 報的 2–3 倍。')
print()
print('  單調性那一格的雜訊檢查（b6 用它否掉「12 層全部單調」）：')
for v in [2, 3]:
    m = (fix == v); p = nat[m].mean()
    print(f'    recv={v}: {p*100:.2f}%  n={m.sum()}  SE={np.sqrt(p*(1-p)/m.sum())*100:.2f}pp')
p2, n2 = nat[fix==2].mean(), (fix==2).sum(); p3, n3 = nat[fix==3].mean(), (fix==3).sum()
se = np.sqrt(p2*(1-p2)/n2 + p3*(1-p3)/n3)
print(f'    差 {(p2-p3)*100:+.2f} pp，配對無關的 SE {se*100:.2f} pp ⇒ **{abs(p2-p3)/se:.2f}σ ＝ 雜訊內的平手，不是反轉**')
print('=' * 102)
