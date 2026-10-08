"""同畫風的三個節點，到底是「持有同一批圖」還是「各拿不重疊的 1/3」？（2026-10-01）

為什麼要查：`0904` 寫「同畫風三節點持有同一批圖」並被引用兩次，但 `util.py` 的
`partition_domain_dataset_for_nodes` 註解寫「split each class bucket evenly across nodes」，
兩邊相反。這會直接改變多處「獨立觀測數」的判讀 ⇒ 必須有決定性證據。

做法：不靠註解、不靠旗標宣告，直接拿 **checkpoint 的 `proto_count`** 對帳。
`update_prototypes_` 每看一個樣本就 +1 ⇒ `proto_count / epoch 數` ＝ 該節點每輪實際看到的樣本數。
再與 `np.array_split` 對該畫風各類別切出來的三份逐格比對。
"""
import os, sys, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAG = 'v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix'
RUN = os.path.join(ROOT, 'exp_result_' + TAG)
d = np.load(os.path.join(ROOT, 'logs/prototype_probe/0826_features_full.npz'))
STYLE = {'art_painting': [0,1,2], 'photo': [3,4,5], 'sketch': [6,7,8]}
EPOCH = 200

print('=' * 88)
print('§1 特徵傾印（0826 npz）裡，同畫風三節點拿到的是不是同一批圖？')
print('=' * 88)
for s, nodes in STYLE.items():
    hs = [np.bincount(d[f'n{i}_src_y'].astype(int), minlength=7).tolist() for i in nodes]
    print(f'  {s:<13} 各節點樣本數 {[len(d[f"n{i}_src_y"]) for i in nodes]}   標籤直方圖全同？ {hs[0]==hs[1]==hs[2]}   {hs[0]}')
print('  ⇒ 三個節點的長度與逐類別直方圖完全相同、且等於該畫風的**整個域** ⇒')
print('    **特徵傾印確實餵了整個域給每個節點**（探針的行為，與訓練無關）。')

print()
print('=' * 88)
print('§2 訓練時呢？用 proto_count 對帳（每看一個樣本 +1，所以 count/200 = 每輪看幾張）')
print('=' * 88)
ok_all = True
for s, nodes in STYLE.items():
    full = np.bincount(d[f'n{nodes[0]}_src_y'].astype(int), minlength=7)[:6]      # 該畫風各類別總數（去 person）
    shards = [np.array([len(x) for x in [np.array_split(np.arange(n), 3)[k] for n in full]]) for k in range(3)]
    print(f'  ── {s}（全域各類別 {full.tolist()}，person 已排除）')
    for k, i in enumerate(nodes):
        ck = torch.load(os.path.join(RUN, f'{TAG}_node_{i}_final.pth'), map_location='cpu', weights_only=False)
        cnt = ck['backbone_state']['proto_count'].sum(1).numpy()
        per_ep = cnt / EPOCH
        cyc = per_ep.sum() / shards[k].sum()                       # 每輪跑了幾趟自己那份
        match = np.allclose(per_ep / cyc, shards[k], rtol=2e-3)
        ok_all &= match
        print(f'     node{i}  每輪看 {per_ep.sum():7.1f} 張 ＝ 自己那份({shards[k].sum()})的 {cyc:.2f} 趟   '
              f'逐類別 {np.round(per_ep/cyc,1).tolist()} vs 切片 {shards[k].tolist()}   吻合={match}')
    tot = sum(sh.sum() for sh in shards)
    print(f'     三份加總 {tot} ＝ 該畫風全域({full.sum()}) ⇒ 不重疊、不遺漏：{tot == full.sum()}')
print()
print(f'  ⇒ 全部吻合：{ok_all}')
print('  ⇒ **訓練時三個同畫風節點各拿不重疊的 1/3（class_balanced）**，`0904` 的「持有同一批圖」是錯的。')
print('  ⇒ fb 對不起來的原因：每輪步數由**最長的那個 loader**（sketch）決定 ⇒ sketch 剛好跑 1.00 趟、')
print('    所以 count/200 正好等於 1/3；art 跑 2.23 趟、photo 2.87 趟，看起來就「不是 1/3」。')
print('=' * 88)
