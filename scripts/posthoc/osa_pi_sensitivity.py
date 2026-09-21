"""OSA 對混入比例 π 的敏感度與交叉點（dany 2026-09-10 要求整理 §4 表格時補）。

為什麼要算：§4.0 的主組合有一個取捨——我方**少誤殺已知圖**（誤拒率 −0.1847、4/4）但
**多放行未知圖**（放行率 +0.1674、0/4）。π 越大，放行未知的代價越重 ⇒ 必須知道邊界在哪。

關鍵性質：`OSA(π) = (1−π)·a_id + π·r_ood`，而 `a_id`／`r_ood` 只由門檻決定、**與 π 無關**
⇒ **OSA 對 π 是線性的** ⇒ 兩條線的交叉點可解析求出，不需要掃描：

    π* = (a_我 − a_對) / [(a_我 − a_對) − (r_我 − r_對)]

⚠️ 兩個對手要分開報，方向相反：
  對【StyleDDG+energy】：我方在低 π 領先，π 超過 π* 後被追上（因為對手的檢測器擋掉更多未知）
  對【StyleDDG 無檢測】：我方在低 π **落後**（沒有未知可擋、檢測只有誤殺成本），π 超過 π* 才划算

用法：./venv_matcha/bin/python scripts/posthoc/osa_pi_sensitivity.py
"""
import numpy as np

PACS = ["art_painting", "cartoon", "photo", "sketch"]
SH = {"art_painting": "art", "cartoon": "cartoon", "photo": "photo", "sketch": "sketch"}
# PACS 各 fold 的【實際】未知類別比例（person 張數 ÷ 全部）——與我們指定的 π 不同
ACTUAL = {"art_painting": 0.219, "cartoon": 0.173, "photo": 0.259, "sketch": 0.041}
N, Q = 9, 0.95
GRID = [0.0, 0.10, 0.20, 0.30, 0.50]


def operating_point(fold, tag, bn, ro, detect=True):
    """回傳 (a_id, r_ood)：兩者皆與 π 無關，π 只是加權。"""
    z = np.load(f"results/osa/{fold}.npz", allow_pickle=True)
    A, R = [], []
    for i in range(N):
        p = f"{tag}__{bn}__{i}"
        s = z[f"{p}__src_{ro}"].astype(np.float64)
        k = z[f"{p}__tgt_known_{ro}"].astype(np.float64)
        u = z[f"{p}__tgt_unk_{ro}"].astype(np.float64)
        c = z[f"{p}__tgt_known_correct"] > 0
        if detect:
            t = np.quantile(s, Q)
            A.append(float(((k <= t) & c).mean())); R.append(float((u > t).mean()))
        else:
            A.append(float(c.mean())); R.append(0.0)      # 不裝檢測器＝全部放行
    return float(np.mean(A)), float(np.mean(R))


def main():
    OURS = {f: operating_point(f, "ours", "avg", "zperp") for f in PACS}
    OPP = [("StyleDDG+energy／原樣（對外靶）", {f: operating_point(f, "baseline", "raw", "energy") for f in PACS}),
           ("StyleDDG 無檢測／原樣（地板）", {f: operating_point(f, "baseline", "raw", "energy", False) for f in PACS})]
    print("★ OSA(π) 對 π 線性 ⇒ 交叉點 π* 可解析求出（我方＝面讀出／平均B）")
    print("  π* ＝ (a_我−a_對) / [(a_我−a_對) − (r_我−r_對)]\n")
    for nm, B in OPP:
        print(f"  ── 對手：{nm} ──")
        print(f"    {'fold':<9}" + "".join(f"{'π='+str(int(p*100))+'%':>9}" for p in GRID)
              + f"{'交叉點π*':>10}{'實際π':>8}{'實際π下':>9}")
        acc = {p: [] for p in GRID}
        for f in PACS:
            ao, ro = OURS[f]; ab, rb = B[f]
            row = [100 * ((1 - p) * (ao - ab) + p * (ro - rb)) for p in GRID]
            for p, v in zip(GRID, row):
                acc[p].append(v)
            da, dr = ao - ab, ro - rb
            star = da / (da - dr) if (da - dr) != 0 else float("nan")
            ap = ACTUAL[f]
            at = 100 * ((1 - ap) * da + ap * dr)
            print(f"    {SH[f]:<9}" + "".join(f"{x:>+9.2f}" for x in row)
                  + f"{star*100:>9.1f}%{ap*100:>7.1f}%{at:>+9.2f}")
        ao = np.mean([OURS[f][0] for f in PACS]); ro = np.mean([OURS[f][1] for f in PACS])
        ab = np.mean([B[f][0] for f in PACS]); rb = np.mean([B[f][1] for f in PACS])
        da, dr = ao - ab, ro - rb
        print(f"    {'平均':<9}" + "".join(f"{np.mean(acc[p]):>+9.2f}" for p in GRID)
              + f"{da/(da-dr)*100:>9.1f}%")
        print()
    print("  ⚠️「實際π下」＝用該 fold 資料集真正的未知比例（非我們指定的 20%）算出的 Δ。")


if __name__ == "__main__":
    main()
