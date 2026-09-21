"""post-hoc probe 共用的 CSV 寫出（append + header 一致性檢查）。

⚠️ 為什麼需要這個檢查（2026-08-14 踩過）：
`csv.DictWriter` 在 append 模式下**不會**去看既有檔案的 header，只按傳入的 fieldnames
順序寫。欄位集合一改（例如本次新增 `ckpt_epoch`），append 到舊檔就會**靜默錯位**——
數字全部落在錯誤的欄名底下，而且不報錯。

同一天踩到的另一半：CSV 原本沒有 `ckpt_epoch` 欄，於是同一個 run 的 ep150 與 final
兩批結果寫進去以後**長得一模一樣、分不出來**。1a-fix 光是 ep150→ep200 誤拒率就自己
漂了 +0.036（比當時要量的效應還大）⇒ 拿錯 epoch 對照會得到相反的結論。
"""
import os
import csv


def write_csv(path, fieldnames, rows):
    """把 rows append 進 path；若既有檔案的 header 不同則中止，不靜默錯位。"""
    fieldnames = list(fieldnames)
    write_header = True
    if os.path.exists(path):
        with open(path, newline="") as f:
            old = next(csv.reader(f), [])
        if old:
            if old != fieldnames:
                raise SystemExit(
                    f"[abort] 既有 CSV 的欄位與本次不符，append 會靜默錯位。\n"
                    f"  檔案：{path}\n"
                    f"  既有：{old}\n"
                    f"  本次：{fieldnames}\n"
                    f"  ⇒ 換一個 --output_csv 檔名（建議把 ckpt_epoch 寫進檔名），"
                    f"或確認舊檔可棄後刪除重跑。")
            write_header = False

    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"\nAppended {len(rows)} rows to {path}")
