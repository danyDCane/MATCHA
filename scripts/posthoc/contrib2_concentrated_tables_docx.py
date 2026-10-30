"""第二貢獻：畫風集中慢詳細表格 → .docx（純標準函式庫組 WordprocessingML，不需安裝套件）。
數字照抄 research/1026_contrib2_cost_and_framing.md §2（來源 1021 主表＋*_source_clean_acc.csv）。
dany 1027 版式：只列理想、EF-HC 雙向、EF-HC 單向、PushCen、按群校正（我們）；拿掉傳輸欄；泛化括號改成相對理想的百分比，其餘仍為差值。"""
import sys, zipfile
from xml.sax.saxutils import escape

OUT = sys.argv[1]
FONT = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="標楷體" w:cs="Times New Roman"/>'
COLS = [2900, 1900, 2100, 1700, 1700, 1800, 2738]          # 合計 14838 = 橫式 A4 去掉左右邊界
HEAD = ['方法', '泛化', '面檢測', 'art 原圖', 'photo 原圖', 'sketch 原圖', '掉最多的畫風']
ORDER = ['全員同速（理想）', 'EF-HC 雙向', 'EF-HC 單向', 'PushCen', '按群校正（我們）']
IDEAL_ACC, NOISE_ACC = 80.09, 1.09
B = '**'   # 前綴 ** ＝粗體（比理想差超過雜訊寬度）

SCEN = [
    ('表一　sketch 集中慢', '慢設備：sketch 三台（n6、n7、n8）＋ n4（基準網路原本就慢）', 'sketch', [
        ('全員同速（理想）', '6,440＊', '80.09', '0.8325', '99.06', '99.27', '93.47', '—'),
        ('按群校正（我們）', '1,600（−75%）', '80.41（+0.32）', '0.8372（+0.0047）', '98.94（−0.12）', '98.95（−0.32）', '93.59（+0.12）', 'photo 98.95（−0.32）'),
        ('照網速送（不校正）', '1,600（−75%）', B + '77.41（−2.68）', B + '0.8001（−0.0324）', '99.75（+0.69）', '99.92（+0.65）', B + '80.62（−12.85）', B + 'sketch 80.62（−12.85）'),
        ('EF-HC 單向（多存 46 MB）', '1,729（−73%）', '79.16（−0.93）', B + '0.8125（−0.0200）', '99.81（+0.75）', '99.43（+0.16）', B + '81.16（−12.31）', B + 'sketch 81.16（−12.31）'),
        ('EF-HC 雙向（多存 46 MB）', '8,048（+25%）', B + '78.70（−1.39）', '0.8266（−0.0059）', '99.62（+0.56）', '99.76（+0.49）', '92.36（−1.11）', 'sketch 92.36（−1.11）'),
        ('PushCen', '1,600（−75%）', '80.09（+0.00）', B + '0.8075（−0.0250）', B + '90.81（−8.25）', B + '95.80（−3.47）', '98.81（+5.34）', B + 'art 90.81（−8.25）'),
        ('同週期送（同傳輸時間）', '3,020（−53%）', '79.94（−0.15）', '0.8235（−0.0090）', '99.44（+0.38）', '99.60（+0.33）', '94.24（+0.77）', 'photo 99.60（+0.33）'),
        ('只看自己（消融）', '1,600（−75%）', '80.06（−0.03）', '0.8271（−0.0054）', '99.26（+0.20）', '99.76（+0.49）', '93.48（+0.01）', 'sketch 93.48（+0.01）'),
    ]),
    ('表二　photo 集中慢', '慢設備：photo 三台（n3、n4、n5；n4 在基準網路原本就慢）', 'photo', [
        ('全員同速（理想）', '6,440＊', '80.09', '0.8325', '99.06', '99.27', '93.47', '—'),
        ('按群校正（我們）', '1,250（−81%）', '79.16（−0.93）', '0.8197（−0.0128）', '99.06（+0.00）', '99.76（+0.49）', '93.52（+0.05）', 'art 99.06（+0.00）'),
        ('照網速送（不校正）', '1,250（−81%）', '81.43（+1.34）', B + '0.8147（−0.0178）', '98.81（−0.25）', B + '97.01（−2.26）', '96.07（+2.60）', B + 'photo 97.01（−2.26）'),
        ('EF-HC 單向（多存 46 MB）', '1,454（−77%）', '80.56（+0.47）', '0.8216（−0.0109）', '99.62（+0.56）', B + '95.64（−3.63）', '95.94（+2.47）', B + 'photo 95.64（−3.63）'),
        ('EF-HC 雙向（多存 46 MB）', '11,163（+73%）', '79.27（−0.82）', '0.8234（−0.0091）', '99.44（+0.38）', '99.68（+0.41）', '92.54（−0.93）', 'sketch 92.54（−0.93）'),
        ('PushCen', '1,250（−81%）', B + '76.53（−3.56）', B + '0.8010（−0.0314）', B + '96.81（−2.25）', '99.92（+0.65）', B + '89.92（−3.55）', B + 'sketch 89.92（−3.55）'),
        ('同週期送（同傳輸時間）', '2,760（−57%）', '79.68（−0.41）', '0.8247（−0.0078）', '99.50（+0.44）', '99.35（+0.08）', '93.50（+0.03）', 'sketch 93.50（+0.03）'),
    ]),
    ('表三　art 集中慢', '慢設備：art 三台（n0、n1、n2）＋ n4（基準網路原本就慢）', 'art', [
        ('全員同速（理想）', '6,440＊', '80.09', '0.8325', '99.06', '99.27', '93.47', '—'),
        ('按群校正（我們）', '1,233（−81%）', '80.29（+0.20）', B + '0.8135（−0.0190）', '99.30（+0.24）', '99.03（−0.24）', '93.22（−0.25）', 'sketch 93.22（−0.25）'),
        ('照網速送（不校正）', '1,233（−81%）', '80.30（+0.21）', '0.8183（−0.0141）', B + '93.68（−5.38）', '99.19（−0.08）', '97.03（+3.56）', B + 'art 93.68（−5.38）'),
        ('EF-HC 單向（多存 46 MB）', '1,153（−82%）', '79.16（−0.93）', B + '0.8136（−0.0188）', B + '91.87（−7.19）', '98.55（−0.72）', '97.42（+3.95）', B + 'art 91.87（−7.19）'),
        ('EF-HC 雙向（多存 46 MB）', '13,225（+105%）', B + '78.49（−1.60）', '0.8228（−0.0097）', '99.56（+0.50）', '99.84（+0.57）', '92.22（−1.25）', 'sketch 92.22（−1.25）'),
        ('PushCen', '1,233（−81%）', B + '78.91（−1.18）', B + '0.8150（−0.0175）', '99.75（+0.69）', '99.11（−0.16）', B + '86.97（−6.50）', B + 'sketch 86.97（−6.50）'),
        ('同週期送（同傳輸時間）', '2,340（−64%）', '79.93（−0.16）', '0.8253（−0.0072）', '99.37（+0.31）', '99.43（+0.16）', '93.44（−0.03）', 'sketch 93.44（−0.03）'),
    ]),
]
NOTES = [
    '註 1　基準（理想）＝全員同速：九台網速相同、照同一週期送。',
    '註 2　泛化＝目標畫風 cartoon 的分類準確率（%）；面檢測＝未知類別檢測 AUROC；art／photo／sketch 原圖＝各來源畫風原圖的分類準確率（%）。皆為第 200 輪、九台平均。',
    '註 3　泛化括號＝相對理想的百分比變化＝（方法 − 理想）÷ 理想 × 100%，理想為 80.09；面檢測與原圖括號＝減去理想的差。',
    '註 4　粗體＝比理想差超過雜訊寬度：泛化 1.09 分（約 1.36%）、面檢測 0.0144、原圖 1.54 分；取自只差送出方式的四組對照的範圍。',
    '註 5　掉最多的畫風＝三個來源畫風中比理想掉最多（或升最少）的那個：是慢畫風自己＝被沖淡；是別的畫風＝被轉嫁。',
    '註 6　網速表：基準網路裡每個畫風各有快、中、慢一台設備（photo 的 n4 為 0.1）；集中慢＝把該畫風三台都改成 0.1，其他設備不動。',
    '註 7　範圍：以 cartoon 為目標畫風的一折、單一隨機起點。',
]


def run(text, bold=False, size=20):
    t = escape(text)
    return (f'<w:r><w:rPr>{FONT}{"<w:b/><w:bCs/>" if bold else ""}<w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr>'
            f'<w:t xml:space="preserve">{t}</w:t></w:r>')


def para(text, bold=False, size=20, align=None, before=0, after=60, keep=False, newpage=False):
    # 順序依 schema：keepNext → pageBreakBefore → spacing → jc
    pp = ('<w:keepNext/>' if keep else '') + ('<w:pageBreakBefore/>' if newpage else '') + f'<w:spacing w:before="{before}" w:after="{after}"/>' + (f'<w:jc w:val="{align}"/>' if align else '')
    return f'<w:p><w:pPr>{pp}</w:pPr>{run(text, bold, size)}</w:p>'


def cell(text, w, fill=None, bold=False, align='center'):
    b = bold or text.startswith(B)
    text = text[len(B):] if text.startswith(B) else text
    shd = f'<w:shd w:val="clear" w:color="auto" w:fill="{fill}"/>' if fill else ''
    return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shd}<w:vAlign w:val="center"/></w:tcPr>'
            f'<w:p><w:pPr><w:spacing w:before="20" w:after="20"/><w:jc w:val="{align}"/></w:pPr>{run(text, b, 19)}</w:p></w:tc>')


def table(rows, slow):
    bd = ''.join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="808080"/>' for s in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV'])
    x = (f'<w:tbl><w:tblPr><w:tblW w:w="{sum(COLS)}" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders>{bd}</w:tblBorders>'
         f'<w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="60" w:type="dxa"/><w:right w:w="60" w:type="dxa"/></w:tblCellMar></w:tblPr>'
         '<w:tblGrid>' + ''.join(f'<w:gridCol w:w="{c}"/>' for c in COLS) + '</w:tblGrid>')
    head = [h + ('（慢）' if h.startswith(slow + ' ') else '') for h in HEAD]
    x += '<w:tr><w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>' + ''.join(cell(h, w, 'D9D9D9', True) for h, w in zip(head, COLS)) + '</w:tr>'
    for r in rows:
        ours = r[0].startswith('按群校正'); ideal = r[0].startswith('全員同速')
        fill = 'DEEAF6' if ours else ('F2F2F2' if ideal else None)
        x += '<w:tr><w:trPr><w:cantSplit/></w:trPr>' + ''.join(
            cell(v, w, fill, bold=(i == 0 and (ours or ideal)), align='left' if i == 0 else 'center')
            for i, (v, w) in enumerate(zip(r, COLS))) + '</w:tr>'
    return x + '</w:tbl>'


def acc_rel(txt):
    """泛化格：'78.70（−1.39）' → '78.70（−1.74%）'；相對理想 (v−80.09)/80.09；粗體仍以 1.09 分為準。"""
    v = float(txt.lstrip('*').split('（')[0])
    if abs(v - IDEAL_ACC) < 1e-9 and '（' not in txt: return txt
    rel = (v - IDEAL_ACC) / IDEAL_ACC * 100
    out = f'{v:.2f}（{rel:+.2f}%）'.replace('-', '−')
    return (B if v - IDEAL_ACC < -NOISE_ACC else '') + out


def reshape(rows):
    by = {}
    for r in rows:
        name = r[0].split('（多存')[0]
        by[name] = (name, acc_rel(r[2])) + tuple(r[3:])
    return [by[n] for n in ORDER]


SCEN = [(t, sub, slow, reshape(rows)) for t, sub, slow, rows in SCEN]
body = para('第二貢獻：畫風集中慢的詳細結果', True, 28, 'center', 0, 120)
body += para('泛化括號為相對理想的百分比變化；其餘括號為減去理想的差；理想那列只寫數值。粗體＝比理想差超過雜訊寬度。詳細定義見表後註。', False, 20, None, 0, 160)
for k, (ttl, sub, slow, rows) in enumerate(SCEN):
    body += para(ttl, True, 22, None, 200 if k else 60, 40, keep=True)   # dany 1027 版式列少，三表同頁
    body += para(sub, False, 20, None, 0, 60, keep=True)
    body += table(rows, slow)
body += para('', after=60)
for n in NOTES:
    body += para(n, False, 18, None, 0, 40)
SECT = ('<w:sectPr><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/>'
        '<w:pgMar w:top="900" w:right="1000" w:bottom="900" w:left="1000" w:header="500" w:footer="500" w:gutter="0"/></w:sectPr>')
DOC = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
       '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
       'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
       f'<w:body>{body}{SECT}</w:body></w:document>')
STYLES = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
          f'<w:docDefaults><w:rPrDefault><w:rPr>{FONT}<w:sz w:val="20"/><w:szCs w:val="20"/><w:lang w:val="en-US" w:eastAsia="zh-TW"/></w:rPr></w:rPrDefault>'
          '<w:pPrDefault><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>'
          '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
          '<w:style w:type="table" w:default="1" w:styleId="TableNormal"><w:name w:val="Normal Table"/><w:tblPr><w:tblInd w:w="0" w:type="dxa"/>'
          '<w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="108" w:type="dxa"/><w:bottom w:w="0" w:type="dxa"/><w:right w:w="108" w:type="dxa"/></w:tblCellMar></w:tblPr></w:style>'
          '</w:styles>')
CT = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
      '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
      '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
      '<Default Extension="xml" ContentType="application/xml"/>'
      '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
      '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
      '</Types>')
RELS = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '</Relationships>')
DRELS = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
         '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
         '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
         '</Relationships>')
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    z.writestr('[Content_Types].xml', CT); z.writestr('_rels/.rels', RELS)
    z.writestr('word/document.xml', DOC); z.writestr('word/styles.xml', STYLES); z.writestr('word/_rels/document.xml.rels', DRELS)
print('寫出', OUT)
