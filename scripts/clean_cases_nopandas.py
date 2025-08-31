# scripts/clean_cases_nopandas.py
import sys, csv, re, unicodedata

if len(sys.argv) != 3:
    print("Usage: python scripts/clean_cases_nopandas.py <in.csv> <out.csv>")
    sys.exit(1)

inp, outp = sys.argv[1], sys.argv[2]

# 允许的列名集合（根据你的 CSV 可增减）
KEEP_COLS = {"query","expected_intent","intent","test_type","type","tags","group_id","step","design_logic"}

# 规范化：去中点/奇怪间隔、全角转半角、空白压缩、去首尾引号
def normalize_query(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", s)
    # 去掉“·”和类似中点
    s = re.sub(r"[·•∙⋅]", "", s)
    # 常见口头禅/多余省略号压缩（可按需保留）
    s = re.sub(r"\s+", " ", s).strip()
    # 去句内重复空格（中文标点两侧空格）
    s = re.sub(r"\s*([，。？！、；：])\s*", r"\1", s)
    # 去首尾成对引号
    s = s.strip().strip("“”'\"")
    return s

# 轻量“去重键”：小写、去标点、去空白
def dedup_key(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\u4e00-\u9fff]", "", s)   # 仅保留字母数字和中日韩统一表意文字
    return s

seen = set()
rows_out = []

with open(inp, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = [c for c in reader.fieldnames if c in KEEP_COLS] or reader.fieldnames
    for row in reader:
        q_col = "query" if "query" in row else ( "Query文本" if "Query文本" in row else None )
        if not q_col: 
            continue
        q = normalize_query(row[q_col])
        if not q: 
            continue
        k = dedup_key(q)
        if k in seen:
            continue
        seen.add(k)
        # 写回规范化后的 query 与统一列名
        out_row = {c: row.get(c, "") for c in fieldnames}
        out_row["query"] = q
        # 兼容 expected_intent / intent
        if "expected_intent" not in out_row and "intent" in out_row:
            out_row["expected_intent"] = out_row.get("intent", "")
        rows_out.append(out_row)

with open(outp, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=sorted(set(sum([list(r.keys()) for r in rows_out],[]))))
    writer.writeheader()
    writer.writerows(rows_out)

print(f"Done. Kept {len(rows_out)} rows. Saved -> {outp}")
