from pathlib import Path
import re
import csv
from datetime import datetime

LOG_DIR = Path("/video_ssd/kongzishang/xutongda/git/IFID/sample_part_logs")

OUT_CSV = LOG_DIR / "without_cfg_fid_summary.csv"
OUT_MD = LOG_DIR / "without_cfg_fid_summary.md"
OUT_DUP_CSV = LOG_DIR / "without_cfg_fid_duplicates.csv"

# 匹配类似：
# part4_sit-xl-repae-400k.log
# part12_sit-b-dcae-400k.log
log_files = sorted(LOG_DIR.glob("part*_sit-*-400k.log"))

fid_re = re.compile(
    r"\bfid\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)

model_re = re.compile(r"^part[^_]*_(sit-.+?-400k)\.log$")

def extract_model_name(path: Path) -> str:
    m = model_re.match(path.name)
    if m:
        return m.group(1)
    # fallback: 去掉 partX_ 前缀和 .log 后缀
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[1]
    return stem

def extract_last_fid(path: Path):
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except Exception as e:
        return None, f"READ_ERROR: {e}"

    # 从后往前找最后一个 fid=
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        m = fid_re.search(line)
        if m:
            return float(m.group(1)), line

    return None, "NO_FID_FOUND"

all_rows = []
for p in log_files:
    model = extract_model_name(p)
    fid, fid_line = extract_last_fid(p)
    stat = p.stat()

    all_rows.append({
        "model": model,
        "fid": fid,
        "fid_line": fid_line,
        "log_file": str(p),
        "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "size_bytes": stat.st_size,
    })

# 如果同一个 model 有多个 log，默认取最新修改的那个作为主结果
by_model = {}
duplicates = []

for r in all_rows:
    model = r["model"]
    if model not in by_model:
        by_model[model] = r
    else:
        duplicates.append(by_model[model])
        duplicates.append(r)

        # 选择 mtime 更新的那一个
        old_mtime = Path(by_model[model]["log_file"]).stat().st_mtime
        new_mtime = Path(r["log_file"]).stat().st_mtime
        if new_mtime > old_mtime:
            by_model[model] = r

summary_rows = list(by_model.values())

# 排序：有 fid 的按 fid 升序；没有 fid 的放最后
def sort_key(r):
    if r["fid"] is None:
        return (1, float("inf"), r["model"])
    return (0, r["fid"], r["model"])

summary_rows = sorted(summary_rows, key=sort_key)

# 输出 CSV
fields = ["model", "fid", "fid_line", "log_file", "mtime", "size_bytes"]

with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in summary_rows:
        rr = dict(r)
        rr["fid"] = "" if rr["fid"] is None else rr["fid"]
        writer.writerow(rr)

# 输出 Markdown
with OUT_MD.open("w") as f:
    f.write("| model | without_cfg_fid | log_file |\n")
    f.write("|---|---:|---|\n")
    for r in summary_rows:
        fid_str = "MISSING" if r["fid"] is None else f"{r['fid']:.12f}"
        f.write(f"| {r['model']} | {fid_str} | {r['log_file']} |\n")

# 输出 duplicate 记录
seen_dup = set()
dup_unique = []
for r in duplicates:
    key = (r["model"], r["log_file"])
    if key not in seen_dup:
        seen_dup.add(key)
        dup_unique.append(r)

with OUT_DUP_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in sorted(dup_unique, key=lambda x: (x["model"], x["mtime"])):
        rr = dict(r)
        rr["fid"] = "" if rr["fid"] is None else rr["fid"]
        writer.writerow(rr)

missing = [r for r in summary_rows if r["fid"] is None]

print("=" * 100)
print(f"Scanned log dir: {LOG_DIR}")
print(f"Matched log files: {len(log_files)}")
print(f"Unique models: {len(summary_rows)}")
print(f"Missing fid logs: {len(missing)}")
print(f"Duplicate model log rows: {len(dup_unique)}")
print(f"Saved CSV: {OUT_CSV}")
print(f"Saved Markdown: {OUT_MD}")
print(f"Saved duplicate CSV: {OUT_DUP_CSV}")
print("=" * 100)

print(f"\n{'model':40s} {'without_cfg_fid':>18s}")
print("-" * 65)
for r in summary_rows:
    fid_str = "MISSING" if r["fid"] is None else f"{r['fid']:.12f}"
    print(f"{r['model']:40s} {fid_str:>18s}")

if missing:
    print("\n" + "=" * 100)
    print("Logs matched but no fid= line found:")
    for r in missing:
        print(f"[NO_FID] {r['model']}  {r['log_file']}  last_status={r['fid_line']}")

if dup_unique:
    print("\n" + "=" * 100)
    print("Duplicate logs exist for some models. Main table uses the newest mtime per model.")
    print(f"Check: {OUT_DUP_CSV}")
