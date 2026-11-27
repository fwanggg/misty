"""Same-moves check and the before/after table, from bench/run_bench.py logs.

  python bench/compare.py ~/bench orig fix1 fix2 fix3

For every seed that the first variant (the original) played: the move-list
diff of each other variant against it, and per variant peak RSS, peak VM
memory, mean and max Misty move time, max |P| and the KMV estimate of the
largest pre-cap |M|.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(root: Path, variant: str) -> dict[int, dict]:
    rows = {}
    log = root / f"{variant}.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["seed"]] = r  # the last run of a seed wins
    return rows


def first_diff(a: list[str], b: list[str]) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"ply {i + 1}: {x} vs {y}"
    if len(a) != len(b):
        return f"length {len(a)} vs {len(b)}"
    return ""


def main() -> int:
    root = Path(sys.argv[1]).expanduser()
    variants = sys.argv[2:]
    data = {v: load(root, v) for v in variants}
    base = variants[0]
    seeds = sorted(data[base])
    print("Same moves (move list of each variant vs", base + "):")
    for s in seeds:
        a = data[base][s]["moves"].split()
        cells = []
        for v in variants[1:]:
            r = data[v].get(s)
            if r is None:
                cells.append(f"{v}: not run")
                continue
            b = r["moves"].split()
            d = first_diff(a, b)
            if not d:
                cells.append(f"{v}: identical ({len(b)} plies)")
            elif data[base][s]["status"] != "ok" and len(b) > len(a) and b[: len(a)] == a:
                cells.append(f"{v}: identical on the {len(a)} plies {base} reached ({base} {data[base][s]['status']})")
            else:
                cells.append(f"{v}: DIFFERENT, {d}")
        print(f"  seed {s}: " + "; ".join(cells))
    print()
    hdr = ("seed", "variant", "status", "plies", "peak RSS MB", "peak VM MB", "mean move s",
           "max move s", "max |P|", "max M est", "cap fires")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for s in seeds:
        for v in variants:
            r = data[v].get(s)
            if r is None:
                continue
            print("| " + " | ".join(str(x) for x in (
                s, v, r["status"], r["plies"], r["peak_rss_mb"], r["peak_vm_used_mb"],
                r["mean_move_s"], r["max_move_s"], f"{r['max_p']:,}", f"{r['max_m_est']:,}",
                r["cap_fires"])) + " |")
    print()
    print("Totals over the seeds every variant played:")
    common = [s for s in seeds if all(s in data[v] for v in variants)]
    for v in variants:
        rs = [data[v][s] for s in common]
        if not rs:
            continue
        n_moves = sum(r["misty_moves"] for r in rs)
        mean = sum((r["mean_move_s"] or 0) * r["misty_moves"] for r in rs) / max(1, n_moves)
        print(f"  {v}: games {len(rs)}, ok {sum(r['status'] == 'ok' for r in rs)}, "
              f"max peak RSS {max(r['peak_rss_mb'] for r in rs)} MB, "
              f"max peak VM {max(r['peak_vm_used_mb'] for r in rs)} MB, "
              f"mean move {mean:.2f} s, max move {max((r['max_move_s'] or 0) for r in rs)} s, "
              f"max |P| {max(r['max_p'] for r in rs):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
