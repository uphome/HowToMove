"""Build the unified GRPO pool from greedy screen rows.

Only maps where the final SFT policy failed the first greedy rollout are kept.
No separate G>1 pre-probe is required; GRPO itself will sample G rollouts and
skip degenerate groups online.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True,
                    help="one or more greedy screen .rows.jsonl files")
    ap.add_argument("--out", default="data/grpo/l1_l4_pool.json")
    args = ap.parse_args()

    rows = []
    seen = set()
    for name in args.rows:
        for line in Path(name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            map_id = row["map_id"]
            if map_id in seen:
                continue
            seen.add(map_id)
            rows.append(row)

    hard = [r for r in rows if not r["success"]]
    by_level = Counter(r["level"] for r in hard)
    all_levels = Counter(r["level"] for r in rows)
    payload = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "scripts/screen_grpo_maps.py",
        "policy": "L4 SFT",
        "selection": "greedy success == false",
        "n_screened": len(rows),
        "n_hard": len(hard),
        "screened_by_level": dict(sorted(all_levels.items())),
        "hard_by_level": dict(sorted(by_level.items())),
        "format_errors": sum(r["format_errors"] for r in hard),
        "rows": hard,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"screened={len(rows)} hard={len(hard)}")
    print(f"hard_by_level={payload['hard_by_level']}")
    print(f"pool -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
