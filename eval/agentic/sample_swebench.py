"""Seeded, stratified sample of SWE-bench Verified.

  python -m eval.agentic.sample_swebench --n 100 --seed 20261116 --out eval/agentic/swebench_verified_100.json

Allocation is proportional by repo (largest remainder), then proportional by the task's
difficulty label within each repo, so the sample mirrors the 500-task population on both.
The sample is a pure function of (population, n, seed); the committed JSON pins the IDs so
every arm runs exactly the same tasks.

The population is read from the Hugging Face datasets-server (no `datasets` dependency), or
from a local JSONL with instance_id / repo / difficulty with --from-jsonl.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import requests

DATASET = "princeton-nlp/SWE-bench_Verified"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_SEED = 20261116
DEFAULT_OUT = Path(__file__).with_name("swebench_verified_100.json")


def fetch_verified(page: int = 100) -> list[dict]:
  rows, offset = [], 0
  while True:
    r = requests.get(ROWS_URL, params={"dataset": DATASET, "config": "default", "split": "test",
                                       "offset": offset, "length": page}, timeout=60)
    r.raise_for_status()
    data = r.json()
    batch = data.get("rows") or []
    for item in batch:
      row = item["row"]
      rows.append({"instance_id": row["instance_id"], "repo": row["repo"],
                   "difficulty": row.get("difficulty") or "unknown"})
    offset += len(batch)
    if not batch or offset >= int(data.get("num_rows_total", 0)):
      return rows


def largest_remainder(sizes: dict[str, int], n: int) -> dict[str, int]:
  """Splits n proportionally to sizes; deterministic tie-break on key."""
  total = sum(sizes.values())
  if n > total:
    raise ValueError(f"cannot take {n} from {total}")
  raw = {k: n * v / total for k, v in sizes.items()}
  out = {k: int(v) for k, v in raw.items()}
  left = n - sum(out.values())
  for k in sorted(sizes, key=lambda k: (-(raw[k] - out[k]), k))[:left]:
    out[k] += 1
  return out


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[str]:
  rng = random.Random(seed)
  by_repo: dict[str, list[dict]] = collections.defaultdict(list)
  for r in sorted(rows, key=lambda r: r["instance_id"]):  # input order must not matter
    by_repo[r["repo"]].append(r)
  repo_quota = largest_remainder({k: len(v) for k, v in by_repo.items()}, n)
  chosen: list[str] = []
  for repo in sorted(by_repo):
    q = repo_quota[repo]
    if not q:
      continue
    by_diff: dict[str, list[dict]] = collections.defaultdict(list)
    for r in by_repo[repo]:
      by_diff[r["difficulty"]].append(r)
    diff_quota = largest_remainder({k: len(v) for k, v in by_diff.items()}, q)
    for diff in sorted(by_diff):
      chosen += [r["instance_id"] for r in rng.sample(by_diff[diff], diff_quota[diff])]
  return sorted(chosen)


def composition(rows: list[dict], ids: list[str]) -> dict:
  keep = set(ids)
  sel = [r for r in rows if r["instance_id"] in keep]
  def count(rs, key):
    return dict(sorted(collections.Counter(r[key] for r in rs).items()))
  return {"population": {"n": len(rows), "repo": count(rows, "repo"), "difficulty": count(rows, "difficulty")},
          "sample": {"n": len(sel), "repo": count(sel, "repo"), "difficulty": count(sel, "difficulty")}}


def main(argv=None) -> int:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--n", type=int, default=100)
  p.add_argument("--seed", type=int, default=DEFAULT_SEED)
  p.add_argument("--out", type=Path, default=DEFAULT_OUT)
  p.add_argument("--from-jsonl", type=Path, help="local population instead of the Hugging Face fetch")
  args = p.parse_args(argv)

  if args.from_jsonl:
    rows = [json.loads(l) for l in args.from_jsonl.read_text().splitlines() if l.strip()]
  else:
    rows = fetch_verified()
  ids = stratified_sample(rows, args.n, args.seed)
  doc = {"dataset": DATASET, "split": "test", "n": len(ids), "seed": args.seed, "method":
         "proportional by repo, then by difficulty within repo (largest remainder), seeded random.sample",
         "composition": composition(rows, ids), "instance_ids": ids}
  args.out.write_text(json.dumps(doc, indent=2) + "\n")
  print(f"wrote {len(ids)} instance ids to {args.out}", file=sys.stderr)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
