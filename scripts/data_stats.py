"""Print summary statistics for generated JSONL files."""

from __future__ import annotations

import json
import statistics
import sys
from math import ceil
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env", override=False)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_gen.config import TRAITS
from data_gen.generate import read_jsonl
from data_gen.llm_client import QuotaCounter


def main() -> None:
    for trait in TRAITS:
        rows = read_jsonl(Path("data/pairs") / f"{trait}.jsonl")
        if not rows:
            continue
        pos = [len(row["pos"].split()) for row in rows]
        neg = [len(row["neg"].split()) for row in rows]
        print(json.dumps({"trait": trait, "count": len(rows), "contexts": dict(Counter(r["context"] for r in rows)), "facets": dict(Counter(r["facet"] for r in rows)), "splits": dict(Counter(r.get("split") for r in rows)), "mean_pos_words": statistics.mean(pos), "mean_neg_words": statistics.mean(neg)}))
    quota = QuotaCounter("data/pairs/.state/quota.json", 1500)
    agreement_path = Path("data/pairs/.state/judge_agreement.json")
    agreements = json.loads(agreement_path.read_text(encoding="utf-8")) if agreement_path.exists() else {}
    used = quota.used()
    remaining = max(0, quota.daily_limit - used)
    print(json.dumps({
        "judge_agreement": agreements,
        "requests_used_today": used,
        "remaining_requests_today": remaining,
        "estimated_remaining_days_at_current_quota": ceil(remaining / quota.daily_limit) if remaining else 0,
    }))


if __name__ == "__main__":
    main()
