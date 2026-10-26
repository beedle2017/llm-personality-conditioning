"""CLI for synthetic Big Five contrastive data."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env", override=False)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_gen.config import TRAITS, load_config
from data_gen.generate import generate_trait, read_jsonl
from data_gen.llm_client import LLMClient
from data_gen.validate import assign_split, filter_pairs, judge_pairs, stable_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait", choices=(*TRAITS, "all"), default="all")
    parser.add_argument("--n-per-trait", type=int, default=700)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--config", default="configs/data_gen.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    traits = TRAITS if args.trait == "all" else (args.trait,)
    client = None if args.dry_run else LLMClient(config)
    for trait in traits:
        path = generate_trait(trait, args.n_per_trait, client, "data/pairs", config.seed, args.dry_run)
        if args.dry_run:
            continue
        raw = read_jsonl(path)
        benchmark = Path("data/behaviorbench_prompts.txt")
        benchmark_lines = benchmark.read_text(encoding="utf-8").splitlines() if benchmark.exists() else []
        accepted, drops = filter_pairs(raw, trait, benchmark_lines, config.behaviorbench_similarity_threshold)
        if args.n_per_trait > 600:
            accepted = accepted[:600]
        if not benchmark.exists():
            logging.warning("BehaviorBench prompt file absent; continuing without overlap filtering.")
        agreement = None
        if not args.skip_judge and accepted:
            if not config.judge_model:
                logging.warning("judge_model is empty; skipping judge. Set it in %s or use --skip-judge.", args.config)
            else:
                accepted, agreement = judge_pairs(accepted, trait, client, config.judge_model, seed=config.seed)
                state = Path("data/pairs/.state")
                state.mkdir(parents=True, exist_ok=True)
                agreements = json.loads((state / "judge_agreement.json").read_text(encoding="utf-8")) if (state / "judge_agreement.json").exists() else {}
                agreements[trait] = agreement
                (state / "judge_agreement.json").write_text(json.dumps(agreements, indent=2), encoding="utf-8")
        split = assign_split(accepted, min(100, len(accepted)), config.seed)
        for pair in split:
            pair["id"] = stable_id(pair)
        path.write_text("".join(json.dumps(pair) + "\n" for pair in split), encoding="utf-8")
        logging.info("%s: kept=%d drops=%s judge_agreement=%s", trait, len(split), dict(drops), agreement)


if __name__ == "__main__":
    main()
