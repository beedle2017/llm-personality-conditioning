"""CLI runner for multi-agent social robustness experiments.

Usage:
    python scripts/run_agents.py --backend {hf,fake} --methods prompting steering rewrite \\
        --traits C E A --pairs 1-5 1-3 3-5 3-3 5-5 --rounds 8 --checkpoints 0 2 4 8 \\
        --probe-items 10 --dialogues-per-pair 6 --batch-size auto --split development \\
        --seed 20261026 --out artifacts/agents [--resume] [--smoke]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agents.analysis import analyze_pair_condition, generate_summary_tables
from agents.conditioning import FakeGenerator
from agents.dialogue import run_dialogues
from rewriting.pipeline import ADAPTED_DIR, behavior_context

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_agents")


def load_tasks(path: Path) -> list[dict[str, Any]]:
    """Load tasks from JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_dev_probe_items(split: str, traits: list[str], limit_per_trait: int, seed: int) -> dict[str, list[Any]]:
    """Load and counterbalance BehaviorBench items for probing."""
    data_path = ADAPTED_DIR / "data" / f"{split}.json"
    items = json.loads(data_path.read_text(encoding="utf-8"))

    probes_by_trait: dict[str, list[Any]] = {}
    for trait in traits:
        subset = [item for item in items if item["trait_code"] == trait]
        if limit_per_trait and limit_per_trait < len(subset):
            # Deterministic subset sampling
            subset = [subset[i * len(subset) // limit_per_trait] for i in range(limit_per_trait)]
        probes_by_trait[trait] = [behavior_context(it, seed) for it in subset]
    return probes_by_trait


def get_auto_batch_size() -> int:
    """Detect free VRAM and choose safe batch size."""
    if torch.cuda.is_available():
        free_bytes = torch.cuda.mem_get_info()[0]
        free_gb = free_bytes / (1024 ** 3)
        logger.info(f"Detected free VRAM: {free_gb:.2f} GB")
        if free_gb >= 10.0:
            return 8
        return 4
    return 4


def load_yaml_config() -> dict[str, Any]:
    cfg_file = ROOT / "configs" / "agents.yaml"
    if cfg_file.exists():
        return yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    return {}


def is_config_completed(out_dir: Path, method: str, trait: str, pair: str) -> bool:
    """Check if config results already exist in probes.csv."""
    probes_csv = out_dir / "probes.csv"
    if not probes_csv.exists():
        return False
    with probes_csv.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("method") == method and row.get("trait") == trait and row.get("pair") == pair:
                return True
    return False


def append_probes_csv(out_dir: Path, probe_rows: list[dict[str, Any]]) -> None:
    """Append probe rows directly to disk."""
    if not probe_rows:
        return
    probes_csv = out_dir / "probes.csv"
    exists = probes_csv.exists() and probes_csv.stat().st_size > 0
    fields = [
        "method", "trait", "pair", "agent", "level", "checkpoint",
        "item_id", "score", "invalid", "dialogue_id", "condition",
        "statement", "extracted"
    ]
    with probes_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(probe_rows)


def append_transcripts_jsonl(out_dir: Path, method: str, trait: str, transcripts: list[dict[str, Any]]) -> None:
    """Append dialogue transcript turns to method/trait jsonl."""
    if not transcripts:
        return
    path = out_dir / f"transcripts_{method}_{trait}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for t in transcripts:
            f.write(json.dumps(t) + "\n")


def order_configs(methods: list[str], traits: list[str], pairs: list[str]) -> list[tuple[str, str, str]]:
    """Order configs so useful results arrive early:

    Prompting on C first, then E and A, then rewrite, then steering on A.
    """
    all_configs: list[tuple[str, str, str]] = []
    # 1. Prompting on C
    if "prompting" in methods and "C" in traits:
        for p in pairs:
            all_configs.append(("prompting", "C", p))
    # 2. Prompting on E and A
    if "prompting" in methods:
        for t in ["E", "A"]:
            if t in traits:
                for p in pairs:
                    all_configs.append(("prompting", t, p))
    # 3. Rewrite
    if "rewrite" in methods:
        for t in ["C", "E", "A", "O", "N"]:
            if t in traits:
                for p in pairs:
                    all_configs.append(("rewrite", t, p))
    # 4. Steering on A
    if "steering" in methods and "A" in traits:
        for p in pairs:
            all_configs.append(("steering", "A", p))
    # Remaining combinations
    for m in methods:
        for t in traits:
            for p in pairs:
                item = (m, t, p)
                if item not in all_configs:
                    all_configs.append(item)
    return all_configs


def main() -> None:
    cfg = load_yaml_config()
    parser = argparse.ArgumentParser(description="Multi-agent social robustness experiment")
    parser.add_argument("--backend", choices=["hf", "fake"], default="hf", help="Model backend")
    parser.add_argument("--load-4bit", action="store_true", help="Load HF model in 4-bit NF4")
    parser.add_argument("--methods", nargs="+", default=["prompting", "steering", "rewrite"], help="Conditioning methods")
    parser.add_argument("--traits", nargs="+", default=["C", "E", "A"], help="Trait codes")
    parser.add_argument("--pairs", nargs="+", default=["1-5", "1-3", "3-5", "3-3", "5-5"], help="Level pairs")
    parser.add_argument("--rounds", type=int, default=8, help="Dialogue rounds")
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[0, 2, 4, 8], help="Probing checkpoints")
    parser.add_argument("--probe-items", type=int, default=10, help="Number of probe items per trait")
    parser.add_argument("--dialogues-per-pair", type=int, default=6, help="Dialogues per pair")
    parser.add_argument("--batch-size", default="auto", help="Batch size or 'auto'")
    parser.add_argument("--split", choices=["development", "test"], default="development", help="Data split")
    parser.add_argument("--confirm-frozen", action="store_true", help="Confirm frozen test split")
    parser.add_argument("--seed", type=int, default=20261026, help="Random seed")
    parser.add_argument("--out", type=Path, default=Path("artifacts/agents"), help="Output directory")
    parser.add_argument("--exploratory", action="store_true", help="Include O and N traits")
    parser.add_argument("--positive-control", action="store_true", help="Run positive control injection at cp 4")
    parser.add_argument("--smoke", action="store_true", help="Smoke test (1 trait, 1 pair, 2 rounds, 2 dialogues)")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume previous incomplete run")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Overwrite previous run")
    parser.add_argument("--max-configs", type=int, default=None, help="Max configs to run")
    args = parser.parse_args()

    if args.split == "test" and not args.confirm_frozen:
        logger.error("Using --split test requires --confirm-frozen flag.")
        sys.exit(1)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Smoke test parameter overrides
    if args.smoke:
        args.traits = ["C"]
        args.pairs = ["1-5"]
        args.rounds = 2
        args.checkpoints = [0, 2]
        args.probe_items = 4
        args.dialogues_per_pair = 2

    if args.exploratory:
        for ex in ["O", "N"]:
            if ex not in args.traits:
                args.traits.append(ex)

    batch_size = get_auto_batch_size() if args.batch_size == "auto" else int(args.batch_size)
    logger.info(f"Effective batch size: {batch_size}")

    # Load tasks and probe items
    tasks_file = ROOT / "data" / "agents" / "tasks.json"
    all_tasks = load_tasks(tasks_file)
    probe_items_by_trait = load_dev_probe_items(args.split, args.traits, args.probe_items, args.seed)

    configs_to_run = order_configs(args.methods, args.traits, args.pairs)
    if args.max_configs is not None:
        configs_to_run = configs_to_run[:args.max_configs]

    logger.info(f"Total configs to run: {len(configs_to_run)}")

    # Initialize generator based on backend
    generator: Any = None
    hf_model: Any = None

    if args.backend == "fake":
        generator = FakeGenerator()
    else:
        from rewriting.llm import HFGenerator
        logger.info(f"Loading HF model in {'4-bit NF4' if args.load_4bit else 'native dtype'}...")
        gen_wrapper = HFGenerator.load(load_4bit=args.load_4bit)
        generator = gen_wrapper
        hf_model = gen_wrapper.model

    # Steering vectors lookup
    steering_vectors: dict[str, torch.Tensor] = {}
    if "steering" in args.methods and args.backend == "hf":
        st_cfg = cfg.get("steering", {})
        vec_paths = st_cfg.get("vector_paths", {})
        for t_code in args.traits:
            path_str = vec_paths.get(t_code)
            if path_str:
                p = ROOT / path_str
                if p.exists():
                    steering_vectors[t_code] = torch.load(p, map_location="cpu", weights_only=True).float()
                    logger.info(f"Loaded steering vector for {t_code} from {p}")

    completed_count = 0
    start_time = time.time()
    per_config_times: list[float] = []

    # Collect outputs in memory for summary generation
    all_probes: list[dict[str, Any]] = []
    all_transcripts: list[dict[str, Any]] = []
    solo_probes_by_trait_level: dict[tuple[str, str, int], list[dict[str, Any]]] = {}

    current_loaded_method: str | None = None

    for idx, (method, trait, pair_str) in enumerate(configs_to_run):
        if args.resume and is_config_completed(out_dir, method, trait, pair_str):
            logger.info(f"Skipping already completed config ({idx+1}/{len(configs_to_run)}): {method}-{trait}-{pair_str}")
            completed_count += 1
            continue

        # Memory cleanup between different methods if requested
        if current_loaded_method is not None and current_loaded_method != method and args.backend == "hf":
            logger.info(f"Transitioning from {current_loaded_method} to {method}; freeing cache.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        current_loaded_method = method

        t0 = time.time()
        logger.info(f"Running config ({idx+1}/{len(configs_to_run)}): method={method}, trait={trait}, pair={pair_str}")

        levels = [int(x) for x in pair_str.split("-")]
        lvl_a, lvl_b = levels[0], levels[1]

        # Select tasks tagged with this trait, or all tasks if not enough
        matching_tasks = [t for t in all_tasks if trait in t.get("traits", [])]
        if len(matching_tasks) < args.dialogues_per_pair:
            matching_tasks = (matching_tasks + all_tasks)[:args.dialogues_per_pair]
        else:
            matching_tasks = matching_tasks[:args.dialogues_per_pair]

        trait_probes = probe_items_by_trait.get(trait, [])
        s_vec = steering_vectors.get(trait)
        k_val = cfg.get("steering", {}).get("k_per_trait", {}).get(trait, 4.0)

        # 1. Run interactive dialogues
        result = run_dialogues(
            method=method,
            trait_code=trait,
            level_a=lvl_a,
            level_b=lvl_b,
            tasks=matching_tasks,
            generator=generator,
            probe_items=trait_probes,
            rounds=args.rounds,
            checkpoints=tuple(args.checkpoints),
            seed=args.seed + idx * 100,
            batch_size=batch_size,
            k=k_val,
            steering_vector=s_vec,
            model=hf_model,
            solo_control=False,
            positive_control=args.positive_control,
        )

        append_probes_csv(out_dir, result["probes"])
        append_transcripts_jsonl(out_dir, method, trait, result["transcripts"])
        all_probes.extend(result["probes"])
        all_transcripts.extend(result["transcripts"])

        # 2. Run solo control if not already run for these levels
        for lvl in (lvl_a, lvl_b):
            solo_key = (method, trait, lvl)
            if solo_key not in solo_probes_by_trait_level:
                logger.info(f"Running solo control for {method}-{trait}-level{lvl}...")
                solo_res = run_dialogues(
                    method=method,
                    trait_code=trait,
                    level_a=lvl,
                    level_b=lvl,
                    tasks=matching_tasks,
                    generator=generator,
                    probe_items=trait_probes,
                    rounds=args.rounds,
                    checkpoints=tuple(args.checkpoints),
                    seed=args.seed + idx * 100 + lvl,
                    batch_size=batch_size,
                    k=k_val,
                    steering_vector=s_vec,
                    model=hf_model,
                    solo_control=True,
                )
                append_probes_csv(out_dir, solo_res["probes"])
                append_transcripts_jsonl(out_dir, method, trait, solo_res["transcripts"])
                solo_probes_by_trait_level[solo_key] = solo_res["probes"]

        elapsed = time.time() - t0
        per_config_times.append(elapsed)
        completed_count += 1

        avg_time = sum(per_config_times) / len(per_config_times)
        remaining_configs = len(configs_to_run) - (idx + 1)
        est_remaining_min = (remaining_configs * avg_time) / 60.0
        logger.info(
            f"Finished config in {elapsed:.1f}s. "
            f"Estimated time remaining: {est_remaining_min:.1f} min ({remaining_configs} configs left)"
        )

    # Statistical analysis & summary generation
    logger.info("Computing analysis metrics and generating summary tables...")
    # Group by (method, trait, pair)
    grouped_probes: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for p in all_probes:
        if p.get("condition") == "interactive":
            key = (p["method"], p["trait"], p["pair"])
            grouped_probes.setdefault(key, []).append(p)

    analyses = []
    for (m, t, pair_str), p_list in grouped_probes.items():
        levels = [int(x) for x in pair_str.split("-")]
        lvl_a, lvl_b = levels[0], levels[1]
        solo_p_a = solo_probes_by_trait_level.get((m, t, lvl_a), [])
        solo_p_b = solo_probes_by_trait_level.get((m, t, lvl_b), [])
        combined_solo = solo_p_a + solo_p_b

        pair_transcripts = [tr for tr in all_transcripts if tr["method"] == m and tr["trait_code"] == t and tr.get("condition") == "interactive"]
        analysis = analyze_pair_condition(
            interactive_probes=p_list,
            solo_probes=combined_solo,
            interactive_transcripts=pair_transcripts,
            n_resamples=1000,
            seed=args.seed,
        )
        if analysis:
            analyses.append(analysis)

    csv_path, md_path = generate_summary_tables(analyses, out_dir)
    logger.info(f"Summary generated at: {csv_path} and {md_path}")
    print("\n" + md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
