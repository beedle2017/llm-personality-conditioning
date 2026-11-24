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


def is_config_completed(out_dir: Path, method: str, trait: str, pair: str, task_type: str = "all") -> bool:
    """Check if config results already exist in probes.csv."""
    probes_csv = out_dir / "probes.csv"
    if not probes_csv.exists():
        return False
    with probes_csv.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row.get("method") == method
                and row.get("trait") == trait
                and row.get("pair") == pair
                and (task_type == "all" or row.get("task_type") == task_type)
            ):
                return True
    return False


def append_probes_csv(out_dir: Path, probe_rows: list[dict[str, Any]]) -> None:
    """Append probe rows directly to disk."""
    if not probe_rows:
        return
    probes_csv = out_dir / "probes.csv"
    exists = probes_csv.exists() and probes_csv.stat().st_size > 0
    fields = [
        "method", "trait", "pair", "task_id", "task_type", "agent", "level", "checkpoint",
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
    parser.add_argument("--task-type", choices=["all", "collaborative", "disagreement"], default="all", help="Task type to filter from tasks.json")
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
    t_load_start = time.time()
    model_load_time = 0.0

    if args.backend == "fake":
        generator = FakeGenerator()
    else:
        from rewriting.llm import HFGenerator
        logger.info(f"Loading HF model in {'4-bit NF4' if args.load_4bit else 'native dtype'}...")
        gen_wrapper = HFGenerator.load(load_4bit=args.load_4bit)
        generator = gen_wrapper
        hf_model = gen_wrapper.model
        model_load_time = time.time() - t_load_start
        logger.info(f"Model loaded in {model_load_time:.1f}s")

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

    # Shared caches across configs
    cp0_probe_cache: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    extraction_cache: dict[tuple[str, str], str] = {}
    greedy_cache: dict[str, str] = {}

    # Cumulative per-phase timing stats
    cumulative_timing = {
        "model_load": model_load_time,
        "dialogue_time": 0.0,
        "probe_time": 0.0,
        "extraction_time": 0.0,
    }

    # Collect outputs in memory for summary generation
    all_probes: list[dict[str, Any]] = []
    all_transcripts: list[dict[str, Any]] = []
    solo_probes_by_trait_level: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    solo_transcripts_by_trait_level: dict[tuple[str, str, int], list[dict[str, Any]]] = {}

    if args.resume:
        probes_csv = out_dir / "probes.csv"
        if probes_csv.exists():
            with probes_csv.open("r", encoding="utf-8", errors="ignore") as f:
                for row in csv.DictReader(f):
                    r = dict(row)
                    r["checkpoint"] = int(r["checkpoint"]) if r.get("checkpoint") else 0
                    r["level"] = int(r["level"]) if r.get("level") else 1
                    r["dialogue_id"] = int(r["dialogue_id"]) if r.get("dialogue_id") else 0
                    r["invalid"] = str(r.get("invalid", "")).lower() == "true"
                    r["score"] = float(r["score"]) if r.get("score") not in ("", None) else None
                    if r.get("condition") == "solo_control":
                        key = (r["method"], r["trait"], r["level"])
                        solo_probes_by_trait_level.setdefault(key, []).append(r)
                    if r.get("checkpoint") == 0 and not r.get("invalid") and r.get("score") is not None:
                        cp0_key = (r["method"], r["trait"], r["level"], r.get("item_id", "item_0"))
                        if cp0_key not in cp0_probe_cache:
                            cp0_probe_cache[cp0_key] = {
                                "statement": r.get("statement", ""),
                                "extracted": r.get("extracted", ""),
                                "score": r.get("score"),
                                "invalid": r.get("invalid"),
                            }
        for tf in out_dir.glob("transcripts_*.jsonl"):
            with tf.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            t = json.loads(line)
                            if t.get("condition") == "solo_control":
                                key = (t["method"], t.get("trait_code", t.get("trait")), t.get("level", 1))
                                solo_transcripts_by_trait_level.setdefault(key, []).append(t)
                        except Exception:
                            pass
        if solo_probes_by_trait_level:
            logger.info(f"Loaded existing solo controls for {len(solo_probes_by_trait_level)} (method, trait, level) conditions from disk.")
        if cp0_probe_cache:
            logger.info(f"Preloaded {len(cp0_probe_cache)} checkpoint-0 probes into cache from disk.")

    current_loaded_method: str | None = None

    for idx, (method, trait, pair_str) in enumerate(configs_to_run):
        if args.resume and is_config_completed(out_dir, method, trait, pair_str, task_type=args.task_type):
            logger.info(f"Skipping already completed config ({idx+1}/{len(configs_to_run)}): {method}-{trait}-{pair_str} (type={args.task_type})")
            completed_count += 1
            continue

        # Memory cleanup between different methods if requested
        if current_loaded_method is not None and current_loaded_method != method and args.backend == "hf":
            logger.info(f"Transitioning from {current_loaded_method} to {method}; freeing cache.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        current_loaded_method = method

        t0 = time.time()
        logger.info(f"Running config ({idx+1}/{len(configs_to_run)}): method={method}, trait={trait}, pair={pair_str}, task_type={args.task_type}")

        levels = [int(x) for x in pair_str.split("-")]
        lvl_a, lvl_b = levels[0], levels[1]

        # Select tasks filtered by task_type and tagged with this trait
        available_tasks = all_tasks
        if args.task_type != "all":
            available_tasks = [t for t in all_tasks if t.get("type") == args.task_type]

        matching_tasks = [t for t in available_tasks if trait in t.get("traits", [])]
        if len(matching_tasks) < args.dialogues_per_pair:
            matching_tasks = (matching_tasks + available_tasks)[:args.dialogues_per_pair]
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
            cp0_probe_cache=cp0_probe_cache,
            extraction_cache=extraction_cache,
            greedy_cache=greedy_cache,
        )

        append_probes_csv(out_dir, result["probes"])
        append_transcripts_jsonl(out_dir, method, trait, result["transcripts"])
        all_probes.extend(result["probes"])
        all_transcripts.extend(result["transcripts"])

        cfg_timing = result.get("timing", {})
        cfg_diag = cfg_timing.get("dialogue_time", 0.0)
        cfg_probe = cfg_timing.get("probe_time", 0.0)
        cfg_ext = cfg_timing.get("extraction_time", 0.0)
        cumulative_timing["dialogue_time"] += cfg_diag
        cumulative_timing["probe_time"] += cfg_probe
        cumulative_timing["extraction_time"] += cfg_ext

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
                    cp0_probe_cache=cp0_probe_cache,
                    extraction_cache=extraction_cache,
                    greedy_cache=greedy_cache,
                )
                append_probes_csv(out_dir, solo_res["probes"])
                append_transcripts_jsonl(out_dir, method, trait, solo_res["transcripts"])
                solo_probes_by_trait_level[solo_key] = solo_res["probes"]
                solo_transcripts_by_trait_level[solo_key] = solo_res["transcripts"]

                s_time = solo_res.get("timing", {})
                cumulative_timing["dialogue_time"] += s_time.get("dialogue_time", 0.0)
                cumulative_timing["probe_time"] += s_time.get("probe_time", 0.0)
                cumulative_timing["extraction_time"] += s_time.get("extraction_time", 0.0)

        elapsed = time.time() - t0
        per_config_times.append(elapsed)
        completed_count += 1

        logger.info(
            f"[TIMING] Config {method}-{trait}-{pair_str} (in {elapsed:.1f}s): "
            f"Dialogue={cfg_diag:.1f}s ({(cfg_diag/max(0.1, elapsed))*100:.1f}%), "
            f"Probes={cfg_probe:.1f}s ({(cfg_probe/max(0.1, elapsed))*100:.1f}%), "
            f"Extraction={cfg_ext:.1f}s ({(cfg_ext/max(0.1, elapsed))*100:.1f}%)"
        )
        logger.info(
            f"[TIMING] Cumulative Phase Totals: "
            f"ModelLoad={cumulative_timing['model_load']:.1f}s, "
            f"Dialogue={cumulative_timing['dialogue_time']:.1f}s, "
            f"Probes={cumulative_timing['probe_time']:.1f}s, "
            f"Extraction={cumulative_timing['extraction_time']:.1f}s"
        )

        avg_time = sum(per_config_times) / len(per_config_times)
        remaining_configs = len(configs_to_run) - (idx + 1)
        est_remaining_min = (remaining_configs * avg_time) / 60.0
        logger.info(
            f"Finished config in {elapsed:.1f}s. "
            f"Estimated time remaining: {est_remaining_min:.1f} min ({remaining_configs} configs left)"
        )

    # Statistical analysis & summary generation
    logger.info("Computing analysis metrics and generating summary tables...")
    # Load all probes from disk (probes.csv) to ensure all completed runs are included
    probes_csv = out_dir / "probes.csv"
    all_disk_probes: list[dict[str, Any]] = []
    if probes_csv.exists():
        with probes_csv.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = dict(row)
                r["checkpoint"] = int(r["checkpoint"]) if r.get("checkpoint") else 0
                r["level"] = int(r["level"]) if r.get("level") else 1
                r["dialogue_id"] = int(r["dialogue_id"]) if r.get("dialogue_id") else 0
                r["invalid"] = str(r.get("invalid", "")).lower() == "true"
                r["score"] = float(r["score"]) if r.get("score") not in ("", None) else None
                all_disk_probes.append(r)
    else:
        all_disk_probes = all_probes

    # Load all transcripts from disk
    all_disk_transcripts: list[dict[str, Any]] = []
    for tf in out_dir.glob("transcripts_*.jsonl"):
        with tf.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_disk_transcripts.append(json.loads(line))
                    except Exception:
                        pass
    if not all_disk_transcripts:
        all_disk_transcripts = all_transcripts

    # Separate solo controls
    solo_disk_probes: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for p in all_disk_probes:
        if p.get("condition") == "solo_control":
            key = (p["method"], p["trait"], p["level"])
            solo_disk_probes.setdefault(key, []).append(p)

    solo_disk_trans: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for t in all_disk_transcripts:
        if t.get("condition") == "solo_control":
            key = (t["method"], t.get("trait_code", t.get("trait")), t.get("level", 1))
            solo_disk_trans.setdefault(key, []).append(t)

    # Group interactive probes by (method, trait, pair, task_type)
    grouped_probes: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for p in all_disk_probes:
        if p.get("condition") == "interactive":
            tt = p.get("task_type", "collaborative")
            key = (p["method"], p["trait"], p["pair"], tt)
            grouped_probes.setdefault(key, []).append(p)

    analyses = []
    for (m, t, pair_str, tt), p_list in grouped_probes.items():
        levels = [int(x) for x in pair_str.split("-")]
        lvl_a, lvl_b = levels[0], levels[1]
        solo_p_a = solo_disk_probes.get((m, t, lvl_a), [])
        solo_p_b = solo_disk_probes.get((m, t, lvl_b), [])
        combined_solo = solo_p_a + solo_p_b

        pair_transcripts = [
            tr for tr in all_disk_transcripts
            if tr.get("method") == m and (tr.get("trait_code") == t or tr.get("trait") == t) and tr.get("pair") == pair_str
            and tr.get("task_type", "collaborative") == tt and tr.get("condition") == "interactive"
        ]
        solo_t_a = solo_disk_trans.get((m, t, lvl_a), [])
        solo_t_b = solo_disk_trans.get((m, t, lvl_b), [])
        combined_solo_trans = solo_t_a + solo_t_b

        analysis = analyze_pair_condition(
            interactive_probes=p_list,
            solo_probes=combined_solo,
            interactive_transcripts=pair_transcripts,
            solo_transcripts=combined_solo_trans,
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
