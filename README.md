# llm-personality-conditioning
Robust personality conditioning in LLMs

## Synthetic data generation

The generator creates behavioral, contrastive OCEAN pairs in
`data/pairs/{trait}.jsonl`. API keys are read only from environment variables;
copy `.env.example` for the variable names, but do not commit a `.env` file.
Gemini uses its OpenAI-compatible endpoint by default. Fill the model names in
`configs/data_gen.yaml` from the current AI Studio model list.

In Windows PowerShell, set a key and run the smoke test:

```powershell
$env:GEMINI_API_KEY = "..."
python scripts/generate_data.py --n-per-trait 10 --dry-run
python scripts/generate_data.py --n-per-trait 10
```

The full run uses approximately 700 requested pairs per trait, writes each
accepted batch immediately, and can be resumed after quota exhaustion:

```powershell
python scripts/generate_data.py --n-per-trait 700
python scripts/data_stats.py
```

Use `--trait openness` to run one trait and `--skip-judge` to defer judging.
The persisted counter is `data/pairs/.state/quota.json`; requests are rate
limited and quota errors back off with jitter. The run plan is: dry-run all
prompts, smoke-test a small generated sample by hand, then run the resumable
full generation and inspect statistics.
