# Repository Guidelines

## Project Structure & Module Organization
- `src/environment/ev_charging_env.py` defines the Gymnasium-compatible grid simulator; reuse this interface when adding environments.
- `src/agents/` contains the clean `DQNAgent` and `BackdooredDQNAgent`; new agents should expose the existing `select_action`, `store_transition`, and `train_step` APIs so experiment scripts keep working.
- `src/detection/` hosts feature builders and detectors; register new detectors in `src/detection/__init__.py` to make them importable across runners.
- `src/utils/` provides metrics, seeding helpers, and stratified splitting. Experiment pipelines live under `experiments/` and write outputs to `experiments/results/` (`results_TIMESTAMP.json`, plots); keep large raw dumps out of version control.

## Build, Test, and Development Commands
- Create an isolated environment and install dependencies:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
- End-to-end baseline (train + detect) from the repo root: `python experiments/run_experiment.py`.
- Multi-seed benchmark run: `python experiments/run_experiment_multiseed.py` (aggregates mean ± std metrics).
- Quick health check before commits: `python test_setup.py`; regenerate visual summaries with `python experiments/visualize_results.py`.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, descriptive `snake_case` for functions, and CapWords for classes; keep module names aligned with existing patterns (`*_agent.py`, `*_detector.py`).
- Use type hints on public APIs and keep numpy/torch tensors obvious with names like `states` or `q_values`.
- Document non-trivial logic with concise docstrings (see `DQNAgent.train_step`) and lightweight inline comments only where reasoning is subtle.
- Prefer deterministic helpers in `src/utils/seed_utils.py` whenever experiments or tests rely on randomness.

## Testing Guidelines
- `python test_setup.py` exercises imports, environment stepping, detectors, and metrics; run it after dependency or API changes.
- For new detectors or metrics, extend that script or add smoke tests under a `tests/` module that mimics the synthetic data patterns already used.
- Capture random seeds via `set_seed(...)` in experiments to make regressions reproducible, and log the generated JSON under `experiments/results/`.
- Validate visualization tooling by regenerating plots after major pipeline edits and spot-checking the saved PNGs.

## Commit & Pull Request Guidelines
- Follow existing history: short, imperative subject lines such as `add multi-seed runner` (~50 chars, lowercase) and keep commits scoped.
- Reference related result artifacts (e.g., `experiments/results/results_YYYYMMDD_HHMMSS.json`) or docs in the body so reviewers can trace outputs.
- PRs should summarize behavioral impact, list commands you ran (e.g., `python test_setup.py`), and attach key metrics or plots from `experiments/results/`.
- Mention any dataset or configuration requirements so reviewers can reproduce results.
