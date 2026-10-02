# Uncertainty-aware MoE diagnostic engine with agentic decision support (BRACS)

Code accompanying the manuscript *"Toward reliable early breast lesion detection:
a state-of-the-art, uncertainty-aware diagnostic engine with agentic decision
support in H&E histopathology"* (under review). Released for peer review under
the **PolyForm Noncommercial License 1.0.0** (see `LICENSE`).

## Contents

| Path | What it is |
|---|---|
| `src/breastpath_copilot/models/moe.py`, `training/moe_trainer.py` | Mixture-of-experts over pathology foundation-model embeddings (sparse top-2 gate, five-seed deep ensemble) |
| `src/breastpath_copilot/training/calibration.py` | Temperature scaling and ECE |
| `src/breastpath_copilot/core/retrieval_index.py` | Patient-disjoint retrieval index |
| `src/breastpath_copilot/api/tools.py`, `mcp_server.py` | Typed tools and the Model Context Protocol server |
| `src/breastpath_copilot/agents/` | Agent harness (`runner.py`), tool registry, the four task definitions (`tasks.py`), LLM client |
| `scripts/` | Scripts that reproduce the reported results (mapping below) |
| `artifacts/agent/` | Deployed MoE checkpoint and retrieval index used by the live agent |
| `artifacts/manifests/` | Case lists with the official BRACS train/val/test split (image paths relative to `data/`) |
| `artifacts/results/` | Result files behind the paper's tables and figures |
| `artifacts/trajectories/` | Full tool-call trajectories of the triage, evidence-report and discrepancy-resolution runs (2,772 runs); cross-cohort routing runs were logged as summary results only (`artifacts/results/agent_tasks2.txt`) |
| `tests/` | Unit tests (`pytest`, 148 tests) |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[fm,mcp,agents]"
pytest -q
```

Python >= 3.10. A CUDA GPU is recommended for embedding extraction; MoE training
on cached embeddings takes minutes on a single GPU.

## Data

Datasets are public and are **not redistributed**. Obtain them from their
maintainers and place them under `data/` (e.g. `data/BRACS/BRACS_RoI/...`), matching the relative paths in `artifacts/manifests/`:

- **BRACS** (official patient-disjoint train/val/test split): https://www.bracs.icar.cnr.it
- **BACH** (ICIAR 2018 Grand Challenge)
- **BreaKHis**

Foundation models UNI and Virchow2 are gated on Hugging Face: accept their terms
and set `HF_TOKEN` in your environment. Paths inside the trajectory files have
been replaced with the placeholder `<BRACS_ROOT>`.

## Reproducing the results

| Paper result | Script |
|---|---|
| Foundation-model embedding extraction | `scripts/extract_fm_embeddings.py` |
| Domain-adapted Phikon-v2 expert | `scripts/finetune_phikon.py` |
| Headline 7/3-class results, per-class F1, confusion matrix, ECE | `scripts/dump_moe_perclass.py` |
| Macro-AUROC | `scripts/benchmark_auroc.py` |
| Routing / expert ablations | `scripts/train_moe.py` |
| Ensemble-size curve | `scripts/ensemble_size_curve.py` |
| Risk-coverage (selective prediction) | `scripts/compute_selective.py` |
| Uncertainty-based deferral policies | `scripts/compute_uncertainty_deferral.py` |
| Conformal prediction sets | `scripts/compute_conformal.py` |
| Out-of-distribution detection (BACH, BreaKHis) | `scripts/compute_ood.py` |
| BACH 5-fold cross-validation | `scripts/bach_moe_kfold.py` |
| Agent checkpoint and retrieval index | `scripts/train_agent_moe_ckpt.py`, `scripts/build_agent_retrieval_index.py` |
| Live agent: triage, evidence report | `scripts/run_agent_eval.py --task {triage,evidence}` |
| Live agent: cross-cohort routing, discrepancy resolution | `scripts/run_agent_tasks2.py --only {routing,discrepancy}` |

Scripts that take options accept `--help`. Scripts write outputs to `artifacts/q1_results/`; compare them with the reference copies in `artifacts/results/`. Scripts expect cached embeddings
under `artifacts/fm_cache` and manifests under `artifacts/manifests`.

### Live agent evaluation

The agent backends are open models served locally through Ollama's
OpenAI-compatible endpoint (no proprietary API):

```bash
ollama pull gemma4:31b
ollama pull qwen3.8:27b
PYTHONPATH=src python scripts/run_agent_eval.py --task triage   --model gemma4:31b --n 570 --max-steps 6
PYTHONPATH=src python scripts/run_agent_eval.py --task evidence --model gemma4:31b --n 570 --max-steps 6
PYTHONPATH=src python scripts/run_agent_tasks2.py --only routing     --model gemma4:31b --per 200 --max-steps 6
PYTHONPATH=src python scripts/run_agent_tasks2.py --only discrepancy --model gemma4:31b --n 246 --max-steps 10 --max-tokens 3000
```

Repeat with `--model qwen3.8:27b`. The recorded trajectories of the reported runs
are in `artifacts/trajectories/`, and the summary lines in
`artifacts/results/agent_triage_eval.txt` and `artifacts/results/agent_tasks2.txt`.

## License

PolyForm Noncommercial License 1.0.0. Noncommercial use (research, teaching,
peer review) is permitted; commercial use requires a separate license from the
authors.
