# Code and data

Supplementary material for *The Compatibility Atlas: A Convex-Geometric Index for LoRA Libraries*. Every headline number in the paper recomputes from what is here.

## Layout

- `reproduce.py` — entry point; forwards to the runner in `peft_atlas_lite/`.
- `peft_atlas_lite/` — the screening runner, the shared-chart probe, and one config per panel under `configs/`. Configs carry adapter lists and seeds; weights are not redistributed.
- `scripts/` — CPU analysis. `panelci.py`, `fidelity.py`, and `cheap.py` assert or print that they reproduce the published values before doing anything else. `envelopes.py` takes `--carto/--gains/--out` so the same computation runs on any dense chart set. `timing.py` and `timing49.py` measure query latency at d=7 and d=49.
- `experiments_gpu/` — the GPU drivers, one per experiment, with panel configs and task files. `GPU_RUNBOOK.md` documents the runs.
- `results_public/` — every result directory the paper draws on, including the Qwen m=50 panels, the five smaller base panels, the Mistral answer-token results, the amortized scale series (`scale_m16`, `scale_m30`, `scale_m50`), the fixed-chart probes, and the maintenance records.

## Re-analysis notes

The `margins` column of every `results.csv` holds per-member margins measured on the held-out validation run, so the sign of its minimum reproduces the feasibility label. It must never be used as a screening-time feature. The fitted-side quantities are `predicted_score`, `primal_upper`, `dual_lower`, and `certificate_gap`; `scripts/baselines.py --feature_modes same_probe` shows the intended equal-access construction.

The four Qwen slice configs use short names; the `slice_id` strings in `results_public/qwen25_m50_balanced/results.csv` use the original run names, and each config's seed field gives the correspondence.

Six claim groups rest on artifacts that predate this release and are not included: the KnOTS comparison, the cosine-baseline cached slices, the coefficient-floor sweep, the license-family column of the manifest table, the TinyLlama cold-start GBM feature and fold specification, and the m=10 TinyLlama jets behind one effective-rank sentence. Their published values are unchanged from the original runs. The pair-vs-triple phenomena table's original pair-label join is likewise not recoverable, since 70 duplicated pair keys carry conflicting held-out labels across slices; the caption discloses this and the conclusion holds under every join variant.

## Hardware

GPU runs on a single AMD MI300X (192 GB HBM3), ROCm 7.0, bf16 weights. CPU analyses run on any x86_64 or arm64 host with the dependencies in `peft_atlas_lite/requirements.txt`.

## Licenses

Model and adapter references come from Hugging Face repositories reporting Apache-2.0 or MIT licenses at audit time. Per-adapter identifiers are in each config's `adapters` field. No private data, scraped user data, or human-subjects data are used.
