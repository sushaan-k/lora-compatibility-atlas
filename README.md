# Code and data

Supplementary material for *The Compatibility Atlas: A Convex-Geometric Index for LoRA Libraries*.

## Check it

```
python3 verify.py
```

recomputes fourteen headline numbers from the released data and compares each against the value printed in the paper. Add `--rerun` to regenerate the analysis outputs from the raw records first, which takes about ten minutes on a laptop. Dependencies are in `peft_atlas_lite/requirements.txt`; numpy, pandas, scipy, scikit-learn, and matplotlib cover everything on the CPU side.

## Layout

- `verify.py` — the reproduction check described above.
- `reproduce.py` — entry point for a fresh screening run; forwards to the runner in `peft_atlas_lite/`.
- `peft_atlas_lite/` — the screening runner, the shared-chart probe, and one config per panel under `configs/`. Configs carry adapter lists and seeds; model weights are not redistributed.
- `scripts/` — CPU analysis, one job per file. `panelci.py`, `fidelity.py`, and `cheap.py` assert or print that they reproduce published values before doing anything else. `envelopes.py` takes `--carto/--gains/--out` so the same computation runs on any dense chart set. `timing.py` and `timing49.py` measure query latency at d=7 and d=49. Files beginning `fig` redraw the paper's figures into `figs_png/`.
- `experiments_gpu/` — the GPU drivers, one per experiment, with panel configs and task files. `GPU_RUNBOOK.md` documents the runs.
- `results_public/` — every result directory the paper draws on. Names follow the experiment: `scale_m16`, `scale_m30`, `scale_m50` for the amortized index at each library size; `cartography` and `cartography_m50` for the dense charts on the original and fifty-adapter substrates, with `deployed_envelopes` and `deployed_envelopes_m50` holding their envelope analyses; `intervention_15q` and `adapter_core_intervention` for the two deployed coefficient tests; `qwen25_m50_balanced` and `qwen25_m50_facepairs` for the Qwen screening panel and its face-pair completion run.

## Paths and inputs

All paths in the configs and scripts are relative to this directory. Run
scripts from the package root, for example `python3 scripts/panelci.py`.

`experiments_gpu/tasks/` holds the task files for the Mistral m=8 and m=16
panels. The larger panels name task files that are not included; `scripts/tasks.py`
regenerates any of them from the public Natural Instructions release, applying
the same rule the originals used (shuffle the instance pool with seed 5, keep
the first N). That rule reproduces every included task file exactly.

Adapter weights are not redistributed. Configs whose `repo` field points into
`local_adapters/` refer to adapters trained for the paper; the Hugging Face
identifiers for all public adapters are in the configs themselves.
`scripts/probes.py` additionally reads a merge list that is not included, so it
is provided for inspection rather than direct execution.

## Re-analysis notes

The `margins` column of every `results.csv` holds per-member margins measured on the held-out validation run, so the sign of its minimum reproduces the feasibility label. It must never be used as a screening-time feature. The fitted-side quantities are `predicted_score`, `primal_upper`, `dual_lower`, and `certificate_gap`; `scripts/baselines.py --feature_modes same_probe` shows the intended equal-access construction.

The four Qwen slice configs use short names; the `slice_id` strings in `results_public/qwen25_m50_balanced/results.csv` use the original run names, and each config's seed field gives the correspondence.

## What is not here

Five claims rest on artifacts that predate this release and would need a fresh run to regenerate: the KnOTS screener comparison, the cosine-baseline table (which needs adapter weights we do not redistribute), the coefficient-floor sweep, the license-family column of the manifest table, and the m=10 TinyLlama sub-library behind one effective-rank sentence. Their published values are unchanged from the original runs.

One table is not fully determined by the released data: the pair-versus-triple phenomena counts depend on a join over 70 duplicated pair keys that carry conflicting held-out labels across slices. `scripts/hybrid.py` fixes the canonical variant, in which a conflicting key counts as feasible if any run labels it feasible, and the table's conclusion holds under every variant in the release.

## Hardware

GPU runs on a single AMD MI300X (192 GB HBM3), ROCm 7.0, bf16 weights. CPU analyses run on any x86_64 or arm64 host.

## Licenses

Model and adapter references come from Hugging Face repositories reporting Apache-2.0 or MIT licenses at audit time. Per-adapter identifiers are in each config's `adapters` field. No private data, scraped user data, or human-subjects data are used.
