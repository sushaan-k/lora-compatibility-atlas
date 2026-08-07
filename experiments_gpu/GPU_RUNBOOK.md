# Queued GPU experiments

**STATUS (2026-07-10, evening).** All queued runs have fired.

- **Experiment B** ran on a Lightning RTX Pro 6000 (both m=8 substrates; ~6 min GPU).
  Qwen-8 single shared projector certifies sigma-adim_0.20=2 / _0.10=3 (residuals
  0.14/0.09); task-diverse TinyLlama-8 needs rank 5 at eps=0.20; (r+1)-core tables
  exact at r=3 on Qwen, errors < 1e-3 at r=2 on TinyLlama. Decision preservation
  VACUOUS on both charts (all 4/5-way queries feasible) — never cite it as evidence.
- **Experiment A** passed (see below): pooled r(L_answer, acc) = -0.72 vs +0.37 for
  L_input; integrated in both manuscripts.
- **m=50 shared chart** DONE (1,275-point singles-and-pairs fit, ~55 min GPU).
  NEGATIVE certification: one projector over all fifty adapters leaves the strict
  max-task residual at 0.48 (r=3), 0.29 (r=10); first < 0.20 at r=23, < 0.10 at
  r=42. Diversity gradient across shared charts: 2 (homogeneous m=8), 5 (diverse
  m=8), 23 (m=50). Certified compression is per-core, not library-wide. Jets +
  certification JSON: `results_public/shared_chart_decision_m50/`. Integrated in
  BOTH manuscripts (contribution (iii) + certification paragraph).
- **SOLVER FIX**: `decide.py` minimax_score previously accepted
  infeasible SLSQP restarts; on qwen8 this corrupted some released rho_full values
  (worst: -842196 vs true -50.1; one subset off by 19.27 with a subset-monotonicity
  violation). A per-restart feasibility guard is now in the script; all six
  qwen8/tinyllama decision JSONs were regenerated with it (paper claims survive:
  exactness now genuine at 9.6e-11; tinyllama values unchanged at 5.7e-13).
- **Interval certificate** (CPU, new): `scripts/certify.py` computes
  the exact per-task sup |q_t - q_t^r| by face enumeration and certifies decisions
  with margin > eps_S. Qwen m=8 r=3: settles 23% of all 211 subset queries at
  deficit 0; TinyLlama: certifies nothing (bound exceeds score spread). Zero
  certified violations in a 201-point tau sweep. In v2 as prop:interval_reduced.
- **CPU query timing** (new): `scripts/timing.py` — 13-46 ms per minimax
  query at m=8, ~0.6 s per five-way query at m=50 (d=49), microsecond core-table
  lookups (Apple M4 Pro).
- **Answer operating point** (CPU, new): `scripts/operating.py` — at
  tau_acc=0.9 (21/45 retained), answer loss recovers 48% recall at 71% precision in
  the top 30%; prompt loss 24%/36%, below the 47% base rate.
- **Experiment C** (answer-token jets, Mistral 5-adapter shared 4-simplex): DONE.
  Jet-predicted answer loss vs measured: r=0.81 (0.79 on the 25 off-design
  triples); AUROC 0.78 vs accuracy retention at tau_acc=0.9. Data:
  `results_public/answer_jets/`.
- **Experiment D** (m=8 answer-token screening panel): DONE (2026-07-11, ran on
  the Exea MI300X via Jupyter after the Lightning studio slept; ~1 GPU-hour).
  Eight Lots-of-LoRAs tasks (expA's five + SST-2 task363, PAWS task400, AG News
  task379), 36-point singles-and-pairs fit, all 28 pairs + 56 triples validated
  (240 rows). RESULTS: all 8 answer-token gains positive (2.1-11.0 nats, median
  5.7) vs 43/50 clamped on prompt-loss Qwen; subset screening vs answer-retention
  AUROC 0.95-0.98 central taus / 0.82 at tau=0.85 (18 infeasible); event-level
  jets vs accuracy retention 0.71-0.78 (replicates expC), measured 0.78-0.89;
  triples-only accuracy discrimination weak (0.53) — cause measured: exactly
  poised 36-pt design at d=7 drops r(pred,meas) to 0.52 from 0.81 at d=4/70 pts;
  denser design is the fix. sigma-adim_0.20=4, _0.10=6 (diverse). Reduced-jet
  preservation now NON-vacuous: 38-96% agreement at r=1..3 — diverse-chart
  compression fails end to end. Data: `results_public/answer_panel_m8/`;
  integrated in both manuscripts (abstract, contributions (v), experiments
  paragraph, limitations, conclusion). NOTE: the Qwen-50 community library
  cannot be relabelled (no task definitions with recoverable answers) — now
  stated in Limitations.

Two runs close the two remaining reviewer blockers. Both are staged here and fire
with a single command once a GPU is free (Exea MI300X or any CUDA box). Neither
touches the validated pipeline in `peft_atlas_lite/` or the released
`results_public/`; each writes to its own output directory and has a CPU-only
self-test that has already been run.

Environment (both): `torch`, `transformers`, `peft`, `numpy`, `scipy`. bf16
(`dtype: bfloat16` in the configs) since ROCm bitsandbytes is unreliable. The
helper `scratchpad/jup.py` drives the Exea Jupyter box if needed.

---

## Experiment B — shared-chart decision preservation (reviewer blocker: the 108x claim)

**What the reviewer asked.** The released `108x` is per-core: twenty five-adapter
cores were each fit on their own `d=4` chart with their own projector. That does
not certify one library-wide rank-`r` projector on one shared chart. Provide: one
shared chart, one shared projection, four- and five-way queries, the observed
error between the full score and the best `(r+1)`-core score, and the percentage of
threshold decisions preserved.

**Design.** Put `k` adapters (default 8) on one shared `(k-1)`-simplex. Fit one
affine-quadratic jet per adapter over that shared chart on a poised design. Take
one shared projector `Pi_r` from the top-`r` eigenspace of the shared mean Hessian.
For every 4- and 5-way query subset, compute the full obstruction score `rho_S`
from the fitted jets and the `(r+1)`-core score `rho_S^{(r+1)}` from the same jets
projected through `Pi_r`, then report `mean|rho_S - rho_S^{(r+1)}|`, the max, and
the fraction of `rho_S <= tau` decisions the projected score preserves over the
deployed `tau` grid.

**Run.**
```
# fit jets on the shared chart (GPU, ~0.5 MI300X-hour for k=8). step 0.25 -> 330 design points (N_d=36 min).
python experiments_gpu/decide.py fit \
    --config experiments_gpu/expB_qwen8_sharedchart.json --step 0.25 \
    --adapters "lhong4759_f68c5c87_934d_4399_b,adammandic87_3d7a58a6_4380_4e3,versil91_01a5620a_fb5f_4bab_94,taronklm_trained_model,versil91_e9f26c47_9b78_4e3f_81,versil91_8444c3be_8779_4769_a0,tuanna08go_6af2bb68_a65f_4331,duyphu_7aba420b_74ac_4209_b514" \
    --out results_public/shared_chart_decision

# decision preservation (CPU, seconds); sweep r over {1,2,3}
python experiments_gpu/decide.py decide \
    --jets results_public/shared_chart_decision/jets.npz --r 2 \
    --out results_public/shared_chart_decision
```
The 8 adapters and their `probes` block are pre-filled in the config (generated
from the real Qwen slice). Generic "conversation" probes are fine here: the test is
the Hessian geometry of the shared coefficient chart, not task accuracy. `--step`
controls the design density (needs >= N_d = 36 points at d=7); `--r` is the shared
rank. Confirm the CPU geometry first with `python experiments_gpu/decide.py selftest`.

**Acceptance / what goes in the paper.** Section "Measured effective rank
compression" currently says "certifying the shared projector ... is the direct
follow-up test." Replace that clause with the measured decision-preservation
fraction at the median `r`. If (as the per-core spectra predict) `rho_S^{(r+1)}`
preserves most decisions at `r=2`-`3`, the `108x` becomes a certified library-wide
reduction rather than a per-core one. Report the number honestly even if the
preservation is partial; the additive slack (`eps L_F D^2`) predicts where it breaks.

---

## Experiment A — answer-token retention labels (reviewer blocker: input-prompt loss != accuracy)

**What the reviewer asked.** The panels score loss on the input prompt, and the
paper's own probe shows input-prompt loss does not predict accuracy (Pearson
`+0.43`) while answer-token loss does (`-0.96`). Rerun the primary tables with
answer-token loss and an actual task metric, on tasks with recoverable answers.

**Design.** Generalize the single-task capability probe
(`scripts/probes.py`, which produced the `-0.96`) to a panel of
answer-bearing tasks. For each merge and each constituent task, measure three
quantities on the same held-out items: `L_input` (CE on the prompt, what the
panels use), `L_answer` (CE on the answer tokens, prompt masked), and `accuracy`
(exact-match generation). Fit the atlas jets on `L_input` and on `L_answer`, label
feasibility both ways, and report: AUROC of the atlas score against answer-token
retention vs against input-prompt retention, and the pooled correlation of each
retention loss with accuracy.

**Run.**
```
python experiments_gpu/panel.py \
    --config experiments_gpu/expA_panel.json \
    --out results_public/answer_token_panel
```
The config lists adapters paired with Lots-of-LoRAs task JSONs (definition +
input/output instances). The Mistral SNLI task (`task190_natural.json`) is already
on the box; add three-plus more answer-bearing tasks with adapters on the same base
so pair/triple merges exist. Start on Mistral-7B (strong base, clean answers), then
extend.

**Acceptance / what goes in the paper.** If answer-token retention AUROC stays high
(`~0.8`) and answer-token retention correlates with accuracy across the panel, the
Downstream and Capability-probe paragraphs upgrade from "one-line fix, shown on one
task" to "answer-token screening tracks accuracy across N tasks", and the abstract's
transfer sentence can name a panel-level number. If it does not hold at panel scale,
the honest current framing (loss-based labels + one-task probe with the prescribed
fix) stays and the negative is reported. Do not overwrite the input-prompt tables;
add the answer-token columns beside them.

---

## Order

Run B first: it is smaller, fully specified, CPU-cheap after the fit, and turns a
standing caveat into a measured number. A is the larger scientific bet (its outcome
is not guaranteed by existing evidence) and needs answer-bearing panel construction.
