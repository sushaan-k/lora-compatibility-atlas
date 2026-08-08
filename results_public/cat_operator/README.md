# Update-space vs factor-space merging at m=16

Paired run on the sixteen-adapter Mistral panel: identical panel, items, and
protocol, differing only in PEFT `combination_type`.

- `cat/` — exact weighted sum of updates, the gauge-invariant merge map.
- `linear/` — factor-space combination, the operator used elsewhere in the paper.

Both arms were fit on the same 136-point singles-and-pairs design and validated
on the same 240 pair and triple merges. Run with `--nshards` workers splitting
the design grid by index (fit) and whole merges (validation); per-point
computation is untouched, so shard counts do not affect the numbers.

Batching was tested and rejected: with sequences of identical length and no
padding, batched bf16 matmuls shift the per-item loss by up to 1.8e-1 nats,
which is large next to the fidelity differences being measured. All numbers
here are batch size 1.

Five of the sixteen task files were reconstructed from the public Natural
Instructions release: `random.Random(5).shuffle` over the instance pool, then
the first N. That rule reproduces all fifteen surviving task files exactly
(N is 400 or 800 depending on the task), and the protocol reads only the first
30 instances.
