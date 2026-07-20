"""Adapter-cluster sign-flip permutation test on atlas vs each learned
baseline. Drops one adapter at a time, recomputes paired AUROC delta,
runs a two-sided sign-flip test over the resulting LOO diffs."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score


def safe_auroc(y, s):
    if y.sum() in (0, len(y)):
        return float("nan")
    return float(roc_auc_score(y, s))


def loo_diffs(df, score_a, score_b, adapters):
    out = []
    for a in adapters:
        mask = ~df.adapter_key.str.split("+").apply(lambda lst: a in lst)
        sub = df[mask]
        if len(sub) < 30 or sub.feasible.nunique() < 2:
            continue
        aa = safe_auroc(sub.feasible.to_numpy(int), sub[score_a].to_numpy(float))
        bb = safe_auroc(sub.feasible.to_numpy(int), sub[score_b].to_numpy(float))
        if not (np.isnan(aa) or np.isnan(bb)):
            out.append(aa - bb)
    return np.array(out)


def sign_flip_p(diffs, n_perm, rng):
    if len(diffs) == 0:
        return float("nan")
    obs = float(diffs.mean())
    hits = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(diffs))
        if abs(float((diffs * signs).mean())) >= abs(obs):
            hits += 1
    return (hits + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.scores)
    rng = np.random.default_rng(args.seed)
    print(f"loaded {len(df)} rows, methods = {sorted(df.method.unique())}")

    learned_cols = [c for c in df.columns
                    if c.endswith("_score") and c != "atlas_score"]
    print(f"learned columns: {learned_cols}")

    adapters = sorted({a for k in df.adapter_key for a in k.split("+")})
    print(f"{len(adapters)} unique adapters")

    out = {"source": args.scores, "n_perm": args.n_perm, "seed": args.seed,
           "per_operator": []}

    print(f"\n{'method':<18} {'baseline':<28} {'full diff':>10}"
          f" {'LOO median':>12} {'LOO range':>20} {'p_perm':>8} {'n_clust':>8}")
    for op in sorted(df.method.unique()):
        sub_op = df[df.method == op]
        block = {"method": op, "n": int(len(sub_op)),
                 "n_positive": int(sub_op.feasible.sum()),
                 "comparisons": []}
        full_atlas = safe_auroc(sub_op.feasible.to_numpy(int),
                                 sub_op.atlas_score.to_numpy(float))
        for col in learned_cols:
            valid = sub_op.dropna(subset=[col])
            if len(valid) < 50:
                continue
            full_learned = safe_auroc(valid.feasible.to_numpy(int),
                                       valid[col].to_numpy(float))
            full_diff = full_atlas - full_learned
            diffs = loo_diffs(valid, "atlas_score", col, adapters)
            if len(diffs):
                med = float(np.median(diffs))
                lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
            else:
                med = lo = hi = float("nan")
            p = sign_flip_p(diffs, args.n_perm, rng)
            block["comparisons"].append({
                "baseline": col,
                "atlas_auroc_full": full_atlas,
                "learned_auroc_full": full_learned,
                "diff_full": full_diff,
                "loo_median_diff": med,
                "loo_2.5_diff": lo, "loo_97.5_diff": hi,
                "n_clusters": int(len(diffs)),
                "p_sign_flip": p,
            })
            print(f"{op:<18} {col:<28} {full_diff:>+10.3f}"
                  f" {med:>+12.3f} [{lo:>+6.3f}, {hi:>+6.3f}]"
                  f" {p:>8.4f} {len(diffs):>8d}")
        out["per_operator"].append(block)

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
