"""Per-operator (predicted, heldout) residual stats: Spearman, regression slope, signed residual mean, fraction o"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_public/qwen25_m50_balanced/results.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.results)
    out = {"source": args.results, "per_operator": []}

    for op in sorted(df.method.unique()):
        sub = df[df.method == op]
        p = sub.predicted_score.to_numpy(float)
        h = sub.heldout_score.to_numpy(float)
        rho_s, _ = spearmanr(p, h)
        rho_p, _ = pearsonr(p, h)
        slope, intercept = np.polyfit(p, h, 1)
        resid = p - h

        for qk in ["pair", "triple"]:
            ssub = sub[sub.query_kind == qk]
            if len(ssub) == 0:
                continue
            ps = ssub.predicted_score.to_numpy(float)
            hs = ssub.heldout_score.to_numpy(float)
            rs = ps - hs
            out["per_operator"].append({
                "method": op, "query_kind": qk, "n": int(len(ssub)),
                "feasible_rate": float(ssub.feasible.mean()),
                "spearman_atlas_heldout": float(spearmanr(ps, hs)[0])
                    if len(ssub) > 2 else None,
                "mean_residual": float(rs.mean()),
                "median_residual": float(np.median(rs)),
                "abs_residual_mean": float(np.abs(rs).mean()),
                "atlas_overestimates_pct": float((rs > 0).mean()),
            })

        out["per_operator"].append({
            "method": op, "query_kind": "all", "n": int(len(sub)),
            "feasible_rate": float(sub.feasible.mean()),
            "spearman_atlas_heldout": float(rho_s),
            "pearson_atlas_heldout": float(rho_p),
            "regression_slope": float(slope),
            "regression_intercept": float(intercept),
            "mean_residual": float(resid.mean()),
            "abs_residual_mean": float(np.abs(resid).mean()),
            "atlas_overestimates_pct": float((resid > 0).mean()),
        })
        print(f"{op:18s} Spearman={rho_s:+.3f} slope={slope:+.3f}"
              f" mean_resid={resid.mean():+.2f}"
              f" overest={float((resid>0).mean()):.2%}")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
