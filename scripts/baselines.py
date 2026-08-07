"""GBM + MLP atlas-free baselines on the Qwen m=50 panel under three feature regimes: cold_start (arity + operato"""
import argparse, json
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def pair_graph_feats(df, leak_scores):
    pair_idx = (df[df.query_kind == "pair"]
                .set_index(["method", "slice_id", "adapter_key"])
                [["feasible", "heldout_score"]])
    pair_feas = pair_idx.feasible.to_dict()
    pair_score = pair_idx.heldout_score.to_dict()

    n_feas, mean_s, max_s = [], [], []
    for _, row in df.iterrows():
        if row.query_kind == "pair":
            n_feas.append(np.nan); mean_s.append(np.nan); max_s.append(np.nan)
            continue
        subs = ["+".join(p) for p in combinations(sorted(row.adapter_list), 2)]
        f = [pair_feas.get((row.method, row.slice_id, k)) for k in subs]
        s = [pair_score.get((row.method, row.slice_id, k)) for k in subs]
        f = [x for x in f if x is not None]
        s = [x for x in s if x is not None]
        n_feas.append(float(sum(f)) if f else np.nan)
        mean_s.append(float(np.mean(s)) if s else np.nan)
        max_s.append(float(np.max(s)) if s else np.nan)

    out = pd.DataFrame({"n_pair_feasible": n_feas}, index=df.index)
    if leak_scores:
        out["mean_pair_heldout"] = mean_s
        out["max_pair_heldout"] = max_s
    return out


def build_features(df, adapters, mode):
    if mode not in {"cold_start", "pair_aware", "full_history", "same_probe"}:
        raise ValueError(mode)

    feats = {
        "arity": df.arity.astype(float).to_numpy(),
        "is_triple": (df.query_kind == "triple").astype(float).to_numpy(),
    }
    for op in sorted(df.method.unique()):
        feats[f"op::{op}"] = (df.method == op).astype(float).to_numpy()
    for sl in sorted(df.slice_id.unique()):
        feats[f"slice::{sl}"] = (df.slice_id == sl).astype(float).to_numpy()
    if mode == "full_history":
        for a in adapters:
            feats[f"adapter::{a}"] = df.adapter_list.apply(
                lambda lst: int(a in lst)).to_numpy(float)
    X = pd.DataFrame(feats, index=df.index)
    if mode in ("pair_aware", "full_history"):
        X = pd.concat([X, pair_graph_feats(df, leak_scores=False).fillna(-1.0)],
                      axis=1)
    if mode == "same_probe":
        # Equal-access regime: the supervised model receives the quantities
        # the atlas computes from its calibration probes at screening time --
        # the predicted minimax deficit and the primal/dual certificate fields
        # of the same convex solve -- on top of cold-start metadata. The
        # `margins` column is excluded: it records per-member HELD-OUT margins
        # (label information), not fitted ones.
        X["atlas_predicted_score"] = df.predicted_score.astype(float).to_numpy()
        for col in ("primal_upper", "dual_lower", "certificate_gap"):
            X[f"probe::{col}"] = df[col].astype(float).fillna(-1.0).to_numpy()
    return X


def per_op(method, y, s_atlas, s_gbm, s_mlp):
    return {
        "method": method, "n": int(len(y)), "n_positive": int(y.sum()),
        "atlas": {"auroc": float(roc_auc_score(y, s_atlas)),
                  "auprc": float(average_precision_score(y, s_atlas))},
        "gbm_strong": {"auroc": float(roc_auc_score(y, s_gbm)),
                       "auprc": float(average_precision_score(y, s_gbm))},
        "mlp_strong": {"auroc": float(roc_auc_score(y, s_mlp)),
                       "auprc": float(average_precision_score(y, s_mlp))},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scores_csv", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feature_modes", nargs="+",
                    default=["cold_start", "pair_aware", "full_history"])
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    df["adapter_list"] = df.adapters.str.split("+").apply(sorted)
    df["adapter_key"] = df.adapter_list.apply(lambda lst: "+".join(lst))
    df["arity"] = df.adapter_list.apply(len)
    adapters = sorted({a for lst in df.adapter_list for a in lst})
    print(f"loaded {len(df)} rows, {len(adapters)} unique adapters,"
          f" feasibility rate {df.feasible.mean():.3f}")

    y = df.feasible.to_numpy(int)
    s_atlas = -df.predicted_score.to_numpy(float)
    groups = df.slice_id.to_numpy()
    n_slices = df.slice_id.nunique()
    gkf = GroupKFold(n_splits=n_slices)

    scores_by_mode = {}
    out = {"source": args.results, "seed": args.seed, "n_folds": n_slices,
           "modes": {}}
    for mode in args.feature_modes:
        print(f"\n=== feature mode: {mode} ===")
        X = build_features(df, adapters, mode=mode)
        print(f"  feature dim: {X.shape[1]}")
        s_gbm = np.full(len(df), np.nan)
        s_mlp = np.full(len(df), np.nan)

        for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
            held = np.unique(groups[te])[0]
            print(f"  fold {fold+1}/{n_slices}: held-out = {held} (n={len(te)})")
            gbm = GradientBoostingClassifier(
                n_estimators=500, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=args.seed)
            gbm.fit(X.iloc[tr], y[tr])
            s_gbm[te] = gbm.predict_proba(X.iloc[te])[:, 1]

            sc = StandardScaler().fit(X.iloc[tr])
            mlp = MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu",
                alpha=1e-3, learning_rate_init=1e-3, max_iter=300,
                early_stopping=True, validation_fraction=0.15,
                random_state=args.seed)
            mlp.fit(sc.transform(X.iloc[tr]), y[tr])
            s_mlp[te] = mlp.predict_proba(sc.transform(X.iloc[te]))[:, 1]

        op_rows = []
        for op in sorted(df.method.unique()):
            m = (df.method == op).to_numpy()
            op_rows.append(per_op(op, y[m], s_atlas[m], s_gbm[m], s_mlp[m]))
        out["modes"][mode] = {"feature_dim": int(X.shape[1]),
                              "per_operator": op_rows}
        scores_by_mode[mode] = {"gbm": s_gbm, "mlp": s_mlp}

        print(f"\n  {'method':<18} {'atlas':>8} {'gbm':>8} {'mlp':>8}"
              f"   {'gbm-atlas':>10} {'mlp-atlas':>10}")
        for r in op_rows:
            a = r["atlas"]["auroc"]; g = r["gbm_strong"]["auroc"]; m = r["mlp_strong"]["auroc"]
            print(f"  {r['method']:<18} {a:>8.3f} {g:>8.3f} {m:>8.3f}"
                  f"   {g-a:>+10.3f} {m-a:>+10.3f}")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")

    if args.scores_csv:
        s = df[["method", "slice_id", "adapter_key", "query_kind", "feasible"]].copy()
        s["atlas_score"] = s_atlas
        for mode in args.feature_modes:
            s[f"gbm_{mode}_score"] = scores_by_mode[mode]["gbm"]
            s[f"mlp_{mode}_score"] = scores_by_mode[mode]["mlp"]
        s.to_csv(args.scores_csv, index=False)
        print(f"wrote {args.scores_csv}")


if __name__ == "__main__":
    main()
