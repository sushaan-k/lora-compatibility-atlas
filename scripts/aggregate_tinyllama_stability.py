"""Concatenate the 5,720-query TinyLlama stability sweep into one DataFrame
and rerun the supervised-baseline comparison (cold_start / pair_aware /
full_history) under leave-one-cell-out CV."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_public/tinyllama_stability")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scores_csv", default=None)
    args = ap.parse_args()

    cells = sorted(Path(args.root, "runs").glob("tau*_seed*"))
    print(f"found {len(cells)} cells")
    dfs = []
    for c in cells:
        f = c / "results.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d["cell_id"] = c.name
        dfs.append(d)
    if not dfs:
        print("no data"); return
    df = pd.concat(dfs, ignore_index=True)
    df["adapter_list"] = df["adapters"].str.split("+").apply(sorted)
    df["adapter_key"] = df["adapter_list"].apply(lambda l: "+".join(l))
    df["arity"] = df["adapter_list"].apply(len)
    print(f"{len(df)} rows from {df.cell_id.nunique()} cells, "
          f"feasibility {df.feasible.mean():.3f}")

    adapters = sorted({a for lst in df.adapter_list for a in lst})
    print(f"{len(adapters)} unique adapters")

    # Feature builds
    def build_feats(df, mode):
        feats = {
            "arity": df.arity.to_numpy(float),
            "is_triple": (df.query_kind == "triple").astype(float).to_numpy(),
        }
        for op in sorted(df.method.unique()):
            feats[f"op::{op}"] = (df.method == op).astype(float).to_numpy()
        for cell in sorted(df.cell_id.unique()):
            feats[f"cell::{cell}"] = (df.cell_id == cell).astype(float).to_numpy()
        if mode == "full_history":
            for a in adapters:
                feats[f"a::{a}"] = df.adapter_list.apply(lambda l: int(a in l)).to_numpy(float)
        fm = pd.DataFrame(feats, index=df.index)
        if mode in ("pair_aware", "full_history"):
            pair_idx = (df[df.query_kind == "pair"]
                        .set_index(["method", "cell_id", "adapter_key"])["feasible"]
                        .to_dict())
            n_pf = []
            for _, r in df.iterrows():
                if r.query_kind == "pair":
                    n_pf.append(np.nan); continue
                subs = ["+".join(p) for p in combinations(sorted(r.adapter_list), 2)]
                vs = [pair_idx.get((r.method, r.cell_id, k)) for k in subs]
                vs = [v for v in vs if v is not None]
                n_pf.append(float(sum(vs)) if vs else np.nan)
            fm["n_pair_feasible"] = pd.Series(n_pf, index=df.index).fillna(-1.0)
        return fm

    y = df.feasible.to_numpy(int)
    s_atlas = -df.predicted_score.to_numpy(float)
    groups = df.cell_id.to_numpy()
    gkf = GroupKFold(n_splits=min(20, df.cell_id.nunique()))

    out = {"source": args.root, "n_rows": int(len(df)),
           "n_cells": int(df.cell_id.nunique()), "modes": {}}
    scores_oof: dict[str, dict[str, np.ndarray]] = {}
    for mode in ["cold_start", "pair_aware", "full_history"]:
        print(f"\n=== {mode} ===")
        X = build_feats(df, mode)
        print(f"  feature dim: {X.shape[1]}")
        s_g = np.full(len(df), np.nan); s_m = np.full(len(df), np.nan)
        for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
            gbm = GradientBoostingClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=0)
            gbm.fit(X.iloc[tr], y[tr])
            s_g[te] = gbm.predict_proba(X.iloc[te])[:, 1]
            sc = StandardScaler().fit(X.iloc[tr])
            mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                max_iter=200, early_stopping=True,
                                validation_fraction=0.15, random_state=0)
            mlp.fit(sc.transform(X.iloc[tr]), y[tr])
            s_m[te] = mlp.predict_proba(sc.transform(X.iloc[te]))[:, 1]
        per_op = []
        for op in sorted(df.method.unique()):
            mask = (df.method == op).to_numpy()
            ya = y[mask]
            if ya.sum() in (0, len(ya)):
                continue
            per_op.append({
                "method": op, "n": int(len(ya)), "n_pos": int(ya.sum()),
                "atlas_auroc": float(roc_auc_score(ya, s_atlas[mask])),
                "gbm_auroc": float(roc_auc_score(ya, s_g[mask])),
                "mlp_auroc": float(roc_auc_score(ya, s_m[mask])),
                "atlas_auprc": float(average_precision_score(ya, s_atlas[mask])),
                "gbm_auprc": float(average_precision_score(ya, s_g[mask])),
                "mlp_auprc": float(average_precision_score(ya, s_m[mask])),
            })
            r = per_op[-1]
            print(f"  {op:18s} atlas {r['atlas_auroc']:.3f}  gbm {r['gbm_auroc']:.3f}  mlp {r['mlp_auroc']:.3f}"
                  f"   diff_gbm={r['atlas_auroc']-r['gbm_auroc']:+.3f}")
        out["modes"][mode] = {"feature_dim": int(X.shape[1]), "per_operator": per_op}
        scores_oof[mode] = {"gbm": s_g, "mlp": s_m}

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    if args.scores_csv:
        s = df[["method", "cell_id", "adapter_key", "query_kind", "feasible"]].copy()
        s["atlas_score"] = s_atlas
        for m in scores_oof:
            s[f"gbm_{m}_score"] = scores_oof[m]["gbm"]
            s[f"mlp_{m}_score"] = scores_oof[m]["mlp"]
        s.to_csv(args.scores_csv, index=False)
        print(f"wrote {args.scores_csv}")


if __name__ == "__main__":
    main()
