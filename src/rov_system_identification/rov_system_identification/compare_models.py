#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _finite_minmax(s: pd.Series, higher_is_better: bool = False) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    finite = x[np.isfinite(x)]
    if finite.empty:
        return pd.Series(np.full(len(x), np.nan), index=s.index)
    lo = float(finite.min()); hi = float(finite.max())
    if hi - lo < 1e-12:
        z = pd.Series(np.zeros(len(x)), index=s.index, dtype=float)
    else:
        z = (x - lo) / (hi - lo)
    if higher_is_better:
        z = 1.0 - z
    return z


def main(argv=None):
    ap = argparse.ArgumentParser(description="Robust model comparison for ROV identification results.")
    ap.add_argument("--root", default="~/rov_system_id_ws/ROV_system_idetification")
    args = ap.parse_args(argv)

    root = Path(os.path.expanduser(args.root)).resolve()
    results = root / "results"
    tables = results / "tables"
    reports = results / "reports"
    plots = results / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    raw_path = tables / "model_comparison_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}; run run_identification first.")

    df = pd.read_csv(raw_path, low_memory=False)
    df["model"] = df["model"].astype(str)
    ok = df["status"].astype(str).eq("ok") if "status" in df.columns else pd.Series(True, index=df.index)

    # Failed fits are excluded from ranking, never rewarded as if metrics were perfect.
    rank_df = df[ok].copy()
    for c in ["mean_rollout_rmse_1s", "mean_rollout_rmse_3s", "mean_rollout_rmse_5s", "mean_next_state_nrmse", "fit_seconds", "n_parameters"]:
        if c in rank_df.columns:
            rank_df[c] = pd.to_numeric(rank_df[c], errors="coerce")

    # Finite rollout metrics only. Missing requested horizons are penalized rather than
    # converted to a score of 1.0 (best). This is especially important for exact 5 s
    # segments with discrete ARX lag history.
    metric_specs = {
        "mean_rollout_rmse_5s": 0.45,
        "mean_rollout_rmse_3s": 0.25,
        "mean_rollout_rmse_1s": 0.15,
        "mean_next_state_nrmse": 0.10,
    }
    score_cols = []
    for col, weight in metric_specs.items():
        if col not in rank_df.columns:
            continue
        score_name = "score_" + col.replace("mean_", "").replace("_rmse", "").replace("_", "")
        # lower error => better score; NaN => worst (1.0)
        z = _finite_minmax(rank_df[col], higher_is_better=False)
        score = z.copy()
        score = score.fillna(1.0)
        # Ensure finite values remain in [0,1], smaller normalized error => lower cost.
        rank_df[score_name] = score
        score_cols.append((score_name, weight))

    # Complexity/runtime are weak tie-breakers, not primary predictive criteria.
    if "fit_seconds" in rank_df.columns:
        rank_df["score_runtime"] = _finite_minmax(rank_df["fit_seconds"], higher_is_better=False).fillna(1.0)
        score_cols.append(("score_runtime", 0.02))
    else:
        rank_df["score_runtime"] = 1.0
    if "n_parameters" in rank_df.columns:
        rank_df["score_complexity"] = _finite_minmax(rank_df["n_parameters"], higher_is_better=False).fillna(1.0)
        score_cols.append(("score_complexity", 0.03))
    else:
        rank_df["score_complexity"] = 1.0

    # Normalize active metric weights so missing columns do not distort the result.
    total_w = sum(w for _, w in score_cols)
    if total_w <= 0:
        rank_df["overall_score"] = np.nan
    else:
        rank_df["overall_score"] = sum(rank_df[c] * (w / total_w) for c, w in score_cols)

    # Coverage: number of requested horizons that are actually available.
    horizon_cols = [c for c in metric_specs if c in rank_df.columns]
    rank_df["rollout_coverage"] = rank_df[horizon_cols].notna().sum(axis=1) / max(1, len(horizon_cols))

    # Sort by overall predictive cost, then prefer higher coverage.
    rank_df = rank_df.sort_values(["overall_score", "rollout_coverage", "mean_next_state_nrmse"], ascending=[True, False, True], na_position="last").reset_index(drop=True)
    rank_df["rank"] = np.arange(1, len(rank_df) + 1)

    # Human-readable report.
    predictive = rank_df[np.isfinite(rank_df.get("mean_rollout_rmse_5s", pd.Series(dtype=float)))] if "mean_rollout_rmse_5s" in rank_df.columns else rank_df.iloc[0:0]
    report = {
        "ranking_method": "finite-metric weighted cost; missing metrics are penalized and failed fits are excluded",
        "weights": metric_specs,
        "tie_breakers": {"fit_seconds": 0.02, "n_parameters": 0.03},
        "requested_horizons_s": [1.0, 3.0, 5.0],
        "predictive_leader_if_5s_available": str(predictive.iloc[0]["model"]) if not predictive.empty else None,
        "balanced_leader": str(rank_df.iloc[0]["model"]) if not rank_df.empty else None,
        "models_ranked": int(len(rank_df)),
        "failed_models_excluded": df.loc[~ok, "model"].astype(str).tolist() if len(df) else [],
        "models": rank_df.to_dict(orient="records"),
    }

    out = df.copy()
    # Merge ranking columns back without dropping failed models.
    rank_cols = ["model", "score_rollout5", "score_rollout3", "score_nrmse", "score_runtime", "score_complexity", "rollout_coverage", "overall_score", "rank"]
    # Map the dynamic score names into the historical expected names.
    if "score_mean_rollout_rmse_5s" in rank_df:
        rank_df["score_rollout5"] = rank_df["score_mean_rollout_rmse_5s"]
    if "score_mean_rollout_rmse_3s" in rank_df:
        rank_df["score_rollout3"] = rank_df["score_mean_rollout_rmse_3s"]
    if "score_mean_next_state_nrmse" in rank_df:
        rank_df["score_nrmse"] = rank_df["score_mean_next_state_nrmse"]
    rank_cols = [c for c in rank_cols if c in rank_df.columns]
    ranked_full = out.merge(rank_df[rank_cols], on="model", how="left", suffixes=("", "_rank"))
    ranked_full.to_csv(tables / "model_comparison_ranked.csv", index=False)
    with open(reports / "model_selection_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=True)

    # Simple ranking plot.
    try:
        import matplotlib.pyplot as plt
        good = rank_df.head(12).copy().sort_values("overall_score", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(good["model"], good["overall_score"])
        ax.set_xlabel("Overall predictive cost (lower is better)")
        ax.set_title("ROV model comparison")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots / "model_ranking_overall.png", dpi=180)
        plt.close(fig)
    except Exception:
        pass

    print(rank_df[["rank", "model", "overall_score", "rollout_coverage", "mean_next_state_nrmse"]].to_string(index=False))
    print(f"\nSaved ranked table: {tables / 'model_comparison_ranked.csv'}")


if __name__ == "__main__":
    main()
