from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    DOF_NAMES,
    add_protocol_split,
    ensure_dirs,
    load_config,
    load_raw_data,
    normalized_rmse,
    r2_score_vec,
    save_json,
)
from .run_identification import _predict_one_step, _rollout


def _excitation_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["phase"].astype(str).str.lower().isin(["pulse", "broadband"])
    ].copy()


def _load_saved_models(root: Path, enabled: list[str]) -> dict[str, object]:
    models = {}
    missing = []
    for name in enabled:
        model_path = root / "results" / "models" / name / "model.pkl"
        if not model_path.exists():
            missing.append(name)
            continue
        try:
            with open(model_path, "rb") as f:
                models[name] = pickle.load(f)
        except Exception as exc:
            print(f"WARNING: could not load {name}: {exc}")

    if missing:
        print("WARNING: missing saved models:")
        for name in missing:
            print(f"  {name}")
    return models


def _evaluate_next_state_by_dof(model, df: pd.DataFrame):
    """Evaluate the already-fitted model exactly through the global one-step path."""
    true_blocks = []
    pred_blocks = []

    for _, g in df.groupby("segment_id", sort=False):
        if len(g) < 5:
            continue
        try:
            result = _predict_one_step(model, g)
        except Exception as exc:
            print(f"WARNING: one-step evaluation failed for {getattr(model, 'name', type(model).__name__)}: {exc}")
            continue
        if result is None:
            continue
        yt, yp, _dt = result
        if len(yt) == 0:
            continue
        true_blocks.append(np.asarray(yt, dtype=float))
        pred_blocks.append(np.asarray(yp, dtype=float))

    if not true_blocks:
        return None

    yt = np.vstack(true_blocks)
    yp = np.vstack(pred_blocks)
    rmse = np.sqrt(np.mean((yt - yp) ** 2, axis=0))
    nrmse = normalized_rmse(yt, yp)
    r2 = r2_score_vec(yt, yp)
    return {
        "true": yt,
        "pred": yp,
        "rmse": np.asarray(rmse, dtype=float),
        "nrmse": np.asarray(nrmse, dtype=float),
        "r2": np.asarray(r2, dtype=float),
    }


def _evaluate_rollout_by_dof(model, df: pd.DataFrame, horizons: list[float]):
    results = {}
    for horizon in horizons:
        true_blocks = []
        pred_blocks = []

        for _, g in df.groupby("segment_id", sort=False):
            if len(g) < 5:
                continue
            try:
                result = _rollout(model, g, float(horizon))
            except Exception:
                result = None
            if result is None:
                continue
            yt, yp, _dt = result
            if len(yp) == 0:
                continue
            true_blocks.append(np.asarray(yt, dtype=float))
            pred_blocks.append(np.asarray(yp, dtype=float))

        key = f"{float(horizon):g}"
        if not true_blocks:
            results[key] = None
            continue

        yt = np.vstack(true_blocks)
        yp = np.vstack(pred_blocks)
        rmse = np.sqrt(np.mean((yt - yp) ** 2, axis=0))
        results[key] = rmse

    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Select the best saved identification model independently for each "
            "ROV DOF using validation data, then evaluate the selected models on test data."
        )
    )
    parser.add_argument("--root", default="~/rov_system_id_ws/ROV_system_idetification")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    root = Path(os.path.expanduser(args.root)).resolve()
    cfg_path = (
        Path(os.path.expanduser(args.config)).resolve()
        if args.config
        else root / "config" / "identification.yaml"
    )
    cfg = load_config(cfg_path)
    dirs = ensure_dirs(root)

    # The global run saves prepared_dataset_with_splits.csv BEFORE calling
    # differentiate_group(), so that CSV does not contain u_filtered...r_filtered.
    # Reproduce the same evaluation preparation here from the raw 50 Hz data.
    data = load_raw_data(dirs["raw"])
    data = add_protocol_split(
        data,
        cfg["analysis"].get("train_max_amplitude", 80),
        cfg["analysis"].get("validation_amplitude", 90),
        cfg["analysis"].get("test_amplitude", 100),
    )

    from .common import differentiate_group

    def _prepare_eval_split(df: pd.DataFrame) -> pd.DataFrame:
        d = _excitation_only(df).copy()
        if d.empty:
            return d
        return differentiate_group(
            d,
            time_col="phase_elapsed_s",
            window=int(cfg["analysis"].get("savgol_window", 21)),
            order=int(cfg["analysis"].get("savgol_order", 3)),
            group_col="segment_id",
        )

    val = _prepare_eval_split(
        data[data["split"].astype(str).str.lower() == "validation"]
    )
    test = _prepare_eval_split(
        data[data["split"].astype(str).str.lower() == "test"]
    )

    enabled = [str(x) for x in cfg["analysis"].get("enable_models", [])]
    models = _load_saved_models(root, enabled)
    if not models:
        raise RuntimeError(
            "No saved models were found. Run the identification pipeline before per-DOF selection."
        )

    # Validation selection uses the same one-step next-state evaluation path as
    # compare_models/run_identification, rather than independently refitting a
    # different set of models.
    validation_rows = []
    validation_results = {}
    for name, model in models.items():
        result = _evaluate_next_state_by_dof(model, val)
        if result is None:
            continue
        validation_results[name] = result
        for j, dof in enumerate(DOF_NAMES):
            validation_rows.append(
                {
                    "model": name,
                    "dof": dof,
                    "validation_rmse": float(result["rmse"][j]),
                    "validation_nrmse": float(result["nrmse"][j]),
                    "validation_r2": float(result["r2"][j]),
                }
            )

    scores = pd.DataFrame(validation_rows)
    if scores.empty:
        raise RuntimeError("No models produced valid validation predictions.")

    selected = {}
    for dof in DOF_NAMES:
        g = scores[scores["dof"] == dof].copy()
        g = g[np.isfinite(g["validation_nrmse"])]
        if g.empty:
            raise RuntimeError(f"No valid model available for DOF {dof}")
        g = g.sort_values(["validation_nrmse", "validation_rmse"])
        selected[dof] = str(g.iloc[0]["model"])

    # Evaluate only the selected model for each DOF on the held-out test set.
    test_results = {}
    rollout_horizons = [float(x) for x in cfg["analysis"].get("rollout_seconds", [1.0, 3.0, 5.0])]
    selected_models = sorted(set(selected.values()))
    for name in selected_models:
        model = models[name]
        result = _evaluate_next_state_by_dof(model, test)
        if result is None:
            continue
        rollout = _evaluate_rollout_by_dof(model, test, rollout_horizons)
        test_results[name] = {"next_state": result, "rollout_rmse": rollout}

    rows = []
    for j, dof in enumerate(DOF_NAMES):
        model_name = selected[dof]
        result = test_results.get(model_name)
        if result is None:
            rows.append(
                {
                    "DOF": dof,
                    "selected_model": model_name,
                    "test_RMSE": np.nan,
                    "test_NRMSE": np.nan,
                    "test_R2": np.nan,
                }
            )
            continue

        row = {
            "DOF": dof,
            "selected_model": model_name,
            "test_RMSE": float(result["next_state"]["rmse"][j]),
            "test_NRMSE": float(result["next_state"]["nrmse"][j]),
            "test_R2": float(result["next_state"]["r2"][j]),
        }
        for horizon in rollout_horizons:
            key = f"{float(horizon):g}"
            rr = result["rollout_rmse"].get(key)
            row[f"test_rollout_RMSE_{key}s"] = float(rr[j]) if rr is not None else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    report = {
        "selection_metric": "validation_next_state_nrmse",
        "selection_source": "saved_global_models",
        "evaluation_path": "same _predict_one_step/_rollout used by global identification",
        "enabled_models": enabled,
        "loaded_models": sorted(models.keys()),
        "selected_model_per_dof": selected,
        "validation_scores": validation_rows,
        "test_summary": rows,
    }
    save_json(dirs["reports"] / "per_dof_model_selection.json", report)
    scores.to_csv(dirs["tables"] / "per_dof_validation_scores.csv", index=False)

    summary.to_csv(dirs["tables"] / "per_dof_test_scores.csv", index=False)


if __name__ == "__main__":
    main()
