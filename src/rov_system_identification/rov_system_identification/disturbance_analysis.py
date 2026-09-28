from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import DOF_NAMES, add_protocol_split, differentiate_group, ensure_dirs, load_config, load_raw_data, save_json


FX_EPS = 1e-9


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 4:
        return float("nan")
    x = x[m]
    y = y[m]
    sx = np.std(x)
    sy = np.std(y)
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _robust_median(values: np.ndarray, mad_scale: float = 3.5):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), values, np.array([], dtype=bool)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad < 1e-12:
        mask = np.ones(values.shape, dtype=bool)
    else:
        sigma = 1.4826 * mad
        mask = np.abs(values - med) <= mad_scale * sigma
    inliers = values[mask]
    return float(np.median(inliers)) if inliers.size else med, inliers, mask


def _individual_fx_slopes(data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Estimate early transient surge-acceleration slopes for +/- Fx pulses."""
    a = cfg.get("analysis", {})
    d = data.copy()
    d = d[
        d["protocol"].astype(str).str.lower().eq("individual")
        & d["phase"].astype(str).str.lower().eq("pulse")
    ].copy()
    if d.empty:
        return pd.DataFrame()

    if "experiment_id" not in d.columns:
        d["experiment_id"] = d["segment_id"].astype(str)
    d = differentiate_group(
        d,
        time_col="phase_elapsed_s",
        window=int(a.get("savgol_window", 21)),
        order=int(a.get("savgol_order", 3)),
        group_col="segment_id",
    )

    rows = []
    early_start = float(a.get("tether_early_start_s", 0.05))
    early_end = float(a.get("tether_early_end_s", 0.60))
    for seg_id, g in d.groupby("segment_id", sort=False):
        cmd = pd.to_numeric(g.get("Fx_cmd"), errors="coerce").to_numpy(float)
        if cmd.size == 0:
            continue
        active = []
        for c in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]:
            v = pd.to_numeric(g.get(f"{c}_cmd"), errors="coerce").to_numpy(float)
            if v.size and np.nanmax(np.abs(v)) > FX_EPS:
                active.append(c)
        if active != ["Fx"]:
            continue
        amplitude = float(abs(cmd[np.nanargmax(np.abs(cmd))]))
        if amplitude <= FX_EPS:
            continue
        t = pd.to_numeric(g["phase_elapsed_s"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(g["u_filtered"], errors="coerce").to_numpy(float)
        mask = np.isfinite(t) & np.isfinite(y) & (t >= early_start) & (t <= early_end)
        if np.sum(mask) < 5:
            continue
        try:
            slope, intercept = np.polyfit(t[mask], y[mask], 1)
        except Exception:
            continue
        rows.append({
            "segment_id": str(seg_id),
            "source_file": str(g["source_file"].iloc[0]) if "source_file" in g.columns else "",
            "session_id": str(g["session_id"].iloc[0]) if "session_id" in g.columns else "",
            "amplitude_N": amplitude,
            "command_N": float(np.nanmedian(cmd)),
            "sign": int(np.sign(np.nanmedian(cmd))),
            "early_slope_mps2": float(slope),
            "early_intercept_mps": float(intercept),
            "start_x_m": float(pd.to_numeric(g["x"], errors="coerce").iloc[0]) if "x" in g.columns else np.nan,
            "start_y_m": float(pd.to_numeric(g["y"], errors="coerce").iloc[0]) if "y" in g.columns else np.nan,
            "start_z_m": float(pd.to_numeric(g["z"], errors="coerce").iloc[0]) if "z" in g.columns else np.nan,
            "repetition": int(g["repetition"].iloc[0]) if "repetition" in g.columns else 1,
        })
    return pd.DataFrame(rows)


def estimate_fx_tether_bias(data: pd.DataFrame, cfg: dict, split: str = "train") -> dict[str, Any]:
    """Estimate approximately constant X-direction tether disturbance.

    For paired +Fx/-Fx trials at the same amplitude:
        gain_i = (s_plus - s_minus) / (2 F)
        bias_i = (s_plus + s_minus) / 2

    The zero-acceleration holding force is F_hold = -bias/gain.  The associated
    estimated tether force is -F_hold.  The estimator is deliberately based on
    symmetric pulses so the common external bias is separated from the input gain.
    """
    train = data[data["split"].astype(str).eq(str(split))].copy() if "split" in data.columns else data.copy()
    slopes = _individual_fx_slopes(train, cfg)
    if slopes.empty:
        raise ValueError("No individual +/-Fx pulse slopes available for tether-bias estimation")

    pair_rows = []
    group_cols = ["source_file", "amplitude_N", "repetition"]
    for key, g in slopes.groupby(group_cols, dropna=False, sort=True):
        pos = g[g["sign"] > 0]["early_slope_mps2"]
        neg = g[g["sign"] < 0]["early_slope_mps2"]
        if pos.empty or neg.empty:
            continue
        s_plus = float(np.median(pos))
        s_minus = float(np.median(neg))
        amp = float(key[1])
        gain = (s_plus - s_minus) / (2.0 * amp)
        bias = 0.5 * (s_plus + s_minus)
        hold = -bias / gain if np.isfinite(gain) and abs(gain) > 1e-12 else np.nan
        starts = g[["start_x_m", "start_y_m", "start_z_m"]].mean(numeric_only=True)
        pair_rows.append({
            "source_file": str(key[0]),
            "amplitude_N": amp,
            "repetition": int(key[2]),
            "s_plus_mps2": s_plus,
            "s_minus_mps2": s_minus,
            "gain_mps2_per_N": float(gain),
            "bias_mps2": float(bias),
            "holding_force_N": float(hold),
            "start_x_m": float(starts.get("start_x_m", np.nan)),
            "start_y_m": float(starts.get("start_y_m", np.nan)),
            "start_z_m": float(starts.get("start_z_m", np.nan)),
        })
    paired = pd.DataFrame(pair_rows)
    if paired.empty:
        raise ValueError("Could not form paired +Fx/-Fx experiments")

    a = cfg.get("analysis", {})
    min_gain = float(a.get("tether_min_gain_mps2_per_N", 1e-5))
    valid = paired[np.isfinite(paired["gain_mps2_per_N"]) & (np.abs(paired["gain_mps2_per_N"]) >= min_gain)].copy()
    if valid.empty:
        raise ValueError("Paired +/-Fx data do not provide a usable surge gain")

    # IMPORTANT: the tether issue is explicitly a low-force effect in the
    # experiment. Do not estimate the constant correction from the full
    # 10--80 N training range, because the vehicle/rope geometry and surge gain
    # change with amplitude. Use only the configurable low-force calibration
    # window for the correction, while retaining all training amplitudes for
    # diagnosing amplitude/position dependence.
    tether_min_amp = float(a.get("tether_calibration_min_amplitude_N", 10.0))
    tether_max_amp = float(a.get("tether_calibration_max_amplitude_N", 40.0))
    low = valid[(valid["amplitude_N"] >= tether_min_amp) & (valid["amplitude_N"] <= tether_max_amp)].copy()
    if len(low) < 2:
        raise ValueError(
            f"Need at least two paired low-force amplitudes in "
            f"[{tether_min_amp}, {tether_max_amp}] N; found {len(low)}"
        )

    gain_pool = low["gain_mps2_per_N"].to_numpy(float)
    gain_robust, gain_inliers, _ = _robust_median(gain_pool)

    bias_pool = low["bias_mps2"].to_numpy(float)
    bias_robust, bias_inliers, _ = _robust_median(bias_pool)

    # Model form: a_x = g_x * Fx_cmd + b_x =
    #              g_x * (Fx_cmd + d_x), where d_x = b_x / g_x.
    # d_x is therefore an equivalent additive command disturbance, not a
    # direct measurement of the physical tether force.
    equivalent_input_disturbance = (
        bias_robust / gain_robust if abs(gain_robust) > 1e-12 else float("nan")
    )
    holding_force = -equivalent_input_disturbance
    tether_force = equivalent_input_disturbance

    paired["holding_force_from_pair_N"] = paired["holding_force_N"]
    paired["low_force_calibration"] = paired["amplitude_N"].between(tether_min_amp, tether_max_amp)
    paired["bias_error_from_global_mps2"] = paired["bias_mps2"] - bias_robust

    # Baseline drift is useful corroborating evidence, but is not used as the sole
    # tether estimate because post-pulse neutral periods may contain transients.
    baseline = train[
        train["phase"].astype(str).str.lower().eq("neutral")
        & train["protocol"].astype(str).str.lower().eq("baseline")
    ].copy()
    baseline_slopes = []
    if not baseline.empty:
        for _, g in baseline.groupby("segment_id", sort=False):
            t = pd.to_numeric(g["phase_elapsed_s"], errors="coerce").to_numpy(float)
            y = pd.to_numeric(g["u"], errors="coerce").to_numpy(float)
            m = np.isfinite(t) & np.isfinite(y)
            if np.sum(m) >= 5:
                try:
                    baseline_slopes.append(float(np.polyfit(t[m], y[m], 1)[0]))
                except Exception:
                    pass

    amp_corr = _safe_corr(valid["amplitude_N"].to_numpy(float), valid["bias_mps2"].to_numpy(float))
    hold_corr = _safe_corr(valid["amplitude_N"].to_numpy(float), valid["holding_force_N"].to_numpy(float))
    x_corr = _safe_corr(valid["start_x_m"].to_numpy(float), valid["bias_mps2"].to_numpy(float))
    low_amp_corr = _safe_corr(low["amplitude_N"].to_numpy(float), low["bias_mps2"].to_numpy(float))
    low_x_corr = _safe_corr(low["start_x_m"].to_numpy(float), low["bias_mps2"].to_numpy(float))

    hold_vals = valid["holding_force_N"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    _, hold_inliers, _ = _robust_median(hold_vals)
    hold_mad = float(np.median(np.abs(hold_inliers - np.median(hold_inliers)))) if hold_inliers.size else float("nan")
    hold_median = float(np.median(hold_inliers)) if hold_inliers.size else float("nan")
    relative_hold_mad = hold_mad / (abs(hold_median) + 1e-9) if np.isfinite(hold_mad) else float("nan")

    corr_threshold = float(a.get("tether_position_correlation_threshold", 0.6))
    variation_threshold = float(a.get("tether_hold_force_relative_mad_threshold", 0.5))
    position_suspected = bool(np.isfinite(x_corr) and abs(x_corr) >= corr_threshold)
    amplitude_suspected = bool(
        (np.isfinite(amp_corr) and abs(amp_corr) >= corr_threshold)
        or (np.isfinite(hold_corr) and abs(hold_corr) >= corr_threshold)
        or (np.isfinite(relative_hold_mad) and relative_hold_mad >= variation_threshold)
    )

    report = {
        "dof": "u / Fx",
        "samples": int(len(slopes)),
        "paired_trials": int(len(valid)),
        "low_force_calibration_trials": int(len(low)),
        "calibration_amplitude_min_N": float(tether_min_amp),
        "calibration_amplitude_max_N": float(tether_max_amp),
        "gain_estimate_mps2_per_N": float(gain_robust),
        "bias_acceleration_estimate_mps2": float(bias_robust),
        "equivalent_input_disturbance_N": float(equivalent_input_disturbance),
        "tether_force_estimate_N": float(tether_force),
        "holding_force_zero_acceleration_N": float(holding_force),
        "zero_crossing_force_N": float(holding_force),
        "baseline_neutral_drift_accel_mean_mps2": float(np.mean(baseline_slopes)) if baseline_slopes else float("nan"),
        "baseline_neutral_drift_accel_median_mps2": float(np.median(baseline_slopes)) if baseline_slopes else float("nan"),
        "gain_pool_min_N": float(np.min(gain_pool)) if gain_pool.size else float("nan"),
        "gain_pool_max_N": float(np.max(gain_pool)) if gain_pool.size else float("nan"),
        "bias_mad_mps2": float(np.median(np.abs(bias_inliers - np.median(bias_inliers)))) if bias_inliers.size else float("nan"),
        "holding_force_relative_mad": float(relative_hold_mad),
        "bias_vs_amplitude_correlation_all_training": float(amp_corr),
        "holding_force_vs_amplitude_correlation_all_training": float(hold_corr),
        "bias_vs_start_x_correlation_all_training": float(x_corr),
        "bias_vs_amplitude_correlation_low_force": float(low_amp_corr),
        "bias_vs_start_x_correlation_low_force": float(low_x_corr),
        "position_dependence_suspected": position_suspected,
        "amplitude_dependence_suspected": amplitude_suspected,
        "interpretation": (
            "The low-force window supports an approximately constant equivalent X-input disturbance; this value is used only as a baseline correction."
            if not (position_suspected or amplitude_suspected)
            else "The training data show amplitude/position dependence, so the low-force equivalent disturbance is treated as a baseline correction only, not as a universal physical tether force."
        ),
        "formulae": {
            "gain": "(s_plus - s_minus)/(2*F)",
            "bias_acceleration": "(s_plus + s_minus)/2",
            "equivalent_input_disturbance": "bias_acceleration/gain",
            "zero_acceleration_command": "-bias_acceleration/gain",
            "corrected_Fx": "Fx_cmd + equivalent_input_disturbance_N",
        },
    }
    return {"report": report, "slopes": slopes, "paired": paired}


def save_disturbance_analysis(data: pd.DataFrame, cfg: dict, dirs: dict, split: str = "train") -> dict:
    result = estimate_fx_tether_bias(data, cfg, split=split)
    report = result["report"]
    slopes = result["slopes"]
    paired = result["paired"]

    paired.to_csv(dirs["tables"] / "fx_tether_bias_by_amplitude.csv", index=False)
    slopes.to_csv(dirs["tables"] / "fx_early_transient_slopes.csv", index=False)
    save_json(dirs["reports"] / "fx_tether_bias_report.json", report)

    try:
        import matplotlib.pyplot as plt

        # Plot 1: signed command against early acceleration slope.
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(slopes["command_N"], slopes["early_slope_mps2"], alpha=0.7, label="Measured pulse slopes")
        x = np.linspace(float(slopes["command_N"].min()), float(slopes["command_N"].max()), 200)
        y = report["gain_estimate_mps2_per_N"] * x + report["bias_acceleration_estimate_mps2"]
        ax.plot(x, y, label="Symmetric +/-Fx bias-aware fit")
        ax.axvline(report["zero_crossing_force_N"], linestyle="--", label="Estimated zero-acceleration command")
        ax.axvspan(report["calibration_amplitude_min_N"], report["calibration_amplitude_max_N"], alpha=0.12, label="Low-force calibration window")
        ax.set_xlabel("Fx command [N]")
        ax.set_ylabel("Early surge acceleration slope [m/s²]")
        ax.set_title("X-axis tether disturbance identification")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(dirs["plots"] / "fx_tether_slope_fit.png", dpi=int(cfg.get("plots", {}).get("dpi", 180)))
        plt.close(fig)

        # Plot 2: inferred hold force versus amplitude.
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(paired["amplitude_N"], paired["holding_force_N"], alpha=0.8, label="Per-pair hold force")
        ax.axhline(report["holding_force_zero_acceleration_N"], linestyle="--", label="Robust estimated hold force")
        ax.set_xlabel("Pulse amplitude [N]")
        ax.set_ylabel("Equivalent Fx zero-acceleration command [N]")
        ax.set_title("Low-force tether bias versus excitation amplitude")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(dirs["plots"] / "fx_tether_hold_force_vs_amplitude.png", dpi=int(cfg.get("plots", {}).get("dpi", 180)))
        plt.close(fig)
    except Exception as exc:
        report["plot_error"] = str(exc)
        save_json(dirs["reports"] / "fx_tether_bias_report.json", report)

    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Estimate the X-axis tether disturbance from symmetric +/-Fx physical trials.")
    parser.add_argument("--root", default="~/rov_system_id_ws/ROV_system_idetification")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    root = Path(os.path.expanduser(args.root)).resolve()
    cfg_path = Path(os.path.expanduser(args.config)).resolve() if args.config else root / "config" / "identification.yaml"
    cfg = load_config(cfg_path)
    dirs = ensure_dirs(root)
    data = load_raw_data(dirs["raw"])
    data = add_protocol_split(
        data,
        cfg["analysis"].get("train_max_amplitude", 80),
        cfg["analysis"].get("validation_amplitude", 90),
        cfg["analysis"].get("test_amplitude", 100),
    )
    result = save_disturbance_analysis(data, cfg, dirs, split="train")
    print("\nX-axis tether disturbance analysis")
    for k in ["gain_estimate_mps2_per_N", "bias_acceleration_estimate_mps2", "tether_force_estimate_N", "zero_crossing_force_N", "position_dependence_suspected", "amplitude_dependence_suspected"]:
        print(f"{k}: {result['report'].get(k)}")
    print(f"\nSaved report: {dirs['reports'] / 'fx_tether_bias_report.json'}")


if __name__ == "__main__":
    main()
