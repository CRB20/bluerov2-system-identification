from __future__ import annotations

import json
import math
import os
import platform
import socket
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

DOF_NAMES = ["u", "v", "w", "p", "q", "r"]
INPUT_NAMES = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
DIRECT_SIGNS = np.array([-1, -1, 1, 1, 1, 1], dtype=int)


def load_config(path: str | os.PathLike) -> dict:
    import yaml
    with open(os.path.expanduser(str(path)), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dirs(root: str | os.PathLike) -> Dict[str, Path]:
    root = Path(os.path.expanduser(str(root))).resolve()
    dirs = {
        "root": root,
        "raw": root / "data" / "raw",
        "metadata": root / "metadata",
        "models": root / "results" / "models",
        "plots": root / "results" / "plots",
        "tables": root / "results" / "tables",
        "reports": root / "results" / "reports",
        "config": root / "config",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def package_root() -> Path:
    # .../src/rov_system_identification/rov_system_identification/common.py
    return Path(__file__).resolve().parents[3]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment_metadata() -> dict:
    metadata = {
        "timestamp_utc": now_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:
        import scipy
        metadata["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        import pandas
        metadata["pandas"] = pandas.__version__
    except Exception:
        pass
    try:
        import sklearn
        metadata["scikit_learn"] = sklearn.__version__
    except Exception:
        pass
    try:
        metadata["ros_distro"] = os.environ.get("ROS_DISTRO", "")
        metadata["ros_version"] = subprocess.check_output(
            ["ros2", "--version"], stderr=subprocess.STDOUT, text=True, timeout=2
        ).strip()
    except Exception:
        pass
    return metadata


def save_json(path: str | os.PathLike, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def safe_savgol(y: np.ndarray, window: int = 21, order: int = 3) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < max(7, order + 3):
        return y.copy()
    w = min(window, n if n % 2 else n - 1)
    min_w = order + 2
    if min_w % 2 == 0:
        min_w += 1
    w = max(w, min_w)
    if w >= n:
        w = n - 1 if n % 2 == 0 else n
    if w <= order or w < 5:
        return y.copy()
    return savgol_filter(y, w, order, mode="interp")


def differentiate_group(df: pd.DataFrame, time_col: str = "phase_elapsed_s", window: int = 21, order: int = 3, group_col: str = "segment_id") -> pd.DataFrame:
    """Differentiate each experiment/trial independently to avoid edge leakage."""
    df = df.copy()
    df["dt_s"] = df.groupby("experiment_id")[time_col].diff()
    df["dt_s"] = df["dt_s"].fillna(1.0 / 50.0)
    for col in DOF_NAMES:
        filt = []
        deriv = []
        for _, g in df.groupby(group_col, sort=False):
            y = g[col].to_numpy(float)
            t = g[time_col].to_numpy(float)
            y_f = safe_savgol(y, window, order)
            dt = np.median(np.diff(t)) if len(t) > 1 else 0.02
            if not np.isfinite(dt) or dt <= 0:
                dt = 0.02
            d = safe_savgol(y, window, order)
            try:
                d = savgol_filter(
                    y, min(window, len(y) if len(y)%2 else len(y)-1), order,
                    deriv=1, delta=dt, mode="interp"
                ) if len(y) >= order + 3 else np.gradient(y, t)
            except Exception:
                d = np.gradient(y_f, t)
            filt.extend(y_f.tolist())
            deriv.extend(d.tolist())
        df[f"{col}_filtered"] = filt
        df[f"d{col}_dt"] = deriv
    return df


def load_raw_data(raw_dir: str | os.PathLike) -> pd.DataFrame:
    paths = sorted(Path(os.path.expanduser(str(raw_dir))).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    frames = []
    for p in paths:
        x = pd.read_csv(p)
        x["source_file"] = p.name
        frames.append(x)
    data = pd.concat(frames, ignore_index=True)
    # Ensure stable numeric columns.
    numeric = [
        "amplitude", "sample_index", "phase_elapsed_s",
        *[f"{a}_cmd" for a in INPUT_NAMES],
        "x", "y", "z", "vx", "vy", "vz",
        "roll", "pitch", "yaw", "roll_rate", "pitch_rate", "yaw_rate",
        *DOF_NAMES,
        *[f"thruster_mix_ch{i}" for i in range(1,9)],
        *[f"pwm_ch{i}" for i in range(1,9)],
        "pwm_received", "pwm_fresh", "pwm_age_s", "pwm_saturated_count", "pwm_min", "pwm_max",
    ]
    for c in numeric:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")
    return data


def add_protocol_split(data: pd.DataFrame, train_max_amp=80.0, validation_amp=90.0, test_amp=100.0) -> pd.DataFrame:
    """Create deterministic splits without mixing samples from the same experiment."""
    data = data.copy()
    protocol = data["protocol"].astype(str).str.lower()
    amp = pd.to_numeric(data["amplitude"], errors="coerce").fillna(-1.0)
    data["split"] = "test"
    data.loc[(amp > 0) & (amp <= train_max_amp) & protocol.isin(["individual", "cross"]), "split"] = "train"
    data.loc[(amp > 0) & (amp == validation_amp) & protocol.isin(["individual", "cross"]), "split"] = "validation"
    data.loc[(amp > 0) & (amp == test_amp) & protocol.isin(["individual", "cross"]), "split"] = "test"
    # Broadband stays test by default: this is an OOD excitation check.
    return data


def make_linear_features(nu: np.ndarray, tau: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    nu = np.asarray(nu, float)
    tau = np.asarray(tau, float)
    X = np.column_stack([np.ones(len(nu)), nu, tau])
    names = ["1", *DOF_NAMES, *INPUT_NAMES]
    return X, names


def make_nonlinear_features(nu: np.ndarray, tau: np.ndarray, include_cross=True) -> Tuple[np.ndarray, List[str]]:
    nu = np.asarray(nu, float)
    tau = np.asarray(tau, float)
    cols = [np.ones(len(nu))]
    names = ["1"]
    for j, n in enumerate(DOF_NAMES):
        cols.append(nu[:, j]); names.append(n)
    for j, n in enumerate(DOF_NAMES):
        cols.append(nu[:, j] * np.abs(nu[:, j])); names.append(f"{n}|{n}|")
    for j, n in enumerate(INPUT_NAMES):
        cols.append(tau[:, j]); names.append(n)
    if include_cross:
        for i in range(6):
            for j in range(i + 1, 6):
                cols.append(nu[:, i] * nu[:, j])
                names.append(f"{DOF_NAMES[i]}*{DOF_NAMES[j]}")
    return np.column_stack(cols), names


def split_training_arrays(data: pd.DataFrame, split: str = "train", filtered=True):
    d = data[data["split"] == split].copy()
    if d.empty:
        raise ValueError(f"No samples available for split={split}")
    ncols = [f"{c}_filtered" for c in DOF_NAMES] if filtered else DOF_NAMES
    nu = d[ncols].to_numpy(float)
    tau = d[INPUT_NAMES].to_numpy(float)
    acc = d[[f"d{c}_dt" for c in DOF_NAMES]].to_numpy(float)
    return d, nu, tau, acc


def normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    err = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    scale = np.sqrt(np.mean(y_true**2, axis=0)) + 1e-12
    return err / scale


def r2_score_vec(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    out = np.full(y_true.shape[1], np.nan)
    for j in range(y_true.shape[1]):
        y = y_true[:, j]
        ss_res = float(np.sum((y - y_pred[:, j]) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot > 1e-15:
            out[j] = 1.0 - ss_res / ss_tot
    return out


def physical_coupling_percent(data: pd.DataFrame) -> pd.DataFrame:
    """Estimate directional percentage coupling from pulse trajectories."""
    rows = []
    for exp_id, g in data.groupby("experiment_id", sort=False):
        if str(g["protocol"].iloc[0]).lower() not in ("individual", "cross"):
            continue
        amp = float(g["amplitude"].iloc[0])
        tau = g[[f"{x}_cmd" for x in INPUT_NAMES]].to_numpy(float)
        active = np.where(np.max(np.abs(tau), axis=0) > 1e-12)[0]
        if len(active) != 1:
            continue
        src = int(active[0])
        X = g[[f"{c}_filtered" if f"{c}_filtered" in g.columns else c for c in DOF_NAMES]].to_numpy(float)
        dX = X - X[0]
        x = dX[:, src]
        denom = np.dot(x, x)
        if denom <= 1e-12:
            continue
        for tgt in range(6):
            if tgt == src:
                continue
            y = dX[:, tgt]
            k = float(np.dot(x, y) / denom)
            rows.append({
                "experiment_id": exp_id,
                "amplitude": amp,
                "source": INPUT_NAMES[src],
                "target": DOF_NAMES[tgt],
                "coupling_percent": 100.0 * k,
            })
    return pd.DataFrame(rows)
