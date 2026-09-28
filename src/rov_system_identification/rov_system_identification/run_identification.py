from __future__ import annotations

import argparse
import json
import os
import pickle
import time
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
from .models import make_model
from .disturbance_analysis import estimate_fx_tether_bias, save_disturbance_analysis
from .decoupling import evaluate as evaluate_decoupling, condition_number, normalized_coupling



def direct_pulse_gain_table(df: pd.DataFrame, early_start=0.05, early_end=0.60):
    rows=[]
    for seg_id,g in df.groupby("segment_id",sort=False):
        if len(g)<10: continue
        active=[i for i,c in enumerate(["Fx","Fy","Fz","Mx","My","Mz"]) if np.max(np.abs(g[f"{c}_cmd"].to_numpy(float)))>1e-9]
        if len(active)!=1: continue
        i=active[0]; name=["Fx","Fy","Fz","Mx","My","Mz"][i]
        t=g["phase_elapsed_s"].to_numpy(float); mask=(t>=early_start)&(t<=early_end)
        if np.sum(mask)<5: continue
        y=g[["u","v","w","p","q","r"][i]+"_filtered"].to_numpy(float); y=y-y[0]
        try: slope,_=np.polyfit(t[mask],y[mask],1)
        except Exception: continue
        cmd=float(g[f"{name}_cmd"].iloc[0]);
        if abs(cmd)<1e-9: continue
        rows.append({"segment_id":seg_id,"input":name,"command":cmd,"early_slope":float(slope),"gain":float(slope/cmd)})
    return pd.DataFrame(rows)


def _eligible_training(data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    a = cfg["analysis"]
    protocols = a.get("train_protocols", ["individual", "cross"])
    d = data[
        (data["split"] == "train")
        & data["protocol"].isin(protocols)
        & (data["phase"].astype(str).str.lower().isin(["pulse", "broadband"]))
    ].copy()
    return d


def _fit_model(name, train_df, cfg):
    from .common import differentiate_group
    analysis = cfg["analysis"]
    # Derivatives are computed per segment/experiment. Neutral segments stay in
    # the CSV but are explicitly excluded from model fitting.
    pulse_df = train_df[train_df["phase"].astype(str).str.lower().isin(["pulse", "broadband"])].copy()
    pulse_df = differentiate_group(
        pulse_df,
        time_col="phase_elapsed_s",
        window=int(analysis.get("savgol_window",21)),
        order=int(analysis.get("savgol_order",3)),
        group_col="segment_id",
    )
    if pulse_df.empty:
        raise ValueError("No excitation samples available for model fitting")
    nu = pulse_df[[f"{x}_filtered" for x in DOF_NAMES]].to_numpy(float)
    tau = pulse_df[[f"{x}_cmd" for x in ["Fx","Fy","Fz","Mx","My","Mz"]]].to_numpy(float)
    acc = pulse_df[[f"d{x}_dt" for x in DOF_NAMES]].to_numpy(float)
    groups = pulse_df["segment_id"].to_numpy()
    times = pulse_df["phase_elapsed_s"].to_numpy(float)
    model = make_model(name, cfg)
    t0 = time.perf_counter()
    model.fit(nu, tau, acc, times=times, groups=groups)
    fit_seconds = time.perf_counter()-t0
    return model, pulse_df, fit_seconds


def _model_tau(model, tau):
    transform = getattr(model, "transform_tau", None)
    return transform(tau) if callable(transform) else tau


def _predict_one_step(model, g, derivative_dt_default=0.02):
    nu = g[[f"{x}_filtered" for x in DOF_NAMES]].to_numpy(float)
    tau_raw = g[[f"{x}_cmd" for x in ["Fx","Fy","Fz","Mx","My","Mz"]]].to_numpy(float)
    t = g["phase_elapsed_s"].to_numpy(float)
    dt = np.median(np.diff(t)) if len(t)>1 else derivative_dt_default
    dt = float(dt if np.isfinite(dt) and dt>0 else derivative_dt_default)
    if getattr(model, "supports_accel", True):
        pred_next = nu + model.predict_accel(nu, tau_raw) * dt
        true_next = np.vstack([nu[1:], nu[-1:]])
        return true_next, pred_next, dt
    if model.name == "arx":
        tau = _model_tau(model, tau_raw)
        n = len(nu); lag=max(model.na,model.nb)
        if n <= lag+1: return None
        pred=[]; true=[]
        for k in range(lag,n):
            f=[]
            for l in range(1,model.na+1): f.extend(nu[k-l])
            for l in range(model.nb): f.extend(tau[k-l])
            p=np.asarray(f)@model.coef_
            pred.append(p); true.append(nu[k])
        return np.asarray(true), np.asarray(pred), dt
    pred_next = model.predict_next_nu(nu[:-1], tau_raw[:-1], dt)
    return nu[1:], pred_next, dt

def _rollout(model, g, horizon_s=5.0):
    nu = g[[f"{x}_filtered" for x in DOF_NAMES]].to_numpy(float)
    tau_raw = g[[f"{x}_cmd" for x in ["Fx","Fy","Fz","Mx","My","Mz"]]].to_numpy(float)
    t = g["phase_elapsed_s"].to_numpy(float)
    if len(nu) < 3:
        return None
    dt = np.median(np.diff(t)); dt = float(dt if np.isfinite(dt) and dt > 0 else 0.02)
    requested_steps = max(1, int(round(horizon_s / dt)))
    pred = []

    if model.name == "arx":
        lag = max(model.na, model.nb)
        available_steps = len(nu) - lag
        if available_steps < 1:
            return None
        steps = min(requested_steps, available_steps)
        history_nu = [x.copy() for x in nu[:lag]]
        history_tau = [x.copy() for x in tau_raw[:lag]]
        for j in range(steps):
            k = lag + j
            current = model.predict_next_nu_from_history(history_nu, history_tau)
            history_nu.append(current.copy())
            history_tau.append(tau_raw[k].copy())
            pred.append(current.copy())
        truth = nu[lag:lag + len(pred)]
        return truth, np.asarray(pred), dt

    steps = min(requested_steps, len(nu) - 1)
    current = nu[0].copy()
    for k in range(steps):
        if getattr(model, "supports_accel", True):
            current = current + model.predict_accel(current[None, :], tau_raw[k][None, :])[0] * dt
        else:
            current = model.predict_next_nu(current[None, :], tau_raw[k][None, :], dt)[0]
        pred.append(current.copy())
    truth = nu[1:1 + len(pred)]
    return truth, np.asarray(pred), dt

def _model_parameter_count(model):
    try: return int(model.parameter_count())
    except Exception: return 0


def evaluate_model(name, model, test_df, cfg):
    from .common import differentiate_group
    rows=[]; next_true=[]; next_pred=[]; rollout_metrics={}
    test_df = test_df[test_df["phase"].astype(str).str.lower().isin(["pulse","broadband"])].copy()
    if test_df.empty:
        return {"status":"failed","reason":"No excitation test samples"}
    test_df = differentiate_group(
        test_df, time_col="phase_elapsed_s",
        window=int(cfg["analysis"].get("savgol_window",21)),
        order=int(cfg["analysis"].get("savgol_order",3)),
        group_col="segment_id",
    )
    groups=[]
    for segment_id, g in test_df.groupby("segment_id", sort=False):
        exp_id = str(g["experiment_id"].iloc[0])
        if len(g) < 5:
            continue
        try:
            one = _predict_one_step(model, g)
        except Exception as exc:
            rows.append({"error": f"one-step failed: {exc}"}); continue
        if one is not None:
            yt,yp,dt=one; next_true.append(yt); next_pred.append(yp)
        group_roll={}
        for horizon in cfg["analysis"].get("rollout_seconds", [1.0,3.0,5.0]):
            try:
                rr=_rollout(model,g,float(horizon))
            except Exception:
                rr=None
            if rr is not None:
                yt,yp,dt=rr
                if len(yp):
                    nr=np.sqrt(np.mean((yt-yp)**2,axis=0))
                    key = f"{float(horizon):g}"
                    group_roll[key] = float(np.mean(nr))
                    group_roll[key + "_effective_s"] = float(len(yp) * dt)
                    for j,n in enumerate(DOF_NAMES):
                        rows.append({
                            "experiment_id": exp_id,
                            "horizon_s": float(horizon),
                            "effective_horizon_s": float(len(yp) * dt),
                            "dof": n,
                            "rollout_rmse": float(nr[j]),
                        })
        groups.append(group_roll)
    if not next_true:
        return {"status":"failed","reason":"No evaluable test sequences"}
    yt=np.vstack(next_true); yp=np.vstack(next_pred)
    rmse=np.sqrt(np.mean((yt-yp)**2,axis=0))
    nrmse=normalized_rmse(yt,yp)
    r2=r2_score_vec(yt,yp)
    out={
        "status":"ok",
        "next_state_rmse":rmse.tolist(),
        "next_state_nrmse":nrmse.tolist(),
        "next_state_r2":r2.tolist(),
        "mean_next_state_rmse":float(np.mean(rmse)),
        "mean_next_state_nrmse":float(np.mean(nrmse)),
        "mean_next_state_r2":float(np.nanmean(r2)),
    }
    for horizon in cfg["analysis"].get("rollout_seconds", [1.0,3.0,5.0]):
        vals=[]; eff=[]
        for gr in groups:
            k=f"{float(horizon):g}"
            if k in gr:
                vals.append(gr[k])
                if k + "_effective_s" in gr:
                    eff.append(gr[k + "_effective_s"])
        out[f"mean_rollout_rmse_{float(horizon):g}s"] = float(np.mean(vals)) if vals else float("nan")
        out[f"mean_rollout_effective_s_{float(horizon):g}s"] = float(np.mean(eff)) if eff else float("nan")
    return out


def main(argv=None):
    parser=argparse.ArgumentParser(description="Fit and save the full ROV system-identification model suite.")
    parser.add_argument("--root",default="~/rov_system_id_ws/ROV_system_idetification")
    parser.add_argument("--config",default=None)
    args=parser.parse_args(argv)
    root=Path(os.path.expanduser(args.root)).resolve()
    cfg_path=Path(os.path.expanduser(args.config)).resolve() if args.config else root/"config"/"identification.yaml"
    cfg=load_config(cfg_path)
    dirs=ensure_dirs(root)
    data=load_raw_data(dirs["raw"])
    data=add_protocol_split(
        data,
        cfg["analysis"].get("train_max_amplitude",80),
        cfg["analysis"].get("validation_amplitude",90),
        cfg["analysis"].get("test_amplitude",100),
    )

    # Calibrate the approximately constant X-axis tether disturbance from the
    # training amplitudes only. This prevents the 90/100 N validation/test trials
    # from leaking into the disturbance estimate.
    try:
        disturbance = save_disturbance_analysis(data, cfg, dirs, split="train")
        tether_report = disturbance["report"]
        cfg["analysis"]["_fx_tether_force_estimate_n"] = float(
            tether_report.get("equivalent_input_disturbance_N", tether_report.get("tether_force_estimate_N", 0.0))
        )
        cfg["analysis"]["_fx_tether_calibration_source"] = "low_force_training_only"
        save_json(dirs["reports"] / "identification_disturbance_calibration.json", tether_report)
        print("\n=== X-axis tether calibration ===")
        print(json.dumps(tether_report, indent=2))
    except Exception as exc:
        cfg["analysis"]["_fx_tether_force_estimate_n"] = 0.0
        cfg["analysis"]["_fx_tether_calibration_source"] = "failed_fallback_zero"
        save_json(dirs["reports"] / "identification_disturbance_calibration.json", {
            "status": "failed",
            "error": str(exc),
            "fallback": "No Fx tether correction applied; uncorrected models remain available."
        })
        print(f"\nTether calibration failed; corrected models will use zero correction: {exc}")

    # Save the exact labeled dataset used for identification.
    data.to_csv(dirs["tables"] / "prepared_dataset_with_splits.csv", index=False)
    from .common import differentiate_group
    pulse_for_diag=data[data["phase"].astype(str).str.lower().eq("pulse")].copy()
    if not pulse_for_diag.empty:
        pulse_for_diag=differentiate_group(pulse_for_diag,time_col="phase_elapsed_s",group_col="segment_id",window=int(cfg["analysis"].get("savgol_window",21)),order=int(cfg["analysis"].get("savgol_order",3)))
        dg=direct_pulse_gain_table(pulse_for_diag)
        dg.to_csv(dirs["tables"] / "direct_pulse_gains.csv",index=False)
    train=_eligible_training(data,cfg)
    test=data[(data["split"]=="test") & data["phase"].astype(str).str.lower().isin(["pulse","broadband"])].copy()
    enabled=cfg["analysis"].get("enable_models",[])
    corrected_models = [m for m in enabled if str(m).endswith("_tether_corrected")]
    summary=[]
    for name in enabled:
        print(f"\n=== {name} ===")
        try:
            model,fit_df,fit_seconds=_fit_model(name,train,cfg)
            metrics=evaluate_model(name,model,test,cfg)
            model_dir=dirs["models"]/name
            if name == "coupled_linear_ridge" and hasattr(model,"matrices"):
                A,B,c=model.matrices()
                sel,cands,feasible=evaluate_decoupling(B)
                save_json(dirs["reports"] / "linear_model_matrices_and_decoupling.json", {
                    "A":A,"B":B,"c":c,"condition_number":condition_number(B)[0],
                    "singular_values":condition_number(B)[1],
                    "normalized_coupling":normalized_coupling(B),
                    "selected_relative_lambda":sel["relative_lambda"],
                    "selected_D":sel["D"],"selected_BD":sel["BD"],
                    "selected_feasible":feasible,
                    "decoupling_candidates":[{k:v for k,v in x.items() if k not in ("D","BD","singular_values")} for x in cands],
                })
            model_dir.mkdir(parents=True,exist_ok=True)
            with open(model_dir/"model.pkl","wb") as f: pickle.dump(model,f,protocol=pickle.HIGHEST_PROTOCOL)
            meta={"model":name,"fit_seconds":fit_seconds,"n_parameters":_model_parameter_count(model),"metrics":metrics}
            save_json(model_dir/"metadata.json",meta)
            summary.append({"model":name,"fit_seconds":fit_seconds,"n_parameters":_model_parameter_count(model),**metrics})
            print(json.dumps(metrics,indent=2))
        except Exception as exc:
            print(f"FAILED: {exc}")
            summary.append({"model":name,"status":"failed","error":str(exc)})
    table=pd.DataFrame(summary)
    if not table.empty:
        table.to_csv(dirs["tables"] / "model_comparison_raw.csv",index=False)
    save_json(dirs["reports"] / "identification_run.json",{
        "root":str(root),
        "config":str(cfg_path),
        "models":summary,
        "train_rows":int(len(train)),
        "test_rows":int(len(test)),
        "tether_equivalent_input_disturbance_N": float(cfg["analysis"].get("_fx_tether_force_estimate_n", 0.0)),
        "tether_calibration_source": cfg["analysis"].get("_fx_tether_calibration_source", "unknown"),
        "corrected_models_enabled": corrected_models,
        "n_corrected_models_enabled": len(corrected_models),
    })
    print(f"\nSaved models/results under {dirs['models']}")

if __name__ == "__main__": main()
