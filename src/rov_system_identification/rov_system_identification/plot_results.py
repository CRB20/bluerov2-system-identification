from __future__ import annotations

import argparse
import ast
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .common import DOF_NAMES, ensure_dirs, load_config, load_raw_data, add_protocol_split, physical_coupling_percent
from .run_identification import _rollout


def _parse_list(value):
    if isinstance(value, list): return value
    try: return ast.literal_eval(value)
    except Exception: return []


def main(argv=None):
    parser=argparse.ArgumentParser(description="Generate plots for ROV system-identification results.")
    parser.add_argument("--root",default="~/rov_system_id_ws/ROV_system_idetification")
    parser.add_argument("--config",default=None)
    args=parser.parse_args(argv)
    root=Path(os.path.expanduser(args.root)).resolve()
    cfg_path=Path(os.path.expanduser(args.config)).resolve() if args.config else root/"config"/"identification.yaml"
    cfg=load_config(cfg_path); dirs=ensure_dirs(root); dpi=int(cfg.get("plots",{}).get("dpi",180))
    ranked=dirs["tables"]/"model_comparison_ranked.csv"
    raw=dirs["tables"]/"model_comparison_raw.csv"
    src=ranked if ranked.exists() else raw
    if not src.exists(): raise FileNotFoundError("Run identification and comparison first.")
    df=pd.read_csv(src)
    ok=df[df.get("status","ok").astype(str).eq("ok")].copy() if "status" in df else df.copy()
    if not ok.empty and "overall_score" in ok:
        fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(len(ok)); ax.bar(x,ok["overall_score"]); ax.set_xticks(x); ax.set_xticklabels(ok["model"],rotation=45,ha="right"); ax.set_ylabel("Overall score (lower is better)"); ax.set_title("ROV system-identification model ranking"); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(dirs["plots"]/"model_ranking_overall.png",dpi=dpi); plt.close(fig)
    # Rollout comparison.
    if not ok.empty:
        horizons=[1.0,3.0,5.0]
        available=[h for h in horizons if f"mean_rollout_rmse_{h:g}s" in ok]
        fig,ax=plt.subplots(figsize=(10,5)); width=0.8/max(1,len(available)); x=np.arange(len(ok))
        for i,h in enumerate(available): ax.bar(x+(i-len(available)/2)*width,ok[f"mean_rollout_rmse_{h:g}s"],width,label=f"{h:g}s")
        ax.set_xticks(x); ax.set_xticklabels(ok["model"],rotation=45,ha="right"); ax.set_ylabel("Velocity rollout RMSE"); ax.set_title("Generalisation by rollout horizon"); ax.legend(); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(dirs["plots"]/"model_rollout_rmse.png",dpi=dpi); plt.close(fig)
        # Per-DOF one-step RMSE.
        rows=[]
        for _,r in ok.iterrows():
            vals=_parse_list(r.get("next_state_rmse",[]))
            for j,n in enumerate(DOF_NAMES):
                if j<len(vals): rows.append({"model":r["model"],"dof":n,"rmse":float(vals[j])})
        if rows:
            m=pd.DataFrame(rows); piv=m.pivot(index="model",columns="dof",values="rmse"); fig,ax=plt.subplots(figsize=(11,6)); piv.plot(kind="bar",ax=ax); ax.set_ylabel("One-step state RMSE"); ax.set_title("Per-DOF model prediction error"); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(dirs["plots"]/"per_dof_rmse.png",dpi=dpi); plt.close(fig)
    # Cross-coupling heatmap.
    data=load_raw_data(dirs["raw"]); data=add_protocol_split(data,cfg["analysis"].get("train_max_amplitude",80),cfg["analysis"].get("validation_amplitude",90),cfg["analysis"].get("test_amplitude",100))
    coupling=physical_coupling_percent(data)
    if not coupling.empty:
        med=coupling.groupby(["source","target"])["coupling_percent"].median().unstack(fill_value=0.0)
        med=med.reindex(index=["Fx","Fy","Fz","Mx","My","Mz"],columns=DOF_NAMES,fill_value=0.0)
        fig,ax=plt.subplots(figsize=(10,6)); im=ax.imshow(med.to_numpy(),aspect="auto"); fig.colorbar(im,ax=ax,label="Coupling [%]"); ax.set_xticks(range(6),DOF_NAMES); ax.set_yticks(range(6),INPUT_NAMES if 'INPUT_NAMES' in globals() else ["Fx","Fy","Fz","Mx","My","Mz"]); ax.set_title("Median directional coupling"); fig.tight_layout(); fig.savefig(dirs["plots"]/"coupling_heatmap.png",dpi=dpi); plt.close(fig)
        coupling.to_csv(dirs["tables"]/"directional_coupling.csv",index=False)
    # Best-model 5 s rollout on one held-out test pulse.
    if not ok.empty and "overall_score" in ok:
        best=str(ok.sort_values("overall_score").iloc[0]["model"])
        model_file=dirs["models"]/best/"model.pkl"
        if model_file.exists():
            with open(model_file,"rb") as f: model=pickle.load(f)
            test=data[(data["amplitude"]==cfg["analysis"].get("test_amplitude",100)) & data["phase"].astype(str).str.lower().isin(["pulse","broadband"])].copy()
            if not test.empty:
                from .common import differentiate_group
                test=differentiate_group(test,time_col="phase_elapsed_s",window=int(cfg["analysis"].get("savgol_window",21)),order=int(cfg["analysis"].get("savgol_order",3)),group_col="segment_id")
                seg=next(iter(test.groupby("segment_id",sort=False)))[1]
                rr=_rollout(model,seg,5.0)
                if rr is not None:
                    truth,pred,dt=rr
                    n=min(len(truth),len(pred)); t=np.arange(n)*dt; fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True); axes=axes.ravel()
                    for j,nm in enumerate(DOF_NAMES): axes[j].plot(t,truth[:n,j],label="measured"); axes[j].plot(t,pred[:n,j],label="predicted",alpha=.8); axes[j].set_ylabel(nm); axes[j].grid(alpha=.25)
                    axes[-2].set_xlabel("Time [s]"); axes[-1].set_xlabel("Time [s]"); axes[0].legend(); fig.suptitle(f"Best-model 5 s rollout: {best}"); fig.tight_layout(); fig.savefig(dirs["plots"]/"best_model_rollout_5s.png",dpi=dpi); plt.close(fig)
    print(f"Plots written to {dirs['plots']}")

if __name__ == "__main__": main()
