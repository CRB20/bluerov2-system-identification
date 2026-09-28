from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import csd, welch
import matplotlib.pyplot as plt

from .common import DOF_NAMES, INPUT_NAMES, ensure_dirs, load_raw_data, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Estimate empirical ROV frequency responses from broadband experiments.")
    parser.add_argument("--root", default="~/rov_system_id_ws/ROV_system_idetification")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    root = Path(os.path.expanduser(args.root)).resolve()
    cfg_path = Path(os.path.expanduser(args.config)).resolve() if args.config else root / "config" / "identification.yaml"
    cfg=load_config(cfg_path); dirs=ensure_dirs(root)
    data=load_raw_data(dirs["raw"])
    d=data[data["protocol"].astype(str).str.lower().eq("broadband")].copy()
    if d.empty:
        raise RuntimeError("No broadband experiments found. Run collect_physical_data first.")
    fs=1.0/float(np.median(np.diff(d["phase_elapsed_s"].to_numpy(float))))
    rows=[]
    for exp_id,g in d.groupby("experiment_id",sort=False):
        g=g[g["phase"].astype(str).str.lower().eq("broadband")].copy()
        if len(g)<32: continue
        for inp in INPUT_NAMES:
            u=g[f"{inp}_cmd"].to_numpy(float)
            if np.std(u)<1e-9: continue
            for out in DOF_NAMES:
                y=g[out].to_numpy(float)
                f,Puu=welch(u,fs=fs,nperseg=min(256,len(u)))
                _,Pyu=csd(y,u,fs=fs,nperseg=min(256,len(u)))
                H=Pyu/(Puu+1e-12)
                for fi,hi in zip(f,H):
                    rows.append({"experiment_id":exp_id,"input":inp,"output":out,"frequency_hz":fi,"magnitude":abs(hi),"phase_deg":np.degrees(np.angle(hi))})
    frf=pd.DataFrame(rows)
    frf.to_csv(dirs["tables"] / "frequency_response.csv",index=False)
    for (inp,out),g in frf.groupby(["input","output"]):
        if len(g)<3: continue
        fig,ax=plt.subplots(figsize=(8,5))
        mag=np.asarray(g["magnitude"],float)
        freq=np.asarray(g["frequency_hz"],float)
        positive=mag>0
        if np.any(positive):
            ax.semilogy(freq[positive],mag[positive])
        else:
            ax.plot(freq,mag)
        ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel(f"|{out}/{inp}|"); ax.grid(True,which="both",alpha=.25)
        ax.set_title(f"Empirical frequency response: {inp} → {out}")
        fig.tight_layout(); fig.savefig(dirs["plots"] / f"FRF_{inp}_to_{out}.png",dpi=int(cfg.get("plots",{}).get("dpi",180))); plt.close(fig)
    print(f"Saved FRF table: {dirs['tables'] / 'frequency_response.csv'}")

if __name__ == "__main__": main()
