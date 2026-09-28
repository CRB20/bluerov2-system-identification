from __future__ import annotations

import argparse
import os
from pathlib import Path

from .common import ensure_dirs, load_config
from .run_identification import main as identify_main
from .compare_models import main as compare_main
from .plot_results import main as plot_main
from .frequency_analysis import main as frequency_main
from .per_dof_selection import main as per_dof_main


def main(argv=None):
    parser=argparse.ArgumentParser(description="Run the full offline ROV system-identification pipeline.")
    parser.add_argument("--root",default="~/rov_system_id_ws/ROV_system_idetification")
    parser.add_argument("--config",default=None)
    parser.add_argument("--skip-frequency",action="store_true")
    args=parser.parse_args(argv)
    root=Path(os.path.expanduser(args.root)).resolve()
    cfg=args.config or str(root/"config"/"identification.yaml")
    identify_main(["--root",str(root),"--config",cfg])
    compare_main(["--root",str(root),"--config",cfg])
    plot_main(["--root",str(root),"--config",cfg])
    try:
        per_dof_main(["--root",str(root),"--config",cfg])
    except Exception as exc:
        print(f"Per-DOF selection skipped/failed: {exc}")
    if not args.skip_frequency:
        try:
            frequency_main(["--root",str(root),"--config",cfg])
        except Exception as exc:
            print(f"Frequency analysis skipped/failed: {exc}")
    print(f"\nFULL PIPELINE COMPLETE: {root}")

if __name__ == "__main__": main()
