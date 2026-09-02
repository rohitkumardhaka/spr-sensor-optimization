import optuna
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import traceback
import time
from pathlib import Path

# Use pathlib for reliable path handling across platforms
db_path = Path(__file__).resolve().parent.joinpath("results", "spr_opt.db")


def export():
    results_dir = db_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    # Use a posix path for the sqlite URL to avoid backslash issues on Windows
    storage_url = f"sqlite:///{db_path.as_posix()}"

    study = optuna.load_study(
        study_name="spr_opt",
        storage=storage_url,
    )

    df = study.trials_dataframe()

    # Retry CSV write in case file is locked (e.g., open in Excel)
    csv_path = str(results_dir.joinpath("best_trials.csv"))
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Try to remove the file first if it exists (to clear locks)
            if Path(csv_path).exists():
                Path(csv_path).unlink()
            df.to_csv(csv_path, index=False)
            break
        except (PermissionError, OSError) as e:
            if attempt < max_retries - 1:
                print(
                    f"CSV write failed (attempt {attempt + 1}/{max_retries}), retrying in 1 second...")
                time.sleep(1)
            else:
                print(f"Failed to write CSV after {max_retries} attempts: {e}")
                raise

    # Visualization requires plotly (and kaleido for write_image). If those
    # packages are not installed, skip plotting but still export the CSV.
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(str(results_dir.joinpath("fom_convergence.png")))

        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_image(str(results_dir.joinpath("parameter_importance.png")))
    except Exception as e:
        print("Skipping visualization due to error:", e)
        print("To enable plots run: python -m pip install plotly kaleido")

    print(f"Exported results to {results_dir}")


if __name__ == "__main__":
    try:
        export()
    except Exception:
        traceback.print_exc()
        print("Export failed. Check that the database exists and required packages are installed (optuna, plotly, kaleido).")
        sys.exit(1)
