import csv
import importlib.util
from pathlib import Path

import torch

DATA_DIR = Path(__file__).resolve().parent
MODULE_PATH = DATA_DIR / "simulate_periodic_ds_captures.py"
CAPTURE_PATH = DATA_DIR / "sdxl_periodic_ds_inputs_training_dout_latent128.pt"
OUTPUT = DATA_DIR / "sdxl_periodic_ds_training_dout_sweep.csv"
RATIO_OUTPUT = DATA_DIR / "sdxl_periodic_ds_training_dout_ratios.csv"

spec = importlib.util.spec_from_file_location("periodic_capture_sim", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

payload = torch.load(CAPTURE_PATH, map_location="cpu", weights_only=False)
rows = []
ratios = []
for record in payload["records"]:
    case_rows, case_ratios = module.run_case(record, CAPTURE_PATH.name, -1)
    for row in case_rows:
        row["sigma_index"] = int(record["sigma_index"])
        row["sigma"] = float(record["sigma"])
        row["training_loss"] = float(record["loss"])
    for row in case_ratios:
        row["sigma_index"] = int(record["sigma_index"])
        row["sigma"] = float(record["sigma"])
        row["training_loss"] = float(record["loss"])
    rows.extend(case_rows)
    ratios.extend(case_ratios)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
with RATIO_OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(ratios[0]))
    writer.writeheader()
    writer.writerows(ratios)

print("sigma_index sigma interval guard dq_rel dk_rel dq_max dk_max sat_ppm zero_pct ds_rmse")
for row in rows:
    print(
        f"{int(row['sigma_index']):5d} {float(row['sigma']):8.4f} {int(row['interval']):3d} "
        f"{float(row['guard']):5.3f} {float(row['dq_rel_vs_exact']):.6f} "
        f"{float(row['dk_rel_vs_exact']):.6f} {float(row['dq_max_vs_exact']):.6f} "
        f"{float(row['dk_max_vs_exact']):.6f} {1.0e6 * float(row['sat_rate']):8.2f} "
        f"{100.0 * float(row['zero_rate']):8.3f} {float(row['ds_rel_rmse']):.6f}"
    )

print("\nscale drift relative to the last exact calibration")
print("sigma_index sigma interval p50 p90 p95 p99 p999 max")
for row in ratios:
    print(
        f"{int(row['sigma_index']):5d} {float(row['sigma']):8.4f} {int(row['interval']):3d} "
        f"{float(row['p50']):.4f} {float(row['p90']):.4f} {float(row['p95']):.4f} "
        f"{float(row['p99']):.4f} {float(row['p999']):.4f} {float(row['max']):.4f}"
    )
print(f"wrote {OUTPUT}")
print(f"wrote {RATIO_OUTPUT}")
