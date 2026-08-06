import csv
import importlib.util
from collections import defaultdict
from pathlib import Path

import torch

DATA_DIR = Path(__file__).resolve().parent
MODULE_PATH = DATA_DIR / "simulate_periodic_ds_captures.py"
CAPTURE_PATHS = (
    DATA_DIR / "sdxl_periodic_ds_inputs_training_dout_latent128.pt",
    DATA_DIR / "sdxl_periodic_ds_inputs_training_dout_latent128_seed2.pt",
)
OUTPUT = DATA_DIR / "sdxl_periodic_ds_training_dout_latent128_multi_sweep.csv"
RATIO_OUTPUT = DATA_DIR / "sdxl_periodic_ds_training_dout_latent128_multi_ratios.csv"

spec = importlib.util.spec_from_file_location("periodic_capture_sim", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

rows = []
ratios = []
for capture_path in CAPTURE_PATHS:
    payload = torch.load(capture_path, map_location="cpu", weights_only=False)
    for record in payload["records"]:
        case_rows, case_ratios = module.run_case(record, capture_path.name, -1)
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

print(
    "sigma interval guard n dq_rel_mean dq_rel_max dk_rel_mean dk_rel_max dq_dyn_mean dk_dyn_mean sat_ppm_max zero_pct_max"
)
groups = defaultdict(list)
for row in rows:
    groups[(int(row["sigma_index"]), int(row["interval"]), float(row["guard"]))].append(row)
for key in sorted(groups):
    sigma_index, interval, guard = key
    values = groups[key]
    print(
        f"{sigma_index:5d} {interval:8d} {guard:5.3f} {len(values):2d} "
        f"{sum(float(r['dq_rel_vs_exact']) for r in values) / len(values):11.5f} "
        f"{max(float(r['dq_rel_vs_exact']) for r in values):11.5f} "
        f"{sum(float(r['dk_rel_vs_exact']) for r in values) / len(values):11.5f} "
        f"{max(float(r['dk_rel_vs_exact']) for r in values):11.5f} "
        f"{sum(float(r['dq_rel_vs_dynamic']) for r in values) / len(values):11.5f} "
        f"{sum(float(r['dk_rel_vs_dynamic']) for r in values) / len(values):11.5f} "
        f"{1e6 * max(float(r['sat_rate']) for r in values):11.2f} "
        f"{100 * max(float(r['zero_rate']) for r in values):11.3f}"
    )
print(f"wrote {OUTPUT}")
print(f"wrote {RATIO_OUTPUT}")
