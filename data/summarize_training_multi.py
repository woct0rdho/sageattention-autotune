import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def f(row, key):
    return float(row[key])


rows = load("sdxl_periodic_ds_training_dout_latent128_multi_sweep.csv")
ratios = load("sdxl_periodic_ds_training_dout_latent128_multi_ratios.csv")
print("LONG MODEL TRAINING SAMPLE: best guard by max dQ/dK error vs exact")
groups = defaultdict(list)
for row in rows:
    groups[(int(row["sigma_index"]), int(row["interval"]))].append(row)
for key in sorted(groups):
    sigma, interval = key
    values = groups[key]
    best = min(values, key=lambda row: max(f(row, "dq_rel_vs_exact"), f(row, "dk_rel_vs_exact")))
    print(
        f"sigma={sigma:3d} interval={interval:3d} guard={f(best, 'guard'):.3f} "
        f"dq={f(best, 'dq_rel_vs_exact'):.4f} dk={f(best, 'dk_rel_vs_exact'):.4f} "
        f"dq-v-dyn={f(best, 'dq_rel_vs_dynamic'):.4f} dk-v-dyn={f(best, 'dk_rel_vs_dynamic'):.4f} "
        f"sat_ppm={1e6 * f(best, 'sat_rate'):.1f} zero_pct={100 * f(best, 'zero_rate'):.1f}"
    )
print("\nLONG MODEL TRAINING SAMPLE: fixed guard=2.0")
for key in sorted(groups):
    sigma, interval = key
    row = next(row for row in groups[key] if abs(f(row, "guard") - 2.0) < 1e-8)
    print(
        f"sigma={sigma:3d} interval={interval:3d} "
        f"dq={f(row, 'dq_rel_vs_exact'):.4f} dk={f(row, 'dk_rel_vs_exact'):.4f} "
        f"dq-v-dyn={f(row, 'dq_rel_vs_dynamic'):.4f} dk-v-dyn={f(row, 'dk_rel_vs_dynamic'):.4f} "
        f"sat_ppm={1e6 * f(row, 'sat_rate'):.1f} zero_pct={100 * f(row, 'zero_rate'):.1f}"
    )
print("\nLONG MODEL TRAINING SAMPLE: scale drift max across two captures")
ratio_groups = defaultdict(list)
for row in ratios:
    ratio_groups[(int(row["sigma_index"]), int(row["interval"]))].append(row)
for key in sorted(ratio_groups):
    sigma, interval = key
    values = ratio_groups[key]
    print(
        f"sigma={sigma:3d} interval={interval:3d} "
        f"p99_max={max(f(r, 'p99') for r in values):.3f} "
        f"max_max={max(f(r, 'max') for r in values):.3f}"
    )
print("\nLONG MODEL TRAINING SAMPLE: dynamic baseline per capture")
seen = {}
for row in rows:
    seen[(row["capture"], int(row["sigma_index"]))] = row
for key in sorted(seen):
    row = seen[key]
    print(f"{Path(key[0]).stem} sigma={key[1]:3d} dq={f(row, 'dynamic_dq_rel'):.4f} dk={f(row, 'dynamic_dk_rel'):.4f}")
