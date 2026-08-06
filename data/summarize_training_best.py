import csv
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
FILES = (
    ("latent64", DATA_DIR / "sdxl_periodic_ds_training_dout_latent64_sweep.csv"),
    ("latent128", DATA_DIR / "sdxl_periodic_ds_training_dout_sweep.csv"),
)


def rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def f(row, key):
    return float(row[key])


for name, path in FILES:
    data = rows(path)
    print(f"{name}: {len(data)} rows")
    groups = defaultdict(list)
    for row in data:
        groups[(int(row["sigma_index"]), int(row["interval"]))].append(row)
    print("sigma interval best_guard best_max_rel dq_rel dk_rel dq_vs_dyn dk_vs_dyn sat_ppm zero_pct p99")
    for key in sorted(groups):
        sigma, interval = key
        values = groups[key]
        best = min(values, key=lambda row: max(f(row, "dq_rel_vs_dynamic"), f(row, "dk_rel_vs_dynamic")))
        print(
            f"{sigma:5d} {interval:8d} {f(best, 'guard'):10.3f} "
            f"{max(f(best, 'dq_rel_vs_exact'), f(best, 'dk_rel_vs_exact')):12.4f} "
            f"{f(best, 'dq_rel_vs_exact'):7.4f} {f(best, 'dk_rel_vs_exact'):7.4f} "
            f"{f(best, 'dq_rel_vs_dynamic'):9.4f} {f(best, 'dk_rel_vs_dynamic'):9.4f} "
            f"{1e6 * f(best, 'sat_rate'):8.2f} {100 * f(best, 'zero_rate'):8.3f} "
            f"{f(best, 'sigma') if 'sigma' in best else 0:.4f}"
        )
    print("fixed guard=2.0")
    for key in sorted(groups):
        sigma, interval = key
        values = [row for row in groups[key] if abs(f(row, "guard") - 2.0) < 1e-8]
        row = values[0]
        print(
            f"sigma={sigma:4d} interval={interval:3d} "
            f"dq={f(row, 'dq_rel_vs_exact'):.4f} dk={f(row, 'dk_rel_vs_exact'):.4f} "
            f"dq_dyn={f(row, 'dq_rel_vs_dynamic'):.4f} dk_dyn={f(row, 'dk_rel_vs_dynamic'):.4f} "
            f"sat_ppm={1e6 * f(row, 'sat_rate'):.2f} zero_pct={100 * f(row, 'zero_rate'):.2f}"
        )
    print()
