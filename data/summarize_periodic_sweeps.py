import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def f(row, key):
    return float(row[key])


def report_training() -> None:
    rows = read_csv(ROOT / "sdxl_periodic_ds_training_dout_sweep.csv")
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["sigma_index"]), int(row["interval"]), float(row["guard"]))].append(row)
    print("TRAINING PER-RECORD SUMMARY")
    print("sigma interval guard dq_rel dk_rel dq_cos dk_cos sat_ppm zero_pct drift_p99")
    ratio_rows = read_csv(ROOT / "sdxl_periodic_ds_training_dout_ratios.csv")
    ratio = {(int(r["sigma_index"]), int(r["interval"])): r for r in ratio_rows}
    for key in sorted(groups):
        sigma_index, interval, guard = key
        values = groups[key]
        worst_dq = max(f(r, "dq_rel_vs_exact") for r in values)
        worst_dk = max(f(r, "dk_rel_vs_exact") for r in values)
        mean_dq = sum(f(r, "dq_rel_vs_exact") for r in values) / len(values)
        mean_dk = sum(f(r, "dk_rel_vs_exact") for r in values) / len(values)
        sat = sum(f(r, "sat_rate") for r in values) / len(values)
        zero = sum(f(r, "zero_rate") for r in values) / len(values)
        ratio_value = ratio[(sigma_index, interval)]
        print(
            f"{sigma_index:5d} {interval:8d} {guard:5.3f} "
            f"{worst_dq:7.4f} {worst_dk:7.4f} "
            f"{1.0 - mean_dq:7.4f} {1.0 - mean_dk:7.4f} "
            f"{1.0e6 * sat:8.2f} {100.0 * zero:8.3f} {f(ratio_value, 'p99'):8.3f}"
        )

    print("\nTRAINING GUARD AGGREGATE ACROSS THE FIVE SIGMAS")
    print("interval guard worst_dq worst_dk mean_dq mean_dk max_sat_ppm max_zero_pct")
    aggregate = defaultdict(list)
    for row in rows:
        aggregate[(int(row["interval"]), float(row["guard"]))].append(row)
    for key in sorted(aggregate):
        interval, guard = key
        values = aggregate[key]
        print(
            f"{interval:8d} {guard:5.3f} "
            f"{max(f(r, 'dq_rel_vs_exact') for r in values):8.4f} "
            f"{max(f(r, 'dk_rel_vs_exact') for r in values):8.4f} "
            f"{sum(f(r, 'dq_rel_vs_exact') for r in values) / len(values):8.4f} "
            f"{sum(f(r, 'dk_rel_vs_exact') for r in values) / len(values):8.4f} "
            f"{1.0e6 * max(f(r, 'sat_rate') for r in values):11.2f} "
            f"{100.0 * max(f(r, 'zero_rate') for r in values):11.3f}"
        )

    print("\nTRAINING BASELINE DYNAMIC METRICS BY SIGMA")
    seen = {}
    for row in rows:
        key = int(row["sigma_index"])
        seen[key] = row
    for key in sorted(seen):
        row = seen[key]
        print(
            f"sigma={key:4d} dynamic_dq_rel={f(row, 'dynamic_dq_rel'):.6f} "
            f"dynamic_dk_rel={f(row, 'dynamic_dk_rel'):.6f} "
            f"dynamic_dq_cos={f(row, 'dynamic_dq_cos'):.6f} "
            f"dynamic_dk_cos={f(row, 'dynamic_dk_cos'):.6f}"
        )


def report_random() -> None:
    rows = read_csv(ROOT / "sdxl_periodic_ds_guard_sweep.csv")
    print("\nRANDOM/MODEL-QKV CAPTURE AGGREGATE")
    print("seq interval guard n worst_dq worst_dk mean_dq mean_dk max_sat_ppm max_zero_pct")
    aggregate = defaultdict(list)
    for row in rows:
        aggregate[(int(row["seq_len"]), int(row["interval"]), float(row["guard"]))].append(row)
    for key in sorted(aggregate):
        seq_len, interval, guard = key
        values = aggregate[key]
        print(
            f"{seq_len:4d} {interval:8d} {guard:5.3f} {len(values):3d} "
            f"{max(f(r, 'dq_rel_vs_exact') for r in values):8.4f} "
            f"{max(f(r, 'dk_rel_vs_exact') for r in values):8.4f} "
            f"{sum(f(r, 'dq_rel_vs_exact') for r in values) / len(values):8.4f} "
            f"{sum(f(r, 'dk_rel_vs_exact') for r in values) / len(values):8.4f} "
            f"{1.0e6 * max(f(r, 'sat_rate') for r in values):11.2f} "
            f"{100.0 * max(f(r, 'zero_rate') for r in values):11.3f}"
        )

    print("\nRANDOM/MODEL-QKV SCALE RATIO AGGREGATE")
    ratios = read_csv(ROOT / "sdxl_periodic_ds_scale_ratios.csv")
    aggregate_ratio = defaultdict(list)
    for row in ratios:
        aggregate_ratio[(int(row["seq_len"]), int(row["interval"]))].append(row)
    print("seq interval n p99_max max_max")
    for key in sorted(aggregate_ratio):
        seq_len, interval = key
        values = aggregate_ratio[key]
        print(
            f"{seq_len:4d} {interval:8d} {len(values):3d} "
            f"{max(f(r, 'p99') for r in values):8.3f} {max(f(r, 'max') for r in values):8.3f}"
        )


report_training()
report_random()
