import importlib.util
from pathlib import Path

import torch

DATA_DIR = Path(__file__).resolve().parent
MODULE_PATH = DATA_DIR / "simulate_periodic_ds_captures.py"
CAPTURE_PATHS = (
    DATA_DIR / "sdxl_periodic_ds_inputs_training_dout_latent128.pt",
    DATA_DIR / "sdxl_periodic_ds_inputs_training_dout_latent128_seed2.pt",
)

spec = importlib.util.spec_from_file_location("periodic_capture_sim", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.__dict__["MAX_HEADS"] = 10

for capture_path in CAPTURE_PATHS:
    payload = torch.load(capture_path, map_location="cpu", weights_only=False)
    for record in payload["records"]:
        rows, _ = module.run_case(record, capture_path.name, -1)
        row = rows[0]
        print(
            f"{capture_path.name} sigma_index={record['sigma_index']} sigma={float(record['sigma']):.6f} "
            f"heads={row['heads']} "
            f"dq_cos={float(row['dynamic_dq_cos']):.9f} dq_rel={float(row['dynamic_dq_rel']):.9f} "
            f"dk_cos={float(row['dynamic_dk_cos']):.9f} dk_rel={float(row['dynamic_dk_rel']):.9f}"
        )
