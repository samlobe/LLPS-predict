#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export sklearn LR + StandardScaler + feature_indices joblib bundle to pure torch .pt checkpoint."
    )
    parser.add_argument(
        "--joblib",
        default="model_development/LLPS_model_latest.joblib",
        help="Input joblib path.",
    )
    parser.add_argument(
        "--out",
        default="model_development/LLPS_model_latest.pt",
        help="Output torch checkpoint path.",
    )
    return parser.parse_args()


def export_joblib_to_pt(joblib_path: Path, out_path: Path) -> None:
    bundle = joblib.load(joblib_path)

    required = ["model", "scaler", "feature_indices"]
    missing = [k for k in required if k not in bundle]
    if missing:
        raise ValueError(f"Invalid joblib bundle {joblib_path}; missing keys: {missing}")

    lr_model = bundle["model"]
    scaler = bundle["scaler"]
    feature_indices = np.asarray(bundle["feature_indices"], dtype=np.int64)

    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    coef = np.asarray(lr_model.coef_, dtype=np.float64).reshape(-1)
    intercept = float(np.asarray(lr_model.intercept_, dtype=np.float64).reshape(-1)[0])

    if coef.shape[0] != feature_indices.shape[0]:
        raise ValueError(
            f"Coefficient length ({coef.shape[0]}) != feature_indices length ({feature_indices.shape[0]})."
        )

    embedding_dim = int(mean.shape[0])
    weight_full = np.zeros(embedding_dim, dtype=np.float32)

    selected_scales = scale[feature_indices]
    selected_means = mean[feature_indices]

    weight_full[feature_indices] = (coef / selected_scales).astype(np.float32)
    bias = np.float32(intercept - np.sum((coef * selected_means) / selected_scales))

    ckpt = {
        "embedding_dim": embedding_dim,
        "feature_indices": torch.as_tensor(feature_indices, dtype=torch.long),
        "weight_full": torch.as_tensor(weight_full, dtype=torch.float32),
        "bias": torch.as_tensor(bias, dtype=torch.float32),
        "meta": {
            "source_joblib": str(joblib_path),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "notes": "scaler folded into weights",
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)


if __name__ == "__main__":
    args = parse_args()
    joblib_path = Path(args.joblib)
    out_path = Path(args.out)
    export_joblib_to_pt(joblib_path, out_path)
    print(f"Exported LR checkpoint: {out_path}")
