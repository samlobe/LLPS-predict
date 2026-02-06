import argparse
import csv
import sys
from time import perf_counter

import torch

from llps_predict.inference import (
    MODEL_NAME_TO_CHECKPOINT,
    MODEL_NAME_TO_LAYER,
    configure_torch_hub_dir,
    describe_esm_checkpoint_state,
    embed_sequences,
    load_esm_model,
    load_inputs,
    load_torch_lr_from_pt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict LLPS propensity for one sequence or a FASTA file of sequences."
    )
    parser.add_argument(
        "--sequence",
        "-s",
        required=True,
        help="A protein sequence string, or a path to a FASTA file.",
    )
    parser.add_argument(
        "--lr_checkpoint",
        default="model_development/LLPS_model_latest.pt",
        help="Path to LR weights (.pt).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="LLPS_propensity.csv",
        help="Output CSV path. Default: LLPS_propensity.csv",
    )
    parser.add_argument(
        "--nogpu",
        action="store_true",
        help="Force CPU inference even when CUDA is available.",
    )
    parser.add_argument(
        "--ESM_model",
        default="3B",
        choices=sorted(MODEL_NAME_TO_LAYER.keys()),
        help="ESM2 backbone to use. Currently supported: 3B",
    )
    parser.add_argument(
        "--esm_weights_dir",
        default=None,
        help=(
            "Optional custom Torch Hub directory for ESM weights cache. "
            "Weights are searched/downloaded at <dir>/checkpoints/."
        ),
    )
    parser.add_argument(
        "--toks_per_batch",
        type=int,
        default=4096,
        help=(
            "Maximum tokens per embedding batch. "
            "Higher values are faster but use more memory."
        ),
    )
    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=1022,
        help="Truncate sequences longer than this length for ESM inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    names, sequences = load_inputs(args.sequence)

    weights_t0 = perf_counter()

    hub_dir = configure_torch_hub_dir(args.esm_weights_dir)
    checkpoint_name = MODEL_NAME_TO_CHECKPOINT[args.ESM_model]
    checkpoint_path = describe_esm_checkpoint_state(hub_dir, checkpoint_name)

    esm_model, alphabet, layer = load_esm_model(args.ESM_model)
    torch_lr = load_torch_lr_from_pt(args.lr_checkpoint)

    if checkpoint_path.exists():
        print(f"Using ESM2 checkpoint: {checkpoint_path}")
    else:
        downloaded_path = hub_dir / "checkpoints" / checkpoint_name
        if downloaded_path.exists():
            print(f"Downloaded ESM2 checkpoint to: {downloaded_path}")
        else:
            print(
                "Warning: ESM2 checkpoint file was not found after model initialization. "
                "fair-esm may have used an alternate cache path."
            )

    weights_loading_seconds = perf_counter() - weights_t0

    predict_t0 = perf_counter()

    use_gpu = torch.cuda.is_available() and not args.nogpu
    device_name = "cuda" if use_gpu else "cpu"
    print(f"Using device: {device_name}")
    if not use_gpu:
        print("CPU mode detected. For faster runtime, see README GPU install instructions.")
    embeddings = embed_sequences(
        names=names,
        sequences=sequences,
        esm_model=esm_model,
        alphabet=alphabet,
        layer=layer,
        use_gpu=use_gpu,
        toks_per_batch=args.toks_per_batch,
        truncation_seq_length=args.truncation_seq_length,
    )

    with torch.no_grad():
        logits = torch_lr(torch.tensor(embeddings, dtype=torch.float32))
        predictions = torch.sigmoid(logits).squeeze(1).cpu().numpy()

    prediction_seconds = perf_counter() - predict_t0

    if len(names) == 1:
        print(f"LLPS probability: {predictions[0]:.4f}")

    with open(args.output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Name", "LLPS Score"])
        for name, score in zip(names, predictions):
            writer.writerow([name, float(score)])
    print(f"LLPS predictions saved to {args.output}")

    print(f"Weights loading time (s): {weights_loading_seconds:.3f}")
    print(f"Prediction time (s): {prediction_seconds:.3f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
