import argparse
import os
import sys
from time import perf_counter

import torch

from llps_predict.inference import DEFAULT_LR_CHECKPOINT, LR_CHECKPOINT_ENV_VAR, MODEL_NAME_TO_LAYER
from llps_predict.per_res import (
    aggregate_per_residue_scores,
    build_fragments,
    predict_fragment_scores,
    require_single_sequence_input,
    write_fragment_scores_csv,
    write_per_res_scores_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict per-residue LLPS scores for a single protein sequence."
    )
    parser.add_argument(
        "--sequence",
        "-s",
        required=True,
        help="A protein sequence string, or a FASTA file containing one sequence.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output CSV path for per-residue scores. Default: <name>_perRes_scores.csv",
    )
    parser.add_argument(
        "--probe_lengths",
        nargs="+",
        type=int,
        default=[15, 25, 40],
        help="Fragment probe lengths. Default: 15 25 40",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding window stride for fragment generation. Default: 1",
    )
    parser.add_argument(
        "--fragment_scores_output",
        default=None,
        help="Optional output CSV for fragment-level LLPS scores.",
    )
    parser.add_argument(
        "--lr_checkpoint",
        default=DEFAULT_LR_CHECKPOINT,
        help=(
            "Path to LR weights (.pt). Relative paths are searched from both the current "
            f"working directory and the installed project root. Env override: {LR_CHECKPOINT_ENV_VAR}."
        ),
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
        help="Maximum tokens per embedding batch. Higher values are faster but use more memory.",
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

    sequence_name, sequence = require_single_sequence_input(args.sequence)
    safe_name = sequence_name.replace(" ", "_")

    if args.output is None:
        args.output = f"{safe_name}_perRes_scores.csv"

    if args.fragment_scores_output is None:
        args.fragment_scores_output = f"{safe_name}_fragment_scores.csv"

    args.probe_lengths = list(dict.fromkeys(args.probe_lengths))

    weights_t0 = perf_counter()
    fragments = build_fragments(
        sequence_name=safe_name,
        sequence=sequence,
        probe_lengths=args.probe_lengths,
        stride=args.stride,
    )

    use_gpu = torch.cuda.is_available() and not args.nogpu
    device_name = "cuda" if use_gpu else "cpu"
    print(f"Using device: {device_name}")
    if not use_gpu:
        print(
            "CPU mode detected. Per-residue runs can be slow on CPU; "
            "see README GPU install instructions."
        )
    fragment_scores = predict_fragment_scores(
        fragments=fragments,
        lr_checkpoint=args.lr_checkpoint,
        esm_model_name=args.ESM_model,
        esm_weights_dir=args.esm_weights_dir,
        use_gpu=use_gpu,
        toks_per_batch=args.toks_per_batch,
        truncation_seq_length=args.truncation_seq_length,
    )
    weights_loading_seconds = perf_counter() - weights_t0

    residues, avg_by_length, overall_avg = aggregate_per_residue_scores(
        sequence_length=len(sequence),
        probe_lengths=args.probe_lengths,
        fragments=fragments,
        fragment_scores=fragment_scores,
    )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fragment_output_dir = os.path.dirname(args.fragment_scores_output)
    if fragment_output_dir:
        os.makedirs(fragment_output_dir, exist_ok=True)

    write_fragment_scores_csv(args.fragment_scores_output, fragments, fragment_scores)
    write_per_res_scores_csv(args.output, residues, args.probe_lengths, avg_by_length, overall_avg)

    print(f"Fragment-level LLPS scores saved to {args.fragment_scores_output}")
    print(f"Per-residue LLPS scores saved to {args.output}")
    print(f"Prediction time (s): {weights_loading_seconds:.3f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
