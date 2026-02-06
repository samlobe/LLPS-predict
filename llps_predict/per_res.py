import csv
from typing import NamedTuple
from typing import Optional

import numpy as np
import torch

from llps_predict.inference import (
    MODEL_NAME_TO_CHECKPOINT,
    configure_torch_hub_dir,
    describe_esm_checkpoint_state,
    embed_sequences,
    load_esm_model,
    load_inputs,
    load_torch_lr_from_pt,
)


class FragmentRecord(NamedTuple):
    name: str
    sequence: str
    probe_length: int
    start_idx: int
    end_idx: int


def require_single_sequence_input(sequence_arg: str) -> tuple[str, str]:
    names, sequences = load_inputs(sequence_arg)
    if len(sequences) != 1:
        raise ValueError(
            "Per-residue mode requires exactly one input sequence. "
            "Provide a single raw sequence or a FASTA with one entry."
        )
    return names[0], sequences[0]


def build_fragments(
    sequence_name: str,
    sequence: str,
    probe_lengths: list[int],
    stride: int,
) -> list[FragmentRecord]:
    if stride <= 0:
        raise ValueError("--stride must be a positive integer.")
    if not probe_lengths:
        raise ValueError("--probe_lengths cannot be empty.")

    unique_lengths = []
    for length in probe_lengths:
        if length <= 0:
            raise ValueError("All probe lengths must be positive integers.")
        if length not in unique_lengths:
            unique_lengths.append(length)

    n_res = len(sequence)
    fragments: list[FragmentRecord] = []
    for length in unique_lengths:
        if length > n_res:
            raise ValueError(
                f"Probe length {length} is longer than sequence length {n_res}."
            )
        for start in range(0, n_res - length + 1, stride):
            end = start + length
            name = f"{sequence_name}_{length}aa_{start + 1}-{end}"
            fragments.append(
                FragmentRecord(
                    name=name,
                    sequence=sequence[start:end],
                    probe_length=length,
                    start_idx=start,
                    end_idx=end,
                )
            )

    if not fragments:
        raise RuntimeError("No fragments were generated. Check probe lengths and stride.")

    return fragments


def predict_fragment_scores(
    fragments: list[FragmentRecord],
    lr_checkpoint: str,
    esm_model_name: str,
    esm_weights_dir: Optional[str],
    use_gpu: bool,
    toks_per_batch: int,
    truncation_seq_length: int,
) -> np.ndarray:
    hub_dir = configure_torch_hub_dir(esm_weights_dir)
    checkpoint_name = MODEL_NAME_TO_CHECKPOINT[esm_model_name]
    checkpoint_path = describe_esm_checkpoint_state(hub_dir, checkpoint_name)

    esm_model, alphabet, layer = load_esm_model(esm_model_name)
    torch_lr = load_torch_lr_from_pt(lr_checkpoint)

    if checkpoint_path.exists():
        print(f"Using ESM2 checkpoint: {checkpoint_path}")

    names = [fragment.name for fragment in fragments]
    sequences = [fragment.sequence for fragment in fragments]

    embeddings = embed_sequences(
        names=names,
        sequences=sequences,
        esm_model=esm_model,
        alphabet=alphabet,
        layer=layer,
        use_gpu=use_gpu,
        toks_per_batch=toks_per_batch,
        truncation_seq_length=truncation_seq_length,
    )

    with torch.no_grad():
        logits = torch_lr(torch.tensor(embeddings, dtype=torch.float32))
        scores = torch.sigmoid(logits).squeeze(1).cpu().numpy()

    return scores


def aggregate_per_residue_scores(
    sequence_length: int,
    probe_lengths: list[int],
    fragments: list[FragmentRecord],
    fragment_scores: np.ndarray,
) -> tuple[list[int], dict[int, np.ndarray], np.ndarray]:
    sums_by_length = {length: np.zeros(sequence_length, dtype=np.float64) for length in probe_lengths}
    counts_by_length = {length: np.zeros(sequence_length, dtype=np.int32) for length in probe_lengths}

    for fragment, score in zip(fragments, fragment_scores):
        sums_by_length[fragment.probe_length][fragment.start_idx : fragment.end_idx] += score
        counts_by_length[fragment.probe_length][fragment.start_idx : fragment.end_idx] += 1

    avg_by_length: dict[int, np.ndarray] = {}
    stacked = []
    for length in probe_lengths:
        avg = np.full(sequence_length, np.nan, dtype=np.float64)
        mask = counts_by_length[length] > 0
        avg[mask] = sums_by_length[length][mask] / counts_by_length[length][mask]
        avg_by_length[length] = avg
        stacked.append(avg)

    stacked_arr = np.vstack(stacked)
    valid_counts = np.sum(~np.isnan(stacked_arr), axis=0)
    overall_sum = np.nansum(stacked_arr, axis=0)
    overall_avg = np.full(sequence_length, np.nan, dtype=np.float64)
    valid_mask = valid_counts > 0
    overall_avg[valid_mask] = overall_sum[valid_mask] / valid_counts[valid_mask]
    residues = list(range(1, sequence_length + 1))
    return residues, avg_by_length, overall_avg


def write_fragment_scores_csv(
    output_path: str,
    fragments: list[FragmentRecord],
    scores: np.ndarray,
) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Name", "LLPS Score"])
        for fragment, score in zip(fragments, scores):
            writer.writerow([fragment.name, float(score)])


def write_per_res_scores_csv(
    output_path: str,
    residues: list[int],
    probe_lengths: list[int],
    avg_by_length: dict[int, np.ndarray],
    overall_avg: np.ndarray,
) -> None:
    headers = ["Residue"] + [f"{length}aa_Avg_Score" for length in probe_lengths] + ["Avg_Score"]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for i, residue in enumerate(residues):
            row = [residue]
            for length in probe_lengths:
                value = avg_by_length[length][i]
                row.append(float(value) if not np.isnan(value) else "")
            avg_value = overall_avg[i]
            row.append(float(avg_value) if not np.isnan(avg_value) else "")
            writer.writerow(row)
