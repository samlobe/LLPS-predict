import re
import os
from pathlib import Path
from typing import Optional

import esm
import numpy as np
import torch
from esm import FastaBatchedDataset

MODEL_NAME_TO_LAYER = {
    "3B": 36,
}
MODEL_NAME_TO_CHECKPOINT = {
    "3B": "esm2_t36_3B_UR50D.pt",
}

ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
DEFAULT_LR_CHECKPOINT = "model_development/LLPS_model_latest.pt"
LR_CHECKPOINT_ENV_VAR = "LLPS_PREDICT_LR_CHECKPOINT"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def is_fasta_path(value: str) -> bool:
    lower = value.lower()
    return lower.endswith(".fasta") or lower.endswith(".fa") or lower.endswith(".faa")


def sanitize_sequence(seq: str) -> str:
    seq = re.sub(r"\s+", "", seq).upper()
    if not seq:
        raise ValueError("Encountered an empty sequence after whitespace cleanup.")
    invalid = sorted(set(ch for ch in seq if ch not in ALLOWED_AA))
    if invalid:
        raise ValueError(
            f"Sequence contains invalid residue codes: {''.join(invalid)}. "
            "Allowed letters are ACDEFGHIKLMNPQRSTVWY and common ambiguous codes BXZJUO."
        )
    return seq


def parse_fasta(path: Path) -> tuple[list[str], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    names: list[str] = []
    seqs: list[str] = []
    current_name: Optional[str] = None
    current_seq: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    seqs.append(sanitize_sequence("".join(current_seq)))
                    names.append(current_name)
                current_name = line[1:].strip() or f"sequence_{len(names) + 1}"
                current_seq = []
            else:
                if current_name is None:
                    raise ValueError(
                        f"Invalid FASTA format in {path}: sequence data appears before first header."
                    )
                current_seq.append(line)

    if current_name is not None:
        seqs.append(sanitize_sequence("".join(current_seq)))
        names.append(current_name)

    if not names:
        raise ValueError(f"No FASTA entries found in {path}")
    return names, seqs


def load_inputs(sequence_arg: str) -> tuple[list[str], list[str]]:
    candidate_path = Path(sequence_arg)
    if candidate_path.exists() or is_fasta_path(sequence_arg):
        if not is_fasta_path(sequence_arg):
            raise ValueError(
                f"Input path exists but does not look like FASTA: {sequence_arg}. "
                "Use .fasta/.fa/.faa, or pass a raw sequence string."
            )
        return parse_fasta(candidate_path)

    sequence = sanitize_sequence(sequence_arg)
    return ["sequence_1"], [sequence]


def configure_torch_hub_dir(custom_dir: Optional[str]) -> Path:
    if custom_dir:
        hub_dir = Path(custom_dir).expanduser().resolve()
        hub_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(hub_dir))
        return hub_dir
    return Path(torch.hub.get_dir()).expanduser().resolve()


def describe_esm_checkpoint_state(hub_dir: Path, checkpoint_name: str) -> Path:
    checkpoint_path = hub_dir / "checkpoints" / checkpoint_name
    if checkpoint_path.exists():
        print(f"ESM2 checkpoint found: {checkpoint_path}")
    else:
        print(
            "ESM2 checkpoint not found.\n"
            f"Searched: {checkpoint_path}\n"
            "Will attempt automatic download via fair-esm now.\n"
            "If this fails, rerun with --esm_weights_dir <writable_dir> to choose another cache location."
        )
    return checkpoint_path


def load_esm_model(model_size: str) -> tuple[torch.nn.Module, object, int]:
    if model_size != "3B":
        raise ValueError(f"Unsupported ESM2 model: {model_size}")
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    model.eval()
    return model, alphabet, MODEL_NAME_TO_LAYER[model_size]


def embed_sequences(
    names: list[str],
    sequences: list[str],
    esm_model: torch.nn.Module,
    alphabet: object,
    layer: int,
    use_gpu: bool,
    toks_per_batch: int = 4096,
    truncation_seq_length: int = 1022,
) -> np.ndarray:
    if toks_per_batch <= 0:
        raise ValueError("--toks_per_batch must be a positive integer.")
    if truncation_seq_length <= 0:
        raise ValueError("--truncation_seq_length must be a positive integer.")

    dataset = FastaBatchedDataset(names, sequences)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    batch_converter = alphabet.get_batch_converter(truncation_seq_length)

    device = torch.device("cuda") if use_gpu else torch.device("cpu")
    esm_model = esm_model.to(device)

    truncated_count = sum(len(seq) > truncation_seq_length for seq in sequences)
    if truncated_count > 0:
        print(
            f"Warning: {truncated_count} sequence(s) exceed {truncation_seq_length} residues "
            "and will be truncated for ESM inference."
        )

    pooled_embeddings: list[Optional[torch.Tensor]] = [None] * len(dataset)

    with torch.no_grad():
        for batch_idx, batch_indices in enumerate(batches):
            if len(batches) > 1:
                print(
                    f"Embedding batch {batch_idx + 1}/{len(batches)} "
                    f"({len(batch_indices)} sequences)"
                )

            batch_records = [dataset[idx] for idx in batch_indices]
            _, _, batch_tokens = batch_converter(batch_records)
            if use_gpu:
                batch_tokens = batch_tokens.to(device=device, non_blocking=True)

            outputs = esm_model(batch_tokens, repr_layers=[layer], return_contacts=False)
            reps_cpu = outputs["representations"][layer].to(device="cpu")

            for row_idx, seq_idx in enumerate(batch_indices):
                trunc_len = min(truncation_seq_length, len(dataset.sequence_strs[seq_idx]))
                pooled_embeddings[seq_idx] = reps_cpu[row_idx, 1 : trunc_len + 1].mean(0).clone()

    if any(emb is None for emb in pooled_embeddings):
        raise RuntimeError("Failed to compute embeddings for one or more sequences.")

    embeddings = torch.stack([emb for emb in pooled_embeddings if emb is not None]).numpy()
    return embeddings


def load_torch_lr_from_pt(pt_path: str) -> torch.nn.Linear:
    resolved_path = resolve_lr_checkpoint_path(pt_path)
    try:
        ckpt = torch.load(resolved_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(resolved_path, map_location="cpu")

    required_keys = ["embedding_dim", "weight_full", "bias"]
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        raise ValueError(f"Invalid LR checkpoint {resolved_path}; missing keys: {missing}")

    embedding_dim = int(ckpt["embedding_dim"])
    weight_full = torch.as_tensor(ckpt["weight_full"], dtype=torch.float32).view(1, embedding_dim)
    bias = torch.as_tensor(ckpt["bias"], dtype=torch.float32).view(1)

    linear = torch.nn.Linear(embedding_dim, 1, bias=True)
    with torch.no_grad():
        linear.weight.copy_(weight_full)
        linear.bias.copy_(bias)
    linear.eval()
    return linear


def _candidate_lr_checkpoint_paths(pt_path: str) -> list[Path]:
    candidates: list[Path] = []

    env_raw = os.environ.get(LR_CHECKPOINT_ENV_VAR)
    if env_raw:
        candidates.append(Path(env_raw).expanduser().resolve())

    raw = Path(pt_path).expanduser()
    if raw.is_absolute():
        candidates.append(raw.resolve())
    else:
        candidates.append((Path.cwd() / raw).resolve())
        candidates.append((PROJECT_ROOT / raw).resolve())

    if raw == Path(DEFAULT_LR_CHECKPOINT):
        candidates.append((PROJECT_ROOT / DEFAULT_LR_CHECKPOINT).resolve())

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def resolve_lr_checkpoint_path(pt_path: str) -> Path:
    candidates = _candidate_lr_checkpoint_paths(pt_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    attempted = "\n".join(f"  - {p}" for p in candidates) if candidates else "  - <none>"
    raise FileNotFoundError(
        "LR checkpoint file not found.\n"
        f"Requested: {pt_path}\n"
        f"Current working directory: {Path.cwd()}\n"
        "Searched:\n"
        f"{attempted}\n"
        "Fix options:\n"
        "  1) pass --lr_checkpoint /absolute/path/to/LLPS_model_latest.pt\n"
        f"  2) set {LR_CHECKPOINT_ENV_VAR} to the checkpoint path\n"
        f"  3) from a source checkout, keep the default file at {DEFAULT_LR_CHECKPOINT}"
    )
