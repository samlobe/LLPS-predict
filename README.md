# LLPS-predict
LLPS propensity prediction for IDRs using ESM2 embeddings and a logistic-regression head.

## Install
Conda-first is recommended for reliable PyTorch setup.

```bash
conda create -n LLPS-predict python=3.9
conda activate LLPS-predict
conda install pytorch
pip install fair-esm
pip install -e .
```

Alternative:

```bash
conda env create -f environment.yml -n LLPS-predict
conda activate LLPS-predict
pip install -e .
```

## CLI Commands
After installation, two console commands are available:
- `llps-predict`
- `llps-predict-per-res`

## Usage
Single sequence score:

```bash
llps-predict --sequence YGQSSYSSYGQSQNTGY
```

FASTA with many sequences:

```bash
llps-predict --sequence example.fasta --output example_sequences_LLPS_propensities.csv
```

Efficient token batching for large FASTA inputs:

```bash
llps-predict \
  --sequence many_sequences.fasta \
  --toks_per_batch 4096 \
  --truncation_seq_length 1022 \
  --output LLPS_propensity.csv
```

Per-residue LLPS profile for a single sequence/FASTA entry:

```bash
llps-predict-per-res \
  --sequence tau.fasta \
  --probe_lengths 15 25 40 \
  --stride 1 \
  --output tau_perRes_scores.csv
```

## Notes
- `--toks_per_batch`: higher is faster but uses more memory.
- `--truncation_seq_length`: sequences longer than this are truncated for ESM inference.
- `llps-predict-per-res` requires exactly one input sequence.

## Export/Update LR Checkpoint
Inference uses a pure torch `.pt` LR checkpoint.
If you retrain the sklearn LR model, export a new checkpoint with:

```bash
conda install scikit-learn=1.5.1 joblib
python scripts/export_lr_joblib_to_pt.py \
  --joblib model_development/LLPS_model_latest.joblib \
  --out model_development/LLPS_model_latest.pt
```

## Acknowledgments
- ESM developers
- Tesei, Lindorff-Larsen et al. ([paper](https://doi.org/10.1038/s41586-023-07004-5))
- CD-CODE contributors
- Scott Shell and Joan-Emma Shea
