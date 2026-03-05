# LLPS-predict
LLPS propensity prediction for IDRs using ESM2 embeddings and a logistic-regression head.

## Install
Choose one of the install paths below.

### Recommended (NVIDIA GPU, fastest)

```bash
conda create -n LLPS-predict python=3.9
conda activate LLPS-predict
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install fair-esm
pip install -e .
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available()); print('cuda_version', torch.version.cuda)"
```

Expected check output includes `cuda True`.

If you have a NVIDIA GPU and this fails, your driver/GPU/platform likely needs a different CUDA-enabled PyTorch build. Use the official selector to pick a compatible install command:
https://pytorch.org/get-started/locally/

### CPU fallback (Mac / no NVIDIA GPU, slower but supported)

```bash
conda create -n LLPS-predict python=3.9
conda activate LLPS-predict
pip install torch
pip install fair-esm
pip install -e .
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available()); print('cuda_version', torch.version.cuda)"
```

Expected check output includes `cuda False`.

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
llps-predict --sequence example_multi_sequences.fasta --output example_sequences_LLPS_propensities.csv
```

You may alter token batching if you run into memory issues:

```bash
llps-predict \
  --sequence example_multi_sequences.fasta \
  --toks_per_batch 4096 \
  --truncation_seq_length 1022 \
  --output LLPS_propensity.csv
```

Per-residue LLPS profile for a single sequence/FASTA entry:

```bash
llps-predict-per-res \
  --sequence example_single_sequence.fasta \
  --probe_lengths 15 25 40 \
  --stride 1 \
  --output tau_perRes_LLPS_scores.csv
```

## Notes
- `--toks_per_batch`: higher is faster but uses more memory.
- `--truncation_seq_length`: sequences longer than this are truncated for ESM2 inference.

## Export/Update LR Checkpoint
Inference uses a pure torch `.pt` LR checkpoint.
If you retrain the sklearn LR model, export a new checkpoint with:

```bash
conda install scikit-learn=1.5.1 joblib
python scripts/export_lr_joblib_to_pt.py \
  --joblib model_development/LLPS_model_latest.joblib \
  --out model_development/LLPS_model_latest.pt
```

## Additional Manuscript Data
- Manuscript datasets and minimal reproducibility scripts are in the companion repository:
  - `https://github.com/samlobe/amyloid-predict/manuscript_data`
- DOI-backed Zenodo archive for the manuscript snapshot will be listed there:
  - `https://doi.org/TBD`

## Acknowledgments
- ESM developers
- Tesei, Lindorff-Larsen et al. ([paper](https://doi.org/10.1038/s41586-023-07004-5))
- CD-CODE contributors
- Scott Shell and Joan-Emma Shea
