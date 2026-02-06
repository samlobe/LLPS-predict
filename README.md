# LLPSpredict
This model predicts LLPS propensity for intrinsically disordered proteins/regions using ESM2 embeddings with a logistic-regression head.

## Quick Start
Use a machine with at least 12 GB RAM for ESM2 3B inference. CPU works; GPU is much faster.

```bash
conda create -n LLPS-predict python=3.9
conda activate LLPS-predict
pip install fair-esm
conda install pytorch
```

Alternative:

```bash
conda env create -f environment.yml -n LLPS-predict
conda activate LLPS-predict
```

## Usage
Single sequence:

```bash
python predict.py --sequence YGQSSYSSYGQSQNTGY
```

FASTA with multiple sequences:

```bash
python predict.py --sequence example.fasta --output example_sequences_LLPS_propensities.csv
```

## Notes if you run into memory/length issues
- `--toks_per_batch`: higher is faster, but uses more memory.
- `--truncation_seq_length`: sequences longer than this are truncated for ESM inference.

## How it works
1. Generate an ESM2 embedding per sequence.
2. Apply a logistic-regression head to predict LLPS propensity.

## Acknowledgments
- The developers of ESM
- Tesei, Lindorff-Larsen et al. for their [work](https://doi.org/10.1038/s41586-023-07004-5)
- CD-CODE contributors
- Scott Shell and Joan-Emma Shea
