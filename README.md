# LLPSpredict
This model was trained to predict intrinsically disordered proteins (IDPs) or intrinsically disordered regions (IDRs) that drive LLPS using an ESM2-based classification model. The training data was from the CD-CODE database (see [paper](https://doi.org/10.1038/s41592-023-01831-0) and [website](https://cd-code.org/)), and human IDR sequences are from Tesei et al. 2024 (see their repo [here](https://github.com/KULL-Centre/_2023_Tesei_IDRome/tree/main)).

## Quick Start
Use a machine with at least 12 GB RAM for ESM2 3B inference. CPU works, but GPU is much faster.

```bash
conda create -n llps-predict python=3.9
conda activate llps-predict
pip install fair-esm
conda install pytorch
```
Essentially you need an environment with python <= 3.9 and pytorch installed.

Alternatively:

```bash
conda env create -f environment.yml
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

`predict.py` outputs a csv with LLPS scores (0-1 scale).


## How it works
I found 118 LLPS driver proteins on CD-CODE (using the API with [code](https://github.com/KULL-Centre/_2023_Tesei_IDRome/blob/main/CD-CODE.ipynb) from Tesei et al.). I considered the human IDRs that were part of those proteins to be positive samples with enriched LLPS propensity: 180 IDRs. Then I considered all other human IDRs to be negative samples.

I weighted each class evenly and fit a logistic regression model on mean embeddings from ESM2 3B. I held out 20% of each class for validation and did feature selection (L1 regularization) and regularization (L2 on selected features), resulting in ROC AUC of ~0.79 on validation. Then I retrained on all data for the final model.

Conceptually, the pipeline is:
1. Generate an ESM2 embedding per sequence
2. Apply a logistic-regression head on that embedding to predict a LLPS propensity

## Acknowledgments
- The developers of ESM
- Tesei, Lindorff-Larsen et al. for their [work](https://doi.org/10.1038/s41586-023-07004-5) that inspired parts of this
- CD-CODE contributors
- My advisors: Scott Shell and Joan-Emma Shea
