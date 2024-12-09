# LLPSpredict
This model was trained to predict intrinsically disordered proteins (IDPs) or intrinsically disordered regions (IDRs) that drive LLPS using an ESM2-based classification model. The training data was from the CD-CODE database (see [paper](https://doi.org/10.1038/s41592-023-01831-0) and [website](https://cd-code.org/)), and the human IDR sequences are from Tesei, et. al. 2024 (see their repo [here](https://github.com/KULL-Centre/_2023_Tesei_IDRome/tree/main)).

## Quick Start 
Use a computer that has 12+ GB of RAM. It works well on my 16GB RAM Mac CPU, but is way faster on my 24GB VRAM GPU.  

```bash
conda create -n LLPS-predict python=3.9
conda activate LLPS-predict
pip install fair-esm # install esm (to get embeddings from 3B parameter ESM2 model)
conda install pytorch pandas scikit-learn=1.5.1 matplotlib tqdm 
```

Alternatively, you may try installing from this environment.yml file:
`conda env create -f environment.yml` 

Note: this is identical to the amyloid-predict environment (see [here](https://github.com/samlobe/amyloid-predict) when it is made public).

After pip install, download the weights of the 3B parameter ESM2 model locally (~12GB of weights, compressed into ~5.3GB) by executing this in python:
```python
import esm
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D() # takes a few min; may fail if you don't have enough free RAM / SWAP
```
Then you can **extract ESM embeddings** and **predict LLPS propensity** of a peptide sequence (one-letter codes) with:  
`python predict.py --sequence VQIVYK`  
which will output a score (between 0-1) for each of the four amyloid classification models.

You can score multiple protein/peptide sequences in a fasta file (e.g. example.fasta) like this:  
`python predict.py --sequence example.fasta`  

If you already have ESM embeddings of sequences you can predict amyloidogenicity faster by pointing to the embeddings file (.pt): 
`python predict.py --embeddingsFile example_embeddings_dir/example_IDR_1.pt`  
or by pointing to the directory with all the embeddings files:  
`python predict.py --embeddingsDir example_embeddings_dir`

Extracting ESM embeddings with **predict.py** is reasonably fast on my Mac's CPUs, but is 3-4 orders of magnitude faster on a GPU. Predicting amyloidogenicity with **predict.py** should be very fast on CPUs or GPUs. 

The easiest/smartest way to extract ESM embeddings for hundreds or thousands of sequences is with the **extract.py** tool, which I altered slightly from the ESM repo. See their original documentation [here](https://github.com/facebookresearch/esm), or do `python extract.py -h` to see how to use it. Make sure to output the mean representations of the embeddings with the `--include mean` flag. 

## How it works
I found 118 LLPS driver proteins on CD-CODE (using the API with [code](https://github.com/KULL-Centre/_2023_Tesei_IDRome/blob/main/CD-CODE.ipynb) from Tesei, et. al.). I considered the human IDRs that were part of those proteins to be positive samples: 180 IDRs. Then I considered all other human IDRs to be negative samples. 
I weighted each class evenly and fit a logistic regression model on the IDRs' mean embeddings (from ESM2 3B model). I held out 20% of each class for validation and did feature selection (with L1 regularization) and regularization (i.e. L2 regularization on selected features) on the training data, resulting in a ROC AUC = 0.87 on the validation set. Then I retrained on all the data for the final model. Note that these IDRs have variable sizes.

To simplify how it works, it's like we use the protein language model (ESM2 3B) to generate a "bar code" for the protein, and then use a simple logistic regression model to predict LLPS propensity from the "bar code".

# Acknowledgments
- The developers of ESM
- Tesei, Lindorff-Larsen, et. al. for their [work](https://doi.org/10.1038/s41586-023-07004-5) that inspired parts of this
- My advisors: Scott Shell & Joan-Emma Shea
