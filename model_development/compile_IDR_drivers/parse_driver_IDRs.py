#%%
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

IDRs = 'IDRs.fasta'

# get list of IDR names
IDR_names = []
for record in SeqIO.parse(IDRs, "fasta"):
    IDR_names.append(record.id)

#%%
# get the part before the | in the IDR names
IDR_proteins = [IDR.split('_')[0] for IDR in IDR_names]

#%%
import pandas as pd
import numpy as np

# load drivers.csv file contents
drivers = list(pd.read_csv("drivers.csv", header=None).values.flatten())

#%%
# how many of the IDR proteins are drivers?
drivers_in_IDRs = [IDR for IDR in IDR_proteins if IDR in drivers]