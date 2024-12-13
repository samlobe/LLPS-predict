#%%
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

ESM_model = 'esm2_3B'; layers = 36; embeddings_dir = '../IDR_embeddings'
# ESM_model = 'esm2_650M'; layers = 33; embeddings_dir = '/home/sam/Research/ESM3/IDRome_embeddings/esm2_650M_IDRome_embeddings'
# ESM_model = 'esm3-large-2024-03'; embeddings_dir = '/home/sam/Research/ESM3/IDRome_embeddings/esm3-large-2024-03_IDRome_embeddings'
# ESM_model = 'esm3-small-2024-03'; embeddings_dir = '/home/sam/Research/ESM3/IDRome_embeddings/esm3-medium-2024-03_IDRome_embeddings'
# ESM_model = 'esmc-6b'; embeddings_dir = '/home/sam/Research/ESM3/IDRome_embeddings/esmc-6b-2024-12_IDRome_embeddings'
# ESM_model = 'esmc-300m'; embeddings_dir = '/home/sam/Research/ESM3/IDRome_embeddings/esmc_300m_IDRome_embeddings.npy'
# ESM_model = 'esmc-600m'; embeddings_dir = '/home/sam/Research/ESM3/IDRome_embeddings/esmc_600m_IDRome_embeddings.npy'

# 118 LLPS drive proteins from CD-CODE
drivers = list(pd.read_csv('drivers.csv', index_col=0, header=None).index)

# parse headers in fasta file
def parse_fasta(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    headers = []
    for line in lines:
        if line[0] == '>':
            headers.append(line[1:].strip())
    return headers

IDRs = parse_fasta('IDRs.fasta')
IDR_proteins = [IDR.split('|')[-1] for IDR in IDRs]

driver_embeddings = {}
non_driver_embeddings = {}

if ESM_model in ['esmc-300m', 'esmc-600m']:
    embeddings = np.load(embeddings_dir)
    for embedding, IDR, IDR_name in tqdm(zip(embeddings, IDRs, IDR_proteins)):
        if IDR_name in drivers:
            driver_embeddings[IDR] = embedding
        else:
            non_driver_embeddings[IDR] = embedding
else:
    for IDR, IDR_name in tqdm(zip(IDRs, IDR_proteins)):
        if ESM_model[:4] == 'esm2':
            if IDR_name in drivers:
                driver_embeddings[IDR] = torch.load(f'{embeddings_dir}/{IDR}.pt')['mean_representations'][layers].cpu().numpy()
            else:
                non_driver_embeddings[IDR] = torch.load(f'{embeddings_dir}/{IDR}.pt')['mean_representations'][layers].cpu().numpy()
        # else:
        elif ESM_model[:4] == 'esm3' or ESM_model == 'esmc-6b':
            if IDR_name in drivers:
                try:
                    driver_embeddings[IDR] = torch.load(f'{embeddings_dir}/{IDR}.pt').cpu().numpy()
                except:
                    print(f'{IDR} not found')
            else:
                try:
                    non_driver_embeddings[IDR] = torch.load(f'{embeddings_dir}/{IDR}.pt').cpu().numpy()
                except:
                    print(f'{IDR} not found')


#%%
# save as csv
driver_df = pd.DataFrame(driver_embeddings).T
non_driver_df = pd.DataFrame(non_driver_embeddings).T
columns = [f'embedding_{i}' for i in range(driver_df.shape[1])]

# set column names
driver_df.columns = columns
non_driver_df.columns = columns

# output as csv
driver_df.to_csv(f'../features/driver_embeddings_{ESM_model}.csv')
non_driver_df.to_csv(f'../features/non_driver_embeddings_{ESM_model}.csv')

#%%