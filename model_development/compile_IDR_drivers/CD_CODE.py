#%%
import requests
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time

# Fetch list of human biomolecular condensates
api_url = "https://cd-code.org/api/condensates?species_tax_id=9606"
headers = {"Authorization": "b537fe97ad384aea9cbfd7b25b38938a"} # using the public key from Tesei et. al.'s github repo
payload = {"page": 1, "size": 1000}
response = requests.get(api_url, headers=headers, params=payload)
dict_cd = json.loads(response.content.decode())

#%%
# Fetch drivers
tik = time()
drivers = np.empty(0)
no_func = 0

for cd_item in tqdm(dict_cd['data']):
    api_url = f"https://cd-code.org/api/condensates/{cd_item['uid']:s}"
    headers = {"Authorization": "b537fe97ad384aea9cbfd7b25b38938a"} # using the public key from Tesei et. al's github repo
    payload = {"page": 1, "size": 1000}
    response = requests.get(api_url, headers=headers, params=payload)
    dict_uid = json.loads(response.content.decode())['data']['protein_functional_type']
    if dict_uid == {}:
        no_func += 1
    drivers = np.append(drivers,[i.split('-')[0] for i in dict_uid if dict_uid[i]=="driver"])
tok = time()
print(f"Time elapsed: {tok-tik:.0f} seconds")

# save drivers and members to separate csv files
np.savetxt("drivers.csv", drivers, fmt="%s", delimiter=",")
