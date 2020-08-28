# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:34:54 2020

@author: jesus
"""
import numpy as np
import json
from itertools import zip_longest

# An optional utility to display a progress bar
# for long-running loops. `pip install tqdm`.
from tqdm import tqdm

from pymatgen import MPRester

mpr = MPRester("cn1xbygDye7gPsW3")

entries = mpr.query({"elements": "O", "nelements": {"$gte": 2}}, ["material_id"])
oxide_mp_ids = [e['material_id'] for e in entries]






print(len(oxide_mp_ids))

# A utility function to "chunk" our queries

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

data = []
mpid_groups = [g for g in grouper(oxide_mp_ids, 1000)]
for group in tqdm(mpid_groups):
    # The last group may have fewer than 1000 actual ids,
    # so filter the `None`s out.
    mpid_list =list( filter(None, group))
    entries = mpr.query({"material_id": {"$in": mpid_list}}, ["material_id", "cif"])
    data.extend(entries)
    
import os

if not os.path.exists('mp_oxide_cifs'):
    os.mkdir('mp_oxide_cifs')

for d in tqdm(data):
    with open("mp_oxide_cifs/{}.cif".format(d["material_id"]), 'w') as f:
        f.write(d["cif"])
        
# Save IDs for saved structures, so you can
# efficiently update later.

with open('oxide_mp_ids.json', 'w') as f:
    json.dump(oxide_mp_ids, f)
    
    
from pymatgen import Structure

Structure.from_file('mp_oxide_cifs/mp-190.cif')


a=oxide_mp_ids
b=np.zeros(len(oxide_mp_ids))
c=[]
for i in range(0, len(oxide_mp_ids)):
    b[i]=i
    c.append(np.hstack((a[i],b[i])))
    

np.savetxt("prueba.csv", c, fmt='%s' ,delimiter="," )