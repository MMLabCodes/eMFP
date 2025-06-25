import sys
import time

import numpy as np
import pandas as pd 

from rdkit import Chem 
from rdkit.Chem import AllChem 
from tqdm import tqdm
from functions import mfp, clean_mols, internal_diversity

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
    

input_file = sys.argv[1]
custom_size = None
if len(sys.argv) >= 3:
    custom_size = int(sys.argv[2])

# Dataframe with smiles 
print('InputFile', input_file)
df = pd.read_csv(input_file)  

# smiles
smiles = df['smiles'].tolist()

# mol object and invalid molecules
print('Obtaining mols')
mols, _ = clean_mols(smiles, stereochem=False)

#nBits from 512 to 32768
if custom_size is not None:
    bit_sizes = [2**custom_size]
    idx_power = [custom_size]
else:
    bit_sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
    idx_power = [ i for i in range(9,20)]



# taken from youtube:
# Center for Computer-Assisted Synthesis
# Guilian Luchini, Dec 6th 2021
# Molecular Fingerprints: Clashing & Clustering
# https://www.youtube.com/watch?v=4V7V2GlxOto
# minute 11:57

print('\n')
print('Obtaining MFP for bits:')
print(bit_sizes)
print('\n')
unique = []
times = []
sumtimes = []
initial_time = time.time()

for iter, bit in enumerate(bit_sizes):
    fps = []
    fps_unf = []
    infos = []
    for mol in tqdm(mols, desc = f'nBits: {bit}'):
        fp, info, fp_unf = mfp(mol, bit, 2)
        fps.append(fp)
        infos.append(info)
        fps_unf.append(fp_unf)
        
    begintime = time.time()

    x = np.logical_or.reduce(fps).sum()
    unique.append( x )

    finaltime = time.time()

    internalDiv = internal_diversity(fps_unf)
    print('case,idx,bit,unique_bits,times,sumtime,mem_size,internal_diversity')
    print(f'{input_file[:-4]},{idx_power[iter]},{bit},{x},{finaltime - begintime},{finaltime - initial_time},{np.array(fps, dtype = np.int8 ).nbytes},{internalDiv}')
    times.append(finaltime - begintime)
    sumtimes.append(finaltime - initial_time)

data = {
    'idx' : idx_power,
    'bit' : bit_sizes,
    'unique_bits': unique,
    'times': times,
    'sumtime' : sumtimes,
     }

if custom_size is not None:
    u = 0
else:
    df_results = pd.DataFrame(data)
    df_results.to_csv('required_bits_' + input_file, index = False )
