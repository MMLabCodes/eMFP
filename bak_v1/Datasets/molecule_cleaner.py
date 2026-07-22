
"""
Clean and canonicalize SMILES from chemical structure databases and write them
to a line-delimited file.
"""

import os
import numpy as np
import pandas as pd
import sys
from itertools import chain
from rdkit import Chem
from tqdm import tqdm

# import functions
from functions import clean_mols, remove_salts_solvents, tokenizer,  filter_smiles, posible_charged
from utils_molecules import get_custom_descriptors, normalize_dataframe

# parse arguments
input_file = sys.argv[1]
smi_column = sys.argv[2]

# Dataframe with smiles 
df = pd.read_csv(input_file)  

# Remove NAN values
df = df.dropna()


# To filter in the same order
initial_cols = df.columns.tolist()

# smiles
smiles = df[smi_column].tolist()

# mol object and invalid molecules
print('')
mols, invalidity = clean_mols(smiles, stereochem=False)

df['mol'] = mols
df['invalidity'] = invalidity

# remove salt and solvents
rem_salt_solvents = [remove_salts_solvents(mol, hac=3)[1] for mol in tqdm(mols)]
df['salt_solvent_status'] = rem_salt_solvents


# Removing molecules that contains not frequent tokens (less than 0.01% )
print('Obtaining vocabulary')
vocabulary = []
for smi in tqdm(smiles):
    vocabulary += tokenizer(smi)
    vocabulary = list( set( vocabulary) )

# Obtaining status of not frequent tokens in smiles
smiles_to_delete = filter_smiles(df, vocabulary, smi_column)
df['no_common_tokens_smiles'] = smiles_to_delete

# Remove smiles that could contain charged atoms, the assumption is 
# the atom is charged if '+]' or '-]' is in the smiles string
df['posible_charged'] = df[smi_column].apply(posible_charged)


df['must_be_removed'] = df['invalidity'] + df['salt_solvent_status'] + df['no_common_tokens_smiles'] + df['posible_charged']

# keep only the smiles that are valid, not salt/solvent, with frequent tokens and not charged
df_clean = df.loc[df['must_be_removed'] == 0].reset_index(drop=True)

# Calculate descriptors and split clean dataset into two files, 
# one where the even rows are selected and the second where the 
# odd rows have been selected

print('Calculate descriptors')
descriptor_list = ['MaxEStateIndex', 'MinEStateIndex', 'MinAbsEStateIndex', 'qed', 'MolWt', 'FpDensityMorgan1', 'BalabanJ', 'HallKierAlpha', 'Ipc', 
                   'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 
                   'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 
                   'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 
                   'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'EState_VSA1', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 
                   'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 
                   'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'FractionCSP3', 
                   'NHOHCount', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 
                   'NumRotatableBonds', 'RingCount', 'MolLogP']

desc = get_custom_descriptors(df_clean['mol'].tolist(), 'mol_object', descriptor_list)
norm_desc = normalize_dataframe(desc)

desc_odd = desc[1::2]
desc_even = desc[::2]

norm_desc_odd = norm_desc[1::2]
norm_desc_even = norm_desc[::2]

# # df_odd  = df[1::2] # odd
# # df_even = df[0::2] # even

if 'pubchem' in input_file:
    # Values for sampled mask of 100k molecules (k =100k)
    
    # n: Number of rows in df
    n = df_clean.shape[0]
    
    # k: Sample size 100k
    k = 100000 
    maskPM6 =  np.zeros(n, dtype = bool)
    maskB3L =  np.zeros(n, dtype = bool)

    # seed for reproducibility
    np.random.seed(42)
    indexPM6 = np.random.choice( n, k, replace = False)
    maskPM6[indexPM6] = True

    # seed for reproducibility, differente set of molecules
    np.random.seed(21)
    indexB3L = np.random.choice( n, k, replace = False)
    maskB3L[indexB3L] = True
  
    # PM6 data
    colspm6   = ['cid', 'smiles', 'homopm6', 'lumopm6', 'gappm6']
    df_pm6 = df_clean[colspm6]
    df_finalpm6 = df_pm6.rename(columns = {'homopm6': 'homo', 'lumopm6': 'lumo', 'gappm6':'gap' })
    df_finalpm6.to_csv(  'clean_' + input_file[:-4] + 'PM6.csv'  , index = False )

    # Descriptors
    desc.to_csv(  'descriptors_clean_' + input_file[:-4] + 'PM6.csv'  , index = False )
    norm_desc.to_csv(  'descNormal_clean_' + input_file[:-4] + 'PM6.csv'  , index = False )

    # B3LYP data
    colsb3lyp = ['cid', 'smiles', 'homo', 'lumo', 'gap']
    df_finalb3lyp = df_clean[colsb3lyp]
    df_finalb3lyp.to_csv('clean_' + input_file[:-4] + 'B3LYP.csv', index = False )

    # Descriptors
    desc.to_csv(  'descriptors_clean_' + input_file[:-4] + 'B3LYP.csv'  , index = False )
    norm_desc.to_csv(  'descNormal_clean_' + input_file[:-4] + 'B3LYP.csv'  , index = False )

    ###################################
    # Saving sample of 100 k molecules
    ###################################

    # PM6 data SAMPLE
    df_finalPM6_sample =  df_finalpm6[maskPM6] 
    df_oddpm6 = df_finalPM6_sample[1::2]
    df_evenpm6 = df_finalPM6_sample[::2]
    df_oddpm6.to_csv(  'trainDb/clean_' + input_file[:-4] + 'PM6_100k_odd.csv'  , index = False )
    df_evenpm6.to_csv(  'validDb/clean_' + input_file[:-4] + 'PM6_100k_even.csv'  , index = False )

    # Descriptors odd and even PM6 SAMPLE
    desc_PM6_sample = desc[maskPM6]
    desc_odd_PM6_sampled =  desc_PM6_sample[1::2]
    desc_even_PM6_sampled =  desc_PM6_sample[::2]
    desc_odd_PM6_sampled.to_csv(  "trainDb/descriptors_clean_" + input_file[:-4] + 'PM6_100k_odd.csv' )
    desc_even_PM6_sampled.to_csv( "validDb/descriptors_clean_" + input_file[:-4] + 'PM6_100k_even.csv'  )
    
    norm_desc_PM6_sample = norm_desc[maskPM6]
    norm_desc_odd_PM6_sampled = norm_desc_PM6_sample[1::2]
    norm_desc_even_PM6_sampled = norm_desc_PM6_sample[::2]
    norm_desc_odd_PM6_sampled.to_csv(  "trainDb/descNormal_clean_" + input_file[:-4] + 'PM6_100k_odd.csv' )
    norm_desc_even_PM6_sampled.to_csv( "validDb/descNormal_clean_" + input_file[:-4] + 'PM6_100k_even.csv'  )


    # B3LYP data SAMPLE
    df_finalB3LYP_sample =  df_finalb3lyp[maskB3L] 
    df_oddb3lyp = df_finalB3LYP_sample[1::2]
    df_evenb3lyp = df_finalB3LYP_sample[::2]
    df_oddb3lyp.to_csv(  'trainDb/clean_' + input_file[:-4] + 'B3LYP_100k_odd.csv'  , index = False )
    df_evenb3lyp.to_csv(  'validDb/clean_' + input_file[:-4] + 'B3LYP_100k_even.csv'  , index = False )

    # Descriptors odd and even B3LYP SAMPLE
    desc_B3LYP_sample = desc[maskB3L]
    desc_odd_B3LYP_sampled =  desc_B3LYP_sample[1::2]
    desc_even_B3LYP_sampled =  desc_B3LYP_sample[::2]
    desc_odd_B3LYP_sampled.to_csv(  "trainDb/descriptors_clean_" + input_file[:-4] + 'B3LYP_100k_odd.csv' )
    desc_even_B3LYP_sampled.to_csv( "validDb/descriptors_clean_" + input_file[:-4] + 'B3LYP_100k_even.csv'  )
    
    norm_desc_B3LYP_sample = norm_desc[maskB3L]
    norm_desc_odd_B3LYP_sampled = norm_desc_B3LYP_sample[1::2]
    norm_desc_even_B3LYP_sampled = norm_desc_B3LYP_sample[::2]
    norm_desc_odd_B3LYP_sampled.to_csv(  "trainDb/descNormal_clean_" + input_file[:-4] + 'B3LYP_100k_odd.csv' )
    norm_desc_even_B3LYP_sampled.to_csv( "validDb/descNormal_clean_" + input_file[:-4] + 'B3LYP_100k_even.csv'  )

elif 'qm9' in input_file:
    # QM9 data
    colsqm9 = ['smiles', 'homo', 'lumo', 'gap']
    df_final = df_clean[colsqm9]
    df_final.to_csv('clean_' + input_file, index = False )
    df_odd = df_final[1::2]
    df_even = df_final[::2]
    df_odd.to_csv('trainDb/clean_' + input_file[:-4] + '_odd.csv', index = False )
    df_even.to_csv('validDb/clean_' + input_file[:-4] + '_even.csv', index = False )

    # Descriptors
    desc.to_csv(  'descriptors_clean_' + input_file, index = False )
    norm_desc.to_csv(  'descNormal_clean_' + input_file, index = False )
    # odd and even, descriptors files
    desc_odd.to_csv(  "trainDb/descriptors_clean_" + input_file[:-4] + '_odd.csv' )
    desc_even.to_csv( "validDb/descriptors_clean_" + input_file[:-4] + '_even.csv'  )
    norm_desc_odd.to_csv(  "trainDb/descNormal_clean_" + input_file[:-4] + '_odd.csv' )
    norm_desc_even.to_csv( "validDb/descNormal_clean_" + input_file[:-4] + '_even.csv'  )

elif 'reddb' in input_file:
    # redDB data
    colsreddb = ["product_smiles", "product_homo", "product_lumo", 'gap']
    df_clean['gap'] = df_clean['product_lumo'] - df_clean['product_homo']

    df_tmp = df_clean[colsreddb]
    df_final = df_tmp.rename( columns = {"product_smiles": "smiles", "product_homo": "homo", "product_lumo": "lumo"})
    df_final.to_csv('clean_' + input_file, index = False )
    df_odd = df_final[1::2]
    df_even = df_final[::2]
    df_odd.to_csv('trainDb/clean_' + input_file[:-4] + '_odd.csv', index = False )
    df_even.to_csv('validDb/clean_' + input_file[:-4] + '_even.csv', index = False )

    # Descriptors
    desc.to_csv(  'descriptors_clean_' + input_file, index = False )
    norm_desc.to_csv(  'descNormal_clean_' + input_file, index = False )
    # odd and even, descriptors files
    desc_odd.to_csv(  "trainDb/descriptors_clean_" + input_file[:-4] + '_odd.csv' )
    desc_even.to_csv( "validDb/descriptors_clean_" + input_file[:-4] + '_even.csv'  )
    norm_desc_odd.to_csv(  "trainDb/descNormal_clean_" + input_file[:-4] + '_odd.csv' )
    norm_desc_even.to_csv( "validDb/descNormal_clean_" + input_file[:-4] + '_even.csv'  )

elif 'nfa' in input_file:
    # NFA data
    colsnfa = ['smiles', 'HOMO_calc', 'LUMO_calc', 'GAP_calc']
    df_tmp = df_clean[colsnfa]
    df_final = df_tmp.rename( columns = {"GAP_calc": "gap", "HOMO_calc": "homo", "LUMO_calc": "lumo"})
    
    df_final.to_csv('clean_' + input_file, index = False )
    df_odd = df_final[1::2]
    df_even = df_final[::2]
    df_odd.to_csv('trainDb/clean_' + input_file[:-4] + '_odd.csv', index = False )
    df_even.to_csv('validDb/clean_' + input_file[:-4] + '_even.csv', index = False )

    # Descriptors
    desc.to_csv(  'descriptors_clean_' + input_file, index = False )
    norm_desc.to_csv(  'descNormal_clean_' + input_file, index = False )
    # odd and even, descriptors files
    desc_odd.to_csv(  "trainDb/descriptors_clean_" + input_file[:-4] + '_odd.csv' )
    desc_even.to_csv( "validDb/descriptors_clean_" + input_file[:-4] + '_even.csv'  )
    norm_desc_odd.to_csv(  "trainDb/descNormal_clean_" + input_file[:-4] + '_odd.csv' )
    norm_desc_even.to_csv( "validDb/descNormal_clean_" + input_file[:-4] + '_even.csv'  )

else:
    df_final = df_clean[initial_cols]
    df_final.to_csv('clean_' + input_file, index = False )
    df_odd = df_final[1::2]
    df_even = df_final[::2]
    df_odd.to_csv('trainDb/clean_' + input_file[:-4] + '_odd.csv', index = False )
    df_even.to_csv('validDb/clean_' + input_file[:-4] + '_even.csv', index = False )

    # Descriptors
    desc.to_csv(  'descriptors_clean_' + input_file, index = False )
    norm_desc.to_csv(  'descNormal_clean_' + input_file, index = False )
    # odd and even, descriptors files
    desc_odd.to_csv(  "trainDb/descriptors_clean_" + input_file[:-4] + '_odd.csv' )
    desc_even.to_csv( "validDb/descriptors_clean_" + input_file[:-4] + '_even.csv'  )
    norm_desc_odd.to_csv(  "trainDb/descNormal_clean_" + input_file[:-4] + '_odd.csv' )
    norm_desc_even.to_csv( "validDb/descNormal_clean_" + input_file[:-4] + '_even.csv'  )

print('Done!')