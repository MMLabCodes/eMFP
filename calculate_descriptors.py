
import os

import pandas as pd 
from tqdm import tqdm
tqdm.pandas()


from rdkit import Chem 
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.*')

from utils_molecules import mol_from_smiles, get_custom_descriptors

import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-file', nargs='?', const='smiles_dataset', help='smiles dataset', default=None)
parser.add_argument('-smicol', nargs='?', const='smiles_column', help='smiles column', default='smiles')


args = parser.parse_args()

directory, fileName = os.path.split(args.file)
name, extension = os.path.splitext(fileName)

df = pd.read_csv(args.file)
try:
    smiles = df[args.smicol].tolist()
except:
    smiles = df['smiles'].tolist()

print('Obtaining Mols')
mols = [mol_from_smiles(smile) for smile in tqdm(smiles, desc = 'SMILES')]

descriptor_list = ['MaxEStateIndex', 'MinEStateIndex', 'MinAbsEStateIndex', 'qed', 'MolWt', 'FpDensityMorgan1', 'BalabanJ', 'HallKierAlpha', 'Ipc', 
                   'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 
                   'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 
                   'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 
                   'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'EState_VSA1', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 
                   'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 
                   'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'FractionCSP3', 
                   'NHOHCount', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 
                   'NumRotatableBonds', 'RingCount', 'MolLogP']


descriptors_path = directory + '/desc_' + fileName 

if os.path.exists(descriptors_path):
    print('File already exists')
else:
    descriptors = get_custom_descriptors(mols, 'mol_object', descriptor_list)
    descriptors.to_csv(descriptors_path, index = False)

