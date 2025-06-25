
import numpy as np

import pandas as pd

import re
import warnings

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem 
from rdkit.Chem import rdmolops
from rdkit.Chem import DataStructs

import itertools


def tokenizer(smile):
    """
    Tokenizes a SMILES string.

    Parameters:
        smile (str): The SMILES string to tokenize.

    Returns:
        list: A list of tokens extracted from the SMILES string.

    Raises:
        AssertionError: If the original SMILES string cannot be reconstructed from the tokens.

    Examples:
        >>> tokenizer('CCO')
        ['C', 'C', 'O']
        >>> tokenizer('[NH4+]')
        ['[NH4+]']
    """
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(smile)]
    assert smile == ''.join(tokens), f"{smile} could not be joined"
    return tokens

def filter_smiles(df, vocabulary, smiles_column):
    """
    Filters a DataFrame of SMILES strings based on token frequency.

    Args:
    - df (pd.DataFrame): DataFrame containing SMILES strings in a column named 'smiles'.
    - vocabulary (list): List of tokens to filter SMILES strings.

    Returns:
    - list: List of integers where each element is 0 if the corresponding SMILES should be kept,
            and 1 if it should be removed.
    """
    n_smiles = len(df)
    delete_indicator = [0] * n_smiles  # Initially, all SMILES are considered to be kept
    
    for token in tqdm(vocabulary):
        token_smiles_indices = []
        for idx, sm in enumerate(df[smiles_column]):
            if token in tokenizer(sm):
                token_smiles_indices.append(idx)
        
        pct_smiles = len(token_smiles_indices) / n_smiles
        # remove any molecules containing tokens found in <0.01% of molecules,
        if pct_smiles <  0.01 / 100 or len(token_smiles_indices) <= 10 :
            for idx in token_smiles_indices:
                delete_indicator[idx] = 1
   
    return delete_indicator

def posible_charged(smiles):
    if '+]' in smiles or '-]' in smiles:
        return 1
    else:
        return 0

def clean_mol(smiles, stereochem=False):
    """
    Construct a molecule from a SMILES string, removing stereochemistry and
    explicit hydrogens, and setting aromaticity.
    """

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("invalid SMILES: " + str(smiles))
    if not stereochem:
        Chem.RemoveStereochemistry(mol)
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    return mol

def clean_mols(all_smiles, stereochem=False):
    """
    Construct a list of molecules from a list of SMILES strings, replacing
    invalid molecules with None in the list.
    """
    mols, invalidity = [], []
    for smiles in tqdm(all_smiles, desc = 'RDKit Mol Object: '):
        try:
            mol = clean_mol(smiles, stereochem)
            mols.append(mol)
            invalidity.append(0)
        except ValueError:
            mols.append(None)
            invalidity(1)
    return mols, invalidity

def remove_salts_solvents(mol, hac=3):
    """
    Remove solvents and ions have max 'hac' heavy atoms.
    This function was obtained from the mol2vec package,
    available at:
        https://github.com/samoturk/mol2vec/blob/master/mol2vec/features.py
    """
    contain_salt_solvents = []
    # split molecule into fragments
    fragments = list(rdmolops.GetMolFrags(mol, asMols = True))
    ## keep heaviest only
    ## fragments.sort(reverse=True, key=lambda m: m.GetNumAtoms())
    # remove fragments with < 'hac' heavy atoms
    fragments = [fragment for fragment in fragments if \
                 fragment.GetNumAtoms() > hac]
    #
    if len(fragments) > 1:
        warnings.warn("molecule contains >1 fragment with >" + str(hac) + \
                      " heavy atoms")
        return None, 1
    elif len(fragments) == 0:
        warnings.warn("molecule contains no fragments with >" + str(hac) + \
                      " heavy atoms")
        return None, 1
    else:
        return fragments[0], 0

def mfp(mol, nbits, rad_size):
    info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=rad_size, nBits = nbits, bitInfo = info, useFeatures = True)
    fp_np = np.zeros((0, ), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, fp_np)
    return fp_np, info, fp


# def internal_diversity(mfps):
#     """
#     Calculate the internal diversity, defined as the mean intra-set Tanimoto
#     coefficient, between a set of Molecular Fingerprints (MFPs) by analyzing all pairs.

#     Parameters:
#     - mfps: List of RDKit MFP objects.

#     Returns:
#     - Mean Tanimoto coefficient between all pairs of MFPs.
#     """
#     tcs = []
    
#     # Obtener todos los pares únicos de huellas moleculares
#     for fp1, fp2 in itertools.combinations(mfps, 2):
#         # Calcula el coeficiente de Tanimoto entre los pares de huellas moleculares
#         tc = DataStructs.FingerprintSimilarity(fp1, fp2)
#         tcs.append(tc)
    
#     # Calcular y devolver el promedio de todos los coeficientes de Tanimoto
#     return np.mean(tcs)


# def internal_diversity(mfps):
#     """
#     Calculate the internal diversity, defined as the mean intra-set Tanimoto
#     coefficient, between a set of Molecular Fingerprints (MFPs) by analyzing all pairs.

#     Parameters:
#     - mfps: List of RDKit MFP objects.

#     Returns:
#     - Mean Tanimoto coefficient between all pairs of MFPs.
#     """
#     tcs = []
    
#     # Obtener todos los pares únicos de huellas moleculares
#     for fp1, fp2 in itertools.combinations(mfps, 2):
#         # Calcula el coeficiente de Tanimoto entre los pares de huellas moleculares
#         tc = DataStructs.FingerprintSimilarity(fp1, fp2)
#         tcs.append(tc)
    
#     # Calcular y devolver el promedio de todos los coeficientes de Tanimoto
#     return np.mean(tcs)

import numpy as np
from rdkit import DataStructs

def internal_diversity(fingerprints):
    """
    Calculate the internal diversity of a list of molecular fingerprints (MFPs).
    
    Parameters:
    fingerprints (list): List of molecular fingerprints (MFPs) as binary vectors.
    
    Returns:
    float: Internal diversity of the set of molecules.
    """
    num_fingerprints = len(fingerprints)
    if num_fingerprints < 2:
        raise ValueError("At least two fingerprints are required to calculate internal diversity.")
    
    dist_matrix = np.zeros((num_fingerprints, num_fingerprints))
    
    # Calculate the Tanimoto distance matrix
    for i in tqdm(range(num_fingerprints)):
        for j in range(i + 1, num_fingerprints):
            similarity = DataStructs.FingerprintSimilarity(fingerprints[i], fingerprints[j])
            distance = 1 - similarity
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance
    
    # Calculate internal diversity as the mean of the upper triangle of the distance matrix
    upper_triangle_indices = np.triu_indices(num_fingerprints, 1)
    diversity = np.mean(dist_matrix[upper_triangle_indices])
    
    return diversity
