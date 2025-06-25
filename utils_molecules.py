
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import combinations
import psutil
import time
import pickle

from random import shuffle

import rdkit 
from rdkit import Chem 
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem import MolFromSmiles
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.*')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score


def mol_from_smiles(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None

def canonical_smiles(smiles):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:
        return None


def custom_descriptors(mol, descriptor_list, missingVal):
    descriptor_dictionary = {}
    for name, function in Descriptors._descList:
        if name in descriptor_list:
            try:
                val = function(mol)
            except:
                import traceback
                traceback.print_exc()
                val = missingVal
            descriptor_dictionary[name] = val
    return descriptor_dictionary

def get_custom_descriptors(molecular, case, descriptor_list):
    '''
    Calculate all descriptors for a list of molecules and organize the results in a DataFrame.
    https://greglandrum.github.io/rdkit-blog/posts/2022-12-23-descriptor-tutorial.html

    Args:
        molecular (list): List of molecular representations (SMILES strings or RDKit molecule objects).
        case (str): Case identifier, either 'smiles' or 'mol_object'.
        descriptor_list (list): List of descriptors to calculate

    Returns:
        pd.DataFrame: A pandas DataFrame where each row represents a molecule and each column
                      represents a molecular descriptor.
    '''
    if case == 'smiles':
        mols = [Chem.MolFromSmiles(smile) for smile in molecular]
        allDescrs = [custom_descriptors(m, descriptor_list, 0) for m in tqdm(mols, desc = 'Mols')]
    elif case == 'mol_object':
        allDescrs = [custom_descriptors(m, descriptor_list, 0) for m in tqdm(molecular, desc = 'Mols')]
    df = pd.DataFrame(allDescrs)
    return df

def get_all_descriptors(molecular, case):
    '''
    Calculate all descriptors for a list of molecules and organize the results in a DataFrame.
    https://greglandrum.github.io/rdkit-blog/posts/2022-12-23-descriptor-tutorial.html

    Args:
        molecular (list): List of molecular representations (SMILES strings or RDKit molecule objects).
        case (str): Case identifier, either 'smiles' or 'mol_object'.

    Returns:
        pd.DataFrame: A pandas DataFrame where each row represents a molecule and each column
                      represents a molecular descriptor.
    '''
    if case == 'smiles':
        mols = [Chem.MolFromSmiles(smile) for smile in molecular]
        allDescrs = [getMolDescriptors(m, 0) for m in tqdm(mols)]
    elif case == 'mol_object':
        allDescrs = [getMolDescriptors(m, 0) for m in tqdm(molecular)]
    df = pd.DataFrame(allDescrs)
    return df

def getMolDescriptors(mol, missingVal=0):
    '''
    Calculate the full list of descriptors for a molecule.
    https://greglandrum.github.io/rdkit-blog/posts/2022-12-23-descriptor-tutorial.html

    Args:
        mol (Chem.Mol): RDKit molecule object for which descriptors are to be calculated.
        missingVal (Optional): Value to be assigned if a descriptor cannot be calculated.

    Returns:
        dict: A dictionary containing molecular descriptors. The keys are descriptor names,
              and the values are the corresponding calculated descriptor values.
    '''
    res = {}
    for nm, fn in Descriptors._descList:
        # Some descriptor functions may raise errors; catch and handle those here:
        try:
            val = fn(mol)
        except Exception:
            # Print the error message:
            import traceback
            traceback.print_exc()
            # Set the descriptor value to whatever missingVal is:
            val = missingVal
        res[nm] = val
    return res

def calculate_morgan_fingerprints(molecules, radius=2, nbits=2048):
    """
    Calculate Morgan fingerprints for a list of molecules in RDKit object format.

    Args:
        molecules (list): List of RDKit Mol objects.
        radius (int): Radius for Morgan fingerprint calculation.
        nbits (int): Length of the Morgan fingerprints.

    Returns:
        list: List of numpy arrays representing the Morgan fingerprints for each molecule.
    """
    fingerprints_list = []

    for mol in tqdm(molecules, desc = 'MFP'):
        # Calculate Morgan fingerprints for each molecule
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, useChirality = True, useFeatures = True)
        fingerprint_array =  np.array(fingerprint)
        
        # # Convert the fingerprint to a numpy array
        # fingerprint_array = np.zeros((1,), dtype=np.int8)
        # ConvertToNumpyArray(fingerprint, fingerprint_array)

        fingerprints_list.append(fingerprint_array)

    return np.array(fingerprints_list)

def single_mfp(mol, radius, nbits):
    info = {}

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, useChirality = True, useFeatures = True)
    x = ''
    for item in fingerprint:
        x += str(item) + '&'
    return x



def convert_fp_to_embV2(vector, size):
    """
    Converts a fingerprint vector into a embedded vector using a specified size.
    Fingerprint shape must be N, M*size, Where N is the number of Fingerprints of
    N molecules and M*size means that the length of the fingerprint must be an 
    integer multiple of size

    Args:
        vector (numpy.ndarray): Input fingerprint vector.
        size (int): Size used for reshaping the fingerprint vector.

    Returns:
        numpy.ndarray: Embedded vector obtained by reshaping and processing the input fingerprint vector.

    Raises:
        ValueError: If the size is not a positive integer.
    """
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Size must be a positive integer. Try size = 16, or size = 32")
    
    if len(vector.shape) == 1:
        rows = 1 
        cols = vector.shape[0]
    else:
        rows, cols = vector.shape

    narrays = cols // size

    mask = 2**np.arange(size, dtype = np.float64 )
    mask = mask/np.sum(mask)
    bigMask = np.tile(mask, (narrays, 1))

    tensorMask = np.tile(bigMask, (rows, 1))
    tensorMask = tensorMask.reshape((rows* narrays, size))
    vector_reshape = vector.reshape((rows* narrays, size))
    
    mfp_masked = tensorMask * vector_reshape
    mfp_maskedDotted = np.sum(mfp_masked, axis=1)
    
    if len(vector.shape) == 1:
        return mfp_maskedDotted.reshape((rows,narrays))
    else:
        return mfp_maskedDotted.reshape((rows,narrays)).squeeze()


def normalize_dataframe(df):
    """
    Normalize the values in each column of a Pandas DataFrame to the range [0, 1].

    Args:
        df (pd.DataFrame): Input DataFrame with numerical values.

    Returns:
        pd.DataFrame: DataFrame with values normalized between 0 and 1.

    Example:
        >>> import pandas as pd
        >>> from sklearn.preprocessing import MinMaxScaler
        >>> df = pd.DataFrame({'column1': [1, 2, 3, 4, 5],
        ...                    'column2': [10, 20, 30, 40, 50],
        ...                    'column3': [100, 200, 300, 400, 500]})
        >>> normalized_df = normalize_dataframe(df)
    """
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Normalize all columns of the DataFrame
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df_normalized


def columns_without_nan_inf(dataframe):
    """
    Returns a list of columns that do not contain NaN or infinity values.

    Parameters:
        dataframe (DataFrame): Input DataFrame.

    Returns:
        list: List of column names without NaN or infinity values.
    """
    nan_columns = dataframe.columns[dataframe.isna().any()].tolist()
    inf_columns = dataframe.columns[dataframe.isin([float('inf'), -float('inf')]).any()].tolist()
    nan_or_inf_columns = list(set(nan_columns + inf_columns))
    columns_without_nan_inf = [col for col in dataframe.columns if col not in nan_or_inf_columns]

    return columns_without_nan_inf

def common_elements(list1, list2):
    """
    Returns a list containing only the elements common to both input lists.

    Parameters:
        list1 (list): First input list.
        list2 (list): Second input list.

    Returns:
        list: List containing elements common to both input lists.
    """
    return list(set(list1) & set(list2))

def remove_nan_inf(df):
    """
    Remove columns containing NaN or Inf values from the dataframe.

    Args:
    df (pd.DataFrame): Input pandas dataframe.

    Returns:
    pd.DataFrame: Dataframe with NaN and Inf values removed.
    """
    # Remove columns with NaN or Inf values
    df_cleaned = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    return df_cleaned


def remove_non_numeric(df):
    """
    Remove columns containing non-numeric values from the dataframe.

    Args:
    df (pd.DataFrame): Input pandas dataframe.

    Returns:
    pd.DataFrame: Dataframe with non-numeric columns removed.
    """
    # Remove columns with non-numeric values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_columns]
    return df_numeric


def remove_correlated_features(df, threshold=0.9):
    """
    Remove correlated features from the dataframe.

    Args:
    df (pd.DataFrame): Input pandas dataframe.
    threshold (float, optional): Threshold for correlation coefficient. Features
        with correlation coefficient higher than this value will be considered
        correlated. Defaults to 0.9.

    Returns:
    pd.DataFrame: Dataframe with correlated features removed.
    """
    # Calculate correlation matrix
    corr_matrix = df.corr().abs()

    # Create a mask to remove the upper triangle of the correlation matrix
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Find index of feature columns with correlation greater than threshold
    features_to_drop = [column for column in corr_matrix.columns if any(corr_matrix[column] > threshold)]

    # Drop highly correlated features
    df_filtered = df.drop(columns=features_to_drop)
    return df_filtered


def get_minimal_columns(dfFeatures, threshold=0.95):
    """
    Returns the minimal list of columns that describe all features by removing highly correlated columns.
    
    Args:
        dfFeatures (pd.DataFrame): DataFrame containing the features.
        threshold (float): Correlation threshold to consider two columns as highly correlated.
        
    Returns:
        list: List of column names that minimally describe all features.
    """
    # Calculate the correlation matrix
    corr_matrix = dfFeatures.corr().abs()
    
    # Create a boolean matrix where True indicates high correlation
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find columns with correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Create a list of remaining columns, removing the highly correlated ones
    remaining_columns = [column for column in dfFeatures.columns if column not in to_drop]
    
    return remaining_columns


def identify_bit_collisions(mol, radius, nbits):
    # Generate the Morgan fingerprint with counts
    info = {}
    fingerprint = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info, useChirality=True, useFeatures=True)
    
    # Initialize a bit count array
    bit_counts = np.zeros(nbits, dtype=int)
    
    for bit, paths in info.items():
        bit_counts[bit] = len(paths)
    
    # Identify bit collisions
    collisions = np.where(bit_counts > 1)[0]
    
    return collisions, bit_counts

