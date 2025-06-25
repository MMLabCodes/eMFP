
import os
import sys

import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

from tqdm import tqdm
import psutil
import gc

import rdkit 
from rdkit import Chem 
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.*')

from utils_molecules import mol_from_smiles, get_custom_descriptors, calculate_morgan_fingerprints, convert_fp_to_embV2, normalize_dataframe

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F

from scipy.stats import ks_2samp
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

import optuna
from optuna.samplers import RandomSampler, TPESampler, CmaEsSampler
from optuna.visualization import plot_contour, plot_param_importances, plot_slice

import random

import matplotlib.pyplot as plt

from models import *

import argparse
parser = argparse.ArgumentParser()

import pickle

from memory_profiler import profile, memory_usage

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = torch.log(torch.cosh(y_pred - y_true))
        return torch.mean(loss)

# File with SMILES
parser.add_argument('-file', required=True, help='Name of the input file containing SMILES')

# Chose between MFP (-mfp) or eMFP (-emfp)
# if -mfp: size is not required
# if -emfp: size is mandatory,  power of 2 greater than 2^2 and smaller than 2^6
#    -size 4, or 
#    -size 8, -size 16, -size 32, -size 64
parser.add_argument('-mfp', dest='mfp',help='MFP', action='store_true',default = None )
parser.add_argument('-emfp', dest='emfp',help='eMFP', action='store_true',default = None )
parser.add_argument('-size', nargs='?', const='embedded compression', help='Compression factor (int 4, 8, 16, 32, 64)', default = 1 )

# Only chose one: -none, or -linear or -gauss
# If -none, input data is withouth FFNN
# If -linear input data is with FFNN
# ig -gauss input data is with FFNN
parser.add_argument('-none', dest='none',help='No FFNN', action='store_true',default = None )
parser.add_argument('-linear', dest='linear',help='Linear FFNN', action='store_true',default = None )
parser.add_argument('-gauss', dest='gauss',help='gauss FFNN', action='store_true',default = None )
parser.add_argument('-order', nargs='?', const='order for FFNN', help='Choose int (1, 2, 3, ...,)', default=1)

# Required training parameters and MFP parameters
parser.add_argument('-nB', type=int, required=True, help='Number of bits for MFP (int 1024, 2048, 4096, etc.)')
parser.add_argument('-rd', type=int, required=True, help='Radius for MFP (int 2, 3, 4, 5)')

parser.add_argument('-model', required=True, help='Name of ML model to train: RF, GBR, KNR, MLP')

args = parser.parse_args()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()
print('Running ffnn_gap.py with parameters: ')
print(args)


if not args.file:
    raise Exception("Dataset with SMILES required, -file /path/to/dataset.csv ")

if not args.mfp and not args.emfp:
    raise Exception("\n\tSelect a case -mfp or -emfp ")
elif args.mfp and args.emfp:
    raise Exception("\n\tSelect only one case -mfp or -emfp ")
elif args.emfp and not args.size:
    raise Exception("\n\tPlease provide size for -emfp ")
elif args.emfp and args.size:
    try:
        assert int(args.size) in [2**i for i in range( 2, 14 )]
    except:
        raise Exception("\n\tTry any of the next values for size: -size 4, -size 8, -size 16, -size 32, -size 64")

if args.size:
    size = int( args.size )
else:
    size = 1

if args.nB:
    try:
        assert int(args.nB) in [2**i for i in range( 9, 21 )]
    except:
        raise Exception("\n\tTry any of the next values for nB (nBits of MFP): -nB 1024, -nB 2048, -nB 4096, ... -nB 2^N, N=[9,21]")

if args.rd:
    try:
        assert int(args.rd) in [i for i in range( 2, 6 )]
    except:
        raise Exception("\n\tWrong value for -rd\n\tTry any of the next values for rd (radius of MFP): -rd 2, -rd 3, -rd 4, -rd 5")

if args.none:
    incNone = 'none'
elif args.linear:
    incNone = 'linear'
elif args.gauss:
    incNone = 'gauss'


if args.nB:
    try:
        assert int(args.nB) in [2**i for i in range( 9, 21 )]
    except:
        raise Exception("\n\tTry any of the next values for nB (nBits of MFP): -nB 1024, -nB 2048, -nB 4096, ... -nB 2^N, N=[9,21]")

if args.rd:
    try:
        assert int(args.rd) in [i for i in range( 2, 6 )]
    except:
        raise Exception("\n\tWrong value for -rd\n\tTry any of the next values for rd (radius of MFP): -rd 2, -rd 3, -rd 4, -rd 5")



head_tail = os.path.split(args.file)

if 'nfa' in head_tail[1]:
    iddb = 'nfa'
    dir_name = f'Models/modelsNFA_{args.model}_{incNone}_{size}'
elif 'qm9' in head_tail[1]:
    iddb = 'qm9'
    dir_name = f'Models/modelsQM9_{args.model}_{incNone}_{size}'
elif 'PM6' in head_tail[1]:
    iddb = 'pm6'
    dir_name = f'Models/modelsPUBPM6_{args.model}_{incNone}_{size}'
elif 'B3LYP' in head_tail[1]:
    iddb = 'lyp'
    dir_name = f'Models/modelsPUBB3LYP_{args.model}_{incNone}_{size}'
elif 'reddb' in head_tail[1]:
    iddb = 'rdb'
    dir_name = f'Models/modelsRedDB_{args.model}_{incNone}_{size}'

if not os.path.exists(dir_name):
    os.makedirs(dir_name) 


def order_of_magnitude(number):
    if number == 0:
        return 0
    else:
        return np.floor(np.log10(np.abs(number)))

# Fix the seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Seed fix to 42, to keep reproducibility
set_seed(42)





# Read file

df = pd.read_csv(args.file)


idxs = np.where(df['lumo'] > df['homo'])[0]
auxZ = np.zeros(len(df), dtype = bool)
for i in idxs:
    auxZ[i] = True
df = df[auxZ]


smiles = df['smiles'].tolist()

# Obtaining Mols
print('Obtaining Mols')
mols = [ mol_from_smiles( smi ) for smi in smiles ]

print('Checking descriptors File')

descNormal_file = head_tail[0] + '/descNormal_' + head_tail[1]
print('Descriptors file', descNormal_file)

output_name = str(head_tail[1])[9:-4]
output_name += '_nbits_' + str( args.nB ) + '_radius_' + str(args.rd)

if os.path.exists(descNormal_file):
    # Check if Descriptor File is already calculated
    print(f'Descriptors file for { args.file } already exists') 
    n_desc = pd.read_csv( descNormal_file )
    n_desc = n_desc[ auxZ ]
    norm_desc = torch.tensor( n_desc.to_numpy(), dtype = torch.float32 )
else:
    raise Exception("\n\t Clean input file to obtain proper descriptor file")


# Calculate morgan fingerprint and reduced morgan fingerprint
print('Calculate Morgan fingerprint')
time_loading_mfp = time.time()

# mem, out = memory_usage((function1, (10,50, )), retval=True)
memory_xmfp, xmfp = memory_usage( (calculate_morgan_fingerprints, (mols, int(args.rd), int(args.nB))), retval = True )
print('MFPMemory', max(memory_xmfp) - min(memory_xmfp))
# print('MFPMemory', memory_xmfp)
# xmfp  = calculate_morgan_fingerprints( mols, radius = int(args.rd), nbits = int(args.nB) ) # Time Consuming
print('Time obtaining MFP:', time.time() - time_loading_mfp)
print(' MFP SizeArray:', xmfp.nbytes / (1024**2) )



if args.emfp:
    print('Obtaining eMFP')
    # rmfp = torch.tensor( convert_fp_to_embV2( xmfp, int(size) ), dtype = torch.float32 )
    memory_emfp, emfp = memory_usage( (convert_fp_to_embV2, (xmfp, int( args.size ) )), retval = True )
    print('eMFP SizeArray:', emfp.nbytes / (1024**2) )
    rmfp = torch.tensor( emfp, dtype = torch.float32 )
    print('eMFPMemory', max(memory_emfp) - min(memory_emfp))
    print('\n')
    print(f'MFP_eMFP_memory {xmfp.nbytes + emfp.nbytes} bytes, {(xmfp.nbytes + emfp.nbytes) / (1024 ** 2)} MB')
    # print('eMFPMemory',  memory_emfp  )


process = psutil.Process(os.getpid())
start_memory = process.memory_info().rss

print("Concatenating MFP and descriptors")
if args.mfp:
    xmfp = torch.from_numpy( xmfp ).float()
    x = torch.hstack(( xmfp, norm_desc ))
    print(' mfp shape', x.shape )
    case = 'xmfp'
else:
    x = torch.hstack(( rmfp, norm_desc ))
    print('emfp shape', x.shape )
    case = 'emfp'

# Concatenating MFP and descriptors
print( 'Obtaining Target: GAP' )
# # Converting to positive to use ReLU
try :
    y =  torch.tensor( df[['gap']].to_numpy(), dtype = torch.float32 )
except:
    y = torch.tensor( df['lumo'].to_numpy() - df['homo'].to_numpy(), dtype=torch.float32 )
    y = y.view(-1, 1)  # Reshaping#

# 80 % Train, 20 % Test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 42)









def metricKS(x_real, x_predicted):
    '''
    Performs Kolmogorov-Smirnov test between xreal and xpredicted
    from pytorch tensors
    '''
    y_pred = []
    y_real = []
    # If values in GPU -> get_device returns 0
    if x_predicted.is_cuda:
        y_pred.extend(x_predicted.cpu().detach().numpy().flatten())
        y_real.extend(x_real.cpu().detach().numpy().flatten())
    else:
        y_pred.extend(x_predicted.detach().numpy().flatten())
        y_real.extend(x_real.detach().numpy().flatten())
    y_pred = np.array(y_pred)
    y_real = np.array(y_real)
    
    # Perform KS test
    ks_statistic, p_value = ks_2samp(y_real, y_pred)
    return ks_statistic, p_value

def should_stop_training(r2_scores, patience=5):
    """
    Determines whether training should be stopped based on R^2 values.

    This function checks if the R^2 values for the last `patience` epochs
    have continuously decreased compared to the R^2 value from `patience` epochs ago.

    Args:
        r2_scores (list of float): A list of R^2 scores recorded for each epoch.
        patience (int): The number of consecutive epochs with decreasing R^2 to trigger early stopping. Default is 5.

    Returns:
        bool: True if the R^2 has continuously decreased for the last `patience` epochs, False otherwise.
    """
    # Check if there are enough values to evaluate (at least `patience` values)
    if len(r2_scores) < patience:
        return False  # Not enough values yet to decide

    # Compare current R^2 values with the value from `patience` epochs ago
    initial_r2 = r2_scores[-patience]
    for i in range(1, patience):
        if r2_scores[-(i + 1)] >= initial_r2:
            return False  # If any value is not decreasing, do not stop

    # If all values have decreased compared to the initial R^2 value, stop training
    return True


def input_mapping(x, B, device, order):
    if B is None:
        return x.to(device)
    else:
        sin_list, cos_list = [], []
        x_proj = torch.matmul(2. * torch.pi * x, B.T).to(device)
        for ord in range( order ):
            sin_list.append( torch.sin( ( ord +1 ) * x_proj ) )
            cos_list.append( torch.cos( ( ord +1 ) * x_proj ) )
        final_list = sin_list + cos_list
        return torch.cat( final_list , dim = -1 ).to(device)


def plotPred(targets, predictions, epoch):
    """
    Plots the model predictions against the target values.

    Args:
        targets (torch.Tensor): Tensor of target values (ground truth).
        predictions (torch.Tensor): Tensor of model predictions.
    """
    # Convert tensors to numpy arrays
    targets_np = targets.detach().cpu().numpy()
    predictions_np = predictions.detach().cpu().numpy()

    # Create scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(targets_np, predictions_np, color='blue', alpha=0.5)
    plt.plot([targets_np.min(), targets_np.max()], [targets_np.min(), targets_np.max()], 'r--', lw=2)  # Ideal line

    plt.xlabel('Target Values (y_true)')
    plt.ylabel('Predictions (y_pred)')
    plt.title('Predictions vs. Target Values')
    plt.grid(True)
    plt.savefig(f'{epoch}_pred.png', dpi = 600)
    # plt.show()


def train_model_rf(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,
                   bootstrap, x_train, x_test, y_train, y_test, device, incNone, order):
    """Train the Random Forest Regressor model.

    Args:
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of the trees.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
        max_features (float): The number of features to consider when looking for the best split.
        bootstrap (bool): Whether bootstrap samples are used when building trees.
        x_train (np.ndarray): Training features.
        x_test (np.ndarray): Testing features.
        y_train (np.ndarray): Training target values.
        y_test (np.ndarray): Testing target values.
        device (str): Device to perform the calculations (e.g., 'cpu', 'cuda').
        incNone (str): Inclusion method for the input mapping.
        order (int): Order of input mapping.

    Returns:
        tuple: Contains the trained regressor and various metrics.
    """  
    criterion = LogCoshLoss()
    criterion_mae = nn.L1Loss()

    B_dict = {}
    if incNone == 'none':
        B_dict['none'] = None
    elif incNone == 'linear':
        B_dict['linear'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)
    elif incNone == 'gauss':
        B_dict['gauss'] = torch.normal(0, 1, size=(x_train.shape[1], x_train.shape[1])).to(device)
    else:
        B_dict['linear'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)



    regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                       max_features=max_features, bootstrap=bootstrap)



    x_train_mapped = input_mapping(x_train, B_dict[incNone], device, order)
    x_test_mapped = input_mapping(x_test, B_dict[incNone], device, order)

    regressor.fit(x_train_mapped, y_train.flatten())

    y_pred_train = torch.tensor(regressor.predict(x_train_mapped)).view(-1, 1)
    y_pred_test = torch.tensor(regressor.predict(x_test_mapped)).view(-1, 1)

    loss_train = criterion(y_pred_train, y_train)
    loss_test = criterion(y_pred_test, y_test)

    loss_train_mae = criterion_mae(y_pred_train, y_train)
    loss_test_mae = criterion_mae(y_pred_test, y_test)

    r2_train = r2_score(y_pred_train.cpu().detach().numpy(), y_train.cpu().detach().numpy())
    r2_test = r2_score(y_pred_test.cpu().detach().numpy(), y_test.cpu().detach().numpy())

    print(f'LossTrain: {loss_train}, lossTest: {loss_test}, R2 Train: {r2_train}, R2 Test: {r2_test}')

    return regressor, loss_train, loss_test, loss_train_mae, loss_test_mae, r2_train, r2_test


def train_model_gbr(n_estimators, max_depth, min_samples_split, min_samples_leaf, subsample,
                    max_features, learning_rate, x_train, x_test, y_train, y_test, device, incNone, order):
    """Train the Gradient Boosting Regressor model.

    Args:
        n_estimators (int): Number of boosting stages to be run.
        max_depth (int): Maximum depth of the individual estimators.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
        subsample (float): Fraction of samples to be used for fitting the individual base learners.
        max_features (float): The number of features to consider when looking for the best split.
        learning_rate (float): Learning rate shrinks the contribution of each tree.
        x_train (np.ndarray): Training features.
        x_test (np.ndarray): Testing features.
        y_train (np.ndarray): Training target values.
        y_test (np.ndarray): Testing target values.
        device (str): Device to perform the calculations (e.g., 'cpu', 'cuda').
        incNone (str): Inclusion method for the input mapping.
        order (int): Order of input mapping.

    Returns:
        tuple: Contains the trained regressor and various metrics.
    """
    criterion = LogCoshLoss()
    criterion_mae = nn.L1Loss()

    B_dict = {}
    if incNone == 'none':
        B_dict['none'] = None
    elif incNone == 'linear':
        B_dict['linear'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)
    elif incNone == 'gauss':
        B_dict['gauss'] = torch.normal(0, 1, size=(x_train.shape[1], x_train.shape[1])).to(device)
    else:
        B_dict['linear'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)

    regressor = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators,
                                           max_depth=max_depth, min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf, subsample=subsample,
                                           max_features=max_features, random_state=42)

    x_train_mapped = input_mapping(x_train, B_dict[incNone], device, order)
    x_test_mapped = input_mapping(x_test, B_dict[incNone], device, order)
    
    regressor.fit(x_train_mapped, y_train.flatten())

    y_pred_train = torch.tensor(regressor.predict(x_train_mapped)).view(-1, 1)
    y_pred_test = torch.tensor(regressor.predict(x_test_mapped)).view(-1, 1)

    loss_train = criterion(y_pred_train, y_train)
    loss_test = criterion(y_pred_test, y_test)

    loss_train_mae = criterion_mae(y_pred_train, y_train)
    loss_test_mae = criterion_mae(y_pred_test, y_test)

    r2_train = r2_score(y_pred_train.cpu().detach().numpy(), y_train.cpu().detach().numpy())
    r2_test = r2_score(y_pred_test.cpu().detach().numpy(), y_test.cpu().detach().numpy())

    print(f'LossTrain: {loss_train}, lossTest: {loss_test}, R2 Train: {r2_train}, R2 Test: {r2_test}')

    return regressor, loss_train, loss_test, loss_train_mae, loss_test_mae, r2_train, r2_test


def train_model_knr(n_neighbors, weights, algorithm, leaf_size, p, metric, 
                    x_train, x_test, y_train, y_test, device, incNone, order):
    """Train the KNeighborsRegressor model.

    Args:
        n_neighbors (int): Number of neighbors to use.
        weights (str): Weight function used in prediction ('uniform' or 'distance').
        algorithm (str): Algorithm used to compute the nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute').
        leaf_size (int): Leaf size for BallTree or KDTree.
        p (int): Power parameter for the Minkowski metric.
        metric (str): The distance metric to use ('minkowski', 'euclidean', 'manhattan').
        x_train (np.ndarray): Training features.
        x_test (np.ndarray): Testing features.
        y_train (np.ndarray): Training target values.
        y_test (np.ndarray): Testing target values.
        device (str): Device to perform the calculations (e.g., 'cpu', 'cuda').
        incNone (str): Inclusion method for the input mapping.
        order (int): Order of input mapping.

    Returns:
        tuple: Contains the trained regressor and various metrics.
    """
    criterion = LogCoshLoss()
    criterion_mae = nn.L1Loss()

    # Prepare input mapping based on incNone argument
    B_dict = {}
    if incNone == 'none':
        B_dict['none'] = None
    elif incNone == 'linear':
        B_dict['linear'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)
    elif incNone == 'gauss':
        B_dict['gauss'] = torch.normal(0, 1, size=(x_train.shape[1], x_train.shape[1])).to(device)
    else:
        B_dict['linear'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)

    # Initialize the KNeighborsRegressor with the suggested hyperparameters
    regressor = KNeighborsRegressor(n_neighbors=n_neighbors, 
                                    weights=weights, 
                                    algorithm=algorithm, 
                                    leaf_size=leaf_size, 
                                    p=p, 
                                    metric=metric)

    # Apply the input mapping to the training and testing data
    x_train_mapped = input_mapping(x_train, B_dict[incNone], device, order)
    x_test_mapped = input_mapping(x_test, B_dict[incNone], device, order)

    # Train the model
    regressor.fit(x_train_mapped, y_train.flatten())

    # Predict the values for both training and testing sets
    y_pred_train = torch.tensor(regressor.predict(x_train_mapped)).view(-1, 1)
    y_pred_test = torch.tensor(regressor.predict(x_test_mapped)).view(-1, 1)

    # Calculate the losses
    loss_train = criterion(y_pred_train, y_train)
    loss_test = criterion(y_pred_test, y_test)

    # Calculate Mean Absolute Error (MAE)
    loss_train_mae = criterion_mae(y_pred_train, y_train)
    loss_test_mae = criterion_mae(y_pred_test, y_test)

    # Calculate R² scores
    r2_train = r2_score(y_pred_train.cpu().detach().numpy(), y_train.cpu().detach().numpy())
    r2_test = r2_score(y_pred_test.cpu().detach().numpy(), y_test.cpu().detach().numpy())

    print(f'LossTrain: {loss_train}, lossTest: {loss_test}, R2 Train: {r2_train}, R2 Test: {r2_test}')

    return regressor, loss_train, loss_test, loss_train_mae, loss_test_mae, r2_train, r2_test


def train_model_mlp(hidden_layer_sizes, activation, solver, alpha, learning_rate, 
                    learning_rate_init, max_iter, momentum, nesterovs_momentum, 
                    x_train, x_test, y_train, y_test, device, incNone, order):
    """Train the MLPRegressor model."""

    criterion = LogCoshLoss()
    criterion_mae = nn.L1Loss()

    B_dict = {}
    if incNone == 'none':
        B_dict['none'] = None
    elif incNone == 'linear':
        B_dict['linear'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)
    elif incNone == 'gauss':
        B_dict['gauss'] = torch.normal(0, 1, size=(x_train.shape[1], x_train.shape[1])).to(device)
    else:
        B_dict['linear'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)

    regressor = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, 
                             alpha=alpha, learning_rate=learning_rate, learning_rate_init=learning_rate_init, 
                             max_iter=max_iter, momentum=momentum, nesterovs_momentum=nesterovs_momentum)

    x_train_mapped = input_mapping(x_train, B_dict[incNone], device, order)
    x_test_mapped = input_mapping(x_test, B_dict[incNone], device, order)

    regressor.fit(x_train_mapped, y_train.flatten())

    y_pred_train = torch.tensor(regressor.predict(x_train_mapped)).view(-1, 1)
    y_pred_test = torch.tensor(regressor.predict(x_test_mapped)).view(-1, 1)

    loss_train = criterion(y_pred_train, y_train)
    loss_test = criterion(y_pred_test, y_test)

    loss_train_mae = criterion_mae(y_pred_train, y_train)
    loss_test_mae = criterion_mae(y_pred_test, y_test)

    r2_train = r2_score(y_pred_train.cpu().detach().numpy(), y_train.cpu().detach().numpy())
    r2_test = r2_score(y_pred_test.cpu().detach().numpy(), y_test.cpu().detach().numpy())

    print(f'LossTrain: {loss_train}, lossTest: {loss_test}, R2 Train: {r2_train}, R2 Test: {r2_test}')

    return regressor, loss_train, loss_test, loss_train_mae, loss_test_mae, r2_train, r2_test


def objective_rf(trial, x_train, x_test, y_train, y_test, order, incNone, model_name, initial_opt_time ):
    """Objective function for optimizing hyperparameters of Random Forest Regressor using Optuna.

    Args:
        trial: An instance of the Optuna trial.
        x_train (np.ndarray): Training features.
        x_test (np.ndarray): Testing features.
        y_train (np.ndarray): Training target values.
        y_test (np.ndarray): Testing target values.
        order (int): Order of input mapping.
        incNone (str): Specifies the inclusion method (e.g., 'none', 'linear', 'gauss').
        model_name (str): Name of the model being trained.

    Returns:
        float: The optimized score based on R² values for the training and testing sets.
    """
  
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_float('max_features', 0.1, 1.0)
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    print("n_estimators      ", n_estimators)
    print("max_depth         ", max_depth)
    print("min_samples_split ", min_samples_split)
    print("min_samples_leaf  ", min_samples_leaf)
    print("max_features      ", max_features)
    print("bootstrap         ", bootstrap)
  
    try:
        print('Training Function')
        outputs = train_model_rf(n_estimators, max_depth, min_samples_split, min_samples_leaf,
                                 max_features, bootstrap, x_train, x_test, y_train, y_test,
                                 device, incNone, order)

        regressor, loss_train, loss_test, loss_train_mae, loss_test_mae, r2_train, r2_test = outputs

        if np.isnan(loss_train) or np.isnan(loss_test) or np.isnan(loss_train_mae) or \
           np.isnan(loss_test_mae) or np.isnan(r2_train) or np.isnan(r2_test):
            output_train = -0.12
        else:
            output_train = (r2_train + r2_test) - 2 * np.abs(np.abs(r2_train) - np.abs(r2_test))

            if output_train < 0:
                output_train = -0.001 - np.exp(output_train)
            else:
                temp_directory_path = dir_name + f'/optuna_models_{iddb}_{model_name}_{case}_{ args.size }/'
                if not os.path.exists(temp_directory_path):
                    os.makedirs(temp_directory_path)

                temp_name = f'{temp_directory_path}/model_{iddb}_size_{size:03}_trialNumber_{trial.number:03}_' + str(output_train)[2:6] 
                temp_name += f'_o_{order}_{incNone}_n_estimators_{n_estimators}_max_depth_{max_depth}_min_samples_split_{min_samples_split}'
                temp_name += f'_min_samples_leaf_{min_samples_leaf}_max_features_{max_features}_bootstrap_{bootstrap}.sav'
                pickle.dump(regressor, open(temp_name, 'wb'))

    except Exception as e:
        print(f"Error occurred: {e}")
        output_train = -0.1

    print(f'Trial { trial.number + 1 }, TimeTrial  {time.time() - initial_opt_time }' )

    return output_train

def objective_gbr(trial, x_train, x_test, y_train, y_test, order, incNone, model_name, initial_opt_time ):
    """Objective function for optimizing hyperparameters of Gradient Boosting Regressor using Optuna.

    Args:
        trial: An instance of the Optuna trial.
        x_train (np.ndarray): Training features.
        x_test (np.ndarray): Testing features.
        y_train (np.ndarray): Training target values.
        y_test (np.ndarray): Testing target values.
        order (int): Order of input mapping.
        incNone (str): Specifies the inclusion method (e.g., 'none', 'linear', 'gauss').
        model_name (str): Name of the model being trained.

    Returns:
        float: The optimized score based on R² values for the training and testing sets.
    """
  
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 40)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    max_features = trial.suggest_float('max_features', 0.1, 1.0)
    learning_rate = trial.suggest_float('learning_rate', 0.00001, 0.3)

    print("n_estimators      ", n_estimators)
    print("max_depth         ", max_depth)
    print("min_samples_split ", min_samples_split)
    print("min_samples_leaf  ", min_samples_leaf)
    print("subsample         ", subsample)
    print("max_features      ", max_features)
    print("learning_rate     ", learning_rate)
  
    try:
        print('Training Function')
        outputs = train_model_gbr(n_estimators, max_depth, min_samples_split, min_samples_leaf,
                                   subsample, max_features, learning_rate, x_train, x_test,
                                   y_train, y_test, device, incNone, order)

        regressor, loss_train, loss_test, loss_train_mae, loss_test_mae, r2_train, r2_test = outputs

        if np.isnan(loss_train) or np.isnan(loss_test) or np.isnan(loss_train_mae) or \
           np.isnan(loss_test_mae) or np.isnan(r2_train) or np.isnan(r2_test):
            output_train = -0.12
        else:
            output_train = (r2_train + r2_test) - 2 * np.abs(np.abs(r2_train) - np.abs(r2_test))

            if output_train < 0:
                output_train = -0.001 - np.exp(output_train)
            else:
                temp_directory_path = dir_name + f'/optuna_models_{iddb}_{model_name}_{case}_{size:03}/'
                if not os.path.exists(temp_directory_path):
                    os.makedirs(temp_directory_path)

                temp_name = f'{temp_directory_path}/model_{iddb}_size_{size:03}_trialNumber_{trial.number:03}_' + str(output_train)[2:6] 
                temp_name += f'_o_{order}_{incNone}_n_estimators_{n_estimators}_max_depth_{max_depth}_min_samples_split_{min_samples_split}_'
                temp_name += f'min_samples_leaf_{min_samples_leaf}_subsample_{subsample}_max_features_{max_features}_learning_rate_{learning_rate}.sav'
                pickle.dump(regressor, open(temp_name, 'wb'))

    except Exception as e:
        print(f"Error occurred: {e}")
        output_train = -0.1
    
    print(f'Trial { trial.number + 1 }, TimeTrial  {time.time() - initial_opt_time }' )

    return output_train

def objective_knr(trial, x_train, x_test, y_train, y_test, order, incNone, model_name, initial_opt_time):
    """Objective function for optimizing hyperparameters of KNeighborsRegressor using Optuna.

    Args:
        trial: An instance of the Optuna trial.
        x_train (np.ndarray): Training features.
        x_test (np.ndarray): Testing features.
        y_train (np.ndarray): Training target values.
        y_test (np.ndarray): Testing target values.
        order (int): Order of input mapping.
        incNone (str): Specifies the inclusion method (e.g., 'none', 'linear', 'gauss').
        model_name (str): Name of the model being trained.

    Returns:
        float: The optimized score based on R² values for the training and testing sets.
    """

    # Suggest hyperparameters for KNeighborsRegressor
    n_neighbors = trial.suggest_int('n_neighbors', 1, 30)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    leaf_size = trial.suggest_int('leaf_size', 10, 100)
    p = trial.suggest_int('p', 1, 2)  # 1 for Manhattan distance, 2 for Euclidean distance
    metric = trial.suggest_categorical('metric', ['minkowski', 'euclidean', 'manhattan'])

    print("n_neighbors       ", n_neighbors)
    print("weights           ", weights)
    print("algorithm         ", algorithm)
    print("leaf_size         ", leaf_size)
    print("p                 ", p)
    print("metric            ", metric)

    try:
        print('Training Function')

        # Train the model
        outputs = train_model_knr(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,device=device, incNone=incNone, order=order)

        regressor, loss_train, loss_test, loss_train_mae, loss_test_mae, r2_train, r2_test = outputs

        if np.isnan(loss_train) or np.isnan(loss_test) or np.isnan(loss_train_mae) or \
           np.isnan(loss_test_mae) or np.isnan(r2_train) or np.isnan(r2_test):
            output_train = -0.12
        else:
            output_train = (r2_train + r2_test) - 2 * np.abs(np.abs(r2_train) - np.abs(r2_test))

            if output_train < 0:
                output_train = -0.001 - np.exp(output_train)
            else:
                temp_directory_path = dir_name + f'/optuna_models_{iddb}_{model_name}_{case}_{size:03}/'
                if not os.path.exists(temp_directory_path):
                    os.makedirs(temp_directory_path)

                temp_name = f'{temp_directory_path}/model_{iddb}_size_{size:03}_trialNumber_{trial.number:03}_' + str(output_train)[2:6]
                temp_name += f'_o_{order}_{incNone}_n_neighbors_{n_neighbors}_weights_{weights}_algorithm_{algorithm}'
                temp_name += f'_leaf_size_{leaf_size}_p_{p}_metric_{metric}.sav'
                pickle.dump(regressor, open(temp_name, 'wb'))

    except Exception as e:
        print(f"Error occurred: {e}")
        output_train = -0.1

    print(f'Trial { trial.number + 1 }, TimeTrial  {time.time() - initial_opt_time }' )

    return output_train

def objective_mlp(trial, x_train, x_test, y_train, y_test, order, incNone, model_name, initial_opt_time ):
    """Objective function for optimizing hyperparameters of MLPRegressor using Optuna."""
    
    hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [[32], [64], [128], [256], [512], [1024], [32,128], [32,256], [32, 512], [32,1024]])
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
    solver = trial.suggest_categorical('solver', ['adam', 'lbfgs', 'sgd'])
    alpha = trial.suggest_float('alpha', 1e-6, 1e-2, log=True)
    learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-5, 1e-1, log=True)
    # max_iter = trial.suggest_int('max_iter', 100, 1000)
    max_iter = 300
    momentum = trial.suggest_float('momentum', 0.5, 0.99)
    nesterovs_momentum = trial.suggest_categorical('nesterovs_momentum', [True, False])

    print("hidden_layer_sizes  ", hidden_layer_sizes)
    print("activation          ", activation)
    print("solver              ", solver)
    print("alpha               ", alpha)
    print("learning_rate       ", learning_rate)
    print("learning_rate_init  ", learning_rate_init)
    print("max_iter            ", max_iter)
    print("momentum            ", momentum)
    print("nesterovs_momentum  ", nesterovs_momentum)

    try:
        print('Training Function')
        outputs = train_model_mlp(hidden_layer_sizes, activation, solver, alpha, learning_rate, learning_rate_init, max_iter, momentum, nesterovs_momentum, x_train, x_test, y_train, y_test, device, incNone, order)

        regressor, loss_train, loss_test, loss_train_mae, loss_test_mae, r2_train, r2_test = outputs

        if np.isnan(r2_train) or np.isnan(r2_test):
            output_train = -0.1
        else:
            output_train = (r2_train + r2_test) - 2 * np.abs(np.abs(r2_train) - np.abs(r2_test))

            if output_train < 0:
                output_train = -0.001 - np.exp(output_train)
            else:
                temp_directory_path = dir_name + f'/optuna_models_{iddb}_{model_name}_{case}_{size:03}/'
                if not os.path.exists(temp_directory_path):
                    os.makedirs(temp_directory_path)

                temp_name = f'{temp_directory_path}/model_{iddb}_size_{size:03}_trialNumber_{trial.number:03}_' + str(output_train)[2:6]
                temp_name += f'_o_{order}_{incNone}_hls_{hidden_layer_sizes}_act_{activation}_solv_{solver}'
                temp_name += f'_lr_{learning_rate}_lrinit_{learning_rate_init}_maxI_{max_iter}'
                temp_name += f'mome_{momentum}_nesteM_{nesterovs_momentum}.sav'
                pickle.dump(regressor, open(temp_name, 'wb'))

    except Exception as e:
        print(f"Error occurred: {e}")
        output_train = -0.1
    
    print(f'Trial { trial.number + 1 }, TimeTrial  {time.time() - initial_opt_time }' )

    return output_train




# Initialize the dataframe that will store the results
results_df = pd.DataFrame(columns=['trial_number', 'value', 'train_time'])

def save_intermediate_results(study, trial):
    """
    Callback function to save the intermediate results after each trial, with parameters as separate columns.
    
    Args:
        study (optuna.study.Study): The study object.
        trial (optuna.trial.FrozenTrial): The current trial object.
    """
    trial_number = trial.number
    value = trial.value
    params = trial.params
    train_time = trial.user_attrs.get('train_time', None)
    
    # Convert the params dictionary to a DataFrame where each parameter becomes a separate column
    params_df = pd.DataFrame([params])

    # Add the trial number and value as columns to the params DataFrame
    params_df['trial_number'] = int( trial_number ) + 1
    params_df['value'] = value
    params_df['train_time'] = train_time

    # Append the new row to the global DataFrame `results_df`
    global results_df
    results_df = pd.concat([results_df, params_df], ignore_index=True)
    
    # Save the updated DataFrame with a customized name
    results_df.to_csv(f"{dir_name}/optuna_results_{iddb}_{args.model}_{args.size}_{incNone}.csv", index=False)

    # Directories for saving plots (ensure they are defined)
    directory_path_png = f"{dir_name}/optuna_visualization_{iddb}_{args.model}_{size}_png/"
    directory_path_svg = f"{dir_name}/optuna_visualization_{iddb}_{args.model}_{size}_svg/"

    if not os.path.exists(directory_path_png):
        os.makedirs(directory_path_png)
        
    if not os.path.exists(directory_path_svg):
        os.makedirs(directory_path_svg)

    # Attempt to generate and save the plots after each trial
    try:
        print(f"Attempting to save Optuna plot for trial {trial_number}...")

        # Ensure the plot is only created if there are enough trials
        if len(study.trials) > 1:
            generate_plots(study, trial_number, directory_path_png, directory_path_svg)
        else:
            print(f"Skipping plots for trial {trial_number}, only 1 trial completed.")
        
    except Exception as e:
        # If an error occurs while saving the plot, print the error message
        print(f"Failed to save Optuna plot for trial {trial_number}. Error: {e}")


def generate_plots(study, trial_number, directory_path_png, directory_path_svg):
    """
    Generate and save plots after each trial with custom configurations.
    
    Args:
        study (optuna.study.Study): The study object.
        trial_number (int): The current trial number.
        directory_path_png (str): Directory path to save PNG plots.
        directory_path_svg (str): Directory path to save SVG plots.
    """
    # Get the best parameters for each trial
    best_params_trial = study.best_params
    custom_name = ''
    for iter, item in enumerate(best_params_trial):
        custom_name += str(item)[:3] + '_' + str(best_params_trial[item]) + '_'

    # Ensure that there are at least two parameters to generate contour plots
    parameters_optuna = [str(item) for item in best_params_trial if isinstance(best_params_trial[item], int) or isinstance(best_params_trial[item], float)]
    pairs_vars = []

    # Create pairs of parameters to plot contour plots
    for i in parameters_optuna:
        for j in parameters_optuna:
            if i != j:
                p = list(set([i, j]))  # Remove duplicates
                if p not in pairs_vars:
                    pairs_vars.append(p)

    # Generate contour plots for each pair of parameters
    for iter, p in enumerate(pairs_vars):
        try:
            print(f"Attempting to save contour plot for parameters: {p[0]} and {p[1]} (Trial {trial_number})...")
            fig = optuna.visualization.plot_contour(study, params=[p[0], p[1]])

            # Customizations for contour plot
            fig.update_traces(colorscale='Blackbody', selector=dict(type='contour'))  # Color palette
            fig.update_traces(line_smoothing=1.15, selector=dict(type='contour'))  # Smooth lines
            fig.update_traces(line_width=0, selector=dict(type='contour'))  # Remove contour lines
            fig.update_traces(marker=dict(size=0.25, color="RoyalBlue"), selector=dict(mode='markers'))  # Marker styling

            # Save the contour plot
            fig.write_image(f"{directory_path_png}{trial_number}_{custom_name}contour_{p[0]}_{p[1]}.png")
            fig.write_image(f"{directory_path_svg}{trial_number}_{custom_name}contour_{p[0]}_{p[1]}.svg")
            print(f"Successfully saved contour plot for parameters {p[0]} and {p[1]} (Trial {trial_number}).")

        except Exception as e:
            print(f"Failed to save contour plot for parameters {p[0]} and {p[1]} (Trial {trial_number}). Error: {e}")

    # Generate parameter importance plot
    try:
        print(f"Attempting to save parameter importances plot (Trial {trial_number})...")
        fig = optuna.visualization.plot_param_importances(study)
        fig.update_layout(template='simple_white')
        fig.write_image(f"{directory_path_png}{trial_number}_{custom_name}param_importances.png")
        fig.write_image(f"{directory_path_svg}{trial_number}_{custom_name}param_importances.svg")
        print(f"Successfully saved parameter importances plot (Trial {trial_number}).")
        
    except Exception as e:
        print(f"Failed to save parameter importances plot for trial {trial_number}. Error: {e}")

    # Generate slice plot for parameters
    try:
        print(f"Attempting to save slice plot (Trial {trial_number})...")
        fig = optuna.visualization.plot_slice(study, params=parameters_optuna)
        fig.update_layout(template='simple_white')
        fig.write_image(f"{directory_path_png}{trial_number}_{custom_name}slice_plot.png")
        fig.write_image(f"{directory_path_svg}{trial_number}_{custom_name}slice_plot.svg")
        print(f"Successfully saved slice plot (Trial {trial_number}).")
        
    except Exception as e:
        print(f"Failed to save slice plot for trial {trial_number}. Error: {e}")



def objective_with_time(trial, x_train, x_test, y_train, y_test, order, incNone, model_name):
    """
    Objective function that includes training time for each trial.

    Args:
        trial (optuna.trial.FrozenTrial): The current trial object.
        x_train (numpy.ndarray): Training data features.
        x_test (numpy.ndarray): Test data features.
        y_train (numpy.ndarray): Training data labels.
        y_test (numpy.ndarray): Test data labels.
        order (int): The order of the polynomial features.
        incNone (str): Control string for inclusion of None.
        model_name (str): The model type (RF, GBR, KNR, or MLP).

    Returns:
        float: The evaluation result of the trial (objective function value).
    """
    start_time = time.time()

    if model_name == 'RF':
        result = objective_rf(trial, x_train, x_test, y_train, y_test, order, incNone, model_name, start_time)
    elif model_name == 'GBR':
        result = objective_gbr(trial, x_train, x_test, y_train, y_test, order, incNone, model_name, start_time)
    elif model_name == 'KNR':
        result = objective_knr(trial, x_train, x_test, y_train, y_test, order, incNone, model_name, start_time)
    elif model_name == 'MLP':
        result = objective_mlp(trial, x_train, x_test, y_train, y_test, order, incNone, model_name, start_time)

    train_time = time.time() - start_time
    trial.set_user_attr('train_time', train_time)

    return result

# Create the Optuna study and optimize the model
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(sampler=sampler, directions=["maximize"], study_name=f"{args.model}_regression")


start_time = time.time()
# Perform the optimization with the callback
if args.model == 'RF':
    study.optimize(lambda trial: objective_with_time(trial, x_train, x_test, y_train, y_test, int(args.order), incNone, args.model), n_trials = 150, callbacks=[save_intermediate_results])
elif args.model == 'GBR':
    study.optimize(lambda trial: objective_with_time(trial, x_train, x_test, y_train, y_test, int(args.order), incNone, args.model), n_trials = 150, callbacks=[save_intermediate_results])
elif args.model == 'KNR':
    study.optimize(lambda trial: objective_with_time(trial, x_train, x_test, y_train, y_test, int(args.order), incNone, args.model), n_trials = 150, callbacks=[save_intermediate_results])
elif args.model == 'MLP':
    study.optimize(lambda trial: objective_with_time(trial, x_train, x_test, y_train, y_test, int(args.order), incNone, args.model), n_trials = 150, callbacks=[save_intermediate_results])

# Finally, print the best results
best_params_trial = study.best_params
print('Best parameters:', best_params_trial)



# final_opt_time = time.time()

# best_params_trial = study.best_params
# print(best_params_trial)



# best_value_trial  = study.best_trial.values[0]

# print('Final Best value:', best_value_trial)
# print('Final Best parameters:')
# print( best_params_trial)


# finalModelTime = time.time()

# if args.model == 'RF':
#     n_estimators      =   int(best_params_trial[ "n_estimators" ])
#     max_depth         =   int(best_params_trial[ "max_depth" ])
#     min_samples_split =   int(best_params_trial[ "min_samples_split" ])
#     min_samples_leaf  =   int(best_params_trial[ "min_samples_leaf" ])
#     max_features      = float(best_params_trial[ "max_features" ])
#     bootstrap         =  bool(best_params_trial[ "bootstrap" ])
#     # Train Model Best Hiperparameters
    
#     outputs = train_model_rf( n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, x_train, x_test, y_train, y_test, device, incNone, int(args.order) )

# elif args.model == 'GBR':
#     n_estimators      = int(best_params_trial['n_estimators'])
#     max_depth         = int(best_params_trial['max_depth'])
#     min_samples_split = int(best_params_trial['min_samples_split'])
#     min_samples_leaf  = int(best_params_trial['min_samples_leaf'])
#     subsample         = float(best_params_trial['subsample'])
#     max_features      = float(best_params_trial['max_features'])
#     learning_rate     = float(best_params_trial['learning_rate'])
#     # Train Model Best Hiperparameters
    
#     outputs = train_model_gbr( n_estimators, max_depth, min_samples_split, min_samples_leaf, subsample, max_features, learning_rate,  x_train, x_test, y_train, y_test, device, incNone, int(args.order) )

# elif args.model == 'KNR':
#     n_neighbors = int(best_params_trial["n_neighbors"])
#     weights     = str(best_params_trial["weights"])
#     algorithm   = str(best_params_trial["algorithm"])
#     leaf_size   = int(best_params_trial["leaf_size"])
#     p           = int(best_params_trial["p"])
#     metric      = str(best_params_trial["metric"])
#     # Train Model with Best Hyperparameters
    
#     outputs = train_model_knr(n_neighbors, weights, algorithm, leaf_size, p, metric, x_train, x_test, y_train, y_test, device, incNone, int(args.order))

# elif args.model == 'MLP':
#     hidden_layer_sizes = tuple(best_params_trial["hidden_layer_sizes"])
#     activation         = str(best_params_trial["activation"])
#     solver             = str(best_params_trial["solver"])
#     alpha              = float(best_params_trial["alpha"])
#     learning_rate      = str(best_params_trial["learning_rate"])
#     learning_rate_init = float(best_params_trial["learning_rate_init"])
#     max_iter           = int(best_params_trial["max_iter"])
#     momentum           = float(best_params_trial["momentum"])
#     nesterovs_momentum = bool(best_params_trial["nesterovs_momentum"])

#     # Train Model with Best Hyperparameters

#     outputs = train_model_mlp(hidden_layer_sizes, activation, solver, alpha, learning_rate, learning_rate_init, max_iter, momentum, nesterovs_momentum, x_train, x_test, y_train, y_test, device, incNone, int(args.order))
# final_model_training_time = time.time()
# print('Training Time', final_model_training_time - finalModelTime)


# regressor, lossTrain, lossTest, lossTrainMAE, lossTestMAE, r2_train, r2_test = outputs

# temp_name  = f'{dir_name}/model_{iddb}_{case}_size_{size:03}'
# temp_name += f'_o_{args.order}_{incNone}_' + custom_name[:-2] + '.sav'

# pickle.dump(regressor, open(temp_name, 'wb'))
















# # B_dict = {}
# # if incNone == 'none':
# #     B_dict['none'] = None
# # elif incNone == 'linear':
# #     B_dict['linear'] = torch.eye(x_train.shape[1], dtype = torch.float32).to(device)
# # elif incNone == 'gauss':
# #     B_dict['gauss'] = torch.normal(0, 1, size = (x_train.shape[1], x_train.shape[1] )).to(device)
# # else:
# #     B_dict['linear'] = torch.eye( x_train.shape[1], dtype = torch.float32).to(device)
    

# # x_train_mapped = input_mapping(x_train, B_dict[incNone], device)
# # x_test_mapped  = input_mapping(x_test,  B_dict[incNone], device)



# # yp_train = regressor.predict(x_train_mapped)
# # yp_test  = regressor.predict(x_test_mapped )


# # yp_train.shape, y.shape


# # print(yp_train)


# # # fig, axs = plt.subplots(1,1, figsize = ( 10,10))

# # # sns.kdeplot(data =  y.flatten().detach().numpy(), ax = axs[0,0], label = 'Real', linestyle='-', linewidth = 1.05, color = 'blue' )
# # # sns.kdeplot(data =  yp_train, ax = axs[0,0], label = 'Real', linestyle='dotted', linewidth = 1.05, color = 'red' )
# # # sns.kdeplot(data =  yp_test, ax = axs[0,0], label = 'Real', linestyle='dashed', linewidth = 1.05, color = 'green' )

# # sns.kdeplot(data =  y.flatten().detach().numpy(), label = 'Real', linestyle='-', linewidth = 1.05, color = 'blue' )
# # sns.kdeplot(data =  yp_train, label = 'Train', linestyle='dotted', linewidth = 1.05, color = 'red' )
# # sns.kdeplot(data =  yp_test, label = 'Test', linestyle='dashed', linewidth = 1.05, color = 'green' )
# # plt.legend()


# # plt.show()


# # loaded_model = pickle.load(open(temp_name, 'rb'))
# # result = loaded_model.score(x_train_mapped, y_train)
# # resulT = loaded_model.score(x_test_mapped, y_test)
# # print(result, resulT)


# # r2_score(y_train, yp_train), r2_score(y_test, yp_test)


# # plt.figure(figsize=(5,5))
# # plt.scatter(y_train, yp_train,s=5, color = 'red', label = 'Train', marker = 'd')
# # plt.scatter(y_test, yp_test, s = 3, color = 'blue', label ='Test')

# # min_val = min(min(y_train.flatten()), min(yp_train.flatten()), min(yp_test.flatten()), min(y_test.flatten()))
# # max_val = max(max(y_train.flatten()), max(yp_train.flatten()), max(yp_test.flatten()), max(y_test.flatten()))
# # # max_val = max(max(y.detach().numpy()), max(yp.detach().numpy()))
# # plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

# # plt.legend()
# # plt.grid(linewidth = 0.1)
# # plt.show()





# process = psutil.Process(os.getpid())
# end_memory = process.memory_info().rss
# print(f'MEMORYPROCESS: {(end_memory - start_memory) / (1024**2)} MB')

