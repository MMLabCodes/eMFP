
import os
import sys
import glob

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

from models import *

import argparse
parser = argparse.ArgumentParser()

import pickle

from memory_profiler import profile, memory_usage




# ==========================
# MAE Loss (Mean Absolute Error)
# ==========================
class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true))

# ==========================
# MSE Loss (Mean Squared Error)
# ==========================
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2)

# ==========================
# MedianAE Loss
# ==========================
class MedianAELoss(nn.Module):
    def __init__(self):
        super(MedianAELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.median(torch.abs(y_pred - y_true))

# ==========================
# RMSE Loss (Root Mean Square Error Loss)
# ==========================
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2))

# ==========================
# LogCosh Loss
# ==========================
class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = torch.log(torch.cosh(y_pred - y_true))
        return torch.mean(loss)


class R2Score(nn.Module):
    def __init__(self):
        super(R2Score, self).__init__()

    def forward(self, y_pred, y_true):
        # Convertir tensores de torch a numpy
        y_pred_np = y_pred.detach().cpu().numpy()
        y_true_np = y_true.detach().cpu().numpy()
        return torch.tensor(r2_score(y_true_np, y_pred_np))
    
# MAE Loss (Mean Absolute Error)
maeLoss   = MAELoss()

# MSE Loss (Mean Squared Error)
mseLoss   = MSELoss()

# MedianAE Loss
medaeLoss = MedianAELoss()

# RMSE Loss
rmseLoss  = RMSELoss()

# R2 (Coefficient of Determination)

r2_scoring = R2Score()

# ========================================================================================================
# ========================================================================================================
# ========================================================================================================

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

def get_fold(x, y, k, N, seed=42):
    """Split the data into N folds and return the training and validation sets
    corresponding to the k-th fold. Works with NumPy arrays or PyTorch tensors.

    Args:
        x (array-like or torch.Tensor): Feature vector (n samples).
        y (array-like or torch.Tensor): Target vector (n samples).
        k (int): Index of the desired fold (0 <= k < N).
        N (int): Total number of folds.
        seed (int, optional): Random seed for shuffling the data. Defaults to 42.

    Returns:
        tuple:
            x_train: Training features (same type as input).
            y_train: Training targets (same type as input).
            x_valid: Validation features (same type as input).
            y_valid: Validation targets (same type as input).
    """
    # Detect backend
    use_torch = torch is not None and isinstance(x, torch.Tensor)

    if use_torch:
        # Convert to numpy for indexing
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
    else:
        x_np = np.array(x)
        y_np = np.array(y)

    # Ensure same length
    assert len(x_np) == len(y_np), "x and y must have the same number of elements"

    # Shuffle indices
    rng = np.random.default_rng(seed)
    indices = np.arange(len(x_np))
    rng.shuffle(indices)

    # Split into folds
    folds = np.array_split(indices, N)

    # Select validation fold
    valid_idx = folds[k]
    train_idx = np.hstack([folds[i] for i in range(N) if i != k])

    # Slice arrays
    x_train, y_train = x_np[train_idx], y_np[train_idx]
    x_valid, y_valid = x_np[valid_idx], y_np[valid_idx]

    # Convert back to torch if input was torch
    if use_torch:
        x_train = torch.tensor(x_train, dtype=x.dtype)
        y_train = torch.tensor(y_train, dtype=y.dtype)
        x_valid = torch.tensor(x_valid, dtype=x.dtype)
        y_valid = torch.tensor(y_valid, dtype=y.dtype)

    return x_train, y_train, x_valid, y_valid

def input_mapping(x, B, device, order):
    if B is None:
        return x.to(device)
    else:
        sin_list, cos_list = [], []
        x_proj = torch.matmul(2. * torch.pi * x, B.T).to(device)
        for ord in range( order ):
            sin_list.append( torch.sin( ( ord + 1 ) * x_proj ) )
            cos_list.append( torch.cos( ( ord + 1 ) * x_proj ) )
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
    plt.savefig("Epoch_{}_pred.png".format(epoch), dpi = 600)
    # plt.show()


def train_model_rf(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,
                   bootstrap, x, y, device, incNone, order):
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

    B_dict = {}
    if incNone == 'without_FFNN':
        B_dict['without_FFNN'] = None

    elif  incNone == 'without_FFNN_with_Descriptors':
        B_dict['without_FFNN_with_Descriptors'] = None

    elif incNone == 'with_FFNN':
        B_dict['with_FFNN'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)

    elif incNone == 'with_FFNN_with_Descriptors':
        B_dict['with_FFNN_with_Descriptors'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)

    elif incNone == 'with_FFNN_Gaussian':
        B_dict['with_FFNN_Gaussian'] = torch.normal(0, 1, size=(x_train.shape[1], x_train.shape[1])).to(device)

    elif incNone == 'with_FFNN_Gaussian_with_Descriptors':
        B_dict['with_FFNN_Gaussian_with_Descriptors'] = torch.normal(0, 1, size=(x_train.shape[1], x_train.shape[1])).to(device)

    else:
        B_dict['with_FFNN'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)


    
    regressor1 = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                       max_features=max_features, bootstrap=bootstrap)
    regressor2 = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                       max_features=max_features, bootstrap=bootstrap)
    regressor3 = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                       max_features=max_features, bootstrap=bootstrap)
    regressor4 = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                       max_features=max_features, bootstrap=bootstrap)
    regressor5 = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                       max_features=max_features, bootstrap=bootstrap)

    # obtain 5 different folds for CV
    x_train1, y_train1, x_valid1, y_valid1 = get_fold(x, y, 1 - 1,  5, seed = 42)
    x_train2, y_train2, x_valid2, y_valid2 = get_fold(x, y, 2 - 1,  5, seed = 42)
    x_train3, y_train3, x_valid3, y_valid3 = get_fold(x, y, 3 - 1,  5, seed = 42)
    x_train4, y_train4, x_valid4, y_valid4 = get_fold(x, y, 4 - 1,  5, seed = 42)
    x_train5, y_train5, x_valid5, y_valid5 = get_fold(x, y, 5 - 1,  5, seed = 42)


    # Apply the corresponding input mapping FFNN to train/validation subsets
    x_train_mapped1 = input_mapping(x_train1, B_dict[incNone], device, order)
    x_train_mapped2 = input_mapping(x_train2, B_dict[incNone], device, order)
    x_train_mapped3 = input_mapping(x_train3, B_dict[incNone], device, order)
    x_train_mapped4 = input_mapping(x_train4, B_dict[incNone], device, order)
    x_train_mapped5 = input_mapping(x_train5, B_dict[incNone], device, order)

    x_valid_mapped1 = input_mapping(x_valid1, B_dict[incNone], device, order)
    x_valid_mapped2 = input_mapping(x_valid2, B_dict[incNone], device, order)
    x_valid_mapped3 = input_mapping(x_valid3, B_dict[incNone], device, order)
    x_valid_mapped4 = input_mapping(x_valid4, B_dict[incNone], device, order)
    x_valid_mapped5 = input_mapping(x_valid5, B_dict[incNone], device, order)

    # Train every regressor for CV
    timer0 = time.time()
    regressor1.fit(x_train_mapped1, y_train1.flatten())

    timer1 = time.time()
    regressor2.fit(x_train_mapped2, y_train2.flatten())
    
    timer2 = time.time()
    regressor3.fit(x_train_mapped3, y_train3.flatten())

    timer3 = time.time()
    regressor4.fit(x_train_mapped4, y_train4.flatten())
    
    timer4 = time.time()
    regressor5.fit(x_train_mapped5, y_train5.flatten())

    timer5 = time.time()

    training_time_fold1 = timer1 - timer0
    training_time_fold2 = timer2 - timer1
    training_time_fold3 = timer3 - timer2
    training_time_fold4 = timer4 - timer3
    training_time_fold5 = timer5 - timer4

    training_times = {
                        'training_InnerFold1' : training_time_fold1,
                        'training_InnerFold2' : training_time_fold2,
                        'training_InnerFold3' : training_time_fold3,
                        'training_InnerFold4' : training_time_fold4,
                        'training_InnerFold5' : training_time_fold5,
                     }


    # Perform predictions after training in train/validation subsets
    y_pred_train1 = torch.tensor(regressor1.predict(x_train_mapped1)).view(-1, 1)
    y_pred_train2 = torch.tensor(regressor2.predict(x_train_mapped2)).view(-1, 1)
    y_pred_train3 = torch.tensor(regressor3.predict(x_train_mapped3)).view(-1, 1)
    y_pred_train4 = torch.tensor(regressor4.predict(x_train_mapped4)).view(-1, 1)
    y_pred_train5 = torch.tensor(regressor5.predict(x_train_mapped5)).view(-1, 1)

    y_pred_valid1 = torch.tensor(regressor1.predict(x_valid_mapped1)).view(-1, 1)
    y_pred_valid2 = torch.tensor(regressor2.predict(x_valid_mapped2)).view(-1, 1)
    y_pred_valid3 = torch.tensor(regressor3.predict(x_valid_mapped3)).view(-1, 1)
    y_pred_valid4 = torch.tensor(regressor4.predict(x_valid_mapped4)).view(-1, 1)
    y_pred_valid5 = torch.tensor(regressor5.predict(x_valid_mapped5)).view(-1, 1)

    # Define Metrics to Use

    # MAE Loss (Mean Absolute Error)
    criterion_maeLoss   = MAELoss()

    # MSE Loss (Mean Squared Error)
    criterion_mseLoss   = MSELoss()

    # MedianAE Loss
    criterion_medaeLoss = MedianAELoss()

    # RMSE Loss
    criterion_rmseLoss  = RMSELoss()

    # R2 Score
    criterion_r2_score = R2Score()

    fn_losses_metrics = {
                        'MAE' : criterion_maeLoss,
                        'MSE' : criterion_mseLoss,
                        'MDAE': criterion_medaeLoss,
                        'RMSE': criterion_rmseLoss,
                        'R2'  : criterion_r2_score
                        }

    
    metric_losses_reg = {}
    regressors = {
                    'reg1': regressor1,
                    'reg2': regressor2,
                    'reg3': regressor3,
                    'reg4': regressor4,
                    'reg5': regressor5,
                 }
    
    # Calculate every loss/metric in fn_losses_metric for every k-th fold
    for name, loss_fn in fn_losses_metrics.items() :
        metric_losses_reg.update( { "{}_train_InnerFold1".format(name) :  loss_fn( y_pred_train1, y_train1 ) } )
        metric_losses_reg.update( { "{}_train_InnerFold2".format(name) :  loss_fn( y_pred_train2, y_train2 ) } )
        metric_losses_reg.update( { "{}_train_InnerFold3".format(name) :  loss_fn( y_pred_train3, y_train3 ) } )
        metric_losses_reg.update( { "{}_train_InnerFold4".format(name) :  loss_fn( y_pred_train4, y_train4 ) } )
        metric_losses_reg.update( { "{}_train_InnerFold5".format(name) :  loss_fn( y_pred_train5, y_train5 ) } )

        metric_losses_reg.update( { "{}_valid_InnerFold1".format(name) :  loss_fn( y_pred_valid1, y_valid1 ) } )
        metric_losses_reg.update( { "{}_valid_InnerFold2".format(name) :  loss_fn( y_pred_valid2, y_valid2 ) } )
        metric_losses_reg.update( { "{}_valid_InnerFold3".format(name) :  loss_fn( y_pred_valid3, y_valid3 ) } )
        metric_losses_reg.update( { "{}_valid_InnerFold4".format(name) :  loss_fn( y_pred_valid4, y_valid4 ) } )
        metric_losses_reg.update( { "{}_valid_InnerFold5".format(name) :  loss_fn( y_pred_valid5, y_valid5 ) } )

    return regressors, metric_losses_reg, training_times



def train_model_gbr(n_estimators, max_depth, min_samples_split, min_samples_leaf, subsample,
                    max_features, learning_rate, x, y, device, incNone, order):
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

    B_dict = {}
    if incNone == 'without_FFNN':
        B_dict['without_FFNN'] = None

    elif  incNone == 'without_FFNN_with_Descriptors':
        B_dict['without_FFNN_with_Descriptors'] = None

    elif incNone == 'with_FFNN':
        B_dict['with_FFNN'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)

    elif incNone == 'with_FFNN_with_Descriptors':
        B_dict['with_FFNN_with_Descriptors'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)

    elif incNone == 'with_FFNN_Gaussian':
        B_dict['with_FFNN_Gaussian'] = torch.normal(0, 1, size=(x_train.shape[1], x_train.shape[1])).to(device)

    elif incNone == 'with_FFNN_Gaussian_with_Descriptors':
        B_dict['with_FFNN_Gaussian_with_Descriptors'] = torch.normal(0, 1, size=(x_train.shape[1], x_train.shape[1])).to(device)

    else:
        B_dict['with_FFNN'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)



    regressor1 = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators,
                                           max_depth=max_depth, min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf, subsample=subsample,
                                           max_features=max_features, random_state=42)
    regressor2 = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators,
                                           max_depth=max_depth, min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf, subsample=subsample,
                                           max_features=max_features, random_state=42)    
    regressor3 = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators,
                                           max_depth=max_depth, min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf, subsample=subsample,
                                           max_features=max_features, random_state=42)    
    regressor4 = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators,
                                           max_depth=max_depth, min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf, subsample=subsample,
                                           max_features=max_features, random_state=42)
    regressor5 = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators,
                                           max_depth=max_depth, min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf, subsample=subsample,
                                           max_features=max_features, random_state=42)    

    # obtain 5 different folds for CV
    x_train1, y_train1, x_valid1, y_valid1 = get_fold(x, y, 1 - 1,  5, seed = 42)
    x_train2, y_train2, x_valid2, y_valid2 = get_fold(x, y, 2 - 1,  5, seed = 42)
    x_train3, y_train3, x_valid3, y_valid3 = get_fold(x, y, 3 - 1,  5, seed = 42)
    x_train4, y_train4, x_valid4, y_valid4 = get_fold(x, y, 4 - 1,  5, seed = 42)
    x_train5, y_train5, x_valid5, y_valid5 = get_fold(x, y, 5 - 1,  5, seed = 42)

    # Apply the corresponding input mapping FFNN to train/validation subsets
    x_train_mapped1 = input_mapping(x_train1, B_dict[incNone], device, order)
    x_train_mapped2 = input_mapping(x_train2, B_dict[incNone], device, order)
    x_train_mapped3 = input_mapping(x_train3, B_dict[incNone], device, order)
    x_train_mapped4 = input_mapping(x_train4, B_dict[incNone], device, order)
    x_train_mapped5 = input_mapping(x_train5, B_dict[incNone], device, order)

    x_valid_mapped1 = input_mapping(x_valid1, B_dict[incNone], device, order)
    x_valid_mapped2 = input_mapping(x_valid2, B_dict[incNone], device, order)
    x_valid_mapped3 = input_mapping(x_valid3, B_dict[incNone], device, order)
    x_valid_mapped4 = input_mapping(x_valid4, B_dict[incNone], device, order)
    x_valid_mapped5 = input_mapping(x_valid5, B_dict[incNone], device, order)

    # Train every regressor for CV
    timer0 = time.time()
    regressor1.fit(x_train_mapped1, y_train1.flatten())

    timer1 = time.time()
    regressor2.fit(x_train_mapped2, y_train2.flatten())
    
    timer2 = time.time()
    regressor3.fit(x_train_mapped3, y_train3.flatten())

    timer3 = time.time()
    regressor4.fit(x_train_mapped4, y_train4.flatten())
    
    timer4 = time.time()
    regressor5.fit(x_train_mapped5, y_train5.flatten())

    timer5 = time.time()

    training_time_fold1 = timer1 - timer0
    training_time_fold2 = timer2 - timer1
    training_time_fold3 = timer3 - timer2
    training_time_fold4 = timer4 - timer3
    training_time_fold5 = timer5 - timer4

    training_times = {
                        'training_InnerFold1' : training_time_fold1,
                        'training_InnerFold2' : training_time_fold2,
                        'training_InnerFold3' : training_time_fold3,
                        'training_InnerFold4' : training_time_fold4,
                        'training_InnerFold5' : training_time_fold5,
                     }

    # Perform predictions after training in train/validation subsets
    y_pred_train1 = torch.tensor(regressor1.predict(x_train_mapped1)).view(-1, 1)
    y_pred_train2 = torch.tensor(regressor2.predict(x_train_mapped2)).view(-1, 1)
    y_pred_train3 = torch.tensor(regressor3.predict(x_train_mapped3)).view(-1, 1)
    y_pred_train4 = torch.tensor(regressor4.predict(x_train_mapped4)).view(-1, 1)
    y_pred_train5 = torch.tensor(regressor5.predict(x_train_mapped5)).view(-1, 1)

    y_pred_valid1 = torch.tensor(regressor1.predict(x_valid_mapped1)).view(-1, 1)
    y_pred_valid2 = torch.tensor(regressor2.predict(x_valid_mapped2)).view(-1, 1)
    y_pred_valid3 = torch.tensor(regressor3.predict(x_valid_mapped3)).view(-1, 1)
    y_pred_valid4 = torch.tensor(regressor4.predict(x_valid_mapped4)).view(-1, 1)
    y_pred_valid5 = torch.tensor(regressor5.predict(x_valid_mapped5)).view(-1, 1)

    # Define Metrics to Use
    # MAE Loss (Mean Absolute Error)
    criterion_maeLoss   = MAELoss()

    # MSE Loss (Mean Squared Error)
    criterion_mseLoss   = MSELoss()

    # MedianAE Loss
    criterion_medaeLoss = MedianAELoss()

    # RMSE Loss
    criterion_rmseLoss  = RMSELoss()

    # R2 Score
    criterion_r2_score = R2Score()
    
    fn_losses_metrics = {
                        'MAE' : criterion_maeLoss,
                        'MSE' : criterion_mseLoss,
                        'MDAE': criterion_medaeLoss,
                        'RMSE': criterion_rmseLoss,
                        'R2'  : criterion_r2_score
                        }

    
    metric_losses_reg = {}
    regressors = {
                    'reg1': regressor1,
                    'reg2': regressor2,
                    'reg3': regressor3,
                    'reg4': regressor4,
                    'reg5': regressor5,
                 }
    
    # Calculate every loss/metric in fn_losses_metric for every k-th fold
    for name, loss_fn in fn_losses_metrics.items() :
        metric_losses_reg.update( { "{}_train_InnerFold1".format(name) :  loss_fn( y_pred_train1, y_train1 ) } )
        metric_losses_reg.update( { "{}_train_InnerFold2".format(name) :  loss_fn( y_pred_train2, y_train2 ) } )
        metric_losses_reg.update( { "{}_train_InnerFold3".format(name) :  loss_fn( y_pred_train3, y_train3 ) } )
        metric_losses_reg.update( { "{}_train_InnerFold4".format(name) :  loss_fn( y_pred_train4, y_train4 ) } )
        metric_losses_reg.update( { "{}_train_InnerFold5".format(name) :  loss_fn( y_pred_train5, y_train5 ) } )
        
        metric_losses_reg.update( { "{}_valid_InnerFold1".format(name) :  loss_fn( y_pred_valid1, y_valid1 ) } )
        metric_losses_reg.update( { "{}_valid_InnerFold2".format(name) :  loss_fn( y_pred_valid2, y_valid2 ) } )
        metric_losses_reg.update( { "{}_valid_InnerFold3".format(name) :  loss_fn( y_pred_valid3, y_valid3 ) } )
        metric_losses_reg.update( { "{}_valid_InnerFold4".format(name) :  loss_fn( y_pred_valid4, y_valid4 ) } )
        metric_losses_reg.update( { "{}_valid_InnerFold5".format(name) :  loss_fn( y_pred_valid5, y_valid5 ) } )

    return regressors, metric_losses_reg, training_times

def train_model_knr(n_neighbors, weights, algorithm, leaf_size, p, metric, 
                    x, y, device, incNone, order):
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

    # Prepare input mapping based on incNone argument
    B_dict = {}
    if incNone == 'without_FFNN':
        B_dict['without_FFNN'] = None

    elif  incNone == 'without_FFNN_with_Descriptors':
        B_dict['without_FFNN_with_Descriptors'] = None

    elif incNone == 'with_FFNN':
        B_dict['with_FFNN'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)

    elif incNone == 'with_FFNN_with_Descriptors':
        B_dict['with_FFNN_with_Descriptors'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)

    elif incNone == 'with_FFNN_Gaussian':
        B_dict['with_FFNN_Gaussian'] = torch.normal(0, 1, size=(x_train.shape[1], x_train.shape[1])).to(device)

    elif incNone == 'with_FFNN_Gaussian_with_Descriptors':
        B_dict['with_FFNN_Gaussian_with_Descriptors'] = torch.normal(0, 1, size=(x_train.shape[1], x_train.shape[1])).to(device)

    else:
        B_dict['with_FFNN'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)


    # Initialize the KNeighborsRegressor with the suggested hyperparameters
    regressor1 = KNeighborsRegressor(n_neighbors=n_neighbors, 
                                    weights=weights, 
                                    algorithm=algorithm, 
                                    leaf_size=leaf_size, 
                                    p=p, 
                                    metric=metric)
    regressor2 = KNeighborsRegressor(n_neighbors=n_neighbors, 
                                    weights=weights, 
                                    algorithm=algorithm, 
                                    leaf_size=leaf_size, 
                                    p=p, 
                                    metric=metric)
    regressor3 = KNeighborsRegressor(n_neighbors=n_neighbors, 
                                    weights=weights, 
                                    algorithm=algorithm, 
                                    leaf_size=leaf_size, 
                                    p=p, 
                                    metric=metric)
    regressor4 = KNeighborsRegressor(n_neighbors=n_neighbors, 
                                    weights=weights, 
                                    algorithm=algorithm, 
                                    leaf_size=leaf_size, 
                                    p=p, 
                                    metric=metric)
    regressor5 = KNeighborsRegressor(n_neighbors=n_neighbors, 
                                    weights=weights, 
                                    algorithm=algorithm, 
                                    leaf_size=leaf_size, 
                                    p=p, 
                                    metric=metric)
            
# obtain 5 different folds for CV
    x_train1, y_train1, x_valid1, y_valid1 = get_fold(x, y, 1 - 1,  5, seed = 42)
    x_train2, y_train2, x_valid2, y_valid2 = get_fold(x, y, 2 - 1,  5, seed = 42)
    x_train3, y_train3, x_valid3, y_valid3 = get_fold(x, y, 3 - 1,  5, seed = 42)
    x_train4, y_train4, x_valid4, y_valid4 = get_fold(x, y, 4 - 1,  5, seed = 42)
    x_train5, y_train5, x_valid5, y_valid5 = get_fold(x, y, 5 - 1,  5, seed = 42)

    # Apply the corresponding input mapping FFNN to train/validation subsets
    x_train_mapped1 = input_mapping(x_train1, B_dict[incNone], device, order)
    x_train_mapped2 = input_mapping(x_train2, B_dict[incNone], device, order)
    x_train_mapped3 = input_mapping(x_train3, B_dict[incNone], device, order)
    x_train_mapped4 = input_mapping(x_train4, B_dict[incNone], device, order)
    x_train_mapped5 = input_mapping(x_train5, B_dict[incNone], device, order)

    x_valid_mapped1 = input_mapping(x_valid1, B_dict[incNone], device, order)
    x_valid_mapped2 = input_mapping(x_valid2, B_dict[incNone], device, order)
    x_valid_mapped3 = input_mapping(x_valid3, B_dict[incNone], device, order)
    x_valid_mapped4 = input_mapping(x_valid4, B_dict[incNone], device, order)
    x_valid_mapped5 = input_mapping(x_valid5, B_dict[incNone], device, order)

    # Train every regressor for CV
    timer0 = time.time()
    regressor1.fit(x_train_mapped1, y_train1.flatten())

    timer1 = time.time()
    regressor2.fit(x_train_mapped2, y_train2.flatten())
    
    timer2 = time.time()
    regressor3.fit(x_train_mapped3, y_train3.flatten())

    timer3 = time.time()
    regressor4.fit(x_train_mapped4, y_train4.flatten())
    
    timer4 = time.time()
    regressor5.fit(x_train_mapped5, y_train5.flatten())

    timer5 = time.time()

    training_time_fold1 = timer1 - timer0
    training_time_fold2 = timer2 - timer1
    training_time_fold3 = timer3 - timer2
    training_time_fold4 = timer4 - timer3
    training_time_fold5 = timer5 - timer4

    training_times = {
                        'training_InnerFold1' : training_time_fold1,
                        'training_InnerFold2' : training_time_fold2,
                        'training_InnerFold3' : training_time_fold3,
                        'training_InnerFold4' : training_time_fold4,
                        'training_InnerFold5' : training_time_fold5,
                     }

    # Perform predictions after training in train/validation subsets
    y_pred_train1 = torch.tensor(regressor1.predict(x_train_mapped1)).view(-1, 1)
    y_pred_train2 = torch.tensor(regressor2.predict(x_train_mapped2)).view(-1, 1)
    y_pred_train3 = torch.tensor(regressor3.predict(x_train_mapped3)).view(-1, 1)
    y_pred_train4 = torch.tensor(regressor4.predict(x_train_mapped4)).view(-1, 1)
    y_pred_train5 = torch.tensor(regressor5.predict(x_train_mapped5)).view(-1, 1)

    y_pred_valid1 = torch.tensor(regressor1.predict(x_valid_mapped1)).view(-1, 1)
    y_pred_valid2 = torch.tensor(regressor2.predict(x_valid_mapped2)).view(-1, 1)
    y_pred_valid3 = torch.tensor(regressor3.predict(x_valid_mapped3)).view(-1, 1)
    y_pred_valid4 = torch.tensor(regressor4.predict(x_valid_mapped4)).view(-1, 1)
    y_pred_valid5 = torch.tensor(regressor5.predict(x_valid_mapped5)).view(-1, 1)

    # Define Metrics to Use

    # MAE Loss (Mean Absolute Error)
    criterion_maeLoss   = MAELoss()

    # MSE Loss (Mean Squared Error)
    criterion_mseLoss   = MSELoss()

    # MedianAE Loss (Median Absolute Error)
    criterion_medaeLoss = MedianAELoss()

    # RMSE Loss (Root Mean Squared Error)
    criterion_rmseLoss  = RMSELoss()

    # R2 Score (Coefficient of Determination)
    criterion_r2_score = R2Score()

    fn_losses_metrics = {
                        'MAE' : criterion_maeLoss,
                        'MSE' : criterion_mseLoss,
                        'MDAE': criterion_medaeLoss,
                        'RMSE': criterion_rmseLoss,
                        'R2'  : criterion_r2_score
                        }

    
    metric_losses_reg = {}
    regressors = {
                    'reg1': regressor1,
                    'reg2': regressor2,
                    'reg3': regressor3,
                    'reg4': regressor4,
                    'reg5': regressor5,
                 }
    
    # Calculate every loss/metric in fn_losses_metric for every k-th fold
    for name, loss_fn in fn_losses_metrics.items() :
        metric_losses_reg.update( { "{}_train_InnerFold1".format(name) :  loss_fn( y_pred_train1, y_train1 ) } )
        metric_losses_reg.update( { "{}_train_InnerFold2".format(name) :  loss_fn( y_pred_train2, y_train2 ) } )
        metric_losses_reg.update( { "{}_train_InnerFold3".format(name) :  loss_fn( y_pred_train3, y_train3 ) } )
        metric_losses_reg.update( { "{}_train_InnerFold4".format(name) :  loss_fn( y_pred_train4, y_train4 ) } )
        metric_losses_reg.update( { "{}_train_InnerFold5".format(name) :  loss_fn( y_pred_train5, y_train5 ) } )
        
        metric_losses_reg.update( { "{}_valid_InnerFold1".format(name) :  loss_fn( y_pred_valid1, y_valid1 ) } )
        metric_losses_reg.update( { "{}_valid_InnerFold2".format(name) :  loss_fn( y_pred_valid2, y_valid2 ) } )
        metric_losses_reg.update( { "{}_valid_InnerFold3".format(name) :  loss_fn( y_pred_valid3, y_valid3 ) } )
        metric_losses_reg.update( { "{}_valid_InnerFold4".format(name) :  loss_fn( y_pred_valid4, y_valid4 ) } )
        metric_losses_reg.update( { "{}_valid_InnerFold5".format(name) :  loss_fn( y_pred_valid5, y_valid5 ) } )

    return regressors, metric_losses_reg, training_times


def train_model_mlp(hidden_layer_sizes, activation, solver, alpha, learning_rate, 
                    learning_rate_init, max_iter, momentum, nesterovs_momentum, 
                    x, y, device, incNone, order):
    """Train the MLPRegressor model."""


    # Prepare input mapping based on incNone argument
    B_dict = {}
    if incNone == 'without_FFNN':
        B_dict['without_FFNN'] = None

    elif  incNone == 'without_FFNN_with_Descriptors':
        B_dict['without_FFNN_with_Descriptors'] = None

    elif incNone == 'with_FFNN':
        B_dict['with_FFNN'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)

    elif incNone == 'with_FFNN_with_Descriptors':
        B_dict['with_FFNN_with_Descriptors'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)

    elif incNone == 'with_FFNN_Gaussian':
        B_dict['with_FFNN_Gaussian'] = torch.normal(0, 1, size=(x_train.shape[1], x_train.shape[1])).to(device)

    elif incNone == 'with_FFNN_Gaussian_with_Descriptors':
        B_dict['with_FFNN_Gaussian_with_Descriptors'] = torch.normal(0, 1, size=(x_train.shape[1], x_train.shape[1])).to(device)

    else:
        B_dict['with_FFNN'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)



    regressor1 = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, 
                             alpha=alpha, learning_rate=learning_rate, learning_rate_init=learning_rate_init, 
                             max_iter=max_iter, momentum=momentum, nesterovs_momentum=nesterovs_momentum)

    regressor2 = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, 
                             alpha=alpha, learning_rate=learning_rate, learning_rate_init=learning_rate_init, 
                             max_iter=max_iter, momentum=momentum, nesterovs_momentum=nesterovs_momentum)

    regressor3 = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, 
                             alpha=alpha, learning_rate=learning_rate, learning_rate_init=learning_rate_init, 
                             max_iter=max_iter, momentum=momentum, nesterovs_momentum=nesterovs_momentum)

    regressor4 = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, 
                             alpha=alpha, learning_rate=learning_rate, learning_rate_init=learning_rate_init, 
                             max_iter=max_iter, momentum=momentum, nesterovs_momentum=nesterovs_momentum)

    regressor5 = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, 
                             alpha=alpha, learning_rate=learning_rate, learning_rate_init=learning_rate_init, 
                             max_iter=max_iter, momentum=momentum, nesterovs_momentum=nesterovs_momentum)


# obtain 5 different folds for CV
    x_train1, y_train1, x_valid1, y_valid1 = get_fold(x, y, 1 - 1,  5, seed = 42)
    x_train2, y_train2, x_valid2, y_valid2 = get_fold(x, y, 2 - 1,  5, seed = 42)
    x_train3, y_train3, x_valid3, y_valid3 = get_fold(x, y, 3 - 1,  5, seed = 42)
    x_train4, y_train4, x_valid4, y_valid4 = get_fold(x, y, 4 - 1,  5, seed = 42)
    x_train5, y_train5, x_valid5, y_valid5 = get_fold(x, y, 5 - 1,  5, seed = 42)

    # Apply the corresponding input mapping FFNN to train/validation subsets
    x_train_mapped1 = input_mapping(x_train1, B_dict[incNone], device, order)
    x_train_mapped2 = input_mapping(x_train2, B_dict[incNone], device, order)
    x_train_mapped3 = input_mapping(x_train3, B_dict[incNone], device, order)
    x_train_mapped4 = input_mapping(x_train4, B_dict[incNone], device, order)
    x_train_mapped5 = input_mapping(x_train5, B_dict[incNone], device, order)

    x_valid_mapped1 = input_mapping(x_valid1, B_dict[incNone], device, order)
    x_valid_mapped2 = input_mapping(x_valid2, B_dict[incNone], device, order)
    x_valid_mapped3 = input_mapping(x_valid3, B_dict[incNone], device, order)
    x_valid_mapped4 = input_mapping(x_valid4, B_dict[incNone], device, order)
    x_valid_mapped5 = input_mapping(x_valid5, B_dict[incNone], device, order)

    # Train every regressor for CV
    timer0 = time.time()
    regressor1.fit(x_train_mapped1, y_train1.flatten())

    timer1 = time.time()
    regressor2.fit(x_train_mapped2, y_train2.flatten())
    
    timer2 = time.time()
    regressor3.fit(x_train_mapped3, y_train3.flatten())

    timer3 = time.time()
    regressor4.fit(x_train_mapped4, y_train4.flatten())
    
    timer4 = time.time()
    regressor5.fit(x_train_mapped5, y_train5.flatten())

    timer5 = time.time()

    training_time_fold1 = timer1 - timer0
    training_time_fold2 = timer2 - timer1
    training_time_fold3 = timer3 - timer2
    training_time_fold4 = timer4 - timer3
    training_time_fold5 = timer5 - timer4

    training_times = {
                        'training_InnerFold1' : training_time_fold1,
                        'training_InnerFold2' : training_time_fold2,
                        'training_InnerFold3' : training_time_fold3,
                        'training_InnerFold4' : training_time_fold4,
                        'training_InnerFold5' : training_time_fold5,
                     }

    # Perform predictions after training in train/validation subsets
    y_pred_train1 = torch.tensor(regressor1.predict(x_train_mapped1)).view(-1, 1)
    y_pred_train2 = torch.tensor(regressor2.predict(x_train_mapped2)).view(-1, 1)
    y_pred_train3 = torch.tensor(regressor3.predict(x_train_mapped3)).view(-1, 1)
    y_pred_train4 = torch.tensor(regressor4.predict(x_train_mapped4)).view(-1, 1)
    y_pred_train5 = torch.tensor(regressor5.predict(x_train_mapped5)).view(-1, 1)

    y_pred_valid1 = torch.tensor(regressor1.predict(x_valid_mapped1)).view(-1, 1)
    y_pred_valid2 = torch.tensor(regressor2.predict(x_valid_mapped2)).view(-1, 1)
    y_pred_valid3 = torch.tensor(regressor3.predict(x_valid_mapped3)).view(-1, 1)
    y_pred_valid4 = torch.tensor(regressor4.predict(x_valid_mapped4)).view(-1, 1)
    y_pred_valid5 = torch.tensor(regressor5.predict(x_valid_mapped5)).view(-1, 1)

    # Define Metrics to Use
    
    # MAE Loss (Mean Absolute Error)
    criterion_maeLoss   = MAELoss()

    # MSE Loss (Mean Squared Error)
    criterion_mseLoss   = MSELoss()

    # MedianAE Loss
    criterion_medaeLoss = MedianAELoss()

    # RMSE Loss
    criterion_rmseLoss  = RMSELoss()

    # R2 Score
    criterion_r2_score = R2Score()

    fn_losses_metrics = {
                        'MAE' : criterion_maeLoss,
                        'MSE' : criterion_mseLoss,
                        'MDAE': criterion_medaeLoss,
                        'RMSE': criterion_rmseLoss,
                        'R2'  : criterion_r2_score
                        }

    
    metric_losses_reg = {}
    regressors = {
                    'reg1': regressor1,
                    'reg2': regressor2,
                    'reg3': regressor3,
                    'reg4': regressor4,
                    'reg5': regressor5,
                 }
    
    # Calculate every loss/metric in fn_losses_metric for every k-th fold
    for name, loss_fn in fn_losses_metrics.items() :
        metric_losses_reg.update( { "{}_train_InnerFold1".format(name) :  loss_fn( y_pred_train1, y_train1 ) } )
        metric_losses_reg.update( { "{}_train_InnerFold2".format(name) :  loss_fn( y_pred_train2, y_train2 ) } )
        metric_losses_reg.update( { "{}_train_InnerFold3".format(name) :  loss_fn( y_pred_train3, y_train3 ) } )
        metric_losses_reg.update( { "{}_train_InnerFold4".format(name) :  loss_fn( y_pred_train4, y_train4 ) } )
        metric_losses_reg.update( { "{}_train_InnerFold5".format(name) :  loss_fn( y_pred_train5, y_train5 ) } )
        
        metric_losses_reg.update( { "{}_valid_InnerFold1".format(name) :  loss_fn( y_pred_valid1, y_valid1 ) } )
        metric_losses_reg.update( { "{}_valid_InnerFold2".format(name) :  loss_fn( y_pred_valid2, y_valid2 ) } )
        metric_losses_reg.update( { "{}_valid_InnerFold3".format(name) :  loss_fn( y_pred_valid3, y_valid3 ) } )
        metric_losses_reg.update( { "{}_valid_InnerFold4".format(name) :  loss_fn( y_pred_valid4, y_valid4 ) } )
        metric_losses_reg.update( { "{}_valid_InnerFold5".format(name) :  loss_fn( y_pred_valid5, y_valid5 ) } )

    return regressors, metric_losses_reg, training_times

def objective_rf(trial, x_train, y_train, order, incNone, model_name, initial_opt_time ):
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
        float: The optimized score based on R2 values for the training and testing sets.
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
    checkpoint = False
  
    try:
        print('Training Function')
        outputs = train_model_rf(n_estimators, max_depth, min_samples_split, min_samples_leaf,
                                 max_features, bootstrap, x_train, y_train,
                                 device, incNone, order)
        

        regressors, losses_metric_dictionary, training_times = outputs

        status_nan = []
        
        for iter_dict, ( name , loss_values ) in enumerate( losses_metric_dictionary.items() ) :
            status_nan.append( int( np.isnan( loss_values ).any() ) ) 

        
        # If is there a NAN in any of the arrays from losses_metric_dictionary: output_train <== (-0.12)
        if np.sum( status_nan ) > 0: 
            output_train = -0.12
        else:
            

            r2_T1, r2_V1 = losses_metric_dictionary['R2_train_InnerFold1'], losses_metric_dictionary['R2_valid_InnerFold1']
            r2_T2, r2_V2 = losses_metric_dictionary['R2_train_InnerFold2'], losses_metric_dictionary['R2_valid_InnerFold2']
            r2_T3, r2_V3 = losses_metric_dictionary['R2_train_InnerFold3'], losses_metric_dictionary['R2_valid_InnerFold3']
            r2_T4, r2_V4 = losses_metric_dictionary['R2_train_InnerFold4'], losses_metric_dictionary['R2_valid_InnerFold4']
            r2_T5, r2_V5 = losses_metric_dictionary['R2_train_InnerFold5'], losses_metric_dictionary['R2_valid_InnerFold5']

            output_train  = ( r2_T1 + r2_V1) - np.abs(np.abs(r2_T1) - np.abs(r2_V1))
            output_train += ( r2_T2 + r2_V2) - np.abs(np.abs(r2_T2) - np.abs(r2_V2))
            output_train += ( r2_T3 + r2_V3) - np.abs(np.abs(r2_T3) - np.abs(r2_V3))
            output_train += ( r2_T4 + r2_V4) - np.abs(np.abs(r2_T4) - np.abs(r2_V4))
            output_train += ( r2_T5 + r2_V5) - np.abs(np.abs(r2_T5) - np.abs(r2_V5))

            print('')
            for name, loss_fn in losses_metric_dictionary.items() :
                # print( f'{name}\t {loss_fn:.8f}' )
                print('{}\t {:.8f}'.format(name, loss_fn))

            
            print('')

            if output_train < 0:
                output_train = -0.001 - np.exp(output_train)
            else:
                temp_dir_paths = []
                for ipk in range(5):
                    # temp_dir_paths.append( dir_name + f'/optuna_models_{databaseName}_{model_name}_{case}_{embeddingSize}_innerFold{ipk+1}/' )
                    temp_dir_paths.append(dir_name + '/optuna_models_{}_{}_{}_{}_innerFold{}/'.format(databaseName, model_name, case, embeddingSize, ipk+1))

                for directory_pk in temp_dir_paths:
                    if not os.path.exists(directory_pk):
                        os.makedirs(directory_pk)
                
                # # for itemp,  directory_pk in enumerate(temp_dir_paths):
                # #     temp_name = f'{directory_pk}/model_{databaseName}_size_{embeddingSize:03}_trialNumber_{trial.number:03}_' + str(output_train)[2:6] 
                # #     temp_name += f'_o_{order}_{incNone}_n_estimators_{n_estimators}_max_depth_{max_depth}_min_samples_split_{min_samples_split}'
                # #     temp_name += f'_min_samples_leaf_{min_samples_leaf}_max_features_{max_features}_bootstrap_{bootstrap}_reg_{itemp+1}.sav'
                # #     pickle.dump(regressors[f'reg{ itemp + 1 }'], open(temp_name, 'wb'))
                # for itemp, directory_pk in enumerate(temp_dir_paths):
                #     temp_name = '{}model_{}_size_{:03}_trialNumber_{:03}'.format(directory_pk + '/', databaseName, embeddingSize, trial.number )
                #     temp_name += '_o_{}_{}_n_estimators_{}_max_depth_{}_min_samples_split_{}'.format(order, incNone, n_estimators, max_depth, min_samples_split )
                #     temp_name += '_min_samples_leaf_{}_max_features_{}_bootstrap_{}_reg_{}.sav'.format(min_samples_leaf, max_features, bootstrap, itemp+1) 
                #     # pickle.dump(regressors['reg{}'.format(itemp+1)], open(temp_name, 'wb'))

                for itemp, directory_pk in enumerate(temp_dir_paths):
                    try:
                        #
                        # This condition will fail in first trial because there 
                        # is not any study to compare, then checkpoint 
                        # will be true, and model will be saved
                        # In the second or larger trials, the study will be able to
                        # compare with existings trials and only will 
                        # save the models if output_train is better than
                        # previous trials
                        #
                        best_trial_info = trial.study.best_trial
                        if output_train > best_trial_info.value:
                            checkpoint == True
                    except:
                        checkpoint = True

                    if checkpoint == True:
                        
                        # Check and delete model from previos "best model" in directory_pk
                        existing_files = glob.glob(os.path.join(directory_pk, '*.sav'))
                        for file_path in existing_files:
                            os.remove(file_path)

                        # Create directory if it does not exists
                        if not os.path.exists(directory_pk):
                            os.makedirs(directory_pk)

                        # Create model file name
                        temp_name = '{}model_{}_size_{:03}_trialNumber_{:03}'.format(directory_pk + '/', databaseName, embeddingSize, trial.number )
                        temp_name += '_o_{}_{}_n_estimators_{}_max_depth_{}_min_samples_split_{}'.format(order, incNone, n_estimators, max_depth, min_samples_split )
                        temp_name += '_min_samples_leaf_{}_max_features_{}_bootstrap_{}_reg_{}.sav'.format(min_samples_leaf, max_features, bootstrap, itemp+1) 

                        # Save Model
                        pickle.dump(regressors['reg{}'.format(itemp+1)], open(temp_name, 'wb'))
                    else:
                        continue


    except Exception as e:
        # print(f"Error occurred: {e}")
        print("Error occurred: {}".format(e))

        output_train = -0.1
        losses_metric_dictionary = {}
        training_times = None


    # print(f'Trial { trial.number  }, TimeTrial  {time.time() - initial_opt_time }' )
    print('Trial {}, TimeTrial {}'.format(trial.number, time.time() - initial_opt_time))


    try:
        return output_train.item(), losses_metric_dictionary, training_times
    except:
        return output_train, losses_metric_dictionary, training_times


def objective_gbr(trial, x_train, y_train, order, incNone, model_name, initial_opt_time ):
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
        float: The optimized score based on R2 values for the training and testing sets.
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
    checkpoint =  False
  
    try:
        print('Training Function')
        outputs = train_model_gbr(n_estimators, max_depth, min_samples_split, min_samples_leaf,
                                   subsample, max_features, learning_rate, x_train,                                    
                                   y_train, device, incNone, order)

        regressors, losses_metric_dictionary, training_times = outputs

        status_nan = []
        
        for iter_dict, ( name , loss_values ) in enumerate( losses_metric_dictionary.items() ) :
            status_nan.append( int( np.isnan( loss_values ).any() ) ) 

        
        # If is there a NAN in any of the arrays from losses_metric_dictionary: output_train <== (-0.12)
        if np.sum( status_nan ) > 0: 
            output_train = -0.12
        else:
            

            r2_T1, r2_V1 = losses_metric_dictionary['R2_train_InnerFold1'], losses_metric_dictionary['R2_valid_InnerFold1']
            r2_T2, r2_V2 = losses_metric_dictionary['R2_train_InnerFold2'], losses_metric_dictionary['R2_valid_InnerFold2']
            r2_T3, r2_V3 = losses_metric_dictionary['R2_train_InnerFold3'], losses_metric_dictionary['R2_valid_InnerFold3']
            r2_T4, r2_V4 = losses_metric_dictionary['R2_train_InnerFold4'], losses_metric_dictionary['R2_valid_InnerFold4']
            r2_T5, r2_V5 = losses_metric_dictionary['R2_train_InnerFold5'], losses_metric_dictionary['R2_valid_InnerFold5']

            output_train  = ( r2_T1 + r2_V1) - np.abs(np.abs(r2_T1) - np.abs(r2_V1))
            output_train += ( r2_T2 + r2_V2) - np.abs(np.abs(r2_T2) - np.abs(r2_V2))
            output_train += ( r2_T3 + r2_V3) - np.abs(np.abs(r2_T3) - np.abs(r2_V3))
            output_train += ( r2_T4 + r2_V4) - np.abs(np.abs(r2_T4) - np.abs(r2_V4))
            output_train += ( r2_T5 + r2_V5) - np.abs(np.abs(r2_T5) - np.abs(r2_V5))

            print('')
            for name, loss_fn in losses_metric_dictionary.items() :
                # print( f'{name}\t {loss_fn:.8f}' )
                print('{}\t {:.8f}'.format(name, loss_fn))
            
            print('')

            if output_train < 0:
                output_train = -0.001 - np.exp(output_train)
            else:
                temp_dir_paths = []
                for ipk in range(5):
                    # temp_dir_paths.append( dir_name + f'/optuna_models_{databaseName}_{model_name}_{case}_{embeddingSize}_innerFold{ipk+1}/' )
                    temp_dir_paths.append(dir_name + '/optuna_models_{}_{}_{}_{}_innerFold{}/'.format(databaseName, model_name, case, embeddingSize, ipk+1))

                for directory_pk in temp_dir_paths:
                    if not os.path.exists(directory_pk):
                        os.makedirs(directory_pk)
                
                # # for itemp,  directory_pk in enumerate(temp_dir_paths):
                # #     temp_name = f'{directory_pk}/model_{databaseName}_size_{embeddingSize:03}_trialNumber_{trial.number:03}'
                # #     temp_name += f'_o_{order}_{incNone}_n_estimators_{n_estimators}_max_depth_{max_depth}_min_samples_split_{min_samples_split}'
                # #     temp_name += f'_min_samples_leaf_{min_samples_leaf}_subsample_{subsample}_max_features_{max_features}_learning_rate_{learning_rate}_reg_{itemp + 1}.sav'
                # #     pickle.dump(regressors[f'reg{ itemp + 1 }'], open(temp_name, 'wb'))
                # for itemp, directory_pk in enumerate(temp_dir_paths):
                #     temp_name = '{}/model_{}_size_{:03}_trialNumber_{:03}'.format(directory_pk, databaseName, embeddingSize, trial.number)
                #     temp_name += '_o_{}_{}_n_estimators_{}_max_depth_{}_min_samples_split_{}'.format(order, incNone, n_estimators, max_depth, min_samples_split)
                #     temp_name += '_min_samples_leaf_{}_subsample_{}_max_features_{}_learning_rate_{}_reg_{}.sav'.format(min_samples_leaf, subsample, max_features, learning_rate, itemp+1)
                #     # pickle.dump(regressors['reg{}'.format(itemp+1)], open(temp_name, 'wb'))

                for itemp, directory_pk in enumerate(temp_dir_paths):
                    try:
                        #
                        # This condition will fail in first trial because there 
                        # is not any study to compare, then checkpoint 
                        # will be true, and model will be saved
                        # In the second or larger trials, the study will be able to
                        # compare with existings trials and only will 
                        # save the models if output_train is better than
                        # previous trials
                        #
                        best_trial_info = trial.study.best_trial
                        if output_train > best_trial_info.value:
                            checkpoint == True
                    except:
                        checkpoint = True

                    if checkpoint == True:
                        
                        # Check and delete model from previos "best model" in directory_pk
                        existing_files = glob.glob(os.path.join(directory_pk, '*.sav'))
                        for file_path in existing_files:
                            os.remove(file_path)

                        # Create directory if it does not exists
                        if not os.path.exists(directory_pk):
                            os.makedirs(directory_pk)

                        # Create model file name
                        temp_name = '{}/model_{}_size_{:03}_trialNumber_{:03}'.format(directory_pk, databaseName, embeddingSize, trial.number)
                        temp_name += '_o_{}_{}_n_estimators_{}_max_depth_{}_min_samples_split_{}'.format(order, incNone, n_estimators, max_depth, min_samples_split)
                        temp_name += '_min_samples_leaf_{}_subsample_{}_max_features_{}_learning_rate_{}_reg_{}.sav'.format(min_samples_leaf, subsample, max_features, learning_rate, itemp+1)

                        # Save Model
                        pickle.dump(regressors['reg{}'.format(itemp+1)], open(temp_name, 'wb'))
                    else:
                        continue


    except Exception as e:
        # print(f"Error occurred: {e}")
        print("Error occurred: {}".format(e))
        output_train = -0.1
        losses_metric_dictionary = {}
        training_times = None

    # print(f'Trial { trial.number  }, TimeTrial  {time.time() - initial_opt_time }' )
    print('Trial {}, TimeTrial {}'.format(trial.number, time.time() - initial_opt_time))

    try:
        return output_train.item(), losses_metric_dictionary, training_times
    except:
        return output_train, losses_metric_dictionary, training_times


def objective_knr(trial, x_train, y_train, order, incNone, model_name, initial_opt_time):
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
        float: The optimized score based on R2 values for the training and testing sets.
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
    checkpoint =  False

    try:
        print('Training Function')

        # Train the model
        outputs = train_model_knr(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric, x=x_train,  y=y_train, device=device, incNone=incNone, order=order)

        regressors, losses_metric_dictionary, training_times = outputs

        status_nan = []
        
        for iter_dict, ( name , loss_values ) in enumerate( losses_metric_dictionary.items() ) :
            status_nan.append( int( np.isnan( loss_values ).any() ) ) 

        
        # If is there a NAN in any of the arrays from losses_metric_dictionary: output_train <== (-0.12)
        if np.sum( status_nan ) > 0: 
            output_train = -0.12
        else:
            

            r2_T1, r2_V1 = losses_metric_dictionary['R2_train_InnerFold1'], losses_metric_dictionary['R2_valid_InnerFold1']
            r2_T2, r2_V2 = losses_metric_dictionary['R2_train_InnerFold2'], losses_metric_dictionary['R2_valid_InnerFold2']
            r2_T3, r2_V3 = losses_metric_dictionary['R2_train_InnerFold3'], losses_metric_dictionary['R2_valid_InnerFold3']
            r2_T4, r2_V4 = losses_metric_dictionary['R2_train_InnerFold4'], losses_metric_dictionary['R2_valid_InnerFold4']
            r2_T5, r2_V5 = losses_metric_dictionary['R2_train_InnerFold5'], losses_metric_dictionary['R2_valid_InnerFold5']

            output_train  = ( r2_T1 + r2_V1) - np.abs(np.abs(r2_T1) - np.abs(r2_V1))
            output_train += ( r2_T2 + r2_V2) - np.abs(np.abs(r2_T2) - np.abs(r2_V2))
            output_train += ( r2_T3 + r2_V3) - np.abs(np.abs(r2_T3) - np.abs(r2_V3))
            output_train += ( r2_T4 + r2_V4) - np.abs(np.abs(r2_T4) - np.abs(r2_V4))
            output_train += ( r2_T5 + r2_V5) - np.abs(np.abs(r2_T5) - np.abs(r2_V5))

            print('')
            for name, loss_fn in losses_metric_dictionary.items() :
                # print( f'{name}\t {loss_fn:.8f}' )
                print('{}\t {:.8f}'.format(name, loss_fn))
            
            print('')

            if output_train < 0:
                output_train = -0.001 - np.exp(output_train)
            else:
                temp_dir_paths = []
                for ipk in range(5):
                    # temp_dir_paths.append( dir_name + f'/optuna_models_{databaseName}_{model_name}_{case}_{embeddingSize}_innerFold{ipk+1}/' )
                    temp_dir_paths.append(dir_name + '/optuna_models_{}_{}_{}_{}_innerFold{}/'.format(databaseName, model_name, case, embeddingSize, ipk+1))

                for directory_pk in temp_dir_paths:
                    if not os.path.exists(directory_pk):
                        os.makedirs(directory_pk)
                
                # # for itemp,  directory_pk in enumerate(temp_dir_paths):
                # #     temp_name = f'{directory_pk}/model_{databaseName}_size_{embeddingSize:03}_trialNumber_{trial.number:03}_' + str(output_train)[2:6] 
                # #     temp_name += f'_o_{order}_{incNone}_n_neighbors_{n_neighbors}_weights_{weights}_algorithm_{algorithm}'
                # #     temp_name += f'_leaf_size_{leaf_size}_p_{p}_metric_{metric}_reg_{itemp +1}.sav'
                # #     pickle.dump(regressors[f'reg{ itemp + 1 }'], open(temp_name, 'wb'))
                # for itemp, directory_pk in enumerate(temp_dir_paths):
                #     temp_name = '{}model_{}_size_{:03}_trialNumber_{:03}'.format(directory_pk + '/', databaseName, embeddingSize, trial.number )
                #     temp_name += '_o_{}_{}_n_neighbors_{}_weights_{}_algorithm_{}'.format(order, incNone, n_neighbors, weights, algorithm)
                #     temp_name += '_leaf_size_{}_p_{}_metric_{}_reg_{}.sav'.format(leaf_size, p, metric, itemp+1)
                #     # pickle.dump(regressors['reg{}'.format(itemp+1)], open(temp_name, 'wb'))

                for itemp, directory_pk in enumerate(temp_dir_paths):
                    try:
                        #
                        # This condition will fail in first trial because there 
                        # is not any study to compare, then checkpoint 
                        # will be true, and model will be saved
                        # In the second or larger trials, the study will be able to
                        # compare with existings trials and only will 
                        # save the models if output_train is better than
                        # previous trials
                        #
                        best_trial_info = trial.study.best_trial
                        if output_train > best_trial_info.value:
                            checkpoint == True
                    except:
                        checkpoint = True

                    if checkpoint == True:
                        
                        # Check and delete model from previos "best model" in directory_pk
                        existing_files = glob.glob(os.path.join(directory_pk, '*.sav'))
                        for file_path in existing_files:
                            os.remove(file_path)

                        # Create directory if it does not exists
                        if not os.path.exists(directory_pk):
                            os.makedirs(directory_pk)

                        # Create model file name
                        temp_name = '{}model_{}_size_{:03}_trialNumber_{:03}'.format(directory_pk + '/', databaseName, embeddingSize, trial.number )
                        temp_name += '_o_{}_{}_n_neighbors_{}_weights_{}_algorithm_{}'.format(order, incNone, n_neighbors, weights, algorithm)
                        temp_name += '_leaf_size_{}_p_{}_metric_{}_reg_{}.sav'.format(leaf_size, p, metric, itemp+1)

                        # Save Model
                        pickle.dump(regressors['reg{}'.format(itemp+1)], open(temp_name, 'wb'))
                    else:
                        continue

    except Exception as e:
        # print(f"Error occurred: {e}")
        print("Error occurred: {}".format(e))
        output_train = -0.1
        losses_metric_dictionary = {}
        training_times = None

    # print(f'Trial { trial.number  }, TimeTrial  {time.time() - initial_opt_time }' )
    print('Trial {}, TimeTrial {}'.format(trial.number, time.time() - initial_opt_time))

    try:
        return output_train.item(), losses_metric_dictionary, training_times
    except:
        return output_train, losses_metric_dictionary, training_times



def objective_mlp(trial, x_train, y_train, order, incNone, model_name, initial_opt_time ):
    """Objective function for optimizing hyperparameters of MLPRegressor using Optuna."""
    
    hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [[32], [64], [128], [256], [512], [1024], [32,128], [32,256], [32, 512], [32,1024]])
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
    solver = trial.suggest_categorical('solver', ['adam', 'lbfgs', 'sgd'])
    alpha = trial.suggest_float('alpha', 1e-6, 1e-2, log=True)
    learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-5, 1e-1, log=True)
    # max_iter = trial.suggest_int('max_iter', 100, 1000)
    
    if databaseName.lower() == 'rdb':
        max_iter = 300
    else:
        max_iter = 1500
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
    checkpoint =  False

    try:
        print('Training Function')
        outputs = train_model_mlp(hidden_layer_sizes, activation, solver, alpha, learning_rate, learning_rate_init, max_iter, momentum, nesterovs_momentum, x_train,  y_train,  device, incNone, order)

        regressors, losses_metric_dictionary, training_times = outputs

        status_nan = []
        
        for iter_dict, ( name , loss_values ) in enumerate( losses_metric_dictionary.items() ) :
            status_nan.append( int( np.isnan( loss_values ).any() ) ) 

        
        # If is there a NAN in any of the arrays from losses_metric_dictionary: output_train <== (-0.12)
        if np.sum( status_nan ) > 0: 

            output_train = -0.12

        else:

            r2_T1, r2_V1 = losses_metric_dictionary['R2_train_InnerFold1'], losses_metric_dictionary['R2_valid_InnerFold1']
            r2_T2, r2_V2 = losses_metric_dictionary['R2_train_InnerFold2'], losses_metric_dictionary['R2_valid_InnerFold2']
            r2_T3, r2_V3 = losses_metric_dictionary['R2_train_InnerFold3'], losses_metric_dictionary['R2_valid_InnerFold3']
            r2_T4, r2_V4 = losses_metric_dictionary['R2_train_InnerFold4'], losses_metric_dictionary['R2_valid_InnerFold4']
            r2_T5, r2_V5 = losses_metric_dictionary['R2_train_InnerFold5'], losses_metric_dictionary['R2_valid_InnerFold5']

            output_train  = ( r2_T1 + r2_V1) - np.abs( r2_T1 -  r2_V1 )
            output_train += ( r2_T2 + r2_V2) - np.abs( r2_T2 -  r2_V2 )
            output_train += ( r2_T3 + r2_V3) - np.abs( r2_T3 -  r2_V3 )
            output_train += ( r2_T4 + r2_V4) - np.abs( r2_T4 -  r2_V4 )
            output_train += ( r2_T5 + r2_V5) - np.abs( r2_T5 -  r2_V5 )

            print('')
            for name, loss_fn in losses_metric_dictionary.items() :
                # print( f'{name}\t {loss_fn:.8f}' )
                print('{}\t {:.8f}'.format(name, loss_fn))

            print('')

            if output_train < 0:
                output_train = -0.001 - np.exp(output_train)
            else:
                temp_dir_paths = []
                for ipk in range(5):
                    # temp_dir_paths.append( dir_name + f'/optuna_models_{databaseName}_{model_name}_{case}_{embeddingSize}_innerFold{ipk+1}/' )
                    temp_dir_paths.append(dir_name + '/optuna_models_{}_{}_{}_{}_innerFold{}/'.format(databaseName, model_name, case, embeddingSize, ipk+1))

                for directory_pk in temp_dir_paths:
                    if not os.path.exists(directory_pk):
                        os.makedirs(directory_pk)
                
                for itemp, directory_pk in enumerate(temp_dir_paths):
                    try:
                        #
                        # This condition will fail in first trial because there 
                        # is not any study to compare, then checkpoint 
                        # will be true, and model will be saved
                        # In the second or larger trials, the study will be able to
                        # compare with existings trials and only will 
                        # save the models if output_train is better than
                        # previous trials
                        #
                        best_trial_info = trial.study.best_trial
                        if output_train > best_trial_info.value:
                            checkpoint == True
                    except:
                        checkpoint = True

                    if checkpoint == True:
                        
                        # Check and delete model from previos "best model" in directory_pk
                        existing_files = glob.glob(os.path.join(directory_pk, '*.sav'))
                        for file_path in existing_files:
                            os.remove(file_path)

                        # Create directory if it does not exists
                        if not os.path.exists(directory_pk):
                            os.makedirs(directory_pk)

                        # Create model file name
                        temp_name = '{}model_{}_size_{:03}_trialNumber_{:03}'.format(directory_pk + '/', databaseName, embeddingSize, trial.number)
                        temp_name += '_o_{}_{}_hls_{}_act_{}_solv_{}'.format(order, incNone, hidden_layer_sizes, activation, solver)
                        temp_name += '_lr_{}_lrinit_{}_maxI_{}'.format(learning_rate, learning_rate_init, max_iter)
                        temp_name += 'mome_{}_nesteM{}_reg_{}.sav'.format(momentum, nesterovs_momentum, itemp+1)

                        # Save Model
                        pickle.dump(regressors['reg{}'.format(itemp+1)], open(temp_name, 'wb'))
                    else:
                        continue

    except Exception as e:
        # print(f"Error occurred: {e}")
        print("Error occurred: {}".format(e))
        output_train = -0.1
        losses_metric_dictionary = {}
        training_times = None

    # print(f'Trial { trial.number  }, TimeTrial  {time.time() - initial_opt_time }' )
    print('Trial {}, TimeTrial {}'.format(trial.number, time.time() - initial_opt_time))

    try:
        return output_train.item(), losses_metric_dictionary, training_times
    except:
        return output_train, losses_metric_dictionary, training_times

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

    trainTime_InnerFold1  = trial.user_attrs.get('training_InnerFold1', None )
    trainTime_InnerFold2  = trial.user_attrs.get('training_InnerFold2', None )
    trainTime_InnerFold3  = trial.user_attrs.get('training_InnerFold3', None )
    trainTime_InnerFold4  = trial.user_attrs.get('training_InnerFold4', None )
    trainTime_InnerFold5  = trial.user_attrs.get('training_InnerFold5', None )    

    MAE_train_InnerFold1  = trial.user_attrs.get('MAE_train_InnerFold1', None )
    MAE_train_InnerFold2  = trial.user_attrs.get('MAE_train_InnerFold2', None )
    MAE_train_InnerFold3  = trial.user_attrs.get('MAE_train_InnerFold3', None )
    MAE_train_InnerFold4  = trial.user_attrs.get('MAE_train_InnerFold4', None )
    MAE_train_InnerFold5  = trial.user_attrs.get('MAE_train_InnerFold5', None )
    MAE_valid_InnerFold1  = trial.user_attrs.get('MAE_valid_InnerFold1', None )
    MAE_valid_InnerFold2  = trial.user_attrs.get('MAE_valid_InnerFold2', None )
    MAE_valid_InnerFold3  = trial.user_attrs.get('MAE_valid_InnerFold3', None )
    MAE_valid_InnerFold4  = trial.user_attrs.get('MAE_valid_InnerFold4', None )
    MAE_valid_InnerFold5  = trial.user_attrs.get('MAE_valid_InnerFold5', None )
    MSE_train_InnerFold1  = trial.user_attrs.get('MSE_train_InnerFold1', None )
    MSE_train_InnerFold2  = trial.user_attrs.get('MSE_train_InnerFold2', None )
    MSE_train_InnerFold3  = trial.user_attrs.get('MSE_train_InnerFold3', None )
    MSE_train_InnerFold4  = trial.user_attrs.get('MSE_train_InnerFold4', None )
    MSE_train_InnerFold5  = trial.user_attrs.get('MSE_train_InnerFold5', None )
    MSE_valid_InnerFold1  = trial.user_attrs.get('MSE_valid_InnerFold1', None )
    MSE_valid_InnerFold2  = trial.user_attrs.get('MSE_valid_InnerFold2', None )
    MSE_valid_InnerFold3  = trial.user_attrs.get('MSE_valid_InnerFold3', None )
    MSE_valid_InnerFold4  = trial.user_attrs.get('MSE_valid_InnerFold4', None )
    MSE_valid_InnerFold5  = trial.user_attrs.get('MSE_valid_InnerFold5', None )
    MDAE_train_InnerFold1 = trial.user_attrs.get('MDAE_train_InnerFold1', None )
    MDAE_train_InnerFold2 = trial.user_attrs.get('MDAE_train_InnerFold2', None )
    MDAE_train_InnerFold3 = trial.user_attrs.get('MDAE_train_InnerFold3', None )
    MDAE_train_InnerFold4 = trial.user_attrs.get('MDAE_train_InnerFold4', None )
    MDAE_train_InnerFold5 = trial.user_attrs.get('MDAE_train_InnerFold5', None )
    MDAE_valid_InnerFold1 = trial.user_attrs.get('MDAE_valid_InnerFold1', None )
    MDAE_valid_InnerFold2 = trial.user_attrs.get('MDAE_valid_InnerFold2', None )
    MDAE_valid_InnerFold3 = trial.user_attrs.get('MDAE_valid_InnerFold3', None )
    MDAE_valid_InnerFold4 = trial.user_attrs.get('MDAE_valid_InnerFold4', None )
    MDAE_valid_InnerFold5 = trial.user_attrs.get('MDAE_valid_InnerFold5', None )
    RMSE_train_InnerFold1 = trial.user_attrs.get('RMSE_train_InnerFold1', None )
    RMSE_train_InnerFold2 = trial.user_attrs.get('RMSE_train_InnerFold2', None )
    RMSE_train_InnerFold3 = trial.user_attrs.get('RMSE_train_InnerFold3', None )
    RMSE_train_InnerFold4 = trial.user_attrs.get('RMSE_train_InnerFold4', None )
    RMSE_train_InnerFold5 = trial.user_attrs.get('RMSE_train_InnerFold5', None )
    RMSE_valid_InnerFold1 = trial.user_attrs.get('RMSE_valid_InnerFold1', None )
    RMSE_valid_InnerFold2 = trial.user_attrs.get('RMSE_valid_InnerFold2', None )
    RMSE_valid_InnerFold3 = trial.user_attrs.get('RMSE_valid_InnerFold3', None )
    RMSE_valid_InnerFold4 = trial.user_attrs.get('RMSE_valid_InnerFold4', None )
    RMSE_valid_InnerFold5 = trial.user_attrs.get('RMSE_valid_InnerFold5', None )
    R2_train_InnerFold1   = trial.user_attrs.get('R2_train_InnerFold1', None )
    R2_train_InnerFold2   = trial.user_attrs.get('R2_train_InnerFold2', None )
    R2_train_InnerFold3   = trial.user_attrs.get('R2_train_InnerFold3', None )
    R2_train_InnerFold4   = trial.user_attrs.get('R2_train_InnerFold4', None )
    R2_train_InnerFold5   = trial.user_attrs.get('R2_train_InnerFold5', None )
    R2_valid_InnerFold1   = trial.user_attrs.get('R2_valid_InnerFold1', None )
    R2_valid_InnerFold2   = trial.user_attrs.get('R2_valid_InnerFold2', None )
    R2_valid_InnerFold3   = trial.user_attrs.get('R2_valid_InnerFold3', None )
    R2_valid_InnerFold4   = trial.user_attrs.get('R2_valid_InnerFold4', None )
    R2_valid_InnerFold5   = trial.user_attrs.get('R2_valid_InnerFold5', None )    

    
    # Convert the params dictionary to a DataFrame where each parameter becomes a separate column
    params_df = pd.DataFrame([params])

    # Add the trial number and value as columns to the params DataFrame
    params_df['trial_number'] = int( trial_number ) + 1
    params_df['value'] = value
    params_df['train_time'] = train_time

    params_df['train_time_InnerFold1'] = trainTime_InnerFold1
    params_df['train_time_InnerFold2'] = trainTime_InnerFold2
    params_df['train_time_InnerFold3'] = trainTime_InnerFold3
    params_df['train_time_InnerFold4'] = trainTime_InnerFold4
    params_df['train_time_InnerFold5'] = trainTime_InnerFold5

    params_df['MAE_train_InnerFold1']  =  MAE_train_InnerFold1  
    params_df['MAE_train_InnerFold2']  =  MAE_train_InnerFold2  
    params_df['MAE_train_InnerFold3']  =  MAE_train_InnerFold3  
    params_df['MAE_train_InnerFold4']  =  MAE_train_InnerFold4  
    params_df['MAE_train_InnerFold5']  =  MAE_train_InnerFold5  
    params_df['MAE_valid_InnerFold1']  =  MAE_valid_InnerFold1  
    params_df['MAE_valid_InnerFold2']  =  MAE_valid_InnerFold2  
    params_df['MAE_valid_InnerFold3']  =  MAE_valid_InnerFold3  
    params_df['MAE_valid_InnerFold4']  =  MAE_valid_InnerFold4  
    params_df['MAE_valid_InnerFold5']  =  MAE_valid_InnerFold5  
    params_df['MSE_train_InnerFold1']  =  MSE_train_InnerFold1  
    params_df['MSE_train_InnerFold2']  =  MSE_train_InnerFold2  
    params_df['MSE_train_InnerFold3']  =  MSE_train_InnerFold3  
    params_df['MSE_train_InnerFold4']  =  MSE_train_InnerFold4  
    params_df['MSE_train_InnerFold5']  =  MSE_train_InnerFold5  
    params_df['MSE_valid_InnerFold1']  =  MSE_valid_InnerFold1  
    params_df['MSE_valid_InnerFold2']  =  MSE_valid_InnerFold2  
    params_df['MSE_valid_InnerFold3']  =  MSE_valid_InnerFold3  
    params_df['MSE_valid_InnerFold4']  =  MSE_valid_InnerFold4  
    params_df['MSE_valid_InnerFold5']  =  MSE_valid_InnerFold5  
    params_df['MDAE_train_InnerFold1'] =  MDAE_train_InnerFold1 
    params_df['MDAE_train_InnerFold2'] =  MDAE_train_InnerFold2 
    params_df['MDAE_train_InnerFold3'] =  MDAE_train_InnerFold3 
    params_df['MDAE_train_InnerFold4'] =  MDAE_train_InnerFold4 
    params_df['MDAE_train_InnerFold5'] =  MDAE_train_InnerFold5 
    params_df['MDAE_valid_InnerFold1'] =  MDAE_valid_InnerFold1 
    params_df['MDAE_valid_InnerFold2'] =  MDAE_valid_InnerFold2 
    params_df['MDAE_valid_InnerFold3'] =  MDAE_valid_InnerFold3 
    params_df['MDAE_valid_InnerFold4'] =  MDAE_valid_InnerFold4 
    params_df['MDAE_valid_InnerFold5'] =  MDAE_valid_InnerFold5 
    params_df['RMSE_train_InnerFold1'] =  RMSE_train_InnerFold1 
    params_df['RMSE_train_InnerFold2'] =  RMSE_train_InnerFold2 
    params_df['RMSE_train_InnerFold3'] =  RMSE_train_InnerFold3 
    params_df['RMSE_train_InnerFold4'] =  RMSE_train_InnerFold4 
    params_df['RMSE_train_InnerFold5'] =  RMSE_train_InnerFold5 
    params_df['RMSE_valid_InnerFold1'] =  RMSE_valid_InnerFold1 
    params_df['RMSE_valid_InnerFold2'] =  RMSE_valid_InnerFold2 
    params_df['RMSE_valid_InnerFold3'] =  RMSE_valid_InnerFold3 
    params_df['RMSE_valid_InnerFold4'] =  RMSE_valid_InnerFold4 
    params_df['RMSE_valid_InnerFold5'] =  RMSE_valid_InnerFold5 
    params_df['R2_train_InnerFold1']   =  R2_train_InnerFold1   
    params_df['R2_train_InnerFold2']   =  R2_train_InnerFold2   
    params_df['R2_train_InnerFold3']   =  R2_train_InnerFold3   
    params_df['R2_train_InnerFold4']   =  R2_train_InnerFold4   
    params_df['R2_train_InnerFold5']   =  R2_train_InnerFold5   
    params_df['R2_valid_InnerFold1']   =  R2_valid_InnerFold1   
    params_df['R2_valid_InnerFold2']   =  R2_valid_InnerFold2   
    params_df['R2_valid_InnerFold3']   =  R2_valid_InnerFold3   
    params_df['R2_valid_InnerFold4']   =  R2_valid_InnerFold4   
    params_df['R2_valid_InnerFold5']   =  R2_valid_InnerFold5   

    # Append the new row to the global DataFrame `results_df`
    global results_df
    results_df = pd.concat([results_df, params_df], ignore_index=True)
    
    # Save the updated DataFrame with a customized name
    # results_df.to_csv(f"{dir_name}/optuna_results_{databaseName}_{modelName}_{embeddingSize}_{incNone}_outterFold_{outterKFold}.csv.gz", index=False, compression = 'gzip')
    results_df.to_csv('{}/optuna_results_{}_{}_{}_{}_outterFold_{}.csv.gz'.format(dir_name, databaseName, modelName, embeddingSize, incNone, outterKFold), index=False, compression='gzip')


    # # Directories for saving plots (ensure they are defined)
    # # directory_path_png = f"{dir_name}/optuna_visualization_{databaseName}_{modelName}_{embeddingSize:03}_png/"
    # # directory_path_svg = f"{dir_name}/optuna_visualization_{databaseName}_{modelName}_{embeddingSize:03}_svg/"
    # directory_path_png = '{}/optuna_visualization_{}_{}_ {:03}_png/'.format(dir_name, databaseName, modelName, embeddingSize)
    # directory_path_svg = '{}/optuna_visualization_{}_{}_ {:03}_svg/'.format(dir_name, databaseName, modelName, embeddingSize)


    # if not os.path.exists(directory_path_png):
    #     os.makedirs(directory_path_png)
        
    # if not os.path.exists(directory_path_svg):
    #     os.makedirs(directory_path_svg)

    # # Attempt to generate and save the plots after each trial
    # try:
    #     # print(f"Attempting to save Optuna plot for trial {trial_number}...")
    #     print("Attempting to save Optuna plot for trial {}...",format(trial_number))

    #     # Ensure the plot is only created if there are enough trials
    #     if len(study.trials) > 1:
    #         generate_plots(study, trial_number, directory_path_png, directory_path_svg)
    #     else:
    #         # print(f"Skipping plots for trial {trial_number}, only 1 trial completed.")
    #         print("Skipping plots for trial {}, only 1 trial completed.".format(trial_number))
        
    # except Exception as e:
    #     # If an error occurs while saving the plot, print the error message
    #     # print(f"Failed to save Optuna plot for trial {trial_number}. Error: {e}")
    #     print("Failed to save Optuna plot for trial {}. Error: {}".format(trial_number, e))



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
        custom_name += str(item) + '_' + str(best_params_trial[item]) + '_'

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
            # print(f"Attempting to save contour plot for parameters: {p[0]} and {p[1]} (Trial {trial_number})...")
            print("Attempting to save contour plot for parameters: {} and {} (Trial {})...".format(p[0], p[1], trial_number))

            fig = optuna.visualization.plot_contour(study, params=[p[0], p[1]])

            # Customizations for contour plot
            fig.update_traces( colorscale = 'Blackbody', selector = dict( type = 'contour' ) )  # Color palette
            fig.update_traces( line_smoothing = 1.15, selector = dict( type = 'contour' ) )  # Smooth lines
            fig.update_traces( line_width = 0, selector = dict( type = 'contour' ) )  # Remove contour lines
            fig.update_traces( marker = dict( size = 0.25, color = "RoyalBlue" ), selector=dict( mode = 'markers' ))  # Marker styling

            # Save the contour plot
            # fig.write_image(f"{directory_path_png}{custom_name}contour_{p[0]}_{p[1]}.png")
            # fig.write_image(f"{directory_path_svg}{custom_name}contour_{p[0]}_{p[1]}.svg")
            # print(f"Successfully saved contour plot for parameters {p[0]} and {p[1]} (Trial {trial_number}).")
            fig.write_image('{}{}contour_{}_{}.png'.format(directory_path_png, custom_name, p[0], p[1]))
            fig.write_image('{}{}contour_{}_{}.svg'.format(directory_path_svg, custom_name, p[0], p[1]))
            print('Successfully saved contour plot for parameters {} and {} (Trial {}).'.format(p[0], p[1], trial_number))

        except Exception as e:
            # print(f"Failed to save contour plot for parameters {p[0]} and {p[1]} (Trial {trial_number}). Error: {e}")
            print('Failed to save contour plot for parameters {} and {} (Trial {}). Error: {}'.format(p[0], p[1], trial_number, e))

    # Generate parameter importance plot
    # try:
    #     # print(f"Attempting to save parameter importances plot (Trial {trial_number})...")
    #     fig = optuna.visualization.plot_param_importances(study)
    #     fig.update_layout(template='simple_white')
    #     fig.write_image(f"{directory_path_png}_{custom_name}param_importances.png")
    #     fig.write_image(f"{directory_path_svg}_{custom_name}param_importances.svg")
    #     print(f"Successfully saved parameter importances plot (Trial {trial_number}).")
        
    # except Exception as e:
    #     print(f"Failed to save parameter importances plot for trial {trial_number}. Error: {e}")

    # # Generate slice plot for parameters
    # try:
    #     print(f"Attempting to save slice plot (Trial {trial_number})...")
    #     fig = optuna.visualization.plot_slice(study, params=parameters_optuna)
    #     fig.update_layout(template='simple_white')
    #     fig.write_image(f"{directory_path_png}_{custom_name}slice_plot.png")
    #     fig.write_image(f"{directory_path_svg}_{custom_name}slice_plot.svg")
    #     print(f"Successfully saved slice plot (Trial {trial_number}).")
        
    # except Exception as e:
    #     print(f"Failed to save slice plot for trial {trial_number}. Error: {e}")
    try:
        print('Attempting to save parameter importances plot (Trial {})...'.format(trial_number))
        fig = optuna.visualization.plot_param_importances(study)
        fig.update_layout(template='simple_white')
        fig.write_image('{}{}_param_importances.png'.format(directory_path_png, custom_name))
        fig.write_image('{}{}_param_importances.svg'.format(directory_path_svg, custom_name))
        print('Successfully saved parameter importances plot (Trial {}).'.format(trial_number))
    except Exception as e:
        print('Failed to save parameter importances plot for trial {}. Error: {}'.format(trial_number, e))
    
    # Generate slice plot for parameters
    try:
        print('Attempting to save slice plot (Trial {})...'.format(trial_number))
        fig = optuna.visualization.plot_slice(study, params=parameters_optuna)
        fig.update_layout(template='simple_white')
        fig.write_image('{}{}_slice_plot.png'.format(directory_path_png, custom_name))
        fig.write_image('{}{}_slice_plot.svg'.format(directory_path_svg, custom_name))
        print('Successfully saved slice plot (Trial {}).'.format(trial_number))
    except Exception as e:
        print('Failed to save slice plot for trial {}. Error: {}'.format(trial_number, e))



def objective_with_time(trial, x_train, y_train,  order, incNone, model_name):
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

    if model_name == 'RF' or model_name == 'rf':
        result, dictionary_loss_r2, training_times =  objective_rf(trial, x_train, y_train,  order, incNone, model_name, start_time)
    elif model_name == 'GBR' or model_name == 'gbr':
        result, dictionary_loss_r2, training_times = objective_gbr(trial, x_train, y_train,  order, incNone, model_name, start_time)
    elif model_name == 'KNR' or model_name == 'knr':
        result, dictionary_loss_r2, training_times = objective_knr(trial, x_train, y_train,  order, incNone, model_name, start_time)
    elif model_name == 'MLP' or model_name == 'mlp':
        result, dictionary_loss_r2, training_times = objective_mlp(trial, x_train, y_train,  order, incNone, model_name, start_time)
    

    train_time = time.time() - start_time
    trial.set_user_attr('train_time', train_time)

    # MAE
    trial.set_user_attr('MAE_train_InnerFold1', dictionary_loss_r2[ 'MAE_train_InnerFold1' ].item() )
    trial.set_user_attr('MAE_train_InnerFold2', dictionary_loss_r2[ 'MAE_train_InnerFold2' ].item() )
    trial.set_user_attr('MAE_train_InnerFold3', dictionary_loss_r2[ 'MAE_train_InnerFold3' ].item() )
    trial.set_user_attr('MAE_train_InnerFold4', dictionary_loss_r2[ 'MAE_train_InnerFold4' ].item() )
    trial.set_user_attr('MAE_train_InnerFold5', dictionary_loss_r2[ 'MAE_train_InnerFold5' ].item() )
    trial.set_user_attr('MAE_valid_InnerFold1', dictionary_loss_r2[ 'MAE_valid_InnerFold1' ].item() )
    trial.set_user_attr('MAE_valid_InnerFold2', dictionary_loss_r2[ 'MAE_valid_InnerFold2' ].item() )
    trial.set_user_attr('MAE_valid_InnerFold3', dictionary_loss_r2[ 'MAE_valid_InnerFold3' ].item() )
    trial.set_user_attr('MAE_valid_InnerFold4', dictionary_loss_r2[ 'MAE_valid_InnerFold4' ].item() )
    trial.set_user_attr('MAE_valid_InnerFold5', dictionary_loss_r2[ 'MAE_valid_InnerFold5' ].item() )

    # MSE
    trial.set_user_attr('MSE_train_InnerFold1', dictionary_loss_r2[ 'MSE_train_InnerFold1' ].item() )
    trial.set_user_attr('MSE_train_InnerFold2', dictionary_loss_r2[ 'MSE_train_InnerFold2' ].item() )
    trial.set_user_attr('MSE_train_InnerFold3', dictionary_loss_r2[ 'MSE_train_InnerFold3' ].item() )
    trial.set_user_attr('MSE_train_InnerFold4', dictionary_loss_r2[ 'MSE_train_InnerFold4' ].item() )
    trial.set_user_attr('MSE_train_InnerFold5', dictionary_loss_r2[ 'MSE_train_InnerFold5' ].item() )
    trial.set_user_attr('MSE_valid_InnerFold1', dictionary_loss_r2[ 'MSE_valid_InnerFold1' ].item() )
    trial.set_user_attr('MSE_valid_InnerFold2', dictionary_loss_r2[ 'MSE_valid_InnerFold2' ].item() )
    trial.set_user_attr('MSE_valid_InnerFold3', dictionary_loss_r2[ 'MSE_valid_InnerFold3' ].item() )
    trial.set_user_attr('MSE_valid_InnerFold4', dictionary_loss_r2[ 'MSE_valid_InnerFold4' ].item() )
    trial.set_user_attr('MSE_valid_InnerFold5', dictionary_loss_r2[ 'MSE_valid_InnerFold5' ].item() )

    # Median AE
    trial.set_user_attr('MDAE_train_InnerFold1', dictionary_loss_r2[ 'MDAE_train_InnerFold1' ].item() )
    trial.set_user_attr('MDAE_train_InnerFold2', dictionary_loss_r2[ 'MDAE_train_InnerFold2' ].item() )
    trial.set_user_attr('MDAE_train_InnerFold3', dictionary_loss_r2[ 'MDAE_train_InnerFold3' ].item() )
    trial.set_user_attr('MDAE_train_InnerFold4', dictionary_loss_r2[ 'MDAE_train_InnerFold4' ].item() )
    trial.set_user_attr('MDAE_train_InnerFold5', dictionary_loss_r2[ 'MDAE_train_InnerFold5' ].item() )
    trial.set_user_attr('MDAE_valid_InnerFold1', dictionary_loss_r2[ 'MDAE_valid_InnerFold1' ].item() )
    trial.set_user_attr('MDAE_valid_InnerFold2', dictionary_loss_r2[ 'MDAE_valid_InnerFold2' ].item() )
    trial.set_user_attr('MDAE_valid_InnerFold3', dictionary_loss_r2[ 'MDAE_valid_InnerFold3' ].item() )
    trial.set_user_attr('MDAE_valid_InnerFold4', dictionary_loss_r2[ 'MDAE_valid_InnerFold4' ].item() )
    trial.set_user_attr('MDAE_valid_InnerFold5', dictionary_loss_r2[ 'MDAE_valid_InnerFold5' ].item() )

    # RMSE
    trial.set_user_attr('RMSE_train_InnerFold1', dictionary_loss_r2[ 'RMSE_train_InnerFold1' ].item() )
    trial.set_user_attr('RMSE_train_InnerFold2', dictionary_loss_r2[ 'RMSE_train_InnerFold2' ].item() )
    trial.set_user_attr('RMSE_train_InnerFold3', dictionary_loss_r2[ 'RMSE_train_InnerFold3' ].item() )
    trial.set_user_attr('RMSE_train_InnerFold4', dictionary_loss_r2[ 'RMSE_train_InnerFold4' ].item() )
    trial.set_user_attr('RMSE_train_InnerFold5', dictionary_loss_r2[ 'RMSE_train_InnerFold5' ].item() )
    trial.set_user_attr('RMSE_valid_InnerFold1', dictionary_loss_r2[ 'RMSE_valid_InnerFold1' ].item() )
    trial.set_user_attr('RMSE_valid_InnerFold2', dictionary_loss_r2[ 'RMSE_valid_InnerFold2' ].item() )
    trial.set_user_attr('RMSE_valid_InnerFold3', dictionary_loss_r2[ 'RMSE_valid_InnerFold3' ].item() )
    trial.set_user_attr('RMSE_valid_InnerFold4', dictionary_loss_r2[ 'RMSE_valid_InnerFold4' ].item() )
    trial.set_user_attr('RMSE_valid_InnerFold5', dictionary_loss_r2[ 'RMSE_valid_InnerFold5' ].item() )

    # R2
    trial.set_user_attr('R2_train_InnerFold1', dictionary_loss_r2[ 'R2_train_InnerFold1' ].item() )
    trial.set_user_attr('R2_train_InnerFold2', dictionary_loss_r2[ 'R2_train_InnerFold2' ].item() )
    trial.set_user_attr('R2_train_InnerFold3', dictionary_loss_r2[ 'R2_train_InnerFold3' ].item() )
    trial.set_user_attr('R2_train_InnerFold4', dictionary_loss_r2[ 'R2_train_InnerFold4' ].item() )
    trial.set_user_attr('R2_train_InnerFold5', dictionary_loss_r2[ 'R2_train_InnerFold5' ].item() )
    trial.set_user_attr('R2_valid_InnerFold1', dictionary_loss_r2[ 'R2_valid_InnerFold1' ].item() )
    trial.set_user_attr('R2_valid_InnerFold2', dictionary_loss_r2[ 'R2_valid_InnerFold2' ].item() )
    trial.set_user_attr('R2_valid_InnerFold3', dictionary_loss_r2[ 'R2_valid_InnerFold3' ].item() )
    trial.set_user_attr('R2_valid_InnerFold4', dictionary_loss_r2[ 'R2_valid_InnerFold4' ].item() )
    trial.set_user_attr('R2_valid_InnerFold5', dictionary_loss_r2[ 'R2_valid_InnerFold5' ].item() )

    trial.set_user_attr('training_InnerFold1', training_times[ 'training_InnerFold1' ] )
    trial.set_user_attr('training_InnerFold2', training_times[ 'training_InnerFold2' ] )
    trial.set_user_attr('training_InnerFold3', training_times[ 'training_InnerFold3' ] )
    trial.set_user_attr('training_InnerFold4', training_times[ 'training_InnerFold4' ] )
    trial.set_user_attr('training_InnerFold5', training_times[ 'training_InnerFold5' ] )

    return result


if __name__ == "__main__":

    # Main program execution
    print("Program started.")
    results_df = pd.DataFrame(columns=['trial_number', 'value', 'train_time'])

    device = torch.device('cpu')

    start_time = time.time()

    # ========================================================================================================
    # Parse Arguments
    # ========================================================================================================
    databaseName    =  str(sys.argv[1])   # str:  rdb, nfa, qm9;                    Default: rdb
    encodingMethod  =  str(sys.argv[2])   # str:  mfp, emfp;                        Default: mfp
    embeddingSize   =  int(sys.argv[3])   # int:  mfp:1, emfp:8,16,32,64,128,256;   Default: 1
    withDescriptors =  str(sys.argv[4])   # bool: True/False;                       Default: False
    ffnnCase        =  str(sys.argv[5])   # str:  none, linear, gauss               Default: none
    ffnnOrder       =  int(sys.argv[6])   # int:  1,2,3,4,...;                      Default: 1
    nBitsMFP        =  int(sys.argv[7])   # int:  1024,2048,4096, ..., 16384, ...;  Default: 16384
    radiusMFP       =  int(sys.argv[8])   # int:  0,1,2,3,4,5,...;                  Default: 2
    modelName       =  str(sys.argv[9])   # str:  RF, GBR, KNR, MLP                 Default: RF
    outterKFold     =  int(sys.argv[10])  # int:  1,2,3,4,5;                        Default: 1
    int_ext_case    =  str(sys.argv[11])  # str:  internal or external;             Default: internal

    # ========================================================================================================
    # File Paths
    # ========================================================================================================
    file                       = 'scaffold_splitting/{}_train.csv.gz'.format(databaseName)
    descNormal_file            = 'scaffold_splitting/desc_{}_train.csv.gz'.format(databaseName)
    file_validation            = 'scaffold_splitting/{}_tests.csv.gz'.format(databaseName)
    descNormal_file_validation = 'scaffold_splitting/desc_{}_tests.csv.gz'.format(databaseName)

    # ========================================================================================================
    # Encoding / FFNN config
    # ========================================================================================================
    if encodingMethod == 'mfp' or encodingMethod == 'MFP':
        embeddingSize = 1
    else:
        try:
            assert int(embeddingSize) in [2**i for i in range(3, 9)]
        except:
            raise Exception("\n\tTry any of the next values for embeddingSize: 8, 16, 32, 64, 128, 256 ..., 2^N, N=[3,8]")

    if ffnnCase == 'none' or ffnnCase == 'None':
        incNone = 'without_FFNN'
    elif ffnnCase == 'linear' or ffnnCase == 'Linear':
        incNone = 'with_FFNN'
    elif ffnnCase in ['gauss', 'Gauss', 'gaussian', 'Gaussian']:
        incNone = 'with_FFNN_Gaussian'

    if withDescriptors == "True":
        incNone += '_with_Descriptors'

    if ffnnOrder > 1:
        ffnnOrder = 1

    print('Database', databaseName)

    # ========================================================================================================
    # Directory and study names
    # ========================================================================================================
    dir_name = 'Models/{database}/{model}_{inc}_{emb}/OutterFold{fold}'.format(
        database=databaseName.upper(), model=modelName, inc=incNone, emb=embeddingSize, fold=outterKFold)

    set_seed(42)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    study_name_custom = "regression_{}_{}_{}_{}_{}".format(
        databaseName, modelName, embeddingSize, incNone, outterKFold)
    storage_url = "sqlite:///{}/{}.db".format(dir_name, study_name_custom)

    # ========================================================================================================
    # Helper: compute Tanimoto distance from each test molecule to its nearest/farthest neighbor in train
    # (copied verbatim from reference_DNN.py)
    # ========================================================================================================
    def compute_tanimoto_distances(smiles_train_list, smiles_test_list, radiusMFP, nBitsMFP):
        """
        For each molecule in smiles_test_list, computes against all molecules in
        smiles_train_list using Tanimoto similarity on Morgan fingerprints.
        Returns two numpy arrays (one per test molecule):
          - min_distances : 1 - max_similarity  (distance to nearest neighbor in train)
          - max_distances : 1 - min_similarity  (distance to farthest neighbor in train)
        Invalid SMILES get np.nan in both arrays.
        """
        from rdkit import DataStructs
        from rdkit.Chem import AllChem, MolFromSmiles as RDKitMolFromSmiles

        print("Computing Tanimoto distances (test vs train)...")

        # Build train fingerprints
        fps_train = []
        for smi in smiles_train_list:
            mol = RDKitMolFromSmiles(smi)
            if mol is not None:
                fps_train.append(AllChem.GetMorganFingerprintAsBitVect(mol, radiusMFP, nBits=nBitsMFP))

        min_distances = []
        max_distances = []
        for smi in smiles_test_list:
            mol = RDKitMolFromSmiles(smi)
            if mol is None or len(fps_train) == 0:
                min_distances.append(np.nan)
                max_distances.append(np.nan)
                continue
            fp_q = AllChem.GetMorganFingerprintAsBitVect(mol, radiusMFP, nBits=nBitsMFP)
            sims = np.array(DataStructs.BulkTanimotoSimilarity(fp_q, fps_train))
            min_distances.append(1.0 - float(np.max(sims)))   # nearest
            max_distances.append(1.0 - float(np.min(sims)))   # farthest

        return np.array(min_distances), np.array(max_distances)

    # ========================================================================================================
    # Helper: save per-molecule predictions CSV
    # (copied verbatim from reference_DNN.py)
    # ========================================================================================================
    def save_per_molecule_csv(smiles_test_list, y_real, y_pred, tanimoto_min, tanimoto_max,
                              dir_name, filename='predictions_per_molecule.csv.gz'):
        """
        Saves a CSV with one row per test molecule containing:
          smiles, y_real, y_pred, abs_error, abs_percentage_error,
          tanimoto_distance_to_train (nearest), tanimoto_max_distance_to_train (farthest)
        """
        abs_err = np.abs(y_real - y_pred)
        # Avoid division by zero for percentage error
        with np.errstate(divide='ignore', invalid='ignore'):
            abs_pct_err = np.where(
                y_real != 0,
                100.0 * abs_err / np.abs(y_real),
                np.nan)

        df_mol = pd.DataFrame({
            'smiles'                        : smiles_test_list,
            'y_real'                        : y_real,
            'y_pred'                        : y_pred,
            'abs_error'                     : abs_err,
            'abs_percentage_error'          : abs_pct_err,
            'tanimoto_distance_to_train'    : tanimoto_min,
            'tanimoto_max_distance_to_train': tanimoto_max,
        })
        out_path = "{}/{}".format(dir_name, filename)
        df_mol.to_csv(out_path, index=False, compression='gzip')
        print("Per-molecule predictions saved to:", out_path)
        return df_mol

    # ========================================================================================================
    # Helper: build a single summary row dict
    # (copied verbatim from reference_DNN.py, column names adapted to Train/Validation/Test scheme)
    # ========================================================================================================
    def build_summary_row(train_val_metrics, n_molecules, pct, test_mae, test_mse, test_mdae, test_r2):
        """
        Builds a single summary row dict with a fixed, explicit column order:
          Train_mae | Train_mse | Train_mdae | Train_r2_all |
          Validation_mae | Validation_mse | Validation_mdae | Validation_r2_all |
          n_molecules | pct |
          Test_mae | Test_mse | Test_mdae | Test_r2_all
        """
        row = {}
        for col, val in train_val_metrics.items():
            row[col] = val
        row['n_molecules'] = n_molecules
        row['pct']         = pct
        row['Test_mae']    = test_mae
        row['Test_mse']    = test_mse
        row['Test_mdae']   = test_mdae
        row['Test_r2_all'] = test_r2
        return row

    # ========================================================================================================
    # Helper: get_best_epoch_row — adapted for sklearn (no results_testing.csv; receives metrics dict)
    # In the DNN version this read a CSV; here we receive the dict directly.
    # ========================================================================================================
    def get_best_epoch_row(train_val_metrics):
        """
        For sklearn there is no epoch-based results CSV. The "best" metrics are simply those
        of the single trained model. Receives a dict with Train_* and Validation_* keys and
        returns it unchanged (mirrors DNN behaviour of returning only train/val columns).
        """
        print("  Train R2 (all)      : {:.5f}".format(train_val_metrics.get('Train_r2_all', float('nan'))))
        print("  Validation R2 (all) : {:.5f}".format(train_val_metrics.get('Validation_r2_all', float('nan'))))
        return dict(train_val_metrics)

    # ========================================================================================================
    # Helper: save_best_epoch_summary — adapted for sklearn
    # (mirrors reference_DNN.py but receives train_val_metrics dict instead of reading a CSV)
    # ========================================================================================================
    def save_best_epoch_summary(train_val_metrics, test_metrics, dir_name, n_molecules=None):
        """
        Writes a temporary file for the pct=100 row (full test set).
        The final consolidated file is assembled by evaluate_closest_subset.
        """
        best_row = get_best_epoch_row(train_val_metrics)
        if best_row is None:
            return

        n_mol = n_molecules if n_molecules is not None else np.nan
        row   = build_summary_row(
            best_row, n_mol, 100,
            test_metrics['Test_mae'],
            test_metrics['Test_mse'],
            test_metrics['Test_mdae'],
            test_metrics['Test_r2_all'])

        df_row       = pd.DataFrame([row])
        # Write temp file; evaluate_closest_subset will pick it up and merge everything
        tmp_path     = "{}/best_epoch_summary_100pct_tmp.csv.gz".format(dir_name)
        df_row.to_csv(tmp_path, index=False, compression='gzip')
        print("\nFull-test-set summary row saved (tmp) to:", tmp_path)
        print("  Test R2 (100%) : {:.5f}".format(test_metrics['Test_r2_all']))

    # ========================================================================================================
    # Helper: evaluate_closest_subset — adapted for sklearn
    # (copied verbatim from reference_DNN.py; uses train_val_metrics dict instead of results_path)
    # ========================================================================================================
    def evaluate_closest_subset(dir_name, databaseName, train_val_metrics):
        """
        Reads the 19 cumulative subset files generated by generate_closest_subset.py.
        Collects all rows (5%..95% + 100%), sorts by pct ascending, writes a single
        best_epoch_summary.csv.gz. Also saves per-molecule CSV for each subset.
        """
        percentiles = list(range(5, 100, 5))    # [5, 10, 15, ..., 95]
        any_found   = False
        all_rows    = []                         # collect here, write once at end

        pred_csv = "{}/predictions_per_molecule.csv.gz".format(dir_name)
        if not os.path.exists(pred_csv):
            print("  WARNING: predictions_per_molecule.csv.gz not found. Skipping subset evaluation.")
            return

        df_pred = pd.read_csv(pred_csv, compression='gzip')

        print("\n" + "="*60)
        print("EVALUATING ON CUMULATIVE CLOSEST SUBSETS")
        print("="*60)
        print("  {:>4}  {:>8}  {:>10}  {:>10}  {:>10}".format(
            "pct", "n_mols", "MAE", "MdAE", "R2"))
        print("  " + "-"*48)

        for pct in percentiles:
            subset_filename = "subset_closest_{:02d}pct_{}.csv.gz".format(pct, databaseName)
            subset_path     = "scaffold_splitting/{}".format(subset_filename)

            if not os.path.exists(subset_path):
                print("  WARNING: {} not found, skipping.".format(subset_filename))
                continue

            any_found = True
            df_sub = pd.read_csv(subset_path, compression='gzip')
            df_sub = df_sub.loc[:, ~df_sub.columns.str.startswith('Unnamed')]

            df_merged = df_sub[['smiles']].merge(
                df_pred[['smiles', 'y_real', 'y_pred', 'abs_error',
                         'abs_percentage_error', 'tanimoto_distance_to_train',
                         'tanimoto_max_distance_to_train']],
                on='smiles', how='inner')

            if len(df_merged) == 0:
                print("  WARNING: No matching SMILES for {}% subset.".format(pct))
                continue

            y_real_s = df_merged['y_real'].to_numpy()
            y_pred_s = df_merged['y_pred'].to_numpy()
            n_mol    = len(df_merged)

            mae  = float(np.mean(np.abs(y_real_s - y_pred_s)))
            mse  = float(np.mean((y_real_s - y_pred_s)**2))
            mdae = float(np.median(np.abs(y_real_s - y_pred_s)))
            r2   = float(r2_score(y_real_s.reshape(-1, 1), y_pred_s.reshape(-1, 1)))

            print("  {:>4}%  {:>8}  {:>10.5f}  {:>10.5f}  {:>10.5f}".format(
                pct, n_mol, mae, mdae, r2))

            # Collect row using train_val_metrics dict (sklearn adaptation)
            best_row = get_best_epoch_row(train_val_metrics)
            if best_row is not None:
                all_rows.append(build_summary_row(best_row, n_mol, pct, mae, mse, mdae, r2))

            # Per-molecule CSV for this subset
            sub_out = "{}/predictions_per_molecule_closest_{:02d}pct.csv.gz".format(dir_name, pct)
            df_merged.to_csv(sub_out, index=False, compression='gzip')

        # Add pct=100 row from tmp file written by save_best_epoch_summary
        tmp_path = "{}/best_epoch_summary_100pct_tmp.csv.gz".format(dir_name)
        if os.path.exists(tmp_path):
            df_100 = pd.read_csv(tmp_path, compression='gzip')
            all_rows.append(df_100.iloc[0].to_dict())
            os.remove(tmp_path)

        # Write single sorted file
        if all_rows:
            df_all       = pd.DataFrame(all_rows)
            df_all       = df_all.sort_values('pct').reset_index(drop=True)
            summary_path = "{}/best_epoch_summary.csv.gz".format(dir_name)
            df_all.to_csv(summary_path, index=False, compression='gzip')

        if not any_found:
            print("  No subset files found. Run generate_closest_subset.py first.")
        else:
            print("  " + "-"*48)
            print("  All results consolidated in:")
            print("  {}/best_epoch_summary.csv.gz".format(dir_name))
        print("="*60)

    # ========================================================================================================
    # Load training data
    # ========================================================================================================
    df = pd.read_csv(file, compression='gzip')

    # Filter valid gap values
    idxs = np.where(df['lumo'] > df['homo'])[0]
    auxZ = np.zeros(len(df), dtype=bool)
    for i in idxs:
        auxZ[i] = True
    df = df[auxZ]

    # Load descriptors if requested
    if withDescriptors == "True":
        print('Checking descriptors File:', descNormal_file)
        if os.path.exists(descNormal_file):
            print('Descriptors file for {} already exists'.format(file))
            n_desc    = pd.read_csv(descNormal_file, compression='gzip')
            n_desc    = n_desc[auxZ]
            norm_desc = torch.tensor(n_desc.to_numpy(), dtype=torch.float32)
            print('Descriptors shape:', norm_desc.shape)
        else:
            raise Exception("\n\t Clean input file to obtain proper descriptor file")

    # Extract SMILES and compute fingerprints
    smiles = df['smiles'].tolist()
    print('Obtaining Mols')
    mols = [mol_from_smiles(smi) for smi in smiles]

    print('Calculate Morgan fingerprint')
    time_loading_mfp = time.time()
    memory_xmfp, xmfp = memory_usage(
        (calculate_morgan_fingerprints, (mols, int(radiusMFP), int(nBitsMFP))), retval=True)
    print('MFPMemory', max(memory_xmfp) - min(memory_xmfp))
    print('Time obtaining MFP:', time.time() - time_loading_mfp)
    print(' MFP SizeArray:', xmfp.nbytes / (1024**2))

    if encodingMethod.lower() == 'emfp':
        print('Obtaining eMFP')
        memory_emfp, emfp = memory_usage(
            (convert_fp_to_embV2, (xmfp, int(embeddingSize))), retval=True)
        print('eMFP SizeArray:', emfp.nbytes / (1024**2))
        rmfp = torch.tensor(emfp, dtype=torch.float32)
        print('eMFPMemory', max(memory_emfp) - min(memory_emfp))
        print('MFP_eMFP_memory {} bytes, {} MB'.format(
            xmfp.nbytes + emfp.nbytes, (xmfp.nbytes + emfp.nbytes) / (1024**2)))

    process      = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss

    if withDescriptors == "True":
        print("Concatenating MFP and descriptors")
        if encodingMethod.lower() == 'mfp':
            xmfp = torch.from_numpy(xmfp).float()
            x    = torch.hstack((xmfp, norm_desc))
            case = 'xmfp'
        elif encodingMethod.lower() == 'emfp':
            x    = torch.hstack((rmfp, norm_desc))
            case = 'emfp'
    else:
        if encodingMethod.lower() == 'mfp':
            x    = torch.from_numpy(xmfp).float()
            case = 'xmfp'
            del xmfp
        elif encodingMethod.lower() == 'emfp':
            x    = 1 * rmfp
            del rmfp
            case = 'emfp'

    print('Obtaining Target: GAP')
    try:
        y = torch.tensor(df[['gap']].to_numpy(), dtype=torch.float32)
    except:
        y = torch.tensor(df['lumo'].to_numpy() - df['homo'].to_numpy(), dtype=torch.float32)
        y = y.view(-1, 1)

    # Outer fold split
    x_train, y_train, x_tests, y_tests = get_fold(x, y, int(outterKFold) - 1, 5, seed=42)

    # ========================================================================================================
    # Optuna hyperparameter optimisation
    # ========================================================================================================
    if int_ext_case == 'internal':
        db_file_path = glob.glob('{}/*.db'.format(dir_name))

        sampler_study = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            sampler=sampler_study, directions=["maximize"],
            study_name=study_name_custom, storage=storage_url, load_if_exists=True)

        total_trials = 150
        if db_file_path:
            existing_trials = len(study.trials)
        else:
            existing_trials = 0
        trials_to_run = total_trials - existing_trials

        study.optimize(
            lambda trial: objective_with_time(trial, x_train, y_train, int(ffnnOrder), incNone, modelName),
            n_trials=trials_to_run,
            callbacks=[save_intermediate_results])

        best_params_trial = study.best_params

    else:
        print('Loading Best Params from .db for External OutterFold')
        sampler_study = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            sampler=sampler_study, directions=["maximize"],
            study_name=study_name_custom, storage=storage_url, load_if_exists=True)

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) == 0:
            raise RuntimeError(
                "No completed trials found in .db: {}".format(storage_url))

        best_params_trial = study.best_params
        print("Best trial value (from .db): {:.5f}".format(study.best_value))

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    print('Best parameters:', best_params_trial)

    # ========================================================================================================
    # Build regressor from best hyperparameters
    # ========================================================================================================
    if modelName == 'RF' or modelName == 'rf':
        regressor = RandomForestRegressor(
            n_estimators      =   int(best_params_trial['n_estimators']),
            max_depth         =   int(best_params_trial['max_depth']),
            min_samples_split =   int(best_params_trial['min_samples_split']),
            min_samples_leaf  =   int(best_params_trial['min_samples_leaf']),
            max_features      = float(best_params_trial['max_features']),
            bootstrap         =  bool(best_params_trial['bootstrap']))

    elif modelName == 'GBR' or modelName == 'gbr':
        regressor = GradientBoostingRegressor(
            learning_rate     = float(best_params_trial['learning_rate']),
            n_estimators      =   int(best_params_trial['n_estimators']),
            max_depth         =   int(best_params_trial['max_depth']),
            min_samples_split =   int(best_params_trial['min_samples_split']),
            min_samples_leaf  =   int(best_params_trial['min_samples_leaf']),
            subsample         = float(best_params_trial['subsample']),
            max_features      = float(best_params_trial['max_features']),
            random_state      = 42)

    elif modelName == 'KNR' or modelName == 'knr':
        regressor = KNeighborsRegressor(
            n_neighbors = int(best_params_trial['n_neighbors']),
            weights     =     best_params_trial['weights'],
            algorithm   =     best_params_trial['algorithm'],
            leaf_size   = int(best_params_trial['leaf_size']),
            p           = int(best_params_trial['p']),
            metric      =     best_params_trial['metric'])

    elif modelName == 'MLP' or modelName == 'mlp':
        if databaseName.lower() == 'rdb':
            max_iter_val = 300
        else:
            max_iter_val = 1500

        regressor = MLPRegressor(
            hidden_layer_sizes = (best_params_trial['hidden_layer_sizes']
                                  if isinstance(best_params_trial['hidden_layer_sizes'], list)
                                  else eval(best_params_trial['hidden_layer_sizes'])),
            activation         =   str(best_params_trial['activation']),
            solver             =   str(best_params_trial['solver']),
            alpha              = float(best_params_trial['alpha']),
            learning_rate      =   str(best_params_trial['learning_rate']),
            learning_rate_init = float(best_params_trial['learning_rate_init']),
            max_iter           = max_iter_val,
            momentum           = float(best_params_trial['momentum']),
            nesterovs_momentum =  bool(best_params_trial['nesterovs_momentum']))

    else:
        raise ValueError("Unknown modelName: {}. Choose RF, GBR, KNR, or MLP.".format(modelName))

    # ========================================================================================================
    # Build B_dict for input mapping (same logic as sklearn_original.py)
    # ========================================================================================================
    B_dict = {}
    if incNone == 'without_FFNN':
        B_dict['without_FFNN'] = None
    elif incNone == 'without_FFNN_with_Descriptors':
        B_dict['without_FFNN_with_Descriptors'] = None
    elif incNone == 'with_FFNN':
        B_dict['with_FFNN'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)
    elif incNone == 'with_FFNN_with_Descriptors':
        B_dict['with_FFNN_with_Descriptors'] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)
    elif incNone == 'with_FFNN_Gaussian':
        B_dict['with_FFNN_Gaussian'] = torch.normal(0, 1, size=(x_train.shape[1], x_train.shape[1])).to(device)
    elif incNone == 'with_FFNN_Gaussian_with_Descriptors':
        B_dict['with_FFNN_Gaussian_with_Descriptors'] = torch.normal(
            0, 1, size=(x_train.shape[1], x_train.shape[1])).to(device)
    else:
        B_dict[incNone] = torch.eye(x_train.shape[1], dtype=torch.float32).to(device)

    # ========================================================================================================
    # Apply input mapping and train
    # ========================================================================================================
    x_train_mapped = input_mapping(x_train, B_dict[incNone], device, ffnnOrder)
    x_tests_mapped = input_mapping(x_tests, B_dict[incNone], device, ffnnOrder)

    print('TRAINING MODEL {} ON OUTER FOLD {}'.format(modelName, outterKFold))
    timer_begin = time.time()
    regressor.fit(x_train_mapped, y_train.flatten())
    train_time = time.time() - timer_begin
    print('Training time: {:.2f}s'.format(train_time))

    # ========================================================================================================
    # Predict on train split and validation split (outer fold)
    # ========================================================================================================
    y_pred_train = torch.tensor(regressor.predict(x_train_mapped)).view(-1, 1)
    y_pred_tests = torch.tensor(regressor.predict(x_tests_mapped)).view(-1, 1)

    # Metric functions
    criterion_mae  = MAELoss()
    criterion_mse  = MSELoss()
    criterion_mdae = MedianAELoss()
    criterion_r2   = R2Score()

    train_mae  = criterion_mae(y_pred_train,  y_train).item()
    train_mse  = criterion_mse(y_pred_train,  y_train).item()
    train_mdae = criterion_mdae(y_pred_train, y_train).item()
    train_r2   = float(r2_score(
        y_train.numpy().flatten(), y_pred_train.detach().numpy().flatten()))

    val_mae  = criterion_mae(y_pred_tests,  y_tests).item()
    val_mse  = criterion_mse(y_pred_tests,  y_tests).item()
    val_mdae = criterion_mdae(y_pred_tests, y_tests).item()
    val_r2   = float(r2_score(
        y_tests.numpy().flatten(), y_pred_tests.detach().numpy().flatten()))

    # Dict of train/validation metrics — mirrors the "best_row" concept from reference_DNN.py
    train_val_metrics = {
        'Train_mae'       : train_mae,
        'Train_mse'       : train_mse,
        'Train_mdae'      : train_mdae,
        'Train_r2_all'    : train_r2,
        'Validation_mae'  : val_mae,
        'Validation_mse'  : val_mse,
        'Validation_mdae' : val_mdae,
        'Validation_r2_all': val_r2,
    }

    print('\n--- Outer Fold {} Metrics ---'.format(outterKFold))
    print('  Train     MAE={:.5f}  MSE={:.5f}  MdAE={:.5f}  R2={:.5f}'.format(
        train_mae, train_mse, train_mdae, train_r2))
    print('  Validation MAE={:.5f}  MSE={:.5f}  MdAE={:.5f}  R2={:.5f}'.format(
        val_mae, val_mse, val_mdae, val_r2))

    # ========================================================================================================
    # Load and preprocess external test set (scaffold_splitting/{db}_tests.csv.gz)
    # ========================================================================================================
    print('\nRUNNING CALCULATION ON EXTERNAL DB WITH DIFFERENT SCAFFOLDS')
    df_validation = pd.read_csv(file_validation, compression='gzip')

    idxs_validation = np.where(df_validation['lumo'] > df_validation['homo'])[0]
    auxZ_validation = np.zeros(len(df_validation), dtype=bool)
    for i in idxs_validation:
        auxZ_validation[i] = True
    df_validation = df_validation[auxZ_validation]

    if withDescriptors == "True":
        print('Checking descriptors File:', descNormal_file_validation)
        if os.path.exists(descNormal_file_validation):
            print('Descriptors file for {} already exists'.format(file_validation))
            n_desc_validation    = pd.read_csv(descNormal_file_validation, compression='gzip')
            n_desc_validation    = n_desc_validation[auxZ_validation]
            norm_desc_validation = torch.tensor(n_desc_validation.to_numpy(), dtype=torch.float32)
            print('Descriptors shape:', norm_desc_validation.shape)
        else:
            raise Exception("\n\t Clean input file to obtain proper descriptor file")

    smiles_validation = df_validation['smiles'].tolist()
    print('Obtaining Mols')
    mols_validation = [mol_from_smiles(smi) for smi in smiles_validation]

    print('Calculate Morgan fingerprint')
    time_loading_mfp_validation = time.time()
    memory_xmfp_validation, xmfp_validation = memory_usage(
        (calculate_morgan_fingerprints, (mols_validation, int(radiusMFP), int(nBitsMFP))), retval=True)
    print('MFPMemory', max(memory_xmfp_validation) - min(memory_xmfp_validation))
    print('Time obtaining MFP:', time.time() - time_loading_mfp_validation)
    print(' MFP SizeArray:', xmfp_validation.nbytes / (1024**2))

    if encodingMethod.lower() == 'emfp':
        print('Obtaining eMFP')
        memory_emfp_validation, emfp_validation = memory_usage(
            (convert_fp_to_embV2, (xmfp_validation, int(embeddingSize))), retval=True)
        print('eMFP SizeArray:', emfp_validation.nbytes / (1024**2))
        rmfp_validation = torch.tensor(emfp_validation, dtype=torch.float32)
        print('eMFPMemory', max(memory_emfp_validation) - min(memory_emfp_validation))
        print('MFP_eMFP_memory {} bytes, {} MB'.format(
            xmfp_validation.nbytes + emfp_validation.nbytes,
            (xmfp_validation.nbytes + emfp_validation.nbytes) / (1024**2)))

    if withDescriptors == "True":
        print("Concatenating MFP and descriptors")
        if encodingMethod.lower() == 'mfp':
            xmfp_validation = torch.from_numpy(xmfp_validation).float()
            x_validation    = torch.hstack((xmfp_validation, norm_desc_validation))
            case = 'xmfp'
        elif encodingMethod.lower() == 'emfp':
            x_validation    = torch.hstack((rmfp_validation, norm_desc_validation))
            case = 'emfp'
    else:
        if encodingMethod.lower() == 'mfp':
            x_validation = torch.from_numpy(xmfp_validation).float()
            case = 'xmfp'
            del xmfp_validation
        elif encodingMethod.lower() == 'emfp':
            x_validation = 1 * rmfp_validation
            del rmfp_validation
            case = 'emfp'

    print('Obtaining Target: GAP')
    try:
        y_validation = torch.tensor(df_validation[['gap']].to_numpy(), dtype=torch.float32)
    except:
        y_validation = torch.tensor(
            df_validation['lumo'].to_numpy() - df_validation['homo'].to_numpy(), dtype=torch.float32)
        y_validation = y_validation.view(-1, 1)

    # Apply input mapping to test set
    x_validation_mapped = input_mapping(x_validation, B_dict[incNone], device, ffnnOrder)

    # Predict on test set
    y_pred_validation = torch.tensor(regressor.predict(x_validation_mapped)).view(-1, 1)

    # Compute test metrics
    test_mae  = criterion_mae(y_pred_validation,  y_validation).item()
    test_mse  = criterion_mse(y_pred_validation,  y_validation).item()
    test_mdae = criterion_mdae(y_pred_validation, y_validation).item()
    test_r2   = float(r2_score(
        y_validation.numpy().flatten(), y_pred_validation.detach().numpy().flatten()))

    test_metrics = {
        'Test_mae'    : test_mae,
        'Test_mse'    : test_mse,
        'Test_mdae'   : test_mdae,
        'Test_r2_all' : test_r2,
    }

    print('\n--- External Test Set Metrics ---')
    print('  Test MAE={:.5f}  MSE={:.5f}  MdAE={:.5f}  R2={:.5f}'.format(
        test_mae, test_mse, test_mdae, test_r2))

    # ========================================================================================================
    # Tanimoto distances: test molecules vs training set
    # ========================================================================================================
    smiles_train_list = df['smiles'].tolist()   # df is already gap-filtered
    tanimoto_min, tanimoto_max = compute_tanimoto_distances(
        smiles_train_list, smiles_validation, radiusMFP, nBitsMFP)

    # ========================================================================================================
    # Save per-molecule predictions CSV (full test set)
    # ========================================================================================================
    y_real_np = y_validation.numpy().flatten()
    y_pred_np = y_pred_validation.detach().numpy().flatten()

    save_per_molecule_csv(
        smiles_validation, y_real_np, y_pred_np,
        tanimoto_min, tanimoto_max, dir_name)

    # ========================================================================================================
    # Save best_epoch_summary.csv.gz (pct=100 tmp row) then consolidate with subset rows
    # ========================================================================================================
    save_best_epoch_summary(
        train_val_metrics, test_metrics, dir_name, n_molecules=len(y_real_np))

    evaluate_closest_subset(dir_name, databaseName, train_val_metrics)

    print('\nRESULTS SUMMARY')
    print('  Train     R2={:.5f}'.format(train_r2))
    print('  Validation R2={:.5f}'.format(val_r2))
    print('  Test      R2={:.5f}'.format(test_r2))
    print('DONE')
