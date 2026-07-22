import os
import sys
import copy
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
# class RMSELoss(nn.Module):
#     def __init__(self):
#         super(RMSELoss, self).__init__()

#     def forward(self, y_pred, y_true):
#         return torch.sqrt(torch.mean((y_pred - y_true) ** 2))

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


class StopIfAboveThresholdCallback:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, study, trial):
        if trial.value is not None and not np.isnan(trial.value) and trial.value > self.threshold:
            study.stop()


def should_stop_training(evaluating_metric_list, patience=5):
    """Checks if training should stop based on early stopping logic.
    
    The function applies early stopping: if the evaluation metric does not 
    improve for `patience` consecutive epochs, it recommends stopping training.
    
    Args:
        evaluating_metric_list (list of float): History of metric values 
            (e.g., accuracy or loss per epoch).
        patience (int, optional): Number of epochs to wait without improvement 
            before stopping. Default is 5.
    
    Returns:
        bool: True if training should stop, False otherwise.
    """
    
    # Not enough values yet
    if len(evaluating_metric_list) <= patience:
        return False  
    
    # Get the best metric value so far
    best_value = max(evaluating_metric_list)
    
    # Check the last `patience` values
    recent_values = evaluating_metric_list[-patience:]
    
    # If none of the recent values are better than the best seen so far (before them)
    # then stop training
    if max(recent_values) < best_value:
        return True
    
    return False

# Input mapping for FFNN
def input_mapping(x, B, device):
    if B is None:
        return x.to(device)
    else:
        sin_list, cos_list = [], []
        x_proj = torch.matmul(2. * torch.pi * x, B.T).to(device)
        for ord in range( int( ffnnOrder ) ):
            sin_list.append( torch.sin( ( ord +1 ) * x_proj ) )
            cos_list.append( torch.cos( ( ord +1 ) * x_proj ) )
        final_list = sin_list + cos_list
        return torch.cat( final_list , dim = -1 ).to(device)

# PlotPredictions
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
    # plt.savefig(f'{epoch}_pred.png', dpi = 600)
    plt.savefig('{}_pred.png'.format(epoch), dpi=600)    
    # plt.show()


# Functions for training Model
def train_model(num_layers, neurons , input_dim, learning_rate, epochs, scaler, B, train_data, valid_data, batch_size, device, scale_factor_neurons, databaseName, embeddingSize, dir_name, order, incNone ):
    
    print('Training parameters:')
    print('num_layers           ', num_layers           )
    print('neurons              ', neurons              )
    print('learning_rate        ', learning_rate        )
    print('epochs               ', epochs               )
    print('batch_size           ', batch_size           )
    print('scale_factor_neurons ', scale_factor_neurons )
    print('B_dict               ', B                    )
    

    # train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True, worker_init_fn=lambda _: set_seed(42))
    # valid_loader = DataLoader(valid_data, batch_size = batch_size, shuffle=True, worker_init_fn=lambda _: set_seed(42))
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_data, batch_size = batch_size, shuffle=True, num_workers=2)

    
    torch.manual_seed(40)

    print('train valid shapes, Train Function', len(train_data), len(valid_data))

    print('Model architechture')
    model = DNN(input_dim * scaler, num_layers, neurons, scale_factor_neurons).to(device)

    print('Model architechture')
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate )

    # Define Metrics to Use

    # MAE Loss (Mean Absolute Error)
    criterion_xmae_Loss = MAELoss()

    # MSE Loss (Mean Squared Error)
    criterion_xmse_Loss = MSELoss()

    # MedianAE Loss
    criterion_meda_Loss = MedianAELoss()

    # RMSE Loss
    # criterion_rmse_Loss = RMSELoss()

    # R2 Score
    criterion_r2_Metric = R2Score()

    # Lists to save results every epoch
    train_xmae_loss = []
    train_xmse_loss = []
    train_mdae_loss = []
    train_r2_metric = []

    valid_xmae_loss = []
    valid_xmse_loss = []
    valid_mdae_loss = []
    valid_r2_metric = []

    r2_train_all = []
    r2_valid_all = []
    metric_kappa_list = []



    best_metric_kappa = -100

    timeEpochList = []

    for epoch in range(epochs):

        train_all_pred = []
        train_all_real = []
        valid_all_pred = []
        valid_all_real = []

        # Train Model
        model.train()

        train_xmae_running_loss = 0.0
        train_xmse_running_loss = 0.0
        train_meda_running_loss = 0.0
        train_xxr2_running_loss = 0.0

        valid_xmae_running_loss = 0.0
        valid_xmse_running_loss = 0.0
        valid_meda_running_loss = 0.0
        valid_xxr2_running_loss = 0.0


        timeEpoch = time.time()

        print('')
        print('Training')

        for current_batch, dataT in enumerate(train_loader):


            x_batch, y_batch = dataT
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_mapped = input_mapping(x_batch, B, device)
           
            optimizer.zero_grad()
            outputs = model(x_mapped)

            
            if device == torch.device("cuda"):

                train_all_pred.append( outputs.detach().cpu().numpy() )
                train_all_real.append( y_batch.detach().cpu().numpy() )

            else:
                train_all_pred.append( outputs.detach().numpy() )
                train_all_real.append( y_batch.detach().numpy() )

            # Evaluate Metrics
            xmae_metric = criterion_xmae_Loss( y_batch, outputs )
            xmse_metric = criterion_xmse_Loss( y_batch, outputs )
            meda_metric = criterion_meda_Loss( y_batch, outputs )
            xxr2_metric = criterion_r2_Metric( y_batch, outputs )



            # Loss Function: MAE
            xmae_metric.backward()

            # Optimizer
            optimizer.step()

            train_xmae_running_loss += xmae_metric.item()
            train_xmse_running_loss += xmse_metric.item()
            train_meda_running_loss += meda_metric.item()
            train_xxr2_running_loss += xxr2_metric.item()

            # print(f'Epoch:{epoch:03}, Batch:{current_batch + 1 :04}/{len(train_loader):04}, MAE: {xmae_metric.item():.4f}, MSE:{xmae_metric.item():.4f}, MedianAE:{meda_metric:.4f} R2:{xxr2_metric:.4f}')
            print('Epoch:{:03}, Batch:{:04}/{:04}, MAE: {:.4f}, MSE:{:.4f}, MedianAE:{:.4f} R2:{:.4f}'.format(epoch, current_batch + 1, len(train_loader), xmae_metric.item(), xmae_metric.item(), meda_metric, xxr2_metric ))            

        all_predT = np.concatenate( train_all_pred ).flatten()
        all_realT = np.concatenate( train_all_real ).flatten()

        all_predT = all_predT.reshape( -1, 1 )
        all_realT = all_realT.reshape( -1, 1 )

        r2_train_all.append( r2_score( all_realT, all_predT ) )
        
        train_xmae_loss.append( train_xmae_running_loss / len( train_loader ) )
        train_xmse_loss.append( train_xmse_running_loss / len( train_loader ) )
        train_mdae_loss.append( train_meda_running_loss / len( train_loader ) )
        train_r2_metric.append( train_xxr2_running_loss / len( train_loader ) )


        model.eval()

        print('')
        print('Validation')
        with torch.no_grad():
            for batchV, dataV in enumerate(valid_loader):
                
                x_batch, y_batch = dataV
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                x_mapped = input_mapping(x_batch, B, device)

                outputs = model(x_mapped)

                valid_all_pred.append( outputs.cpu().numpy() )
                valid_all_real.append( y_batch.cpu().numpy() )

                
                xmae_metric_valid = criterion_xmae_Loss( y_batch, outputs )
                xmse_metric_valid = criterion_xmse_Loss( y_batch, outputs )
                meda_metric_valid = criterion_meda_Loss( y_batch, outputs )
                xxr2_metric_valid = criterion_r2_Metric( y_batch, outputs )


                valid_xmae_running_loss += xmae_metric_valid.item()
                valid_xmse_running_loss += xmse_metric_valid.item()
                valid_meda_running_loss += meda_metric_valid.item()
                valid_xxr2_running_loss += xxr2_metric_valid.item()

                # print(f'Epoch:{epoch:03}, Batch:{ batchV + 1 :04}/{len(valid_loader):04}, MAE: {xmae_metric_valid.item():.4f}, MSE: {xmae_metric_valid:.4f}, MedianAE: {meda_metric_valid:.4f}, R2: {xxr2_metric_valid:.4f}')
                print('Epoch:{:03}, Batch:{:04}/{:04}, MAE: {:.4f}, MSE: {:.4f}, MedianAE: {:.4f}, R2: {:.4f}'.format(epoch, batchV + 1, len(valid_loader), xmae_metric_valid.item(), xmae_metric_valid, meda_metric_valid, xxr2_metric_valid ))                 
        

        all_predV = np.concatenate( valid_all_pred ).flatten()
        all_realV = np.concatenate( valid_all_real ).flatten()
        all_predV = all_predV.reshape( -1, 1 )
        all_realV = all_realV.reshape( -1, 1 )


        r2_valid_all.append( r2_score( all_realV, all_predV ) )

        timeEpochList.append(time.time() - timeEpoch)

        valid_xmae_loss.append( valid_xmae_running_loss / len( valid_loader ) )
        valid_xmse_loss.append( valid_xmse_running_loss / len( valid_loader ) )
        valid_mdae_loss.append( valid_meda_running_loss / len( valid_loader ) )
        valid_r2_metric.append( valid_xxr2_running_loss / len( valid_loader ) )

        # KAPPA: R2_train + R2_valid - | R2_train - R2_valid |

        
        metric_kappa = r2_train_all[-1] + r2_valid_all[-1] - np.abs( r2_train_all[-1] - r2_valid_all[-1] )
        metric_kappa_list.append( metric_kappa )


        # If metric_kappa improves, save the model
        quality_improve = len(metric_kappa_list) - np.argmax( metric_kappa_list )


        print('')
        print('Summary')
        message_epoch  = ''
        #message_epoch += f"Epoch:{epoch:03}, Quality Improve: {quality_improve}, "
        message_epoch += "Epoch:{:03}, Quality Improve: {}, ".format(epoch, quality_improve)        
        #message_epoch += f"MAE_Train: {train_xmae_loss[-1]:.4f}, MAE_Valid: {valid_xmae_loss[-1]:.4f} "
        message_epoch += "MAE_Train: {:.4f}, MAE_Valid: {:.4f} ".format(train_xmae_loss[-1], valid_xmae_loss[-1])        
        #message_epoch += f"MSE_Train: {train_xmae_loss[-1]:.4f}, MSE_Valid: {valid_xmse_loss[-1]:.4f} "
        message_epoch += "MSE_Train: {:.4f}, MSE_Valid: {:.4f} ".format(train_xmae_loss[-1], valid_xmse_loss[-1])        
        #message_epoch += f"MedianAE_Train: {train_xmae_loss[-1]:.4f}, MedianAE_Valid: {valid_mdae_loss[-1]:.4f} "
        message_epoch += "MedianAE_Train: {:.4f}, MedianAE_Valid: {:.4f} ".format(train_xmae_loss[-1], valid_mdae_loss[-1])        
        #message_epoch += f"R2_Train: {train_r2_metric[-1]:.4f}, R2_Valid: {valid_r2_metric[-1]:.4f} "
        message_epoch += "R2_Train: {:.4f}, R2_Valid: {:.4f} ".format(train_r2_metric[-1], valid_r2_metric[-1])        
        print(message_epoch)

        print('')


        if quality_improve == 1:
            # Saving best weigths of models
            best_model_wts = copy.deepcopy(model.state_dict())
            best_R2_train = r2_train_all[-1] 
            best_R2_valid = r2_valid_all[-1]
            best_epoch = epoch


        
        # Stop Training if model loss is lower than 0.0001
        if train_xmae_loss[-1] < 0.0001  and valid_xmae_loss[-1] < 0.0001 and train_r2_metric[-1] > 0.925 and valid_r2_metric[-1] > 0.925:
            # print(f'Loss reached at epoch {epoch}: Training Loss MAE: {train_xmae_loss[-1]}, Validation Loss MAE: {valid_xmae_loss[-1]}')
            print('Loss reached at epoch {}: Training Loss MAE: {}, Validation Loss MAE: {}'.format(epoch, train_xmae_loss[-1], valid_xmae_loss[-1] ))             
            break





        # Early stop criteria
        # Stop Training if model has not improved in 30 epochs
        if should_stop_training( metric_kappa_list, patience = 10 ):
            print('Early stopping, R2 starts to decrease', metric_kappa_list[-31:-1]) 
            break
        
    
    results_dict = {
                    "Epoch_time"            : timeEpochList,      #
                    "Train_mae"             : train_xmae_loss,
                    "Train_mse"             : train_xmse_loss,
                    "Train_mdae"            : train_mdae_loss,
                    "Train_r2_mean_batches" : train_r2_metric,
                    "Train_r2_all"          : r2_train_all,
                    "Valid_mae"             : valid_xmae_loss,
                    "Valid_mse"             : valid_xmse_loss,
                    "Valid_mdae"            : valid_mdae_loss,
                    "Valid_r2_mean_batches" : valid_r2_metric,
                    "Valid_r2_all"          : r2_valid_all,
                    "Kappa"                 : metric_kappa_list,
                    }

    # RETURN BEST MODEL and Train/Valid Losses and Metrics
    # Loading Best Weights accordingly to quality_improve

    model.load_state_dict( best_model_wts )
    return model, results_dict, best_R2_train, best_R2_valid, best_epoch
    # return model.load_state_dict( best_model_wts ), results_dict, best_R2_train, best_R2_valid, best_epoch


def objective(trial, x, y, order, incNone, xshape, dir_name, databaseName, embeddingSize, start_time):

    best_obj = -1000

    # Number of features to capture in Fourier Mapping
    mapping_size = xshape
    input_dim = xshape

    if incNone == 'none':
        k = 1
    else:
        k = 2

    epochs = 30

    max_power = int( input_dim      ).bit_length()
    min_power = int( input_dim // 2 ).bit_length()

    if 2**max_power < input_dim:
        max_power += 1

    if 2**min_power > input_dim//2 :
        min_power -= 2


    num_layers           =           trial.suggest_int('num_layers'   ,         3, 10       , step = 1 )   # 3, 10
    neurons              = int( 2 ** trial.suggest_int('neurons'      ,         4, max_power, step = 1 ) )
    batch_size           = int( 2 ** trial.suggest_int('batch_size'   ,         6, 10       , step = 1 ) ) #### 4:16, 5:32, 6:64, 7:128, 8:256, 9:512
    scale_factor_neurons = trial.suggest_int('scale_factor_neurons', 50, 75       , step = 1 )
    learning_rate        = trial.suggest_int('learning_rate',         1, 100000   , step = 1 )

    print("learning_rate        ", 1e-7 * learning_rate )
    print("neurons              ", neurons       )
    print("num_layers           ", num_layers    )
    print("batch_size           ", batch_size    )
    print("scale_factor_neurons ",    1e-2 * scale_factor_neurons )
    checkpoint = False

    print('input_dim, mapping_size', input_dim, mapping_size )

    B_dict = {}
    if incNone == 'without_FFNN':
        B_dict['without_FFNN'] = None

    elif  incNone == 'without_FFNN_with_Descriptors':
        B_dict['without_FFNN_with_Descriptors'] = None

    elif incNone == 'with_FFNN':
        B_dict['with_FFNN'] = torch.eye(x.shape[1], dtype=torch.float32).to(device)

    elif incNone == 'with_FFNN_with_Descriptors':
        B_dict['with_FFNN_with_Descriptors'] = torch.eye(x.shape[1], dtype=torch.float32).to(device)

    elif incNone == 'with_FFNN_Gaussian':
        B_dict['with_FFNN_Gaussian'] = torch.normal(0, 1, size=(x.shape[1], x.shape[1])).to(device)

    elif incNone == 'with_FFNN_Gaussian_with_Descriptors':
        B_dict['with_FFNN_Gaussian_with_Descriptors'] = torch.normal(0, 1, size=(x.shape[1], x.shape[1])).to(device)

    else:
        B_dict['with_FFNN'] = torch.eye(x.shape[1], dtype=torch.float32).to(device)


    # Take x_train_outter, y_train_outter from outter fold to obtain inner folds
    x_train1_inner, y_train1_inner, x_valid1_inner, y_valid1_inner = get_fold( x, y, 1 - 1,  5, seed = 42)
    x_train2_inner, y_train2_inner, x_valid2_inner, y_valid2_inner = get_fold( x, y, 2 - 1,  5, seed = 42)
    x_train3_inner, y_train3_inner, x_valid3_inner, y_valid3_inner = get_fold( x, y, 3 - 1,  5, seed = 42)
    x_train4_inner, y_train4_inner, x_valid4_inner, y_valid4_inner = get_fold( x, y, 4 - 1,  5, seed = 42)
    x_train5_inner, y_train5_inner, x_valid5_inner, y_valid5_inner = get_fold( x, y, 5 - 1,  5, seed = 42)

    # Obtaining dataloader
    print('Preparing dataloader for every inner Fold')
    trainIN1 = TensorDataset( x_train1_inner, y_train1_inner )
    trainIN2 = TensorDataset( x_train2_inner, y_train2_inner )
    trainIN3 = TensorDataset( x_train3_inner, y_train3_inner )
    trainIN4 = TensorDataset( x_train4_inner, y_train4_inner )
    trainIN5 = TensorDataset( x_train5_inner, y_train5_inner )

    validIN1 = TensorDataset( x_valid1_inner, y_valid1_inner )
    validIN2 = TensorDataset( x_valid2_inner, y_valid2_inner )
    validIN3 = TensorDataset( x_valid3_inner, y_valid3_inner )
    validIN4 = TensorDataset( x_valid4_inner, y_valid4_inner )
    validIN5 = TensorDataset( x_valid5_inner, y_valid5_inner )

    outputs = {}
    
    for k, B in B_dict.items():
        start2 = time.time()
        #print(f"Training with B: {k}")
        print("Training with B: {}".format(k))        
        if k == None or k == 'none' or 'without_FFNN' in k:
            scaler = 1 
        else:
            scaler = 2 * int(order)

        print('Scaler' , scaler)

        try:
            # try to run
            print('Training Function')
            model_Fold1, results_Fold1, bestR2train1, bestR2valid1, bestEpoch1 = train_model( num_layers, neurons , input_dim, 1e-7 * learning_rate, epochs, scaler, B_dict[incNone], trainIN1, validIN1, batch_size, device, 1e-2 * scale_factor_neurons,  databaseName, embeddingSize, dir_name, order, incNone)
            model_Fold2, results_Fold2, bestR2train2, bestR2valid2, bestEpoch2 = train_model( num_layers, neurons , input_dim, 1e-7 * learning_rate, epochs, scaler, B_dict[incNone], trainIN2, validIN2, batch_size, device, 1e-2 * scale_factor_neurons,  databaseName, embeddingSize, dir_name, order, incNone)
            model_Fold3, results_Fold3, bestR2train3, bestR2valid3, bestEpoch3 = train_model( num_layers, neurons , input_dim, 1e-7 * learning_rate, epochs, scaler, B_dict[incNone], trainIN3, validIN3, batch_size, device, 1e-2 * scale_factor_neurons,  databaseName, embeddingSize, dir_name, order, incNone)
            model_Fold4, results_Fold4, bestR2train4, bestR2valid4, bestEpoch4 = train_model( num_layers, neurons , input_dim, 1e-7 * learning_rate, epochs, scaler, B_dict[incNone], trainIN4, validIN4, batch_size, device, 1e-2 * scale_factor_neurons,  databaseName, embeddingSize, dir_name, order, incNone)
            model_Fold5, results_Fold5, bestR2train5, bestR2valid5, bestEpoch5 = train_model( num_layers, neurons , input_dim, 1e-7 * learning_rate, epochs, scaler, B_dict[incNone], trainIN5, validIN5, batch_size, device, 1e-2 * scale_factor_neurons,  databaseName, embeddingSize, dir_name, order, incNone)

            regressors = {
                          'reg1': model_Fold1,
                          'reg2': model_Fold2,
                          'reg3': model_Fold3,
                          'reg4': model_Fold4,
                          'reg5': model_Fold5,
                         }
            metrics =    {
                          'met1' : pd.DataFrame( results_Fold1 ),
                          'met2' : pd.DataFrame( results_Fold2 ),
                          'met3' : pd.DataFrame( results_Fold3 ),
                          'met4' : pd.DataFrame( results_Fold4 ),
                          'met5' : pd.DataFrame( results_Fold5 ),

                         }
            best_values_ETV = {
                              'Fold'          : list(range(1,6)),
                              'Best_Epoch'    : [ bestEpoch1, bestEpoch2, bestEpoch3, bestEpoch4, bestEpoch5 ],
                              'Best_R2_Train' : [ bestR2train1, bestR2train2, bestR2train3, bestR2train4, bestR2train5 ],
                              'Best_R2_Valid' : [ bestR2valid1, bestR2valid2, bestR2valid3, bestR2valid4, bestR2valid5 ] 
                              }
            

            df_summary = pd.DataFrame( best_values_ETV )
            


            status_nan = []
        
            for iter_dict, ( name , loss_values ) in enumerate( results_Fold1.items() ) :
                status_nan.append( int( np.isnan( loss_values ).any() ) ) 

            for iter_dict, ( name , loss_values ) in enumerate( results_Fold2.items() ) :
                status_nan.append( int( np.isnan( loss_values ).any() ) ) 

            for iter_dict, ( name , loss_values ) in enumerate( results_Fold3.items() ) :
                status_nan.append( int( np.isnan( loss_values ).any() ) ) 

            for iter_dict, ( name , loss_values ) in enumerate( results_Fold4.items() ) :
                status_nan.append( int( np.isnan( loss_values ).any() ) ) 

            for iter_dict, ( name , loss_values ) in enumerate( results_Fold5.items() ) :
                status_nan.append( int( np.isnan( loss_values ).any() ) ) 

            
            if np.sum( status_nan ) > 0: 

                outputTrain = -0.12

            else:

                outputTrain  = bestR2train1 + bestR2valid1  - np.abs( bestR2train1 - bestR2valid1 )
                outputTrain += bestR2train2 + bestR2valid2  - np.abs( bestR2train2 - bestR2valid2 )
                outputTrain += bestR2train3 + bestR2valid3  - np.abs( bestR2train3 - bestR2valid3 )
                outputTrain += bestR2train4 + bestR2valid4  - np.abs( bestR2train4 - bestR2valid4 )
                outputTrain += bestR2train5 + bestR2valid5  - np.abs( bestR2train5 - bestR2valid5 )
                
                print('')
                print('')
                print('OUTPUT TRAIN', trial.number, outputTrain )
                print('')
                print('')
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
                    if outputTrain > best_trial_info.value:
                        checkpoint = True

                except:

                    checkpoint = True


                if checkpoint == True:
                     
                    temp_dir_paths = []
                    for ipk in range(5):
                        try:

                            # path =  f'{dir_name}/optuna_models_{databaseName}_dnn_{case}_{embeddingSize:03}_innerFold_{ipk+1}/'
                            path = '{}/optuna_models_{}_dnn_{}_{:03}_innerFold_{}/'.format(dir_name, databaseName, case, embeddingSize, ipk + 1)                             
                            temp_dir_paths.append( path )

                        except Exception as e:
                            #print(f"Error in path. Error: {e}")
                            print("Error in path. Error: {}".format(e))                            

                    for itemp, directory_pk in enumerate(temp_dir_paths):

                        if not os.path.exists(directory_pk):
                            os.makedirs(directory_pk)

                        label = best_values_ETV["Best_Epoch"][itemp]

                        # temp_name  = f'model_{databaseName}_size_{embeddingSize:03}_o_{order}_{incNone}_' 
                        temp_name = 'model_{}_size_{:03}_o_{}_{}_'.format( databaseName, embeddingSize, order, incNone )                        
                        # temp_name1 = f'{directory_pk}{temp_name}'
                        temp_name1 = '{}{}'.format(directory_pk, temp_name)                        
                        # paths_pattern = glob.glob(f'{temp_name1}*')
                        paths_pattern = glob.glob('{}*'.format(temp_name1))                        


                        for file_name in paths_pattern:
                            if temp_name in file_name:
                                os.remove(file_name)



                        # temp_name1 += f'trialNumber_{trial.number:03}_'
                        temp_name1 += 'trialNumber_{:03}_'.format(trial.number)                        
                        #temp_name1 += f"best_epoch_{label}_"
                        temp_name1 += "best_epoch_{}_".format(label)                        
                        #temp_name1 += f"lr_{learning_rate}_"
                        temp_name1 += "lr_{}_".format(learning_rate)                        
                        #temp_name1 += f"neurons_{neurons}_"
                        temp_name1 += "neurons_{}_".format(neurons)                        
                        #temp_name1 += f"N_layers_{num_layers}_"
                        temp_name1 += "N_layers_{}_".format(num_layers)                        
                        #temp_name1 += f"batch_size_{batch_size}_"
                        temp_name1 += "batch_size_{}_".format(batch_size)                        
                        #temp_name1 += f"scale_factor_neurons_{scale_factor_neurons}_innerFold_{itemp+1}"
                        temp_name1 += "scale_factor_neurons_{}_innerFold_{}".format(scale_factor_neurons, itemp + 1)                        



                        # Save best model
                        #torch.save(regressors[f"reg{itemp+1}"], f"{temp_name1}.pth"  )
                        torch.save(regressors["reg{}".format(itemp + 1)], "{}.pth".format(temp_name1))                        


                        # Save metrics

                        #metrics[ f"met{itemp+1}" ].to_csv( f"{temp_name1}.csv", index = False )
                        metrics["met{}".format(itemp + 1)].to_csv("{}.csv".format(temp_name1), index=False)                        

                    else:

                        continue

                #df_summary.to_csv(f"{dir_name}/summary_metrics.csv", index = False )
                df_summary.to_csv("{}/summary_metrics.csv".format(dir_name), index=False)                

        except:
            # if it fails, return -0.1 
            outputTrain = 0
            print('')
            print('###################################')
            print('#')
            print('# FAILLING COLLECTING RESULTS')
            print('#')            
            print('###################################')
            print('')
        # model, train_losses, trainMAE_losses, R2Train, valid_losses, validMAE_losses, R2Valid = outputTrain
    # print(f'Time { time.time() - start_time }' )
    print('Time {}'.format(time.time() - start_time))    
    print('FINALOUT', outputTrain )
    return outputTrain


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
    #results_df.to_csv(f"{dir_name}/optuna_results_{databaseName}_dnn_{embeddingSize}_{incNone}.csv", index=False)
    results_df.to_csv("{}/optuna_results_{}_dnn_{}_{}.csv".format(dir_name, databaseName, embeddingSize, incNone), index=False)    

    # # Directories for saving plots (ensure they are defined)
    ## directory_path_png = f"{dir_name}/optuna_visualization_{databaseName}_dnn_{embeddingSize:03}_png/"
    # directory_path_png = "{}/optuna_visualization_{}_dnn_{:03}_png/".format(dir_name, databaseName, embeddingSize)    
    ## directory_path_svg = f"{dir_name}/optuna_visualization_{databaseName}_dnn_{embeddingSize:03}_svg/"
    # directory_path_svg = "{}/optuna_visualization_{}_dnn_{:03}_svg/".format(dir_name, databaseName, embeddingSize)    

    # if not os.path.exists(directory_path_png):
    #     os.makedirs(directory_path_png)
    # if not os.path.exists(directory_path_svg):
    #     os.makedirs(directory_path_svg)

    # # Attempt to generate and save the plots after each trial
    # try:
    ##     print(f"Attempting to save Optuna plot for trial {trial_number}...")
    # print("Attempting to save Optuna plot for trial {}...".format(trial_number))    

    #     # Ensure the plot is only created if there are enough trials
    #     if len(study.trials) > 1:
    #         generate_plots(study, trial_number, directory_path_png, directory_path_svg)
    #     else:
    ##         print(f"Skipping plots for trial {trial_number}, only 1 trial completed.")
    # print("Skipping plots for trial {}, only 1 trial completed.".format(trial_number))    
        
    # except Exception as e:
    #     # If an error occurs while saving the plot, print the error message
    ##     print(f"Failed to save Optuna plot for trial {trial_number}. Error: {e}")
    # print("Failed to save Optuna plot for trial {}. Error: {}".format(trial_number, e))    

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
            #print(f"Attempting to save contour plot for parameters: {p[0]} and {p[1]} (Trial {trial_number})...")
            print("Attempting to save contour plot for parameters: {} and {} (Trial {})...".format(p[0], p[1], trial_number))            
            fig = optuna.visualization.plot_contour(study, params=[p[0], p[1]])

            # Customizations for contour plot
            fig.update_traces(colorscale='Blackbody', selector=dict(type='contour'))  # Color palette
            fig.update_traces(line_smoothing=1.15, selector=dict(type='contour'))  # Smooth lines
            fig.update_traces(line_width=0, selector=dict(type='contour'))  # Remove contour lines
            fig.update_traces(marker=dict(size=0.25, color="RoyalBlue"), selector=dict(mode='markers'))  # Marker styling

            # Save the contour plot
            #fig.write_image(f"{directory_path_png}{trial_number}_{custom_name}contour_{p[0]}_{p[1]}.png")
            fig.write_image("{}{}_{}contour_{}_{}.png".format(directory_path_png, trial_number, custom_name, p[0], p[1]))            
            #fig.write_image(f"{directory_path_svg}{trial_number}_{custom_name}contour_{p[0]}_{p[1]}.svg")
            fig.write_image("{}{}_{}contour_{}_{}.svg".format(directory_path_svg, trial_number, custom_name, p[0], p[1]))            
            #print(f"Successfully saved contour plot for parameters {p[0]} and {p[1]} (Trial {trial_number}).")
            print("Successfully saved contour plot for parameters {} and {} (Trial {}).".format(p[0], p[1], trial_number))            

        except Exception as e:
            #print(f"Failed to save contour plot for parameters {p[0]} and {p[1]} (Trial {trial_number}). Error: {e}")
            print("Failed to save contour plot for parameters {} and {} (Trial {}). Error: {}".format(p[0], p[1], trial_number, e))            

    # Generate parameter importance plot
    try:
        #print(f"Attempting to save parameter importances plot (Trial {trial_number})...")
        print("Attempting to save parameter importances plot (Trial {})...".format(trial_number))        
        fig = optuna.visualization.plot_param_importances(study)
        fig.update_layout(template='simple_white')
        #fig.write_image(f"{directory_path_png}trial_{trial_number}_{custom_name}param_importances.png")
        fig.write_image("{}trial_{}_{}param_importances.png".format(directory_path_png, trial_number, custom_name))        
        #fig.write_image(f"{directory_path_svg}trial_{trial_number}_{custom_name}param_importances.svg")
        fig.write_image("{}trial_{}_{}param_importances.svg".format(directory_path_svg, trial_number, custom_name))        
        #print(f"Successfully saved parameter importances plot (Trial {trial_number}).")
        print("Successfully saved parameter importances plot (Trial {}).".format(trial_number))        
        
    except Exception as e:
        #print(f"Failed to save parameter importances plot for trial {trial_number}. Error: {e}")
        print("Failed to save parameter importances plot for trial {}. Error: {}".format(trial_number, e))        

    # Generate slice plot for parameters
    try:
        #print(f"Attempting to save slice plot (Trial {trial_number})...")
        print("Attempting to save slice plot (Trial {})...".format(trial_number))        
        fig = optuna.visualization.plot_slice(study, params=parameters_optuna)
        fig.update_layout(template='simple_white')
        #fig.write_image(f"{directory_path_png}trial_{trial_number}_{custom_name}slice_plot.png")
        fig.write_image("{}trial_{}_{}slice_plot.png".format(directory_path_png, trial_number, custom_name))        
        #fig.write_image(f"{directory_path_svg}trial_{trial_number}_{custom_name}slice_plot.svg")
        fig.write_image("{}trial_{}_{}slice_plot.svg".format(directory_path_svg, trial_number, custom_name))        
        #print(f"Successfully saved slice plot (Trial {trial_number}).")
        print("Successfully saved slice plot (Trial {}).".format(trial_number))        
        
    except Exception as e:
        #print(f"Failed to save slice plot for trial {trial_number}. Error: {e}")
        print("Failed to save slice plot for trial {}. Error: {}".format(trial_number, e))        

def objective_with_time(trial, train_data, valid_data, order, incNone, xshape, dir_name, databaseName, embeddingSize):

# def objective_with_time(trial, x_train, x_test, y_train, y_test, order, incNone, model_name):
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
    result = objective(trial, train_data, valid_data, order, incNone, xshape, dir_name, databaseName, embeddingSize, start_time)
    # result = objective(trial, x_train, x_test, y_train, y_test, order, incNone, model_name, start_time)
    train_time = time.time() - start_time
    trial.set_user_attr('train_time', train_time)

    return result






if __name__ == "__main__":
    print("Program started.")

    # ========================================================================================================
    # Parse Arguments
    # ========================================================================================================
    databaseName    =  str(sys.argv[1])
    encodingMethod  =  str(sys.argv[2])
    embeddingSize   =  int(sys.argv[3])
    withDescriptors =  str(sys.argv[4])
    ffnnCase        =  str(sys.argv[5])
    ffnnOrder       =  int(sys.argv[6])
    nBitsMFP        =  int(sys.argv[7])
    radiusMFP       =  int(sys.argv[8])
    modelName       =  str(sys.argv[9])
    outterKFold     =  int(sys.argv[10])
    int_ext_case    =  str(sys.argv[11])

    # ========================================================================================================
    # File Paths
    # ========================================================================================================
    file                        = 'scaffold_splitting/{}_train.csv.gz'.format(databaseName)
    descNormal_file             = 'scaffold_splitting/desc_{}_train.csv.gz'.format(databaseName)
    file_validation             = 'scaffold_splitting/{}_tests.csv.gz'.format(databaseName)
    descNormal_file_validation  = 'scaffold_splitting/desc_{}_tests.csv.gz'.format(databaseName)

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

    if withDescriptors == 'True':
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

    model_path  = "{}/model_testing_eval.pth".format(dir_name)
    db_path     = "{}/{}.db".format(dir_name, study_name_custom)

    db_exists    = os.path.exists(db_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========================================================================================================
    # Helper: preprocess a dataframe (train or validation) into tensors x, y
    # ========================================================================================================
    def preprocess_dataframe(df_input, descNormal_file_path, withDescriptors, encodingMethod,
                              embeddingSize, radiusMFP, nBitsMFP):
        """
        Returns x (tensor), y (tensor), and input_dim (int).
        Applies the same preprocessing pipeline as the training data:
          - gap filter
          - MFP / eMFP
          - optional descriptors
          - target y = gap
        """
        # Filter gap
        idxs  = np.where(df_input['lumo'] > df_input['homo'])[0]
        auxZ  = np.zeros(len(df_input), dtype=bool)
        for i in idxs:
            auxZ[i] = True
        df_filtered = df_input[auxZ]

        # Descriptors
        norm_desc_tensor = None
        if withDescriptors == "True":
            print('Loading descriptors from', descNormal_file_path)
            if os.path.exists(descNormal_file_path):
                n_desc_full = pd.read_csv(descNormal_file_path, compression='gzip')
                n_desc_full = n_desc_full.loc[:, ~n_desc_full.columns.str.startswith('Unnamed')]

                if 'original_index' in df_input.columns:
                    # Subset case: use stored original row indices to align descriptors
                    # original_index refers to position in the raw (pre-gap-filter) file,
                    # so we apply the same gap filter to the descriptor file first,
                    # then select by the subset's original_index values.
                    gap_mask     = df_input['lumo'] > df_input['homo']
                    valid_orig   = df_input.loc[gap_mask, 'original_index'].tolist()
                    # Build a mapping: original_index -> descriptor row
                    # The descriptor file rows correspond 1:1 with the raw test CSV rows
                    n_desc = n_desc_full.iloc[valid_orig].reset_index(drop=True)
                else:
                    # Full dataset case: apply gap mask by position (original behavior)
                    n_desc = n_desc_full[auxZ].reset_index(drop=True)

                norm_desc_tensor = torch.tensor(n_desc.to_numpy(), dtype=torch.float32)
                print('Descriptors shape:', norm_desc_tensor.shape)
            else:
                raise Exception("\n\tDescriptor file not found: {}".format(descNormal_file_path))

        # Fingerprints
        smiles = df_filtered['smiles'].tolist()
        mols   = [mol_from_smiles(smi) for smi in smiles]
        print('Calculating Morgan fingerprints...')
        _, xmfp = memory_usage(
            (calculate_morgan_fingerprints, (mols, int(radiusMFP), int(nBitsMFP))), retval=True)

        if encodingMethod.lower() == 'emfp':
            print('Obtaining eMFP...')
            _, emfp = memory_usage((convert_fp_to_embV2, (xmfp, int(embeddingSize))), retval=True)
            rmfp = torch.tensor(emfp, dtype=torch.float32)

        # Build x
        if withDescriptors == "True":
            if encodingMethod.lower() == 'mfp':
                xmfp_t = torch.from_numpy(xmfp).float()
                x      = torch.hstack((xmfp_t, norm_desc_tensor))
            else:
                x = torch.hstack((rmfp, norm_desc_tensor))
        else:
            if encodingMethod.lower() == 'mfp':
                x = torch.from_numpy(xmfp).float()
            else:
                x = 1 * rmfp

        # Target y
        try:
            y = torch.tensor(df_filtered[['gap']].to_numpy(), dtype=torch.float32)
        except:
            y = torch.tensor(
                df_filtered['lumo'].to_numpy() - df_filtered['homo'].to_numpy(), dtype=torch.float32)
            y = y.view(-1, 1)

        return x, y

    # ========================================================================================================
    # Helper: evaluate a model on a TensorDataset and return a metrics dict
    # ========================================================================================================
    def evaluate_model_on_dataset(model, dataset, device, B, batch_size=512):
        """
        Runs model.eval() on the given TensorDataset.
        Returns:
          - ext_metrics dict  (aggregated metrics, one value each)
          - all_real_np       (numpy array of real values, per molecule)
          - all_pred_np       (numpy array of predicted values, per molecule)

        B must be the same matrix used during training (None for without_FFNN cases).
        ffnnOrder is read from the outer scope (same as training).
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        criterion_mae  = MAELoss()
        criterion_mse  = MSELoss()
        criterion_mdae = MedianAELoss()
        criterion_r2   = R2Score()

        mae_running  = 0.0
        mse_running  = 0.0
        mdae_running = 0.0
        r2_running   = 0.0

        all_pred = []
        all_real = []

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                x_mapped = input_mapping(x_batch, B, device)

                outputs = model(x_mapped)

                all_pred.append(outputs.cpu().numpy())
                all_real.append(y_batch.cpu().numpy())

                mae_running  += criterion_mae(y_batch,  outputs).item()
                mse_running  += criterion_mse(y_batch,  outputs).item()
                mdae_running += criterion_mdae(y_batch, outputs).item()
                r2_running   += criterion_r2(y_batch,   outputs).item()

        n = len(loader)
        all_pred_np = np.concatenate(all_pred).flatten()
        all_real_np = np.concatenate(all_real).flatten()
        r2_all      = r2_score(all_real_np.reshape(-1, 1), all_pred_np.reshape(-1, 1))

        ext_metrics = {
            "Test_mae"             : [mae_running  / n],
            "Test_mse"             : [mse_running  / n],
            "Test_mdae"            : [mdae_running / n],
            "Test_r2_all"          : [r2_all],
        }
        return ext_metrics, all_real_np, all_pred_np

    # ========================================================================================================
    # Helper: build B_dict and scaler from incNone (needed to reconstruct model input)
    # ========================================================================================================
    def build_B_dict(incNone, x_dim, ffnnOrder, device):
        B_dict = {}
        if incNone == 'without_FFNN':
            B_dict['without_FFNN'] = None
            scaler = 1
        elif incNone == 'without_FFNN_with_Descriptors':
            B_dict['without_FFNN_with_Descriptors'] = None
            scaler = 1
        elif incNone == 'with_FFNN':
            B_dict['with_FFNN'] = torch.eye(x_dim, dtype=torch.float32).to(device)
            scaler = 2 * int(ffnnOrder)
        elif incNone == 'with_FFNN_with_Descriptors':
            B_dict['with_FFNN_with_Descriptors'] = torch.eye(x_dim, dtype=torch.float32).to(device)
            scaler = 2 * int(ffnnOrder)
        elif incNone == 'with_FFNN_Gaussian':
            B_dict['with_FFNN_Gaussian'] = torch.normal(0, 1, size=(x_dim, x_dim)).to(device)
            scaler = 2 * int(ffnnOrder)
        elif incNone == 'with_FFNN_Gaussian_with_Descriptors':
            B_dict['with_FFNN_Gaussian_with_Descriptors'] = torch.normal(0, 1, size=(x_dim, x_dim)).to(device)
            scaler = 2 * int(ffnnOrder)
        else:
            B_dict[incNone] = torch.eye(x_dim, dtype=torch.float32).to(device)
            scaler = 2 * int(ffnnOrder)
        return B_dict, scaler

    # ========================================================================================================
    # Helper: compute Tanimoto distance from each test molecule to its nearest neighbor in train
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
    # Helper: save per-molecule predictions CSV with real, predicted and Tanimoto distance
    # ========================================================================================================
    def save_per_molecule_csv(smiles_test_list, y_real, y_pred, tanimoto_min, tanimoto_max,
                              dir_name, filename='predictions_per_molecule.csv.gz'):
        """
        Saves a CSV with one row per test molecule containing:
          smiles, y_real, y_pred, abs_error, abs_percentage_error,
          tanimoto_distance_to_train (nearest), tanimoto_max_distance_to_train (farthest)
        """
        abs_err  = np.abs(y_real - y_pred)
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
    # Helper: evaluate metrics on closest-10% subset and save dedicated files
    # ========================================================================================================
    def evaluate_closest_subset(dir_name, databaseName, results_path):
        """
        Reads the 19 cumulative subset files generated by generate_closest_subset.py.
        Collects all rows (5%..95% + 100%), sorts by pct ascending, writes a single
        best_epoch_summary.csv.gz. Also saves per-molecule CSV for each subset.
        """
        percentiles = list(range(5, 105, 5))    # [5, 10, 15, ..., 95]
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

            # Collect row
            best_row = get_best_epoch_row(results_path)
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
    # Helper: read results_testing.csv, extract best epoch row, combine with ext_metrics, save summary
    # ========================================================================================================
    def get_best_epoch_row(results_path):
        """
        Reads results_testing.csv and returns a dict with the metrics
        of the best epoch (highest Kappa). Returns None if not found.
        """
        if not os.path.exists(results_path):
            print("WARNING: {} not found".format(results_path))
            return None

        df_results    = pd.read_csv(results_path, compression='gzip')
        df_train_rows = df_results[df_results['Kappa'].notna()].copy()

        if df_train_rows.empty:
            print("WARNING: No valid Kappa rows found in results_testing.csv")
            return None

        best_idx = df_train_rows['Kappa'].idxmax()
        best_row = df_train_rows.loc[best_idx].to_dict()

        print("  Best epoch index : {}".format(best_idx))
        print("  Best Kappa       : {:.5f}".format(best_row['Kappa']))
        print("  Train R2 (all)   : {:.5f}".format(best_row.get('Train_r2_all', float('nan'))))
        print("  Valid R2 (all)   : {:.5f}".format(best_row.get('Valid_r2_all', float('nan'))))

        # Keep only train/validation columns
        train_val_cols = [c for c in best_row.keys() if not c.startswith('Test_')]
        return {c: best_row[c] for c in train_val_cols}

    def build_summary_row(best_row, n_molecules, pct, mae, mse, mdae, r2):
        """
        Builds a single summary row dict with a fixed, explicit column order:
          [train/val cols] | n_molecules | pct | Test_mae | Test_mse |
          Test_mdae | Test_r2_all
        """
        row = {}
        for col, val in best_row.items():
            row[col] = val
        row['n_molecules']               = n_molecules
        row['pct']                       = pct
        row['Test_mae']              = mae
        row['Test_mse']              = mse
        row['Test_mdae']             = mdae
        row['Test_r2_all']           = r2
        return row

    def save_best_epoch_summary(results_path, ext_metrics, dir_name, n_molecules=None):
        """
        Appends one row (pct=100, full test set) to best_epoch_summary.csv.
        Uses build_summary_row to guarantee fixed column order.
        """
        best_row = get_best_epoch_row(results_path)
        if best_row is None:
            return

        n_mol = n_molecules if n_molecules is not None else np.nan
        row   = build_summary_row(
            best_row, n_mol, 100,
            ext_metrics['Test_mae'][0],
            ext_metrics['Test_mse'][0],
            ext_metrics['Test_mdae'][0],
            ext_metrics['Test_r2_all'][0])

        df_row       = pd.DataFrame([row])
        summary_path = "{}/best_epoch_summary.csv.gz".format(dir_name)

        if os.path.exists(summary_path):
            df_existing = pd.read_csv(summary_path, compression='gzip')
            df_out      = pd.concat([df_existing, df_row], ignore_index=True)
        else:
            df_out = df_row

        df_out.to_csv(summary_path, index=False, compression='gzip')
        print("\nBest epoch summary saved to:", summary_path)
        print("  External R2 (100%) : {:.5f}".format(ext_metrics['Test_r2_all'][0]))

    # ========================================================================================================
    # SCENARIO 2: Only .db EXISTS (no model_testing.pth)
    #   → Read best trial from .db, retrain on full outer fold, evaluate on external set
    # ========================================================================================================
    if db_exists:
        print("\n[SCENARIO 2] Only trials_info.db found. Reading best trial, retraining, then evaluating.\n")

        # --- Load training data ---
        print("Loading training data...")
        df_train = pd.read_csv(file, compression='gzip')
        df_train = df_train.loc[:, ~df_train.columns.str.startswith('Unnamed')]
        x, y = preprocess_dataframe(
            df_train, descNormal_file, withDescriptors,
            encodingMethod, embeddingSize, radiusMFP, nBitsMFP)

        # --- Outer fold split ---
        x_train_outter, y_train_outter, x_tests_outter, y_tests_outter = get_fold(
            x, y, outterKFold - 1, 5, seed=42)

        # --- Load best params from .db, fall back to CSV if .db is corrupt ---
        def _load_best_params_from_csv(dir_name, databaseName, embeddingSize, incNone):
            """Try both compressed and uncompressed CSV variants."""
            for csv_path, kwargs in [
                ("{}/optuna_results_{}_dnn_{}_{}.csv.gz".format(
                    dir_name, databaseName, embeddingSize, incNone),
                 {"compression": "gzip"}),
                ("{}/optuna_results_{}_dnn_{}_{}.csv".format(
                    dir_name, databaseName, embeddingSize, incNone),
                 {}),
            ]:
                if os.path.exists(csv_path):
                    print("  Loading best params from CSV:", csv_path)
                    df_csv = pd.read_csv(csv_path, **kwargs)
                    return df_csv.loc[df_csv['value'].idxmax()].to_dict()
            raise FileNotFoundError(
                "No optuna CSV found in {} for {}_{}_{})".format(
                    dir_name, databaseName, embeddingSize, incNone))

        sampler_study = optuna.samplers.TPESampler(seed=42)
        try:
            study = optuna.create_study(
                sampler=sampler_study, directions=["maximize"],
                study_name=study_name_custom, storage=storage_url,
                load_if_exists=True)
            completed = [t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE]
            if len(completed) > 0:
                best_params = study.best_params
                print("Best params loaded from .db:", best_params)
            else:
                print("No completed trials in .db, falling back to CSV...")
                best_params = _load_best_params_from_csv(
                    dir_name, databaseName, embeddingSize, incNone)
                print("Best params loaded from CSV:", best_params)
        except Exception as e:
            print("  WARNING: .db corrupted or inaccessible ({}), "
                  "falling back to CSV...".format(e))
            best_params = _load_best_params_from_csv(
                dir_name, databaseName, embeddingSize, incNone)
            print("Best params loaded from CSV:", best_params)

        # --- Build B_dict and scaler ---
        B_dict, scaler = build_B_dict(incNone, x.shape[1], ffnnOrder, device)

        # --- Retrain on full outer fold ---
        print("\nRetraining on outer fold {}...".format(outterKFold))
        trainOUT = TensorDataset(x_train_outter, y_train_outter)
        testsOUT = TensorDataset(x_tests_outter, y_tests_outter)


        # ── Memory cap: prevent OOM for large models on big datasets ──────────
        # Measure actual GPU memory AFTER loading all data tensors
        # (fingerprints ~3GiB + identity B matrix ~4GiB for large MFP+FFNN).
        # Using memory_allocated() here gives the true remaining budget.
        _input_dim    = x.shape[1]
        _neurons_raw  = 2 ** int(best_params['neurons'])
        _batch_raw    = 2 ** int(best_params['batch_size'])
        _scale        = 1e-2 * best_params['scale_factor_neurons']
        _num_layers   = int(best_params['num_layers'])

        torch.cuda.synchronize()
        _used_gb         = torch.cuda.memory_allocated(device) / (1024 ** 3)
        _total_gpu_gb    = torch.cuda.get_device_properties(device).total_memory                            / (1024 ** 3)
        _model_budget_gb = _total_gpu_gb - _used_gb - 2.0
        if _input_dim >= 16384 and _neurons_raw > 8192:
            print("  Hard cap: input_dim={} forces neurons {} -> 8192, batch -> 128".format(
                  _input_dim, _neurons_raw))
            _neurons_raw  = 8192
            _batch_capped = 128
        print("  GPU used by data: {:.2f} GiB, model budget: {:.2f} GiB".format(
              _used_gb, _model_budget_gb))

        def _est_mem_gb(n0, nl, sf):
            total = 0
            cur = _input_dim
            nxt = n0
            for _ in range(nl):
                total += cur * nxt + nxt
                cur = nxt
                nxt = max(1, int(nxt * sf))
            total += cur + 1
            return (total * 4 * 3) / (1024 ** 3)

        _mem_est = _est_mem_gb(_neurons_raw, _num_layers, _scale)

        if _mem_est > _model_budget_gb:
            _orig_neurons = _neurons_raw
            while _neurons_raw > 512 and \
                  _est_mem_gb(_neurons_raw, _num_layers, _scale) > _model_budget_gb:
                _neurons_raw = _neurons_raw // 2
            _batch_capped = min(_batch_raw, 128)
            print("  WARNING: neurons reduced {} -> {} to fit {:.1f} GiB "
                  "(model was {:.2f} GiB)".format(
                  _orig_neurons, _neurons_raw, _model_budget_gb, _mem_est))
        else:
            _batch_capped = _batch_raw

        torch.cuda.empty_cache()
        import gc; gc.collect()
        # ─────────────────────────────────────────────────────────────────────
        outputs_external = train_model(
            int(best_params['num_layers']),
            _neurons_raw,
            x.shape[1],
            1e-7 * best_params['learning_rate'],
            1000,
            scaler,
            B_dict[incNone],
            trainOUT,
            testsOUT,
            _batch_capped,
            device,
            1e-2 * best_params['scale_factor_neurons'],
            databaseName,
            embeddingSize,
            dir_name,
            ffnnOrder,
            incNone)

        outter_k_fold_model         = outputs_external[0]
        outter_k_fold_results_dict  = outputs_external[1]
        outter_k_fold_best_R2_train = outputs_external[2]
        outter_k_fold_best_R2_valid = outputs_external[3]
        outter_k_fold_best_epoch    = outputs_external[4]

        kp = (outter_k_fold_best_R2_train + outter_k_fold_best_R2_valid
              - np.abs(outter_k_fold_best_R2_train - outter_k_fold_best_R2_valid))

        print('R2 Score Train:  {:.5f}'.format(outter_k_fold_best_R2_train))
        print('R2 Score Valid:  {:.5f}'.format(outter_k_fold_best_R2_valid))
        print('Metric Kappa:    {:.5f}'.format(kp))

        # --- Save model and training results ---
        torch.save(outter_k_fold_model.state_dict(), model_path)
        print("Model saved to:", model_path)

        df_fold = pd.DataFrame(outter_k_fold_results_dict)

        # ── Free training tensors after training to recover GPU memory ────────
        # The model (outter_k_fold_model) stays in GPU memory intentionally.
        # We delete all training/validation tensors, optimizer states and
        # intermediate outputs that are no longer needed before loading the
        # external validation set, which can be large for nfa/qm9.
        del trainOUT, testsOUT
        del x_train_outter, y_train_outter, x_tests_outter, y_tests_outter
        del outputs_external
        import gc; gc.collect()
        torch.cuda.empty_cache()
        print("  GPU memory after freeing train tensors: {:.2f} GiB allocated".format(
              torch.cuda.memory_allocated(device) / (1024 ** 3)))
        # ─────────────────────────────────────────────────────────────────────

        # --- Preprocess external validation set ---
        print("\nPreprocessing external validation set...")
        df_val = pd.read_csv(file_validation, compression='gzip')
        df_val = df_val.loc[:, ~df_val.columns.str.startswith('Unnamed')]
        x_val, y_val = preprocess_dataframe(
            df_val, descNormal_file_validation, withDescriptors,
            encodingMethod, embeddingSize, radiusMFP, nBitsMFP)
        smiles_val = df_val[df_val['lumo'] > df_val['homo']]['smiles'].tolist()

        val_dataset = TensorDataset(x_val, y_val)

        # --- Evaluate on external validation set ---
        print("\nEvaluating on external validation set...")
        ext_metrics, y_real_np, y_pred_np = evaluate_model_on_dataset(
            outter_k_fold_model, val_dataset, device, B=B_dict[incNone])

        print("External MAE:   {:.5f}".format(ext_metrics["Test_mae"][0]))
        print("External MSE:   {:.5f}".format(ext_metrics["Test_mse"][0]))
        print("External MdAE:  {:.5f}".format(ext_metrics["Test_mdae"][0]))
        print("External R2:    {:.5f}".format(ext_metrics["Test_r2_all"][0]))

        # --- Tanimoto distances for test molecules ---
        smiles_train_list = pd.read_csv(file, compression='gzip')
        smiles_train_list = smiles_train_list.loc[:, ~smiles_train_list.columns.str.startswith('Unnamed')]
        smiles_train_list = smiles_train_list[smiles_train_list['lumo'] > smiles_train_list['homo']]['smiles'].tolist()
        tanimoto_min, tanimoto_max = compute_tanimoto_distances(
            smiles_train_list, smiles_val, radiusMFP, nBitsMFP)

        # --- Save per-molecule CSV ---
        save_per_molecule_csv(smiles_val, y_real_np, y_pred_np,
                              tanimoto_min, tanimoto_max, dir_name)

        # --- Save training results only ---
        results_path = "{}/results_testing.csv.gz".format(dir_name)
        df_fold.to_csv(results_path, index=False, compression='gzip')
        print("\nResults (training) saved to:", results_path)
        save_best_epoch_summary(results_path, ext_metrics, dir_name, n_molecules=len(x_val))
        evaluate_closest_subset(dir_name, databaseName, results_path)
        print("DONE")

    # ========================================================================================================
    # SCENARIO 3: Neither .db nor model_testing.pth exist
    #   → Normal full training flow (original code)
    # ========================================================================================================
    else:
        print("\n[SCENARIO 3] No checkpoint found. Running full nested CV + Optuna optimization.\n")

        # ---- Metrics instances ----
        maeLoss   = MAELoss()
        mseLoss   = MSELoss()
        medaeLoss = MedianAELoss()
        r2_scoring = R2Score()

        start_time = time.time()

        # ---- Load training data ----
        df = pd.read_csv(file, compression='gzip')

        idxs = np.where(df['lumo'] > df['homo'])[0]
        auxZ = np.zeros(len(df), dtype=bool)
        for i in idxs:
            auxZ[i] = True
        df = df[auxZ]

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

        process       = psutil.Process(os.getpid())
        start_memory  = process.memory_info().rss

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

        x_train_outter, y_train_outter, x_tests_outter, y_tests_outter = get_fold(
            x, y, outterKFold - 1, 5, seed=42)

        if int_ext_case == 'internal':
            db_file_path = glob.glob('{}/*.db'.format(dir_name))

            sampler_study = optuna.samplers.TPESampler(seed=42)
            results_df    = pd.DataFrame(columns=['trial_number', 'value', 'train_time'])

            try:
                study = optuna.create_study(
                    sampler=sampler_study, directions=["maximize"],
                    study_name=study_name_custom, storage=storage_url,
                    load_if_exists=True)
                db_corrupted = False
            except Exception as e:
                print("  WARNING: .db corrupted or inaccessible ({}), "
                      "will read best params from CSV after optimization.".format(e))
                db_corrupted = True
                study = optuna.create_study(
                    sampler=sampler_study, directions=["maximize"],
                    study_name=study_name_custom)

            total_trials    = 150
            existing_trials = len(study.trials) if (db_file_path and not db_corrupted) else 0
            trials_to_run   = total_trials - existing_trials

            study.optimize(
                lambda trial: objective_with_time(
                    trial, x_train_outter, y_train_outter,
                    int(ffnnOrder), incNone, x.shape[1], dir_name, databaseName, embeddingSize),
                n_trials=trials_to_run,
                callbacks=[save_intermediate_results])

            if not db_corrupted:
                best_params_trial = study.best_params
                print('Best parameters:', best_params_trial)
            else:
                print("  Falling back to CSV for best params (db was corrupted)...")
                best_params_trial = _load_best_params_from_csv(
                    dir_name, databaseName, embeddingSize, incNone)
                print('Best parameters from CSV:', best_params_trial)
            print('TRAINING ON EXTERNAL FOLD {}'.format(outterKFold))

        else:
            print('Loading Best Params for External OutterFold')
            df_trial_information = pd.read_csv(
                "{}/optuna_results_{}_dnn_{}_{}.csv".format(
                    dir_name, databaseName, embeddingSize, incNone),
                compression='gzip')
            best_params_trial = df_trial_information.loc[
                df_trial_information['value'].idxmax()]

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', -1)

        print('Best parameters:', best_params_trial)

        B_dict, scaler = build_B_dict(incNone, x.shape[1], ffnnOrder, device)

        trainOUT = TensorDataset(x_train_outter, y_train_outter)
        testsOUT = TensorDataset(x_tests_outter, y_tests_outter)


        # ── Memory cap: prevent OOM for large models on big datasets ──────────
        # Measure actual GPU memory AFTER loading all data tensors
        # (fingerprints ~3GiB + identity B matrix ~4GiB for large MFP+FFNN).
        # Using memory_allocated() here gives the true remaining budget.
        _input_dim    = x.shape[1]
        _neurons_raw  = 2 ** int(best_params_trial['neurons'])
        _batch_raw    = 2 ** int(best_params_trial['batch_size'])
        _scale        = 1e-2 * best_params_trial['scale_factor_neurons']
        _num_layers   = int(best_params_trial['num_layers'])

        torch.cuda.synchronize()
        _used_gb         = torch.cuda.memory_allocated(device) / (1024 ** 3)
        _total_gpu_gb    = torch.cuda.get_device_properties(device).total_memory                            / (1024 ** 3)
        _model_budget_gb = _total_gpu_gb - _used_gb - 2.0
        if _input_dim >= 16384 and _neurons_raw > 8192:
            print("  Hard cap: input_dim={} forces neurons {} -> 8192, batch -> 128".format(
                  _input_dim, _neurons_raw))
            _neurons_raw  = 8192
            _batch_capped = 128
        print("  GPU used by data: {:.2f} GiB, model budget: {:.2f} GiB".format(
              _used_gb, _model_budget_gb))

        def _est_mem_gb(n0, nl, sf):
            total = 0
            cur = _input_dim
            nxt = n0
            for _ in range(nl):
                total += cur * nxt + nxt
                cur = nxt
                nxt = max(1, int(nxt * sf))
            total += cur + 1
            return (total * 4 * 3) / (1024 ** 3)

        _mem_est = _est_mem_gb(_neurons_raw, _num_layers, _scale)

        if _mem_est > _model_budget_gb:
            _orig_neurons = _neurons_raw
            while _neurons_raw > 512 and \
                  _est_mem_gb(_neurons_raw, _num_layers, _scale) > _model_budget_gb:
                _neurons_raw = _neurons_raw // 2
            _batch_capped = min(_batch_raw, 128)
            print("  WARNING: neurons reduced {} -> {} to fit {:.1f} GiB "
                  "(model was {:.2f} GiB)".format(
                  _orig_neurons, _neurons_raw, _model_budget_gb, _mem_est))
        else:
            _batch_capped = _batch_raw

        torch.cuda.empty_cache()
        import gc; gc.collect()
        # ─────────────────────────────────────────────────────────────────────
        outputs_external = train_model(
            int(best_params_trial['num_layers']),
            _neurons_raw,
            x.shape[1],
            1e-7 * best_params_trial['learning_rate'],
            1000,
            scaler,
            B_dict[incNone],
            trainOUT,
            testsOUT,
            _batch_capped,
            device,
            1e-2 * best_params_trial['scale_factor_neurons'],
            databaseName,
            embeddingSize,
            dir_name,
            ffnnOrder,
            incNone)

        outter_k_fold_model         = outputs_external[0]
        outter_k_fold_results_dict  = outputs_external[1]
        outter_k_fold_best_R2_train = outputs_external[2]
        outter_k_fold_best_R2_valid = outputs_external[3]
        outter_k_fold_best_epoch    = outputs_external[4]

        kp = (outter_k_fold_best_R2_train + outter_k_fold_best_R2_valid
              - np.abs(outter_k_fold_best_R2_train - outter_k_fold_best_R2_valid))

        print('SAVING MODEL TESTED ON {} FOLD'.format(outterKFold))
        torch.save(outter_k_fold_model.state_dict(), model_path)

        print('SAVING METRICS EVALUATED ON {} FOLD'.format(outterKFold))
        df_fold = pd.DataFrame(outter_k_fold_results_dict)

        # ── Free training tensors after training to recover GPU memory ────────
        del trainOUT, testsOUT
        del x_train_outter, y_train_outter, x_tests_outter, y_tests_outter
        del outputs_external
        import gc; gc.collect()
        torch.cuda.empty_cache()
        print("  GPU memory after freeing train tensors: {:.2f} GiB allocated".format(
              torch.cuda.memory_allocated(device) / (1024 ** 3)))
        # ─────────────────────────────────────────────────────────────────────

        # --- Preprocess and evaluate on external validation set ---
        print("\nPreprocessing external validation set...")
        df_val = pd.read_csv(file_validation, compression='gzip')
        x_val, y_val = preprocess_dataframe(
            df_val, descNormal_file_validation, withDescriptors,
            encodingMethod, embeddingSize, radiusMFP, nBitsMFP)
        smiles_val = df_val[df_val['lumo'] > df_val['homo']]['smiles'].tolist()

        val_dataset = TensorDataset(x_val, y_val)

        print("\nEvaluating on external validation set...")
        ext_metrics, y_real_np, y_pred_np = evaluate_model_on_dataset(
            outter_k_fold_model, val_dataset, device, B=B_dict[incNone])

        print("External MAE:   {:.5f}".format(ext_metrics["Test_mae"][0]))
        print("External MSE:   {:.5f}".format(ext_metrics["Test_mse"][0]))
        print("External MdAE:  {:.5f}".format(ext_metrics["Test_mdae"][0]))
        print("External R2:    {:.5f}".format(ext_metrics["Test_r2_all"][0]))

        # --- Tanimoto distances for test molecules ---
        smiles_train_list = df['smiles'].tolist()
        tanimoto_min, tanimoto_max = compute_tanimoto_distances(
            smiles_train_list, smiles_val, radiusMFP, nBitsMFP)

        # --- Save per-molecule CSV ---
        save_per_molecule_csv(smiles_val, y_real_np, y_pred_np,
                              tanimoto_min, tanimoto_max, dir_name)

        # Save training results only
        results_path = "{}/results_testing.csv.gz".format(dir_name)
        df_fold.to_csv(results_path, index=False, compression='gzip')
        save_best_epoch_summary(results_path, ext_metrics, dir_name, n_molecules=len(x_val))
        evaluate_closest_subset(dir_name, databaseName, results_path)

        print('Model trained until epoch {}'.format(outter_k_fold_best_epoch))
        print('R2 Score Train:  {:.5f}'.format(outter_k_fold_best_R2_train))
        print('R2 Score Valid:  {:.5f}'.format(outter_k_fold_best_R2_valid))
        print('Metric Kappa:    {:.5f}'.format(kp))
        print('\nResults saved to:', results_path)
        print('DONE')

