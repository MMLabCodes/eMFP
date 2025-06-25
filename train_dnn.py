
import os
import sys

import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

from tqdm import tqdm
import psutil


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

model_name = 'DNN'

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    sizeemb = int(args.size )
else:
    sizeemb = 1


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
    dir_name = f'Models/modelsNFA/{model_name}_{incNone}_{sizeemb}'
elif 'qm9' in head_tail[1]:
    iddb = 'qm9'
    dir_name = f'Models/modelsQM9/{model_name}_{incNone}_{sizeemb}'
elif 'reddb' in head_tail[1]:
    iddb = 'rdb'
    dir_name = f'Models/modelsRedDB/{model_name}_{incNone}_{sizeemb}'

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

# Seed fix to 42, to keep reproducibility, 40 for QM9 emfp(128)
set_seed(40)





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

mfp  = calculate_morgan_fingerprints( mols, radius = int(args.rd), nbits = int(args.nB) ) # Time Consuming
print('Time obtaining MFP:', time.time() - time_loading_mfp)








if args.emfp:
    print('Obtaining eMFP')
    rmfp = torch.tensor( convert_fp_to_embV2( mfp, int(args.size) ), dtype = torch.float32 )












print("Concatenating MFP and descriptors")
if args.mfp:
    mfp = torch.from_numpy( mfp ).float()
    x = torch.hstack(( mfp, norm_desc ))
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


# Obtaining dataloader
print('Preparing dataloader')
dataset = TensorDataset(x, y)

# Split Training/Validation percentage 80/20
print('Split Training/Validation percentage 80/20')
train_size = int( 0.8 * len(dataset) )  # 0.8 -> 80 %
val_size = len(dataset) - train_size
train, valid = random_split( dataset, [ train_size, val_size ] )

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


def input_mapping(x, B, device):
    if B is None:
        return x.to(device)
    else:
        sin_list, cos_list = [], []
        x_proj = torch.matmul(2. * torch.pi * x, B.T).to(device)
        for ord in range( int( args.order ) ):
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



def train_model(num_layers, neurons , input_dim, learning_rate, epochs, scaler, B, train_data, valid_data, batch_size, device, scale_factor_neurons, iddb, sizeemb, dir_name, order, incNone ):
    
    print('Training parameters:')
    print('num_layers', num_layers)
    print('neurons ', neurons )
    print('learning_rate', learning_rate)
    print('epochs', epochs)
    print('batch_size', batch_size)
    print('scale_factor_neurons', scale_factor_neurons)
    

    # train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True, worker_init_fn=lambda _: set_seed(42))
    # valid_loader = DataLoader(valid_data, batch_size = batch_size, shuffle=True, worker_init_fn=lambda _: set_seed(42))
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_data, batch_size = batch_size, shuffle=True, num_workers=2)
    
    torch.manual_seed(40)

    print('train valid shapes, Train Function', len(train_data), len(valid_data))

    print('Model architechture')
    model = FSNN(input_dim * scaler, num_layers, neurons, scale_factor_neurons).to(device)

    print('Model architechture')
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate )
    # criterion = nn.MSELoss()
    # criterion = nn.HuberLoss( delta = 1.35)
    criterion = LogCoshLoss()
    # criterion = nn.GaussianNLLLoss( )
    criterion_mae = nn.L1Loss()

    train_losses = []
    maeT_losses = []
    r2T_list = []
    val_losses = []
    maeV_losses = []
    r2V_list = []
    ksT_list, ksV_list = [], []
    pT_list , pV_list  = [], []
    metric_stop = []
    timeEpochList = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        mae_running_loss = 0.0
        r2_running = 0.0
        ksrun = 0.0
        ptrun = 0.0
        mstop = 0.0
        timeEpoch = time.time()
        
        for current_batch, dataT in enumerate(train_loader):
            x_batch, y_batch = dataT
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_mapped = input_mapping(x_batch, B, device)

            optimizer.zero_grad()

            outputs = model(x_mapped)
            # outputs_mean, outputs_var = model(x_mapped)


            # Performs Kolmogorov-Smirnov test between xreal and xpredicted
            ksT, pT = metricKS(y_batch, outputs )
            # ksT, pT = metricKS(y_batch, outputs_mean )
            ksrun += ksT
            ptrun += pT
            
            # MAE
            mae = criterion_mae(outputs, y_batch)
            # mae = criterion_mae(outputs_mean, y_batch)

            # Calculate R^2
            try:
                r2 = r2_score(y_batch.cpu().detach().numpy(), outputs.cpu().detach().numpy() )
                # r2 = r2_score(y_batch.cpu().detach().numpy(), outputs_mean.cpu().detach().numpy() )
            except:
                r2 = -100.0

            loss = criterion(outputs, y_batch) #+ mae + mstop
            # loss = criterion(outputs_mean, y_batch, outputs_var) #+ mae

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            mae_running_loss += mae.item()
            r2_running += r2
            print(f'Epoch:\t{epoch}, Batch:\t{current_batch}/{len(train_loader)}, Loss: {loss.item():.3f}, MAE: {mae:.3f}, R2: {r2:.3f}, KST: {ksT:.3f} pT: {pT:.3f}')
            
        ksT_list.append(ksT/ len(valid_loader))
        pT_list.append(pT/ len(valid_loader))
        train_losses.append(running_loss / len(train_loader))
        maeT_losses.append(mae_running_loss / len(train_loader))
        r2T_list.append(r2_running / len(train_loader))

        model.eval()
        val_loss = 0.0
        maeV_running = 0.0
        r2V_running = 0.0
        ksVrun = 0.0 
        pVrun = 0.0

        fplt1 = []
        fplt2 = []

        with torch.no_grad():
            for batchV, dataV in enumerate(valid_loader):
                x_batch, y_batch = dataV
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                x_mapped = input_mapping(x_batch, B, device)

                outputs = model(x_mapped)
                # outputs_mean, outputs_var = model(x_mapped)

                # Performs Kolmogorov-Smirnov test between xreal and xpredicted
                ksV, pV = metricKS(y_batch, outputs )
                # ksV, pV = metricKS(y_batch, outputs_mean )
                ksVrun += ksV 
                pVrun += pV

                # MAE
                maeV = criterion_mae(outputs, y_batch)

                # if epoch %10 == 0:
                #     fplt1 = fplt1 + outputs.tolist() 
                #     fplt2 = fplt2 + y_batch.tolist() 

                # maeV = criterion_mae(outputs_mean, y_batch)
    
                # Calculate R^2
                try:
                    r2V = r2_score(y_batch.cpu().detach().numpy(), outputs.cpu().detach().numpy() )
                    # r2V = r2_score(y_batch.cpu().detach().numpy(), outputs_mean.cpu().detach().numpy() )
                except:
                    r2V = -99.99
                lossV = criterion(outputs, y_batch) #+ maeV
                # lossV = criterion(outputs_mean, y_batch, outputs_var)# + maeV

                val_loss += lossV.item()
                maeV_running += maeV.item()
                r2V_running += r2V

                if epoch < 5:
                    mstop += 3 
                else:
                    mstop += 0.1*abs(ksV) + 0.1*abs(1 - pV) + 0.8*abs(1 - r2V)


                print(f'Epoch:\t{epoch}, Batch:\t{batchV}/{len(valid_loader)}, LossV: {lossV.item():.3f}, MAE_V: {maeV:.3f}, R2_V: {r2V:.3f}, KSV: {ksV:.3f} pV: {pV:.3f}')
            # if epoch % 10 == 0:
            #     # plotPred( torch.tensor(fplt1), torch.tensor(fplt2), epoch )
            #     print(f'LenTarget {len(fplt2)}, LenPred {len(fplt1)}, meanTarget {np.mean(fplt2)}, mean pred {np.mean(fplt1)}')
        
        timeEpochList.append(time.time() - timeEpoch)
        metric_stop.append(mstop / len(valid_loader))
        ksV_list.append(ksV / len(valid_loader))
        pV_list.append(pV / len(valid_loader))
        val_losses.append(val_loss / len(valid_loader))
        maeV_losses.append(maeV_running/len(valid_loader))
        r2V_list.append(r2V_running/len(valid_loader))


        print(f"Epoch [{epoch}/{epochs}], LossT: {train_losses[-1]:.4f}, LossV: {val_losses[-1]:.4f}, MAET: {maeT_losses[-1]:.4f}, MAEV: {maeV_losses[-1]:.4f}, R2T: {r2T_list[-1]:.4f}, R2V: {r2V_list[-1]:.4f},  KST: {ksT_list[-1]:.3f},  KSV: {ksV_list[-1]:.3f} 1-pT {pT_list[-1]:.3f}, 1-pV: {pV_list[-1]:.3f}, metricStop: {metric_stop[-1]:.3f}")
        print('')

        # Stop Training if model loss is lower than 0.0001
        if train_losses[-1] < 0.0001  and val_losses[-1] < 0.0001 and r2V_list[-1] > 0.925 and r2T_list[-1] > 0.925:
            print(f'Loss reached at epoch {epoch}: Training Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}')
            break

        # Stop Training if model has not improved in 30 epochs
        quality_increase = len(metric_stop) - np.argmin(metric_stop)
        quality_r2v = len(r2V_list) - np.argmax(r2V_list)

        # if quality_r2v == 0 and r2V_list[-1] > 0:
        #     strsize = str(sizeemb)
        #     if len(strsize) < 3:
        #         strsize = '0' * (3 - len(strsize)) + strsize
        #     if len(trialNumber) < 3:
        #         trialNumber = '0' * ( 3 - len(trialNumber) ) + trialNumber
        #     if len(str(epoch)) < len(str(epochs)):
        #         xep = '0' * ( len(epochs) - len(str(epoch)) )

            
        #     temp_name  = f'{dir_name}/modeltmp{iddb}_epoch_{xep}_size_{strsize}'
        #     temp_name += f'_o_{order}_{incNone}_lr_{learning_rate}_neurons_{neurons}'
        #     temp_name += f'_nLayers_{num_layers}_bs_{batch_size}_scale_factor_neurons_{scale_factor_neurons}.pth'
        #     print('Inside train quality R2V', temp_name )
        
        #     torch.save(model.state_dict(), temp_name  )
        #     print('Inside Train, model saved')



        print('Quality Increase', quality_increase, epoch)
        if quality_increase > 15 and train_losses[-1] < 0.075  and val_losses[-1] < 0.075 and r2V_list[-1] > 0.925 and r2T_list[-1] > 0.925: 
            print('Early stopping criteria')
            break

        if r2T_list[-1] > 0.92 and r2V_list[-1] > 0.92 and quality_increase > 15:
            print('Early stopping criteria, not improvement in R2 during 15 epoch:', r2V_list[-1], 'Current epoch:', epoch)
            break

        if np.abs(train_losses[-1]) > 100  and np.abs(val_losses[-1]) > 100 and epoch > 10:
            print('Early stopping criteria... Not Converging')
            print(f'Epoch {epoch}: Training Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}')
            break

        if should_stop_training(r2V_list, patience = 15):
            print('Early stopping, R2 starts to decrease', r2V_list[-31:-1])    
            break

        # patience_quality = 50
        # if quality_increase > patience_quality and should_stop_training(r2V_list, patience = 10):
        #     print('')
        #     print('Early Stop ...\n')
        #     print(f'Epoch: {epoch}. Quality does not increase in {patience_quality} epochs. ')
        #     print('')
        #     break

        # if r2V_list[-1] < 0.1 and epoch > 5:
        #     print('Early stopping criteria, not improvement in R2, Bad Model:', r2V_list[-1], 'epoch:', epoch)
        #     break

    return model, train_losses,  maeT_losses,  r2T_list, ksT_list, pT_list, val_losses,  maeV_losses,  r2V_list, ksV_list, pV_list, timeEpochList

# class StopIfAboveThresholdCallback:
#     def __init__(self, threshold):
#         self.threshold = threshold

#     def __call__(self, study, trial):
#         if trial.value > self.threshold:
#             study.stop()





class StopIfAboveThresholdCallback:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, study, trial):
        if trial.value is not None and not np.isnan(trial.value) and trial.value > self.threshold:
            study.stop()

def objective(trial, train_data, valid_data, order, incNone, xshape, dir_name, iddb, sizeemb, start_time):

    # Number of features to capture in Fourier Mapping
    mapping_size = xshape
    input_dim = xshape

    if incNone == 'none':
        k = 1
    else:
        k = 2

    epochs = 500

    max_power = int( input_dim      ).bit_length()
    min_power = int( input_dim // 2 ).bit_length()

    if 2**max_power < input_dim:
        max_power += 1

    if 2**min_power > input_dim//2 :
        min_power -= 2


    num_layers           =           trial.suggest_int('num_layers'   ,         3, 10,        step = 1 )   # 3, 10
    neurons              = int( 2 ** trial.suggest_int('neurons'      ,         4, max_power, step = 1 ) )
    batch_size           = int( 2 ** trial.suggest_int('batch_size'   ,         6, 10,        step = 1 ) ) #### 4:16, 5:32, 6:64, 7:128, 8:256, 9:512
    scale_factor_neurons =    1e-2 * trial.suggest_int('scale_factor_neurons', 50, 75,        step = 1 )
    learning_rate        =    1e-7 * trial.suggest_int('learning_rate',         1, 100000,    step = 1 )

    print("learning_rate ", learning_rate )
    print("neurons ", neurons       )
    print("num_layers ", num_layers    )
    print("batch_size ", batch_size    )
    print("scale_factor_neurons ", scale_factor_neurons )

    print('input_dim,mapping_size',input_dim,mapping_size)
    
    B_dict = {}
    if incNone == 'none':
        B_dict['none'] = None
        print('Bshape', B_dict[incNone])
    elif incNone == 'linear':
        B_dict['linear'] = torch.eye(input_dim, dtype = torch.float32).to(device)
        print('Bshape', B_dict[incNone].shape)
    elif incNone == 'gauss':
        B_dict['gauss'] = torch.normal(0, 1, size = (mapping_size, input_dim)).to(device)
        print('Bshape', B_dict[incNone].shape)
    else:
        B_dict['linear'] = torch.eye(input_dim, dtype = torch.float32).to(device)
        print('Bshape', B_dict[incNone].shape)
    
    outputs = {}
    
    for k, B in B_dict.items():
        start2 = time.time()
        print(f"Training with B: {k}")
        if k == None or k == 'none':
            scaler = 1 
        else:
            scaler = 2 * int(order)

        print('Scaler' , scaler)

        try:
            # try to run
            print('Training Function')
            outputs = train_model( num_layers, neurons , input_dim, learning_rate, epochs, scaler, B_dict[incNone], train_data, valid_data, batch_size, device, scale_factor_neurons,  iddb, sizeemb, dir_name, order, incNone)

            # model = outputs[0]
            # train_losses = outputs[1]
            # trainMAE_losses = outputs[2]
            # R2Train = outputs[3]
            # KSTrain = outputs[4]
            # PTrain = outputs[5]
            # valid_losses = outputs[6]
            # validMAE_losses = outputs[7]
            # R2Valid = outputs[8]
            # KSValid = outputs[9]
            # PValid = outputs[10]
            # timeEpochList = outputs[11]

            if np.isnan(outputs[8][-1]) or np.isnan(outputs[1][-1]) or np.isnan(outputs[6][-1]):
                outputTrain = -0.12
            else:
                # outputTrain = outputs[8][-1] - 0.1 * abs( outputs[1][-1] + outputs[6][-1] )/2 ##  R2V -  [( 10 % ) (LossT + LossV)/2]
                outputTrain = (outputs[3][-1] +  outputs[8][-1])  - 2 * np.abs( np.abs(outputs[3][-1]) -  np.abs(outputs[8][-1]) )

                if outputTrain < 0 :
                    outputTrain = -0.001 - np.exp(outputTrain)
                else:
                    # R2 train > 0, and, R2 valid > 0
                    temp_directory_path = dir_name + f'/optuna_models_{iddb}_{sizeemb}/'
                    if not os.path.exists(temp_directory_path):
                        os.makedirs(temp_directory_path)

                    temp_name  = f'{temp_directory_path}/model_{iddb}_size_{sizeemb}_trialNumber_{trial.number}_' + str(outputTrain)[2:6] 
                    temp_name += f'_o_{order}_{incNone}_lr_{learning_rate}_neurons_{neurons}'
                    temp_name += f'_nLayers_{num_layers}_bs_{batch_size}_scale_factor_neurons_{scale_factor_neurons}'

                    tmodel_name = temp_name + '.pth'
                    name_loss = temp_name + '.csv'
                    print(trial.number, temp_name)
                    torch.save(outputs[0].state_dict(), tmodel_name  ) ### model = outputs[0]

                    losses_file = {
                                    'Epoch' : list( range( len( outputs[1] ) ) ),
                                    'TimePerEpoch' : outputs[11], #timeEpochList
                                    'MSELossT' : outputs[1], # train_losses
                                    'MAELossT' : outputs[2], # trainMAE_losses
                                    'R2T'      : outputs[3], # R2Train
                                    'KSTrain'  : outputs[4], # KSTrain
                                    'PTrain'   : outputs[5], # PTrain
                                    'MSELossV' : outputs[6], # valid_losses
                                    'MAELossV' : outputs[7], # validMAE_losses
                                    'R2V'      : outputs[8], # R2Valid
                                    'KSValid'  : outputs[9], # KSValid
                                    'PValid'   : outputs[10], # PValid
                    }
                    lossdf = pd.DataFrame(losses_file)

                    lossdf.to_csv( name_loss, index = False)        


        except:
            # if it fails, return -0.1 
            outputTrain = - 0.1
        # model, train_losses, trainMAE_losses, R2Train, valid_losses, validMAE_losses, R2Valid = outputTrain
    print(f'Time { time.time() - start_time }' )
    return outputTrain





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
    results_df.to_csv(f"{dir_name}/optuna_results_{iddb}_{model_name}_{args.size}_{incNone}.csv", index=False)

    # Directories for saving plots (ensure they are defined)
    directory_path_png = f"{dir_name}/optuna_visualization_{iddb}_{model_name}_{sizeemb}_png/"
    directory_path_svg = f"{dir_name}/optuna_visualization_{iddb}_{model_name}_{sizeemb}_svg/"

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

def objective_with_time(trial, train_data, valid_data, order, incNone, xshape, dir_name, iddb, sizeemb):

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
    result = objective(trial, train_data, valid_data, order, incNone, xshape, dir_name, iddb, sizeemb, start_time)
    # result = objective(trial, x_train, x_test, y_train, y_test, order, incNone, model_name, start_time)
    train_time = time.time() - start_time
    trial.set_user_attr('train_time', train_time)

    return result



# custom_name  = f'_{output_name}_case_{case}_type_{incNone}_embSize_{sizeemb:03}_'


# # thresholdR2 = 0.95
# # stop_callback = StopIfAboveThresholdCallback(thresholdR2)
# print('')
# print(f'Optuna tuning: ffnn_gap_{output_name}')
# print('train valid shapes, before tunning', len(train), len(valid))
sampler = optuna.samplers.TPESampler(seed = 40)
study = optuna.create_study(sampler=sampler, directions=["maximize"], study_name=f"{model_name}_regression")

start_time = time.time()

study.optimize(lambda trial: objective_with_time(trial, train, valid, int(args.order), incNone, x.shape[1], dir_name, iddb, sizeemb), n_trials = 150, callbacks=[save_intermediate_results])

# Finally, print the best results
best_params_trial = study.best_params
print('Best parameters:', best_params_trial)



# ffnn_gap = optuna.create_study(sampler = sampler, directions = [ "maximize" ], study_name = f"r{output_name}_ffnn_gap")
# # ffnn_gap.optimize(lambda trial: objective(trial, train, valid, int(args.order), incNone, x.shape[1]), n_trials = 600,  callbacks=[stop_callback] )
# ffnn_gap.optimize(lambda trial: objective(trial, train, valid, int(args.order), incNone, x.shape[1], dir_name, iddb, sizeemb), n_trials = 200  )




































# custom_name  = f'_{output_name}_case_{case}_type_{incNone}_embSize_{sizeemb:03}_'


# # thresholdR2 = 0.95
# # stop_callback = StopIfAboveThresholdCallback(thresholdR2)
# print('')
# print(f'Optuna tuning: ffnn_gap_{output_name}')
# print('train valid shapes, before tunning', len(train), len(valid))
# sampler = optuna.samplers.TPESampler(seed = 44)
# ffnn_gap = optuna.create_study(sampler = sampler, directions = [ "maximize" ], study_name = f"r{output_name}_ffnn_gap")
# # ffnn_gap.optimize(lambda trial: objective(trial, train, valid, int(args.order), incNone, x.shape[1]), n_trials = 600,  callbacks=[stop_callback] )
# ffnn_gap.optimize(lambda trial: objective(trial, train, valid, int(args.order), incNone, x.shape[1], dir_name, iddb, sizeemb), n_trials = 200  )


# directory_path = dir_name + f'/optuna_visualization_{iddb}/'
# if not os.path.exists(directory_path):
#     os.makedirs(directory_path)


# # Save optuna visualization plots
# parameters_optuna = ["neurons", "num_layers", "learning_rate", "batch_size", "scale_factor_neurons"]
# pairs_vars = []

# for i in parameters_optuna:
#     for j in parameters_optuna:
#         if i != j:
#             p = list( set( [i, j]) ) 
#             if p not in pairs_vars:
#                 pairs_vars.append(p)

# cont_value = 0
# for iter, p in enumerate(pairs_vars):
#     cont_value = iter
#     fig = optuna.visualization.plot_contour(ffnn_gap, params=[ p[0], p[1]])
    
#     # Choose a differente color pallete
#     fig.update_traces(colorscale='Blackbody', selector=dict(type='contour'))  
#     # Smooth colormap 
#     fig.update_traces(line_smoothing=1.15, selector=dict(type='contour')) 
#     # Remove contour lines
#     fig.update_traces(line_width=0, selector=dict(type='contour')) 
#     # Update marker sizes and color
#     fig.update_traces(marker=dict(size=0.25, color="RoyalBlue"), selector=dict(mode='markers'))

#     fig.write_image( directory_path + str(iter) + custom_name + '_'+ str(p[0]) + '_'+ str(p[1]) + '_'+ "img_optuna.png")
#     fig.write_image( directory_path + str(iter) + custom_name + '_'+ str(p[0]) + '_'+ str(p[1]) + '_'+ "img_optuna.svg")
    
# print('cont_value',cont_value)
# fig = optuna.visualization.plot_param_importances(ffnn_gap)
# fig.update_layout(template='simple_white')
# fig.write_image( directory_path + str(cont_value + 1) + custom_name +  'img_optuna.png')
# fig.write_image( directory_path + str(cont_value + 1) + custom_name +  'img_optuna.svg')

# fig = optuna.visualization.plot_slice(ffnn_gap, params = parameters_optuna)
# fig.update_layout(template='simple_white')
# fig.write_image( directory_path + str( cont_value + 2 ) + custom_name +  'slice_img_optuna.png')
# fig.write_image( directory_path + str( cont_value + 2 ) + custom_name +  'slice_img_optuna.svg')


# best_params_trial = ffnn_gap.best_params
# best_value_trial  = ffnn_gap.best_trial.values[0]

# print('Final Best value:', best_value_trial)
# print('Final Best parameters:')
# print( best_params_trial)

# learning_rate = 1e-7 * float(best_params_trial['learning_rate'])
# neurons       =  2** int(best_params_trial['neurons'])
# num_layers    = int(best_params_trial['num_layers'])
# batch_size    = 2**int(best_params_trial['batch_size'])
# scale_factor_neurons       = 0.01* float(best_params_trial['scale_factor_neurons'])


# print('')
# print('Preparing Training')

# epochs = 500

# # Number of features to capture in Fourier Mapping
# mapping_size = x.shape[1]
# input_dim = x.shape[1]

# B_dict = {}
# if args.none:
#     B_dict['none'] = None
#     print('B shape', B_dict[incNone])
# elif args.linear:
#     B_dict['linear'] = torch.eye(input_dim, dtype = torch.float32).to(device)
#     print('B shape', B_dict[incNone].shape)
# elif args.gauss:
#     B_dict['gauss'] = torch.normal(0, 1, size = (mapping_size, input_dim)).to(device)
#     print('B shape', B_dict[incNone].shape)
# else:
#     B_dict['linear'] = torch.eye(input_dim, dtype = torch.float32).to(device)
#     print('B shape', B_dict[incNone].shape)

# outputs = {}

# for k, B in B_dict.items():
#     start2 = time.time()
#     print(f"Training with B: {k}")
#     if k == None or k == 'none':
#         scaler = 1 
#     else:
#         scaler = 2 * int(args.order)

# custom_name += f'lr_{learning_rate}_layer_{neurons}_nL_{num_layers}_bs_{batch_size}_dp_{scale_factor_neurons}'

# name_model   = f'model' + custom_name + '.pth'
# name_loss    = f'loss' + custom_name + '.csv'
# params_name = 'best_params' + custom_name + '.csv'
                                        
# if best_value_trial > -0.1:

#     print('Scaler' , scaler)
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         torch.cuda.set_device(0)
#         torch.cuda.reset_max_memory_allocated(device=device)  # Reset memory tracker
#         torch.cuda.synchronize()
#         # Keep track of memory usage before the model is trained
#         start_memory = torch.cuda.max_memory_allocated(device=device)
#     else:
#         device = torch.device("cpu")
#         # Keep track of memory usage before the model is trained
#         process = psutil.Process(os.getpid())
#         start_memory = process.memory_info().rss
    
    
    
#     model, train_losses, trainMAE_losses, R2Train, KSTrain, PTrain, valid_losses, validMAE_losses, R2Valid, KSValid, PValid, timeEpochList = train_model( num_layers, neurons , input_dim, learning_rate, epochs, scaler, B_dict[incNone], train, valid, batch_size, device, scale_factor_neurons,  iddb, sizeemb, dir_name, int(args.order), incNone )
    
#     end_time = time.time()
    
#     if device == torch.device("cuda"):
#         torch.cuda.synchronize()  # Ensure all GPU operations are complete
#         # Keep track of memory usage after the model is trained
#         end_memory = torch.cuda.max_memory_allocated(device=device)
#     else:
#         process = psutil.Process(os.getpid())
#         end_memory = process.memory_info().rss
#     if args.order ==1:
#         x = 0
#     else:
#         x=1
#     print('train_losses',len(train_losses))
#     print('trainMAE_losses',len(trainMAE_losses))
#     print('R2Train',len(R2Train))
#     print('KSTrain',len(KSTrain))
#     print('PTrain',len(PTrain))
#     print('valid_losses',len(valid_losses))
#     print('validMAE_losses',len(validMAE_losses))
#     print('R2Valid',len(R2Valid))
#     print('KSValid',len(KSValid))
#     print('PValid',len(PValid))
#     print('TimeEpoch',len(timeEpochList))
#     losses_file = {'Epoch':[i for i in range(len(train_losses))]}
#     losses_file.update({
#                'TimePerEpoch' : timeEpochList,
#                'MSELossT'     : train_losses,
#                'MAELossT'     : trainMAE_losses,
#                'R2T'          : R2Train,
#                'KST'          : KSTrain, 
#                'PT'           : PTrain,
#                'MSELossV'     : valid_losses,
#                'MAELossV'     : validMAE_losses,
#                'R2V'          : R2Valid,
#                'KSV'          : KSValid, 
#                'PV'           : PValid,
#     })
   
#     torch.save(model.state_dict(), dir_name + '/' + name_model)
#     final_time_train = time.time()
#     print(f'Training Time with B {k}: {final_time_train - start2} seconds')
#     print('Model Saved:', name_model)
    
#     print(f'Total running time: {final_time_train- start_time} seconds')
#     print('Max Memory Allocated:', end_memory - start_memory, "bytes", (end_memory - start_memory)/1024**2, 'MB')
    
#     lossdf = pd.DataFrame(losses_file)
#     lossdf.to_csv(dir_name + '/' + name_loss, index = False)

#     best_params_dict = {
#                     "learning_rate" : learning_rate,
#                     "neurons" : neurons,
#                     "num_layers" : num_layers,
#                     "batch_size" : batch_size,
#                     "embSize": int(sizeemb),
#                     "scale_factor_neurons" : scale_factor_neurons,
#                     "Memory_Allocated": end_memory - start_memory,
#                     "Training_Time": final_time_train - start2,
#                     "R2T": R2Train[-1],
#                     "R2V": R2Valid[-1],
#                     }
#     dfParams = pd.DataFrame(best_params_dict, index=[0])
#     dfParams.to_csv( dir_name + '/' + params_name, index = False )

    
#     print('Loss file saved in :', dir_name + '/' + name_loss )
#     print('DirectoryModelLoss:', dir_name )

# else:
#     print('Run Again')