import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FSNN(nn.Module):
    def __init__(self, input_dim, num_layers, neurons, scale_factor ):
        super(FSNN, self).__init__()
        
        layers = []
        current_units = neurons

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else prev_units, current_units))
            if i == 1:
                layers.append(nn.Dropout( p = 0.2 ) )
            if i < num_layers - 2:
                layers.append(nn.PReLU(current_units))
            else:
                layers.append(nn.ReLU())

            prev_units = current_units
            current_units = max(2, 2**(int(current_units * scale_factor).bit_length() -1) )  
            if prev_units == 2:
                break

        layers.append(nn.Linear(prev_units, 1))  
        # layers.append(nn.ReLU())
        

        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

