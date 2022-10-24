import torch
from torch import nn
import sklearn

class LinearNet(nn.Module):
    
    def __init__(self, input_dim:int, nbr_classes:int, layer_size:int= 128, dropout_rate:float=0.3, *args, **kwargs):
        super(LinearNet, self).__init__()
        layer_size = layer_size
        d_rate = dropout_rate

        self.fc = nn.Sequential(
            nn.Linear(input_dim, layer_size),
            nn.ReLU(),
            nn.Dropout(d_rate),
            nn.Linear(layer_size, nbr_classes)
        )
        
    def forward(self, x):
        logits = self.fc(x)
        return logits


class DeepNet(nn.Module):

    def __init__(self, input_dim:int, nbr_classes:int, dropout_rate:float=0.3, layer_sizes = None,
                 *args, **kwargs):
        super(DeepNet, self).__init__()
        if layer_sizes is None:    
            layer_sizes = []
            assert 'L1' in kwargs, 'If layer size is None, then L1 needs to be specified in kwargs.'
            i = 1
            while(f'L{i}' in kwargs):
                layer_sizes.append(kwargs[f'L{i}'])
                i += 1
        layer_sizes = [input_dim] + layer_sizes
            
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1])),
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(layer_sizes[-1], nbr_classes))    

        self.fc = nn.Sequential(
            *layers
        )

    def forward(self, x):
        logits = self.fc(x)
        return logits


