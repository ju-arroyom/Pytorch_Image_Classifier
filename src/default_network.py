
from torch import nn
import torch.nn.functional as F
class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drops=None):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drops: a list of floats between 0 and 1, dropout probability for layers
        '''
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        if drops:
            self.dropout = nn.ModuleList([nn.Dropout(p=drop) for drop in drops])
        else:
            self.dropout = None
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for index, linear in enumerate(self.hidden_layers):
            x = F.relu(linear(x))
            if self.dropout:
                x = self.dropout[index](x)
        
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)