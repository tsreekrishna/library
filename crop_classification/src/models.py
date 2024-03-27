from torch import nn
import torch

class RNNModel(nn.Module):
    '''
    Creating a class which encapsulate Simple RNN.
    '''
    def __init__(self, input_size, hidden_size, hidden_layers, output_size):
        super(RNNModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        # RNN layers
        self.rnn = nn.RNN(
            input_size, hidden_size, hidden_layers, batch_first=True
        )
        # Fully connected layer
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        '''
        Method function that performs forward pass.
        '''
        batch_size = x.size(0)

        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.hidden_layers, batch_size, self.hidden_size)
        
        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0)
        
        #Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        out = out[:,-1,:]
        out = self.relu(out)
        out = self.fc(out)
        out = self.softmax(out)

        return out