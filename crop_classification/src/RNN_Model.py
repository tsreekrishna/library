import pkg_resources
import pandas as pd
import pickle
from copy import deepcopy
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch
from .data_preprocess import get_model_data, conformal_prediction, generate_label_set_map

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, output_size):
        super(RNNModel, self).__init__()

        # Defining the number of laye
        # rs and the nodes in each layer
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
        # Initializing hidden state for first input with zeros
        
        # device = 'cpu'

        # x.to(device)
        
        batch_size = x.size(0)
        
        # h0 = torch.zeros(self.hidden_layers, batch_size, self.hidden_size).to(device)
        h0 = torch.zeros(self.hidden_layers, batch_size, self.hidden_size)
        
        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0)
        
        #Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        out = out[:,-1,:]
        out = self.relu(out)
        out = self.fc(out)
        out = self.softmax(out)

        return out
    
scaler_path = pkg_resources.resource_filename('crop_classification', 'src/models/RNN_standard_scaler.pkl')
# scaler_path = 'models/RNN_standard_scaler.pkl'
scaler = pickle.load(open(scaler_path, 'rb'))
params = {'hidden_layers': 1, 'hidden_size': 16, 'input_size': 8, 'output_size': 2}   
classifier = RNNModel(**params)
classifier_weights_path = pkg_resources.resource_filename('crop_classification', 'src/models/RNN_oct_2f-feb_1f.pth')
# classifier_weights_path = 'models/RNN_oct_2f-feb_1f.pth'
classifier.load_state_dict(torch.load(classifier_weights_path))

def batch_prediction_prob(data, n_features, batch_size, trained_classifier):
    tensor = torch.Tensor(data.values)
    data_loader = DataLoader(tensor, batch_size=batch_size)
    with torch.no_grad():
        pred_prob = []
        for batch in data_loader:
            batch = batch.view([batch.shape[0], -1, n_features])
            trained_classifier.eval()
            pred_prob.append(trained_classifier.forward(batch))

    pred_prob = np.vstack([np.array(pred_prob[:-1]).reshape(-1,2), pred_prob[-1]])
    return pred_prob

def point_predictions(data):
    '''
    Classifies the data into Wheat and Mustard

    Parameters
    -----------
    data : Data which has to fed to the classifier (Has to be in the form 
    of Pandas Dataframe)

    Returns
    -------
    point predictions based on max prob among all the classes

    '''
    scaled_data = pd.DataFrame(scaler.transform(data), columns=data.columns)
    scaled_data = scaled_data.loc[:, 'oct_2f':'feb_1f']
    # device = torch.device('cpu' if torch.cuda.is_available() else 'gpu')
    # device = torch.device('cpu')
    # classifier = classifier.to(device)
    batch_size = 8
    n_features = scaled_data.shape[1]

    pred_prob = batch_prediction_prob(scaled_data, n_features, batch_size, classifier)
    point_pred = np.argmax(pred_prob, axis=1)
    crop_map = {'0':'Mustard', '1':'Wheat'}
    labels = list(map(lambda label: crop_map[str(label)], point_pred))

    return pred_prob, labels

def conformal_predictions(data, alpha=0.05):
    '''
    Classifies the data into wheat, mustard, wheat/mustard or NONE. 

    Parameters
    -----------
    data : Data which has to fed to the classifier (Has to be in the form 
    of Pandas Dataframe)

    Returns 
    -------
    set predictions using conformal_predictions algorithm

    '''
    scaled_data = pd.DataFrame(scaler.transform(data), columns=data.columns)
    scaled_data = scaled_data.loc[:, 'oct_2f':'feb_1f']
    # device = torch.device('cpu' if torch.cuda.is_available() else 'gpu')
    # device = torch.device('cpu')
    # classifier = classifier.to(device)
    _, val, _ = get_model_data()
    X_cal, y_cal = val.drop('crop_name', axis=1), val['crop_name']
    scaled_X_cal = pd.DataFrame(scaler.transform(X_cal), columns=X_cal.columns)
    scaled_X_cal = scaled_X_cal.loc[:, 'oct_2f':'feb_1f']
    cp = conformal_prediction(classifier, 'RNN')
    cp.fit(scaled_X_cal, y_cal)
    _, sets = cp.predict(scaled_data, alpha)
    crop_map = {'0':'Mustard', '1':'Wheat'}
    crop_set_map = generate_label_set_map(crop_map)
    labels = list(map(lambda label: crop_set_map[label] if type(label) == str else label, sets))
    return labels 
    
if __name__ == '__main__':
    df = pd.read_csv('data_files/test-4.csv').drop('crop_name', axis=1)
    print(point_predictions(df.head(5)))
