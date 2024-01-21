import numpy as np

class InvalidInputError(Exception):
    pass

def data_check(arr):
    # Is the input a 2d array with 14 columns?""
    if not (isinstance(arr, list) or isinstance(arr, np.ndarray)):
        raise InvalidInputError("Input must be a list or a NumPy array.")
    if not ((isinstance(arr, list) and all(isinstance(row, list) for row in arr)) or (isinstance(arr, np.ndarray) and arr.ndim == 2)):
        raise InvalidInputError("Input must be a 2D array.")
    if not (len(arr) > 0):
        raise InvalidInputError("Empty 2D array")
    if not (len(arr[0]) == 14):
        raise InvalidInputError("Input must have exactly 14 columns")

def algo_check(algo):
    if algo not in {'xgb', 'rnn'}:
        raise InvalidInputError("Invalid algorithm name. Please choose from 'xgb' or 'rnn'")

def classifier_type_check(classifier_type):
    if classifier_type not in {'mw', 'mwp'}:
        raise InvalidInputError("Invalid classifier type. Please choose from 'mw' or'mwp'")
    
def batch_size_check(algo, batch_size, data):
    if algo == 'rnn':
        if not (isinstance(batch_size, int) and batch_size > 0):
            raise InvalidInputError("Batch size must be a positive integer")
        if not (batch_size <= len(data)):
            raise InvalidInputError("Batch size cannot be greater than the length of the input data")
        
def alphas_check(alphas):
    if not (isinstance(alphas, list) or isinstance(alphas, np.ndarray)):
        raise InvalidInputError("Input must be a list or a NumPy array.")
    if not ((isinstance(alphas, list) and all(isinstance(alpha, float) for alpha in alphas)) or (isinstance(alphas, np.ndarray) and alphas.ndim == 1)):
        raise InvalidInputError("Input must be a 1D array of floats.")
    if not (len(alphas) > 0):
        raise InvalidInputError("Empty 1D array")
    if not (all(alpha > 0 and alpha < 1 for alpha in alphas)):
        raise InvalidInputError("Input must be a 1D array of floats between 0 and 1.")