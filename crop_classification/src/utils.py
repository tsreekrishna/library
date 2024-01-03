from copy import deepcopy
from typing import Dict, List
from itertools import combinations
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd


def _generate_label_set_map(label_map: Dict[str, str]) -> Dict[str, str]:
    set_map = {}
    for i in range(1,len(label_map)+1):
        for comb in combinations(label_map.keys(), i):
            key_comb = ''
            value_comb = ''
            for items in comb:
                key_comb =  key_comb + ' ' + items
                value_comb =  value_comb + ' ' + label_map[items]
            key_comb, value_comb = key_comb.lstrip(), value_comb.lstrip()
            set_map[key_comb] = value_comb
    return set_map

def _dip_impute(row):
    lst = deepcopy(row)
    act_strt_idx = lst.index.get_loc('oct_2f')
    for i in range(len(lst.loc['oct_2f':'apr_1f'])):
        actual_idx = i + act_strt_idx
        if ((lst[actual_idx-1] - lst[actual_idx]) >= 20) and ((lst[actual_idx+1] - lst[actual_idx]) >= 20):
            lst[actual_idx] = (lst[actual_idx-1] + lst[actual_idx+1])/2
    return lst

def _sowing_period(row):
    sowing_periods = row.loc['oct_1f':'dec_2f'].index
    sowing_periods_NDVI = row.loc['oct_1f':'dec_2f']
    minima = np.argmin(sowing_periods_NDVI)
    ndvi_values = row.loc['oct_1f':'apr_2f']
    i = minima
    while i < len(sowing_periods):
        if ndvi_values[i] in set(np.arange(110, 141)):
            if (ndvi_values[i+1] - ndvi_values[i]) > 5:
                if ((ndvi_values[i+1] - ndvi_values[i+4]) < 30):
                    return sowing_periods[i]
        i += 1 
    return 'Unknown'

def _harvest_period(row):
    sowing_period_idx = row.index.get_loc(row['sowing_period'])
    i = sowing_period_idx + 6
    while i < len(row.loc[:'apr_2f']):
        if row[i] < 140:
            return row.index[i-1]
        i += 1
    return 'Unknown'

def _less_than_150_drop(row):
    sp_loc = row.index.get_loc(row['sowing_period'])
    hp_loc = row.index.get_loc(row['harvest_period'])
    if max(row.iloc[sp_loc+1:hp_loc]) < 150:
        return False
    return True

def _batch_prediction_prob(data, n_features, batch_size, trained_classifier):
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