from copy import deepcopy
from typing import Dict, List
from itertools import combinations
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
import geopandas as gp

def generate_label_set_map(label_map: Dict[str, str]) -> Dict[str, str]:
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

def find_district(data, dist_poly_path):
    shp_file = gp.read_file(dist_poly_path)

    data.crs = "epsg:4326"
    data = data.to_crs("WGS84")

    shp_file = shp_file.to_crs("WGS84")

    districts = gp.sjoin(data, shp_file, how='inner', predicate='within')['dtname']
    return districts