import pandas as pd
import numpy as np
from typing import List
from copy import deepcopy
from crop_classification.src.utils.checks import data_check
from crop_classification.src.utils.constants import rabi_season_fns

class Rabi_NDVI_Preprocessor:
    '''
    Preprocess the input data and filter non-crop data points.

    Parameters
    ----------
    data : np.ndarray
        NDVI recorded every fortnight from the 1st fortnight 
        of October to 2nd fortnight of April (Must be a 2d array).
    '''
    def __init__(self, data: np.ndarray|List):
        # Checking parameters for exceptions
        data_check(data)
        self.fortnight_list = rabi_season_fns
        self.spectral_data = pd.DataFrame(data=data, columns=self.fortnight_list)
        # self.n_fns = self.spectral_data.shape[1]

    def preprocess(self) -> (pd.DataFrame, pd.DataFrame):
        '''
        Preprocess the input data and filter non-crop data points.

        Returns
        -------
        (pd.DataFrame, pd.DataFrame)
            Two pandas DataFrames: preprocessed crop data and non-crop outliers.
        '''
        # Initializing outliers dataframe
        self.outliers = pd.DataFrame(columns=self.fortnight_list)
        
        # Imputing the NDVI fortnights with the averages if the dip is greater than 20 when compared to both adjacents
        self.spectral_data = self.spectral_data.apply(self._dip_impute, axis=1)

        # Determining sowing period (S.P). If the S.P is not found, then it is regarded as a non-crop.
        self.spectral_data['sowing_period'] = self.spectral_data.apply(self._sowing_period, axis=1)
        new_outliers = self.spectral_data[self.spectral_data.sowing_period == 'Unknown']
        self.outliers = pd.concat([self.outliers, new_outliers])
        self.spectral_data.drop(new_outliers.index, inplace=True)

        # Determining harvest period (H.P). If the H.P is not found, then it is regarded as a non-crop.
        self.spectral_data['harvest_period'] = self.spectral_data.apply(self._harvest_period, axis=1)
        new_outliers = self.spectral_data[self.spectral_data.harvest_period == 'Unknown']
        self.outliers = pd.concat([self.outliers, new_outliers])
        self.spectral_data.drop(new_outliers.index, inplace=True)

        # Dropping the rows which have max NDVI values less than 150 for all the values between sowing and harvest period.
        new_outliers = self.spectral_data[self.spectral_data.apply(self._less_than_150_drop, axis=1) == False]
        self.outliers = pd.concat([self.outliers, new_outliers])
        self.spectral_data = self.spectral_data.drop(new_outliers.index)

        # Dropping the duplicates (if any)
        self.spectral_data = self.spectral_data.drop_duplicates()

        return self.spectral_data, self.outliers

    def _dip_impute(self, row):
        lst = deepcopy(row)
        act_strt_idx = lst.index.get_loc('oct_2f')
        for i in range(len(lst.loc['oct_2f':'apr_1f'])):
            actual_idx = i + act_strt_idx
            if ((lst[actual_idx-1] - lst[actual_idx]) >= 20) and ((lst[actual_idx+1] - lst[actual_idx]) >= 20):
                lst[actual_idx] = (lst[actual_idx-1] + lst[actual_idx+1])/2
        return lst

    def _sowing_period(self, row):
        sowing_periods = row.loc['oct_1f':'dec_2f'].index
        sowing_periods_NDVI = row.loc['oct_1f':'dec_2f']
        minima = np.argmin(sowing_periods_NDVI)
        ndvi_values = row.loc['oct_1f':'mar_1f']
        i = minima
        while i < len(sowing_periods):
            if 110 <= ndvi_values[i] <= 151:
                if (ndvi_values[i+1] - ndvi_values[i]) >= 5:
                    if i + 4 < len(ndvi_values):
                        if ((ndvi_values[i+4] - ndvi_values[i]) >= 25):
                            return sowing_periods[i]
                    else:
                        return sowing_periods[i]
            i += 1
        return 'Unknown'

    def _harvest_period(self, row):
        sowing_period_idx = row.index.get_loc(row['sowing_period'])
        i = sowing_period_idx + 6
        while i < len(row.loc[:'apr_2f']):
            if row[i] < 140:
                return row.index[i-1]
            i += 1
        return 'Unknown'

    def _less_than_150_drop(self, row):
        sp_loc = row.index.get_loc(row['sowing_period'])
        hp_loc = row.index.get_loc(row['harvest_period'])
        if max(row.iloc[sp_loc+1:hp_loc]) < 150:
            return False
        return True
