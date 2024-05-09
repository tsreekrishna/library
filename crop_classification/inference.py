from typing import List
import os
from glob import glob
import numpy as np
import pandas as pd
import geopandas as gp
import shapely
from huggingface_hub import hf_hub_download
from crop_classification.src.utils.helper import find_district
from crop_classification.src.raster_export import SpectralDataHarvester
from crop_classification.src.raster_extraction import SpectralDataExtractor
from crop_classification.src.preprocess import Rabi_NDVI_Preprocessor
from crop_classification.src.models import CropTypeClassifier
    

class CropTypeClassification(SpectralDataHarvester, SpectralDataExtractor, Rabi_NDVI_Preprocessor, CropTypeClassifier):

    def __init__(self, data_path: str, season: str) -> None:
        df = pd.read_csv(data_path)
        df['geometry'] = df['geometry'].apply(shapely.wkt.loads)
        self.data = gp.GeoDataFrame(data=df, geometry='geometry')
        self.data['geometry'] = self.data['geometry'].apply(lambda geom : geom.centroid if type(geom) == shapely.geometry.polygon.Polygon else geom)
        self.geom_list = self.data['geometry'] 
        self.districts = find_district(self.geom_list, hf_hub_download(repo_id='krishnamlexp/gt_curation_datasets', filename='state_district_boundaries_2020.geojson',  repo_type="dataset"))
        self.season = season
        
    def harvest_raster_data(self, data_type: str, end_fn: str, year: int, dir_path: str, storage_type: str, bucket_name: str = None) -> None:
        """
        Generate raster stack for each district fortnightly according to the specified season.

        Args:
            season (str): The season for which raster stacks are to be generated. Should be either 'rabi' or 'kharif'.
            end_fn (str): The end fortnight(exclusive)
            year (int): Year when the season started.
            dir_path (str): Directory path to export raster stacks.
            storage_type (str): Storage type (either 'drive' or 'cloud').
            bucket_name (str, optional): Bucket name if exporting to cloud storage.
        """
        return super().generate_district_spectral_data_perfn(data_type, self.season, end_fn, year, dir_path, storage_type, bucket_name)
        
    def extract_raster_data(self, raster_dir_path: str, data_type: str='ndvi') -> List[np.ndarray]:
        """
        Extract NDVI values for each point in the dataset.

        Args:
            data_type (str): The type of data to be extracted, either 'ndvi' or 'sar'. Defaults to 'ndvi'.
        Returns:
            List[np.ndarray]: A list of arrays containing the spectral data for each point in the dataset.
        """
        self.raster_path_list = glob(os.path.join(raster_dir_path, '*.tif'))
        spectral_data = super().extract_raster_values_for_dataset(data_type=data_type)
        return spectral_data
    
    def reject_ood(self, ndvi_data: List[np.ndarray]) -> (pd.DataFrame, pd.DataFrame):
        """
        Filter out-of-distribution data points that do not belong to crops or crop types other than wheat, Mustard and potato.

        Args:
            data (List[np.ndarray]): A list of arrays containing the spectral data for each point in the dataset.
        Returns:
            (pd.DataFrame, pd.DataFrame): A tuple of two dataframes, one containing the filtered dataset and the other containing the rejected data points.
        Note:
            This process currently works for NDVI data and rabi season only. Planning to extend it to other seasons and other spectral data types.
        """
        self.spectral_data = ndvi_data
        return super().preprocess()
    
    def crop_type_classification(self, end_fn: str, conformal: bool=False, alpha: float = 0.1) -> (np.ndarray, np.ndarray):
        """
        Classify each point in the dataset into one of the crop types.
        Args:
            end_fn (str): The last fortnight(inclusive) you want to use for classification.
            conformal (bool, optional): Whether to use conformal prediction. Defaults to False.
        Returns:
        (np.ndarray, np.ndarray): A tuple of two arrays, prediction probabilities of each crop type and model argmax/conformal predictions.
        """
        if conformal:
            return super().get_conformal_predictions(end_fn, alpha)
        else:
            return super().get_croptype_predictions(end_fn)