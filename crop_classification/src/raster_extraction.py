from typing import List
import rasterio
from rasterio.windows import Window
import numpy as np
from glob import glob
import os
import pandas as pd
import geopandas as gp
from shapely import wkt
from helper import find_district
from constants import rabi_season_fns, kharif_season_fns

class SpectralDataExtractor:
    """A class for extracting spectral data (e.g., NDVI/VH values) from raster images."""
    
    def __init__(self, raster_dir_path: str, geom_list: List[str], ind_dist_shp: str):
        """
        Initialize the SpectralDataExtractor object.

        Args:
            raster_dir_path (str): The directory path containing the raster files.
        """
        self.raster_path_list = glob(os.path.join(raster_dir_path, '*.tif'))
        geom_list = pd.Series(geom_list).astype(str).apply(wkt.loads)
        self.data = gp.GeoDataFrame(geometry=geom_list)
        self.data['district'] = find_district(self.data, ind_dist_shp)
    
    def extract_raster_values_for_point(self, raster_path: str, point: gp.GeoSeries, season_fns: List[int]) -> np.ndarray:
        """
        Extract raster values for a given point from a raster file.

        Args:
            raster_path (str): The path to the raster file.
            point (gpd.GeoSeries): The point coordinates for which raster values are to be extracted.

        Returns:
            np.ndarray: An array containing the raster values for the given point.
        """
        with rasterio.open(raster_path) as raster:
            row, col = raster.index(point.x, point.y)
            raster_values = raster.read(window=Window(col, row, 1, 1)).reshape(-1)
            if len(raster_values) == 0:
                raster_values = np.zeros(raster.count)
        return pd.Series(raster_values, index=season_fns)
    
    def extract_raster_values_for_dataset(self, data_type: str = 'ndvi') -> List[np.ndarray]:
        """
        Extract raster values for each point in the dataset.

        Args:
            data_type (str, optional): The type of data to be extracted, either 'ndvi' or 'sar'. Defaults to 'ndvi'.
        Returns:
            List[np.ndarray]: A list of arrays containing the raster values for each point in the dataset.
        """
        district_raster_path_map = {(path.split('/')[-1].split('_')[0], path.split('/')[-1].split('_')[2][:-1]): path for path in self.raster_path_list}
        start = self.raster_path_list[0].split('/')[-1].split('_')[1][:-2]
        end = self.raster_path_list[0].split('/')[-1].split('_')[2][:-2]
        start_fn, end_fn = start[:3] + '_' + start[3:],  end[:3] + '_' + end[3:]
        if start_fn == 'oct_1f':
            season_fns = list(map(lambda fn:fn+'_'+data_type, rabi_season_fns[:rabi_season_fns.index(end_fn)+1]))
        elif start_fn == 'jun_1f':
            season_fns =  list(map(lambda fn:fn+'_'+data_type, kharif_season_fns[:kharif_season_fns.index(end_fn)+1]))
        raster_values = self.data.apply(lambda row: self.extract_raster_values_for_point(district_raster_path_map[(row['district'], data_type)], row['geometry'], season_fns=season_fns), axis=1)
        return raster_values
