import pandas as pd
import ee
from shapely import wkt
import geopandas as gp
from typing import List
from huggingface_hub import hf_hub_download
from crop_classification.src.utils.helper import find_district
from crop_classification.src.utils.constants import rabi_season_fns, kharif_season_fns, fns_dates_map

class SpectralDataHarvester:
    def __init__(self, geom_list: List[str]) -> None:
        """Initialize the SpectralDataHarvester instance.
        
        Args:
            geom_list (List[str]): List of geometries.
            ind_dist_shp (str): Path to the shapefile containing individual district boundaries.
        """
        self.geom_list = pd.Series(geom_list).astype(str).apply(wkt.loads)
        self.geom_list = gp.GeoDataFrame(geometry=self.geom_list)
        self.districts = find_district(self.geom_list, hf_hub_download(repo_id='krishnamlexp/gt_curation_datasets', filename='state_district_boundaries_2020.geojson',  repo_type="dataset"))
    
    def initialize_ee(self, ee_project: str) -> None:
        """Initialize Earth Engine."""
        ee.Authenticate()
        ee.Initialize(project=ee_project)
        self.ee_project = ee_project
    
    def generate_district_spectral_data(self, data_type: str, start_date: str, end_date: str, frequency: int, dir_path: str, storage_type: str, bucket_name: str = None) -> List[ee.Image]:
        """Generate raster stack for each district and export to specified storage.
        
        Args:
            data_type (str): Type of spectral data to generate, either 'ndvi' or 'sar'.
            start_date (pd.Timestamp): Start date for raster collection.
            end_date (pd.Timestamp): End date for raster collection.
            frequency (int): Frequency of raster collection in days.
            dir_path (str): Directory path to export raster stacks.
            storage_type (str): Storage type (either 'drive' or 'cloud').
            bucket_name (str, optional): Bucket name if exporting to cloud storage.
        
        Returns:
            List[ee.Image]: List of raster stacks, one per district.
        """
        
        start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
        no_of_readings = ((end_date - start_date).days + 1) // frequency
        
        for dist_name in self.districts.unique():
            dist_boundary = ee.FeatureCollection(f'projects/{self.ee_project}/assets/INDIA_DISTRICTS').filter(ee.Filter.eq('dtname', dist_name))
            stacked_data = None
            
            for _ in range(no_of_readings):
                end_date = start_date + pd.to_timedelta(frequency, unit='D')
                if data_type == 'ndvi':
                    data = self._getNDVI(start_date, end_date, dist_boundary)
                elif data_type == 'sar':
                    data = self._getVH(start_date, end_date, dist_boundary)              
                if stacked_data is None:
                    stacked_data = data
                else:
                    stacked_data = stacked_data.addBands(data)
                
                start_date = end_date
            
            # Export to storage
            file_name = f'{dist_name}_{data_type}' 
            self.export_to_storage(stacked_data, file_name, dist_boundary, dir_path, storage_type, bucket_name)
                
        print("All raster stacks for districts have been queued on earth engine successfully.")
    
    def generate_district_spectral_data_perfn(self, data_type: str, season: str, end_fn: str, year: int, dir_path: str, storage_type: str, bucket_name: str = None) -> List[ee.Image]:
        """
        Generate raster stack for each district fortnightly according to the specified season.

        Args:
            data_type (str): The type of data to be generated. Should be either 'ndvi' or 'vh(SAR)'.
            season (str): The season for which raster stacks are to be generated. Should be either 'rabi' or 'kharif'.
            end_fn (str): The end fortnight(exclusive)
            year (int): Year when the season started.
            dir_path (str): Directory path to export raster stacks.
            storage_type (str): Storage type (either 'drive' or 'cloud').
            bucket_name (str, optional): Bucket name if exporting to cloud storage.

        Returns:
            None
        """
        if season == 'rabi':
            start_year, end_year = year, year+1
            fns_to_extract = rabi_season_fns[:rabi_season_fns.index(end_fn)+1]
        elif season == 'kharif':
            start_year, end_year = year, year
            fns_to_extract = kharif_season_fns[:kharif_season_fns.index(end_fn)+1]
        
        for dist_name in self.districts.unique():
            dist_boundary = ee.FeatureCollection(f'projects/{self.ee_project}/assets/INDIA_DISTRICTS').filter(ee.Filter.eq('dtname', dist_name))
            stacked_data = None
            for fn in fns_to_extract:
                start_date, end_date = str(year)+'-'+fns_dates_map[fn][0], str(year)+'-'+fns_dates_map[fn][1]
                if data_type == 'ndvi':
                    data = self._getNDVI(start_date, end_date, dist_boundary)
                elif data_type == 'vh':
                    data = self._getVH(start_date, end_date, dist_boundary)
                if stacked_data is None:
                    stacked_data = data
                else:
                    stacked_data = stacked_data.addBands(data)
                if fn == 'dec_2f':
                    year += 1
            # Export to storage
            file_name = f'{dist_name}_{fns_to_extract[0]}{start_year[-2:]}_{fns_to_extract[0]}{end_year[-2:]}_{data_type}' 
            self.export_to_storage(stacked_data, file_name, dist_boundary, dir_path, storage_type, bucket_name)
                
        print("All raster stacks for districts have been queued on earth engine successfully.")
    
    def export_to_storage(self, data_stack: ee.Image, file_name: str, dist_boundary: ee.FeatureCollection, dir_path: str, storage_type: str, bucket_name: str = None) -> None:
        """Export raster stack to specified storage.
        
        Args:
            data_stack (ee.Image): raster stack to export.
            file_name (str): File name for exported raster stack.
            dist_boundary (ee.FeatureCollection): District boundary.
            dir_path (str): Directory path to export raster stacks.
            storage_type (str): Storage type (either 'drive' or 'cloud').
            bucket_name (str, optional): Bucket name if exporting to cloud storage.
        """
        if storage_type not in ['drive', 'cloud']:
            raise ValueError("Invalid storage type. Must be either 'drive' or 'cloud'.")
        
        export_params = {'image': data_stack, 'description': file_name,
                         'region': dist_boundary.geometry(), 'scale': 10, 'maxPixels': 1e13, "fileFormat": 'GeoTIFF'}
        
        if storage_type == 'drive':
            task = ee.batch.Export.image.toDrive(**export_params, folder=dir_path)
        else:
            raster_path = f'{dir_path}/{file_name}'
            export_params['fileNamePrefix'] = raster_path
            export_params['bucket'] = bucket_name
            task = ee.batch.Export.image.toCloudStorage(**export_params)
        
        task.start()
        print(f"Queuing {file_name} for export to {storage_type} Storage...")
        
        if task.status()["state"] == "FAILED":
            print("Error message:", task.status()["error_message"])
            
    def _calcNDVI(self, image):
        # Adds NDVI band to the image
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)

    def _getNDVI(self, start_date, end_date, dist_boundary):
        # Picks the greenest pixel for the given timeperiod
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
                        .filterDate(start_date, end_date)\
                        .filterBounds(dist_boundary)\
                        .map(self._calcNDVI)\
                        .qualityMosaic('NDVI')\
                        .select(['NDVI'])\
                        .clip(dist_boundary)
        scaled_collection = collection.multiply(100).add(100).uint8()
        return scaled_collection
    
    def _getVH(start_date, end_date, dist_boundary):
        collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) \
            .filterBounds(dist_boundary) \
            .select('VH') \
            .mean() \
            .clip(dist_boundary) \
            .rename('VH_' + str(start_date))
        scaled_collection = collection.multiply(100).toInt16().divide(100)
        return scaled_collection