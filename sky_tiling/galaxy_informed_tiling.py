#Creator: Kruthi Krishna
#Date: 3 Jan 2024
#Description: This file contains the implementation of a galaxy-informed tiling algorithm.

import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import scipy
import os
import pickle
import numpy as np
import pandas as pd

import healpy as hp
import configparser
from scipy import interpolate
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.table import Table
from astropy.utils.console import ProgressBar
from .utilities import AllSkyMap_basic
from .ranked_tiling import RankedTileGenerator
from astropy.utils.console import ProgressBar
import importlib.resources
from pathlib import Path


class GalaxyTileGenerator(RankedTileGenerator):

    def __init__(self, configfile, skymapfile=None, tileFile=None, outdir=None):        
        super().__init__(configfile, skymapfile=skymapfile, tileFile=tileFile, outdir=outdir)
        
    def append_tile_indices_to_catalog(self, galaxy_catalog, telescope):
            """
            Append the telescope tile index for each galaxy in the galaxy catalog.

            Parameters:
            - galaxy_catalog (astropy Table): The catalog of galaxies
            - telescope (str): The name of the telescope; should be the same as the config file.

            Returns:
            - astropy Table: The updated galaxy catalog with "<telescope>_tile_index" column appended
            """
            RA_tile = self.tileData['ra_center'] 
            Dec_tile = self.tileData['dec_center']
            galaxy_catalog_new = galaxy_catalog.copy()
            galaxy_catalog_new[telescope + "tile_index"] = np.full(len(galaxy_catalog_new), np.nan)
            with ProgressBar(len(galaxy_catalog_new)) as bar:
                for row in galaxy_catalog_new:
                    ra = row['ra']
                    dec = row['dec']
                    s = np.arccos( np.sin(np.pi*dec/180.)\
                        * np.sin(np.pi*Dec_tile/180.)\
                        + np.cos(np.pi*dec/180.)\
                        * np.cos(np.pi*Dec_tile/180.) \
                        * np.cos(np.pi*(RA_tile - ra)/180.) )
                    closestTileIndex = np.argmin(s) ### minimum angular distance index
                    row[telescope + "_tile_index"] = closestTileIndex
                    bar.update()
                    
            return galaxy_catalog_new
    
    def get_ranked_galaxies():
        """function that returns the ranked galaxies
        """
        
        return None
        
    def get_galaxy_informed_tiles(self, catalog_with_indices, telescope,  save_csv=False, tag=None, CI=0.9):
        """
        Reorders probability-ranked-tiles based on galaxy information.

        Parameters:
        - catalog_with_indices (astropy Table): The catalog with tile indices for the given telescope
        - telescope (str): The telescope name.
        - csv_file_name (str): The name of the CSV file to save the results.

        Returns:
        - df_summed_fields (DataFrame): The DataFrame containing the galaxy-informed tiles.
        """
        df_ranked_tiles = self.getRankedTiles(CI=CI)
        df_summed_fields = df_ranked_tiles.sort_values(by='tile_index').copy()
        df_summed_fields.reset_index(inplace=True, drop=True)
        
        df_summed_fields['tile_Mstar'] = np.full(len(df_summed_fields), np.nan)
        df_summed_fields['tile_Mstar_x_tile_prob'] = np.full(len(df_summed_fields), np.nan)
        
        grouped_data = catalog_with_indices.group_by(telescope+'_tile_index')
        tile_indices_from_catalog = grouped_data.groups.keys[telescope+'_tile_index']
        df_summed_fields.loc[tile_indices_from_catalog, "tile_Mstar"] =  grouped_data['Mstar'].groups.aggregate(np.sum)
        
        df_summed_fields['tile_Mstar_x_tile_prob'] = df_summed_fields['tile_Mstar']*df_summed_fields['tile_prob']
        
        if save_csv:
            if tag is None: 
                tag = self.configParser.get('plot', 'filenametag')
            df_summed_fields.to_csv(self.outdir+tag+"_galaxy_informed_tiles.csv", index=False,  na_rep='NaN')
        
        return df_summed_fields
        