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

    def __init__(self, skymapfile, configfile, tileFile=None, outdir=None):        
        super().__init__(skymapfile, configfile, tileFile=tileFile, outdir=outdir)
        
    def append_tile_indices_to_catalog(self, galaxy_catalog, telescope):
            """
            Append the tile index for each galaxy in the galaxy catalog

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
        
        
    