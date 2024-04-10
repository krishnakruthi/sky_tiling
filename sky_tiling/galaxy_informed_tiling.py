#Creator: Kruthi Krishna
#Date: 3 Jan 2024
#Description: This file contains the implementation of a galaxy-informed tiling algorithm.

import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.table import Table
from astropy.utils.console import ProgressBar
from .utilities import AllSkyMap_basic
from .ranked_tiling import RankedTileGenerator
from astropy.utils.console import ProgressBar
from ligo.skymap.io import read_sky_map
from ligo.skymap.postprocess import crossmatch
from pathlib import Path


def crossmatch_galaxies(ra, dec, dist_mpc, galaxy_catalog, skymapfile, CI=0.9, save_csv=False,
                        tag = "crossmatched_catalog"):
    """
    Crossmatches the galaxy catalog with the skymapfile. The output file is similar to NED-LVS GWF 
    Service (https://ned.ipac.caltech.edu/NED::GWFoverview/).

    Parameters:
    - ra (array-like): Array of right ascension values of galaxies.
    - dec (array-like): Array of declination values of galaxies.
    - dist_mpc (array-like): Array of distance values of galaxies in megaparsecs.
    - galaxy_catalog (astropy Table): Table containing the galaxy catalog.
    - skymapfile (str): Path to the skymap file.
    - CI (float): The skymap confidence interval for the crossmatch. Default is 0.9.
    - save_csv (bool): If True, the crossmatched catalog is saved as a CSV file.
    - tag (str): The tag to be used for the saved CSV file.

    Returns:
    - result: ligo.skymap.postprocess.crossmatch.CrossmatchResult
    """
    
    skymap = read_sky_map(skymapfile, moc=True)
    coordinates = SkyCoord(ra, dec, dist_mpc, unit=(u.deg, u.deg, u.Mpc))
    result = crossmatch(skymap, coordinates)
    CI_cutoff = result.searched_prob_vol < CI
    galaxy_catalog_CI = galaxy_catalog[CI_cutoff]
    galaxy_catalog_CI["dP_dA"] = result.probdensity[CI_cutoff]
    galaxy_catalog_CI["dP_dV"] = result.probdensity_vol[CI_cutoff]
    galaxy_catalog_CI["P_2D"] = result.searched_prob[CI_cutoff] #this is not tile probability
    galaxy_catalog_CI["P_3D"] = result.searched_prob_vol[CI_cutoff]
    
    area_cutoff = result.searched_prob < CI
    sm, meta = read_sky_map(skymapfile)
    cat_area_cutoff = galaxy_catalog[area_cutoff]
    cat_area_cutoff = cat_area_cutoff[cat_area_cutoff["DistMpc"] < meta["distmean"]+ 3*meta["diststd"]]
    sum_Mstar = cat_area_cutoff["Mstar"].sum()
    sum_SFR = cat_area_cutoff["SFR_W4"].sum()
    sum_Lum_W1 = cat_area_cutoff["Lum_W1"].sum()
    
    galaxy_catalog_CI["P_Mstar"] = galaxy_catalog_CI["Mstar"]/sum_Mstar
    galaxy_catalog_CI["P_SFR_W4"] = galaxy_catalog_CI["SFR_W4"]/sum_SFR
    galaxy_catalog_CI["P_Lum_W1"] = galaxy_catalog_CI["Lum_W1"]/sum_Lum_W1
    
    if save_csv:
        galaxy_catalog_CI.write(tag+".csv", format='ascii.csv', overwrite=True)
    
    return galaxy_catalog_CI


class GalaxyTileGenerator(RankedTileGenerator):

    def __init__(self, configfile, skymapfile=None, tileFile=None, outdir=None):        
        super().__init__(configfile, skymapfile=skymapfile, tileFile=tileFile, outdir=outdir)
        self.skymapfile = skymapfile
        
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
        galaxy_catalog_new[telescope + "_tile_index"] = np.full(len(galaxy_catalog_new), np.nan)
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
    


    def get_galaxy_targeted_tiles(self, cat_with_indices, telescope, unique_tiles = True, sort_metric = 'Mstar', CI=0.9,
                                  sort_by_metric_times_P_3D = False, save_csv=False, save_crossmatched_csv = False):
        """
        Retrieves galaxy-targeted tiles based on a galaxy catalog with telescope tile indices appended. 
        Currently only supports NED-LVS [Cook et. al (2023), 10.26132/NED8]

        Parameters:
        - cat_with_indices (astropy Table): Galaxy catalog table containing columns - 'ra', 'dec', 'DistMpc', and telescope tile indices
        - telescope (str): Name of the telescope.
        - unique_tiles (bool): Flag indicating whether to return only unique tiles. Default is True.
        - sort_metric (str): Metric used for sorting the tiles. Default is 'Mstar'.
        - sort_by_metric_times_P_3D (bool): Flag indicating whether to sort by the product of the metric and P_3D. Default is False.
        - save_csv (bool): Flag indicating whether to save the resulting dataframe as a CSV file. Default is False.
        - CI (float): Confidence interval for crossmatching galaxies. Default is 0.9.

        Returns:
        - df_gal_targeted (DataFrame): DataFrame containing the galaxy-targeted tiles, including 'tile_index', 'objname', 'RA', and 'Dec'.

        Note:
        - If unique_tiles is set to True, only the first galaxy info in each tile is shown.
        
        """
        
        crossmatched_cat_with_indices = crossmatch_galaxies(cat_with_indices['ra'], cat_with_indices['dec'], cat_with_indices['DistMpc'], cat_with_indices, self.skymapfile, CI=CI, save_csv=save_crossmatched_csv)
        
        if sort_by_metric_times_P_3D:
            sort_metric_final = 'P_3D_'+sort_metric
        else:
            sort_metric_final = 'P_'+sort_metric
        
        df_gal_targeted = crossmatched_cat_with_indices[telescope+'_tile_index', "objname", sort_metric_final].to_pandas()
        df_gal_targeted.sort_values(by=sort_metric_final, ascending=False, inplace=True, ignore_index=True)
    
        if unique_tiles:
            df_gal_targeted.drop_duplicates(subset=[telescope+'_tile_index'], inplace=True)
            tag = '_unique'
            print("Warning: unique_tiles is set to True. Only the first galaxy info in each tile is shown.")
        
        RA_tile = self.tileData['ra_center'] 
        Dec_tile = self.tileData['dec_center']
        selected_tile_indices = df_gal_targeted[telescope+'_tile_index'].values
        df_gal_targeted["RA"] = RA_tile[selected_tile_indices.astype(int)]
        df_gal_targeted["Dec"] = Dec_tile[selected_tile_indices.astype(int)]
        df_gal_targeted.rename(columns={telescope+'_tile_index': 'tile_index'}, inplace=True)
        df_gal_targeted['objname'] = df_gal_targeted['objname'].str.decode("utf-8")
        
        if save_csv:
            df_gal_targeted.to_csv(self.outdir+telescope+"_galaxy_targeted_tiles"+tag+".csv", index=False,  na_rep='NaN')
        
        return df_gal_targeted
    
        
    def get_galaxy_informed_tiles(self, catalog_with_indices, telescope, sort_metric = 'Mstar',  CI=0.9,
                                  sort_by_metric_times_tile_prob = False, save_csv=False, tag=None, res=256):
        """
        Reorders probability-ranked-tiles based on galaxy information.

        Parameters:
        - catalog_with_indices (astropy Table): The catalog with tile indices for the given telescope
        - telescope (str): The telescope name.
        - sort_metric (str): The galaxy property to sort the tiles by. The string has to be a column in the galaxy catalog.
        - sort_by_metric_times_tile_prob (bool): If True, the tiles are sorted by the product of the galaxy property specified and the tile probability. Default is False.
        - csv_file_name (str): The name of the CSV file to save the results.

        Returns:
        - df_summed_fields (DataFrame): The DataFrame containing the galaxy-informed tiles.
        """
        df_ranked_tiles = self.getRankedTiles(CI=CI, resolution=res) #get ranked tiles within CI area
        df_summed_fields = df_ranked_tiles.copy()
        grouped_data = catalog_with_indices.group_by(telescope+'_tile_index') #group galaxies by telescope tile index
        sum_by_sort_metric = grouped_data[sort_metric].groups.aggregate(np.sum) #add up "sort_metric" within tile index
        tile_indices_grouped = grouped_data.groups.keys[telescope+'_tile_index']
        df_summed_indices = df_summed_fields["tile_index"].values
        indices = np.searchsorted(tile_indices_grouped, df_summed_indices)
        df_summed_fields['tile_'+sort_metric] = sum_by_sort_metric[indices] 
        df_summed_fields['tile_'+sort_metric+'*tile_prob'] = df_summed_fields['tile_'+sort_metric]*df_summed_fields['tile_prob']
        
        if sort_by_metric_times_tile_prob:
            final_sorting_metric = 'tile_'+sort_metric+'*tile_prob'
        else:
            final_sorting_metric = 'tile_'+sort_metric
        
        print("Sorting by: ", final_sorting_metric)
        df_summed_fields.sort_values(by=final_sorting_metric, ascending=False, inplace=True, ignore_index=True)
        
        if save_csv:
            if tag is None: 
                tag = self.configParser.get('plot', 'filenametag')
            df_summed_fields.to_csv(self.outdir+tag+"_galaxy_informed_tiles.csv", index=False,  na_rep='NaN')
        
        return df_summed_fields