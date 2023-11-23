# Copyright (C) 2017 Shaon Ghosh, David Kaplan, Shasvath Kapadia, Deep Chatterjee
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""

Creates ranked tiles for a given gravitational wave trigger. Sample steps are below:


tileObj = rankedTilesGenerator.RankedTileGenerator('bayestar.fits.gz')
[ranked_tile_index, ranked_tile_probs] = tileObj.getRankedTiles(resolution=512)

This gives the ranked tile indices and their probabilities for the bayestar sky-map.
The resolution is 512, thus ud_grading to this value from the actual sky-map resolution.
The code expects the file ZTF_tiles_set1_nowrap_indexed.dat and the pickled file 
preComputed_pixel_indices_512.dat to be in the same path. 

"""

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



def getTileBounds(FOV, ra_cent, dec_cent):
    dec_down = dec_cent - 0.5*np.sqrt(FOV)
    dec_up = dec_cent + 0.5*np.sqrt(FOV)

    ra_down_left = ra_cent - 0.5*(np.sqrt(FOV)/(np.cos(dec_down*(np.pi/180.))))
    ra_down_right = ra_cent + 0.5*(np.sqrt(FOV)/(np.cos(dec_down*(np.pi/180.))))
    ra_up_left = ra_cent - 0.5*(np.sqrt(FOV)/(np.cos(dec_up*(np.pi/180.))))
    ra_up_right = ra_cent + 0.5*(np.sqrt(FOV)/(np.cos(dec_up*(np.pi/180.))))
    
    return([dec_down, dec_up, ra_down_left, ra_down_right, ra_up_left, ra_up_right])



class RankedTileGenerator:
		
	def __init__(self, skymapfile, configfile, tileFile=None, outdir=None):
		'''
		skymapfile :: The GW sky-localization map for the event
		path	   :: Path to the preCoputed files
		preComputeFiles  :: A list of all the precompute files
		'''
		
		self.outdir = outdir
		self.configParser = configparser.ConfigParser()
		self.configParser.read(configfile)

		preComputed_64 = self.configParser.get('pixelTileMap', 'preComputed_64')
		preComputed_128 = self.configParser.get('pixelTileMap', 'preComputed_128')
		preComputed_256 = self.configParser.get('pixelTileMap', 'preComputed_256')
		preComputed_512 = self.configParser.get('pixelTileMap', 'preComputed_512')
		preComputed_1024 = self.configParser.get('pixelTileMap', 'preComputed_1024')
		preComputed_2048 = self.configParser.get('pixelTileMap', 'preComputed_2048')
		

		
		self.skymap = hp.read_map(skymapfile)
		npix = len(self.skymap)
		self.nside = hp.npix2nside(npix)
		if tileFile is None:
			tileFile = self.configParser.get('tileFiles', 'tileFile')
		self.tileData = np.recfromtxt(tileFile, names=True)
		
		self.preCompDictFiles = {64:preComputed_64, 128:preComputed_128,
					256:preComputed_256, 512:preComputed_512,
					1024:preComputed_1024, 2048:preComputed_2048}
				

	def sourceTile(self, ra, dec):
		'''
		METHOD     :: This method takes the position of the injected 
					  event and returns the tile index
					  
		ra		   :: Right ascension of the source in degrees
		dec		   :: Declination angle of the source in degrees
		tiles	   :: The tile coordinate file (in the following format)
			      ID	ra_center	dec_center	
			      0  	24.714290	-85.938460
			      1  	76.142860	-85.938460
			      ...
		'''

		Dec_tile = self.tileData['dec_center']
		RA_tile =  self.tileData['ra_center']
		ID = self.tileData['ID']
		s = np.arccos( np.sin(np.pi*dec/180.)\
			* np.sin(np.pi*Dec_tile/180.)\
			+ np.cos(np.pi*dec/180.)\
			* np.cos(np.pi*Dec_tile/180.) \
			* np.cos(np.pi*(RA_tile - ra)/180.) )
		index = np.argmin(s) ### minimum angular distance index
		

		return ID[index] ### Since the indexing begins with 1.

	
	def searchedArea(self, ra, dec, resolution=None, verbose=True):
		'''
		METHOD     :: This method takes the position of the injected 
			      event and the sky-map. It returns the searched 
			      area of the sky-map to reach to the source lo-
			      cation. The searched area constitutes both the
			      total area (sq. deg) that needed to be search-
			      ed to reach the source location, and the total
			      localization probability covered in the process.
					  
		ra		   :: Right ascension of the source in degrees
		dec		   :: Declination angle of the source in degrees
		resolution 	   :: The value of the nside, if not supplied, 
				      the default skymap is used.
		output     :: [Searched area, Searched Probability]
		'''
		if not resolution:
			resolution = self.nside
		n = np.log(resolution)/np.log(2)
		resolution = int(2 ** round(n)) ## resolution in powers of 2
		if resolution > 2048: resolution = 2048
		if resolution < 64: resolution = 64
		filename = self.preCompDictFiles[resolution]
		if not os.path.isfile(filename):
			if verbose: print("Precomputed pickle file for this resolution is not found")
			if verbose: print("Reverting to default resolution (= 256)")
			resolution = 256
			filename = self.preCompDictFiles[resolution]

		File = open(filename, 'rb')
		data = pickle.load(File)
		tile_index = np.arange(len(data))
		skymapUD = hp.ud_grade(self.skymap, resolution, power=-2)
		npix = len(skymapUD)
		theta, phi = hp.pix2ang(resolution, np.arange(0, npix))
		ra_map = np.rad2deg(phi) # Construct ra array
		dec_map = np.rad2deg(0.5*np.pi - theta) # Construct dec array
		pVal = skymapUD[np.arange(0, npix)]
		order = np.argsort(-pVal)
		ra_map = ra_map[order]
		dec_map = dec_map[order]
		pVal = pVal[order]
		s = np.arccos( np.sin(np.pi*dec/180.)\
			* np.sin(np.pi*dec_map/180.)\
			+ np.cos(np.pi*dec/180.)\
			* np.cos(np.pi*dec_map/180.) \
			* np.cos(np.pi*(ra_map - ra)/180.) )
		index = np.argmin(s) ### minimum angular distance index
		coveredProb = np.sum(pVal[0:index])
		searchedArea = index*hp.nside2pixarea(resolution, degrees=True)
		return [searchedArea, coveredProb]

	
	def getRankedTiles(self, resolution=None, verbose=True, save_csv=False, tag=None):
		'''
		METHOD		:: This method returns two numpy arrays, the first
				   contains the tile indices of telescope and the second
				   contains the probability values of the corresponding 
				   tiles. The tiles are sorted based on their probability 
				   values.
		
		resolution  :: The value of the nside, if not supplied, 
			       the default skymap is used.
		'''
		if not resolution:
			resolution = self.nside
		n = np.log(resolution)/np.log(2)
		resolution = int(2 ** round(n)) ## resolution in powers of 2
		if resolution > 2048: resolution = 2048
		if resolution < 64: resolution = 64
		if verbose: print('Using resolution of ' + str(resolution))
		filename = self.preCompDictFiles[resolution]
		if verbose: print("Using file: "+filename)
		if not os.path.isfile(filename):
			if verbose: print("Precomputed pickle file for this resolution is not found")
			if verbose: print("Reverting to default resolution (= 256)")
			resolution = 256
			filename = self.preCompDictFiles[resolution]

		File = open(filename, 'rb')
			
		data = pickle.load(File)
		tile_index = np.arange(len(data))
		skymapUD = hp.ud_grade(self.skymap, resolution, power=-2)
		npix = len(skymapUD)
		pVal = skymapUD[np.arange(0, npix)]

		allTiles_probs = []
		for ii in range(0, len(data)):
			pTile = np.sum(pVal[data[ii]])
			allTiles_probs.append(pTile)

		allTiles_probs = np.array(allTiles_probs)
		index = np.argsort(-allTiles_probs)

		Dec_tile = self.tileData['dec_center']
		RA_tile = self.tileData['ra_center']
		allTiles_probs_sorted = allTiles_probs[index]
		tile_index_sorted = tile_index[index]
		self.df = pd.DataFrame({"tile_index": tile_index_sorted, "RA": RA_tile[index], "Dec": Dec_tile[index], "tile_prob" : allTiles_probs_sorted})

		if save_csv:
			if tag is None: 
				tag = self.configParser.get('plot', 'filenametag')
			self.df.to_csv(self.outdir+tag+"_ranked_tiles.csv", index=False)
		
		return self.df


	def plotTiles(self, FOV=None,resolution=None, tileEdges=False, CI=0.9,
	       save_plot=False, tag=None, highlight=None, event=None, title=None, fig=None):
		'''
		METHOD 		:: This method plots the ranked-tiles on a hammer projection
				       skymap. 
		ranked_tile_indices    :: The index of he ranked-tiles
		allTiles_probs_sorted  :: The probabilities of the ranked-tiles
		tileFile    :: The file with tile indices and centers
			       		ID	ra_center	dec_center	
			       		0  	24.714290	-85.938460
			       		1  	76.142860	-85.938460
		
		FOV			:: Field of view of the telescopes. If not supplied,
				   		tile boundaries will not be plotted.

		resolution  :: The resolution of the skymap to be used.
		tileEdges	:: Allows plotting of the tile edges. Default is False.
		tag 		:: Extra tag to file name
		highlight	:: Specify the tile index and this tile will be highlighted
		event		:: [RA, Dec] of the event (for injections)
		title		:: Title of the plot (optional)
		size		:: (Optional) Size of the plot if plot show option is used
		'''			

		ranked_tile_indices = self.df["tile_index"]
		allTiles_probs_sorted = self.df["tile_prob"]

		skymap = self.skymap
		if resolution:
			skymap = hp.ud_grade(skymap, resolution, power=-2)
		npix = len(skymap)
		nside = hp.npix2nside(npix)
		theta, phi = hp.pix2ang(nside, np.arange(0, npix))
		ra = np.rad2deg(phi)
		dec = np.rad2deg(0.5*np.pi - theta)
		pVal = skymap[np.arange(0, npix)]
		order = np.argsort(-pVal)
		ra = ra[order]
		dec = dec[order]
		pVal = pVal[order]
		include = np.cumsum(pVal) < CI
		include[np.sum(include)] = True
		ra_CI = ra[include]
		dec_CI = dec[include]
		# pVal_CI = pVal[include]

		if fig is None:
			fig = plt.figure(figsize=(8, 5))

		if title:
			plt.title(title)
		
		m = AllSkyMap_basic.AllSkyMap(projection='hammer')
		RAP_map, DecP_map = m(ra_CI, dec_CI)
		if event: RAP_event, DecP_event = m(event[0], event[1])
		m.drawparallels(np.arange(-90.,120.,20.), color='grey', 
						labels=[False,True,True,False], labelstyle='+/-')
		m.drawmeridians(np.arange(0.,420.,30.), color='grey')
		m.drawmapboundary(fill_color='white')
		lons = np.arange(-150,151,30)

		m.plot(RAP_map, DecP_map, color='mistyrose', marker='.', linewidth=0, markersize=0.5, alpha=1) 
		m.label_meridians(lons, fontsize=12, vnudge=1, halign='left', hnudge=-1)
		if event:
			m.plot(RAP_event, DecP_event, color='b', marker='*', linewidth=1, markersize=1, alpha=1.0)
			source_index = self.sourceTile(event[0], event[1])
			RA_event_tilecenter = self.tileData['ra_center'][source_index]
			Dec_event_tilecenter = self.tileData['dec_center'][source_index]
			print(RA_event_tilecenter, Dec_event_tilecenter, source_index)
			[dec_down, dec_up,ra_down_left, ra_down_right, ra_up_left, ra_up_right] = getTileBounds(FOV, RA_event_tilecenter, Dec_event_tilecenter)
			RAP1, DecP1 = m(ra_up_left, dec_up)
			RAP2, DecP2 = m(ra_up_right, dec_up)
			RAP3, DecP3 = m(ra_down_left, dec_down)
			RAP4, DecP4 = m(ra_down_right, dec_down)
			m.plot([RAP1, RAP2], [DecP1, DecP2],'b-', linewidth=0.5, alpha=0.5,) 
			m.plot([RAP2, RAP4], [DecP2, DecP4],'b-', linewidth=0.5, alpha=0.5,) 
			m.plot([RAP4, RAP3], [DecP4, DecP3],'b-', linewidth=0.5, alpha=0.5,) 
			m.plot([RAP3, RAP1], [DecP3, DecP1],'b-', linewidth=0.5, alpha=0.5,)


		Dec_tile = self.tileData['dec_center']
		RA_tile = self.tileData['ra_center']
		
		include_tiles = np.cumsum(allTiles_probs_sorted) < CI
		include_tiles[np.sum(include_tiles)] = True
		ranked_tile_indices = ranked_tile_indices[include_tiles]

		if FOV is None:
			tileEdges = False

		alpha=1.0
		lw=0.5
		if highlight:
			alpha = 0.2
		if highlight:
			print('Tile center of this pointing: (' + str(RA_tile[highlight])+','+ str(Dec_tile[highlight])+')')
			RAP_hcenter, DecP_hcenter = m(RA_tile[highlight], Dec_tile[highlight])
			if tileEdges:
				[dec_down_h, dec_up_h,\
				ra_down_left_h, ra_down_right_h,\
				ra_up_left_h, ra_up_right_h] = getTileBounds(FOV, RA_tile[highlight], Dec_tile[highlight])
				m.plot(RAP_hcenter, DecP_hcenter, 'co', markersize=4, mew=1)
				RAP1_h, DecP1_h = m(ra_up_left_h, dec_up_h)
				RAP2_h, DecP2_h = m(ra_up_right_h, dec_up_h)
				RAP3_h, DecP3_h = m(ra_down_left_h, dec_down_h)
				RAP4_h, DecP4_h = m(ra_down_right_h, dec_down_h)
				m.plot([RAP1_h, RAP2_h], [DecP1_h, DecP2_h],'r-', linewidth=lw*3) 
				m.plot([RAP2_h, RAP4_h], [DecP2_h, DecP4_h],'r-', linewidth=lw*3) 
				m.plot([RAP4_h, RAP3_h], [DecP4_h, DecP3_h],'r-', linewidth=lw*3)
				m.plot([RAP3_h, RAP1_h], [DecP3_h, DecP1_h],'r-', linewidth=lw*3) 
			else:
				m.plot(RAP_hcenter, DecP_hcenter, 'co', markersize=5*lw, mew=1)

		for ii in ranked_tile_indices:

			RAP_peak, DecP_peak = m(RA_tile[ii], Dec_tile[ii])
			if tileEdges:
				[dec_down, dec_up,
				ra_down_left, ra_down_right, 
				ra_up_left, ra_up_right] = getTileBounds(FOV, RA_tile[ii], Dec_tile[ii])
				# m.plot(RAP_peak, DecP_peak, 'k.', markersize=4, mew=1, alpha=alpha)

				RAP1, DecP1 = m(ra_up_left, dec_up)
				RAP2, DecP2 = m(ra_up_right, dec_up)
				RAP3, DecP3 = m(ra_down_left, dec_down)
				RAP4, DecP4 = m(ra_down_right, dec_down)
				m.plot([RAP1, RAP2], [DecP1, DecP2],'k-', linewidth=0.5, alpha=0.5) 
				m.plot([RAP2, RAP4], [DecP2, DecP4],'k-', linewidth=0.5, alpha=0.5) 
				m.plot([RAP4, RAP3], [DecP4, DecP3],'k-', linewidth=0.5, alpha=0.5) 
				m.plot([RAP3, RAP1], [DecP3, DecP1],'k-', linewidth=0.5, alpha=0.5)

			else:
				m.plot(RAP_peak, DecP_peak, 'ko', markersize=lw, mew=1, alpha=alpha)

		if save_plot:
			extension = self.configParser.get('plot', 'extension')
			if tag is None: 
				tag = self.configParser.get('plot', 'filenametag')
			plt.savefig(self.outdir + tag + '_skyTiles' + '.' + extension, dpi=600)
			plt.close()

		else:
			return fig


	def rankGalaxies2D(self, catalog, resolution=None):
		'''
		METHOD  :: This method takes as input a galaxy catalog pickle file
				   that is generated by running the createCatalog.py script.
				   The output is the IDs of the galaxies from the catalog 
				   ranked based on their localization probability.
		
		catalog	:: A pickle file which stores a 7 col numpy array with. The 
				   columns of this array are defined below:
				   		col1 : galaxy ID
				   		col2 : distance to the galaxy
				   		col3 : Declination angle of the galaxy
				   		col4 : Right ascencion of the galaxy
				   		col5 : Closest BAYESTAR Healpix pixel to the galaxy
				   		col6 : Declination angle of the closest pixel
				   		col7 : Right ascencion of the closest pixel
				   		
		resolution :: Optional argument. allows you to fix the resolution of 
					  the skymap. Currently the catalog file has only been 
					  generated for resolution of 512. Use this value.
				   
		'''

		if not resolution:
			resolution = self.nside
		n = np.log(resolution)/np.log(2)
		resolution = int(2 ** round(n)) ## resolution in powers of 2
		if resolution > 2048: resolution = 2048
		if resolution < 64: resolution = 64
		filename = self.preCompDictFiles[resolution]
		File = open(filename, 'rb')
		data = pickle.load(File)
		# tile_index = np.arange(len(data))
		skymapUD = hp.ud_grade(self.skymap, resolution, power=-2)
		npix = len(skymapUD)
		theta, phi = hp.pix2ang(resolution, np.arange(0, npix))
		ra_map = np.rad2deg(phi) # Construct ra array
		dec_map = np.rad2deg(0.5*np.pi - theta) # Construct dec array
		pVal = skymapUD[np.arange(0, npix)]
		
		catalogFile = open(catalog, 'rb')
		catalogData = pickle.load(catalogFile)
		
		indices = catalogData[:,4].astype('int') ### Indices of pixels for all galaxies
		galaxy_probs = pVal[indices] ### Probability values of the galaxies in catalog
		order = np.argsort(-galaxy_probs) ### Sorting in descending order of probability
		galaxy_indices = catalogData[:,0].astype('int') ### Indices of galaxies
		ranked_galaxies = galaxy_indices[order]
		galaxy_probs = galaxy_probs[order]
		
		return [ranked_galaxies, galaxy_probs]


	def getSamplesInTiles(self, samples):
		'''
		METHOD	:: This method, can be used for any set of indexed points in the sky.
				   The output is the set of these point indices contained in each tiles.
				   Potential use, LALinference samples that are present in each tile, or
				   galaxies from a galaxy catalog that are located within the FOV of each
				   tile

		samples	::	Just three columns array samples[:,0] = index, samples[:,1] = ra,
					samples[:,2] = dec
		'''
		
		Dec_tile = self.tileData['dec_center']
		RA_tile = self.tileData['ra_center']
		tile_index = self.tileData['ID']
		closestTileIndex = []

		sampleIndex = samples[:,0]
		sample_ras = samples[:,1]*u.radian
		sample_decs = samples[:,2]*u.radian
		
		
		with ProgressBar(len(samples)) as bar:
			for sample, ra, dec in zip(sampleIndex, sample_ras.to(u.degree).value, sample_decs.to(u.degree).value):
				s = np.arccos( np.sin(np.pi*dec/180.)\
					* np.sin(np.pi*Dec_tile/180.)\
					+ np.cos(np.pi*dec/180.)\
					* np.cos(np.pi*Dec_tile/180.) \
					* np.cos(np.pi*(RA_tile - ra)/180.) )
				index = np.argmin(s) ### minimum angular distance index
				closestTileIndex.append(tile_index[index])
				bar.update()

		closestTileIndex = np.array(closestTileIndex)
		uniqueTiles = np.unique(closestTileIndex)
		samplesInTile = []
		for tile in uniqueTiles:
			whereThisTile = tile == closestTileIndex ### pixels indices in this tile
			samplesInTile.append(sampleIndex[whereThisTile])
			
		return [uniqueTiles, samplesInTile]
			

	def integrationTime(self, T_obs, pValTiles=None, func=None):
		'''
		METHOD :: This method accepts the probability values of the ranked tiles, the 
			  total observation time and the rank of the source tile. It returns 
			  the array of time to be spent in each tile which is determined based
			  on the localizaton probability of the tile. How the weight factor is 
			  computed can also be supplied in functional form. Default is linear.
				  
		pValTiles :: The probability value of the ranked tiles. Obtained from getRankedTiles 
					 output
		T_obs     :: Total observation time available for the follow-up.
		func	  :: functional form of the weight. Default is linear. 
					 For example, use x**2 to use a quadratic function.
		'''
		if pValTiles is None:
			pValTiles = self.allTiles_probs_sorted
		
		if func is None:
			f = lambda x: x
		else:
			f = lambda x: eval(func)
		fpValTiles = f(pValTiles)
		modified_prob = fpValTiles/np.sum(fpValTiles)
		t_tiles = modified_prob * T_obs ### Time spent in each tile if not constrained
		t_tiles[t_tiles > 1200.0] = 1200.0 ### Upper limit of exposure time
		t_tiles[t_tiles < 60] = 60.0 ### Lower limit of exposure time
		Obs = np.cumsum(t_tiles) <= T_obs ### Tiles observable in T_obs seconds
		time_per_tile = t_tiles[Obs] ### Actual time spent per tile
		
		return time_per_tile
		

	def optimize_time(self, T, M, range, pValTiles=None):
		'''
		METHOD	:: This method accepts the total duration of time, the absolute mag,
				   range for optimization(e.g: [0.0, 0.1]) and the probability values
				   of the tiles and returns an optimized array of time per tile.
		'''
		time_data, limmag_data, _ = np.loadtxt('timeMagnitude_new.dat', unpack=True)
		s = interpolate.UnivariateSpline(np.log(time_data), limmag_data, k=5)
		AA = np.linspace(range[0], range[1], 10000) ### Variable for optimization
		if pValTiles is None:
			pValTiles = self.allTiles_probs_sorted
		kappa_sum = []		
		for aa in AA:
			# time_per_tile = self.integrationTime(T, pValTiles, func='x + ' + str(aa))
			time_per_tile = self.integrationTime(T, pValTiles, func='x**' + str(aa))
			limmag = s(np.log(time_per_tile))
			dists = 10**(1.0 + (limmag - M)/5)
			#kappas = (dists**3)*pValTiles[:len(dists)]
			kappas = limmag*pValTiles[:len(dists)]
			kappa_sum.append(np.sum(kappas))
		kappa_sum = np.array(kappa_sum)
		maxIndex = np.argmax(kappa_sum)
		a_max = AA[maxIndex]
		kappa_max = kappa_sum[maxIndex]
		time_per_tile_max = self.integrationTime(T, pValTiles, func='x**' + str(a_max))
		return [time_per_tile_max, a_max]
		





