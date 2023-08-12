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


import numpy as np
import pandas as pd

import configparser
from scipy import interpolate

from astropy.time import Time
from astropy import units as u
from astropy.coordinates import get_sun
from astropy.coordinates import get_moon
from astropy.coordinates import get_body

from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from .ranked_tiling import RankedTileGenerator


############ UNDER CONSTRUCTION ############
	
class Scheduler(RankedTileGenerator):
	'''
	The scheduler class: Inherits from the RankedTileGenerator class. If no attribute 
	is supplied while creating schedular objects, a default instance of ZTF scheduler 
	is created. To generate scheduler for other telescopes use the corresponding site
	names which can be obtaine from astropy.coordinates.EarthLocation.get_site_names().
	The tile tile coordinate file also needs to be supplied to the variable tileCoord.
	This file needs to have at least three columns, the first being an ID (1, 2, ...),
	the second should be the tile center's ra value and the third the dec value of the 
	same. The utcoffset is the time difference between UTC and the site in hours. 
	'''
	def __init__(self, skymapFile, configfile, astropy_site_location=None, outdir=None, resolution=None):

		configParser = configparser.ConfigParser()
		configParser.read(configfile)

		self.tileCoord = configParser.get('tileFiles', 'tileFile')
		self.outdir = outdir

		if astropy_site_location is not None:
			self.Observatory = astropy_site_location
		
		else:
			site = configParser.get('observation', 'site')
			self.Observatory = EarthLocation.of_site(site)
		
		self.tileData = np.recfromtxt(self.tileCoord, names=True)
		self.skymapfile = skymapFile
		
		self.tileObj = RankedTileGenerator(skymapFile, configfile)
		df_ranked_tiles = self.tileObj.getRankedTiles(resolution=resolution)
		self.tileIndices = df_ranked_tiles["tile_index"].values
		self.tileProbs = df_ranked_tiles["tile_prob"].values

		self.tiles = SkyCoord(ra = self.tileData['ra_center'][self.tileIndices.astype(int)]*u.degree, 
					    dec = self.tileData['dec_center'][self.tileIndices.astype(int)]*u.degree, 
					    frame = 'icrs') ### Tile(s) 

	def tileVisibility(self, t, gps=False):
		'''
		METHOD	:: This method takes as input the time (gps or mjd) of observation
				   and the observatory site name, and returns the alt and az of the 
				   ranked tiles. It also returns the alt and az of the sun.
			  t	:: The time at which observation is made. Default is mjd. If time is 
				   given in gps then set gps to True.

		'''
		if gps: time = Time(t, format='gps') ### If time is given in GPS format
		else: time = Time(t, format='mjd') ### else time is assumed in mjd format
		altAz_tile = self.tiles.transform_to(AltAz(obstime=time, location=self.Observatory))
		altAz_sun = get_sun(time).transform_to(AltAz(obstime=time, location=self.Observatory))
		
		#isSunDown = altAz_sun.alt.value < -18.0 ### Checks if it is past twilight.
		whichTilesUp = altAz_tile.alt.value > 20.0  ### Checks which tiles are up		
		
        #return [altAz_tile, self.tileProbs, altAz_sun]
		return [self.tileIndices[whichTilesUp], self.tileProbs[whichTilesUp], altAz_tile[whichTilesUp], altAz_sun]
		

	def advanceToSunset(self, eventTime, intTime):
		'''
		This method is called when the observation scheduler determines that the sun is 
		above horizon. It finds the nearest time prior to the next sunset within +/- 
		integration time and then advances the scheduler code to that point. 
		This speeds up the code by refraining from computing pointings during the daytime.
		Currently only works with GPS time. In the future mjd will also be included.
		
		eventTime	:: The GPS time for which the advancement is to be computed.
		intTim		:: The integration time for the obsevation
		'''
		
		dt = np.arange(0, 24*3600 + intTime, intTime)
		time = Time(eventTime + dt, format='gps')
		altAz_sun = get_sun(time).transform_to(AltAz(obstime=time, location=self.Observatory))
		timeBeforeSunset = (eventTime + dt)[altAz_sun.alt.value < -18.0][0]
		return timeBeforeSunset


	############### NOT TESTED AND NOT USED CURRENTLY ############
	def whenThisTileSets(self, index, currentTime, duration, gps=False):
		'''
		This method approximately computes the amount of time left in seconds for a tile
		to set below 20 degrees.
		
		index		::	The index of the tile for which setting tile is to be found
		currentTime	::	The current time when this tile is scheduled
		'''
		# if gps: time = Time(currentTime, format='gps')
		# else: time = Time(currentTime, format='mjd')
		thisTile = SkyCoord(ra = self.tileData['ra_center'][index]*u.degree, 
					    dec = self.tileData['dec_center'][index]*u.degree, 
					    frame = 'icrs') ### Tile(s)
		dt = np.arange(0, duration + 1.0, 1.0)
		times = Time(currentTime + dt, format='gps')
		altAz_tile = thisTile.transform_to(AltAz(obstime=times,
												location=self.Observatory))
		
		setTime = None
		if altAz_tile.alt.value[-1] < 20.0:
			s = interpolate.UnivariateSpline(altAz_tile.alt.value, times.value, k=3)
			setTime = s(20.0)
			
		return setTime

	def observationSchedule(self, duration, eventTime, integrationTime=120, CI=0.9,
							observedTiles=None, plot=False, verbose=True, save_schedule=False, tag=None):
		'''
		METHOD	:: This method takes the duration of observation, time of the GW trigger
				   integration time per tile as input and outputs the observation
				   schedule.
				   
		duration   		 :: Total duration of the observation in seconds.
		eventTime  		 :: The gps time of the time of the GW trigger.
		integrationTime  :: Time spent per tile in seconds (default == 120 seconds)
		observedTiles	 :: (Future development) Array of tile indices that has been 
							observed in an earlier epoch
		plot			 :: (optional, future development) Plots the tile centers that are observed.
		verbose			 :: Toggle verbose flag for print statements.
				   
		
		'''
		
		includeTiles = np.cumsum(self.tileProbs) < CI
		includeTiles[np.sum(includeTiles)] = True
		
		thresholdTileProb = self.tileProbs[includeTiles][-1]

		observedTime = 0 ## Initiating the observed times
		elapsedTime = 0  ## Initiating the elapsed times. Time since observation begun.
		scheduled = np.array([]) ## tile indices scheduled for observation
		obs_tile_altAz = []
		ObsTimes = []
		pVal_observed = []
		ii = 0
		observed_count = 0
		sun_ra = []
		sun_dec = []
		moon_ra = []
		moon_dec = []
		lunar_illumination = []
		
		
		[_, _, _, altAz_sun] = self.tileVisibility(eventTime, gps=True)
		## This will be changed in the future. The argument of time that 
		## will be passed should be an astropy time quantity so that there
		## is no need to specify what kind of time format is used.

		localTime = Time(eventTime, format='gps') ## This variable name is incorrect!
		
		if altAz_sun.alt.value >= -18.0:
			if verbose: print('Local event time: '+ str(localTime.utc.datetime)+'; Sun is above the horizon')
			eventTime = self.advanceToSunset(eventTime, integrationTime)
			if verbose:
				localTime = Time(eventTime, format='gps')
				print('Scheduling observations starting: ' + str(localTime.utc.datetime))
		else:
			if verbose:
				print('Local event time: '+ str(localTime.utc.datetime)+'; Scheduling observations right away!')
		
		while elapsedTime <= duration: 
			[tileIndices, tileProbs, altAz_tile, altAz_sun] = self.tileVisibility(eventTime, gps=True)
			localTime = Time(eventTime, format='gps') ## This variable name is incorrect!
			
			if altAz_sun.alt.value < -18.0: 
				# if verbose: 
				# 	print(str(localTime.utc.datetime) + ': Observation mode')
				for jj in np.arange(len(tileIndices)):
					if tileIndices[jj] not in scheduled:
						if tileProbs[jj] >= thresholdTileProb:
							scheduled = np.append(scheduled, tileIndices[jj])
							obs_tile_altAz.append(altAz_tile[jj])
							ObsTimes.append(localTime)
							pVal_observed.append(tileProbs[jj])
							Sun = get_sun(Time(eventTime, format='gps'))
							sun_ra.append(Sun.ra.value)
							sun_dec.append(Sun.dec.value)
							Moon = get_moon(Time(eventTime, format='gps'))
							sunMoonAngle = Sun.separation(Moon)
							phaseAngle = np.arctan2(Sun.distance*np.sin(sunMoonAngle), 
										Moon.distance - Sun.distance *
										np.cos(sunMoonAngle))
							illumination = 0.5*(1.0 + np.cos(phaseAngle))
							
							# if verbose: print('Lunar illumination = ' + str(illumination))
							lunar_illumination.append(illumination)
							
							moon_ra.append(Moon.ra.value)
							moon_dec.append(Moon.dec.value)
							observedTime += integrationTime ## Tracking observations
							break
				
			else:
				if verbose: 
					localTime = Time(eventTime, format='gps')
					print("Epoch completed!")
					print(str(localTime.utc.datetime) + ': Sun above the horizon')
				eventTime = self.advanceToSunset(eventTime, integrationTime)
				if verbose:
					localTime = Time(eventTime, format='gps')
					print('Advancing time to ' + str(localTime.utc.datetime))

			ii += 1
			eventTime += integrationTime
			elapsedTime += integrationTime
			# if verbose:
			# 	print('elapsedTime --->' + str(elapsedTime))
			# 	print('observedTime --->' + str(observedTime))
				
		if np.any(scheduled) == False: 
			print("The input tiles are not visible from the given site")
			return None

		else: 
			tile_obs_times = []
			airmass = []
			alttiles = []
			for ii in np.arange(len(scheduled)):
				tile_obs_times.append(ObsTimes[ii].utc.datetime)
				# if verbose: print(str(ObsTimes[ii].utc.datetime) + '\t' + str(int(scheduled[ii])))
				altAz_tile = self.tiles[int(scheduled[ii])].transform_to(AltAz(obstime=\
										ObsTimes[ii], location=self.Observatory))
				alttiles.append(obs_tile_altAz[ii].alt.value)
				airmass.append(obs_tile_altAz[ii].secz)
			
			pVal_observed = np.array(pVal_observed)
			sun_ra = np.array(sun_ra)
			sun_dec = np.array(sun_dec)
			moon_ras = np.array(moon_ra)
			moon_decs = np.array(moon_dec)
			moonTile = []
			for moon_ra, moon_dec in zip(moon_ras, moon_decs):
				moonTile.append(self.tileObj.sourceTile(moon_ra, moon_dec))

			alttiles = np.array(alttiles)
			moonTile = np.array(moonTile)
			
			## Angular separation = arccos(sin(dec1)*sin(dec2) + (cos(dec1)*cos(dec2)*cos(ra1 - ra2))
			
			RA_scheduled_tile = np.deg2rad(self.tileData['ra_center']\
								[np.isin(self.tileData['ID'], scheduled.astype('int'))])
			Dec_scheduled_tile = np.deg2rad(self.tileData['dec_center']\
								[np.isin(self.tileData['ID'], scheduled.astype('int'))])
			RA_Moontile = np.deg2rad(self.tileData['ra_center']\
							[np.isin(self.tileData['ID'], moonTile)])
			Dec_Moontile = np.deg2rad(self.tileData['dec_center']\
							[np.isin(self.tileData['ID'], moonTile)])

			# moonTileDist = np.rad2deg(np.arccos(np.sin(Dec_scheduled_tile)*np.sin(Dec_Moontile) +\
			# 			  (np.cos(Dec_scheduled_tile)*np.cos(Dec_Moontile)*\
			# 			  np.cos(RA_scheduled_tile - RA_Moontile))))

			# moonDist = np.rad2deg(np.arccos(np.sin(Dec_scheduled_tile)*np.sin(np.deg2rad(moon_decs)) +\
			# 			  (np.cos(Dec_scheduled_tile)*np.cos(np.deg2rad(moon_decs))*\
			# 			  np.cos(RA_scheduled_tile - np.deg2rad(moon_ras)))))

			## Slewing angle computation ##
			slewDist = [0.0]
			for ii in range(1, len(tile_obs_times)):
				slewDist.append(np.rad2deg(np.arccos(np.sin(Dec_scheduled_tile[ii])*np.sin(Dec_scheduled_tile[ii-1]) +\
							(np.cos(Dec_scheduled_tile[ii])*np.cos(Dec_scheduled_tile[ii-1])*\
							np.cos(RA_scheduled_tile[ii] - RA_scheduled_tile[ii-1])))))


			slewDist = np.array(slewDist)
			df = pd.DataFrame(np.vstack((tile_obs_times, scheduled.astype('int'), np.rad2deg(RA_scheduled_tile), np.rad2deg(Dec_scheduled_tile), pVal_observed, slewDist,\
											airmass, moonTile, lunar_illumination)).T,\
											columns=['Observation_Time', 'Tile_Index', 'RA', 'Dec', 'Tile_Probs', 'Slew Angle (deg)',\
											'Air_Mass', 'Lunar-tile', \
											'Lunar_Illumination'])
			

			if save_schedule:
				if tag is None: 
					tag = self.configParser.get('plot', 'filenametag')
				df.to_csv(self.outdir+tag+"_schedule.csv")

			return df
