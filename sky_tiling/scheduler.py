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
import logging

import configparser
from scipy import interpolate

from astropy.time import Time
from astropy import units as u
from astropy.coordinates import get_sun
from astropy.coordinates import get_body
from .ranked_tiling import RankedTileGenerator

from astropy.coordinates import SkyCoord, EarthLocation, AltAz


############ UNDER CONSTRUCTION ############
	
class Scheduler(RankedTileGenerator):
	'''
	The scheduler class: Inherits from the RankedTileGenerator class. If no attribute 
	is supplied while creating schedular objects, a default instance of ZTF scheduler 
	is created. To generate scheduler for other telescopes use the corresponding site
	names which can be obtaine from astropy.coordinates.EarthLocation.get_site_names().
	The tile tile coordinate file also needs to be supplied to the variable tileCoord.
	This file needs to have at least three columns, the first being an ID (0, 1, 2, ...),
	the second should be the tile center's ra value and the third the dec value of the 
	same. The utcoffset is the time difference between UTC and the site in hours. 
	'''
	def __init__(self, configfile, ranked_tiles_csv, astropy_site_location=None, outdir=None, logfile=None):

		configParser = configparser.ConfigParser()
		configParser.read(configfile)
		logging.basicConfig(filename=logfile, level=logging.INFO, format='%(message)s')

		self.tileCoord = configParser.get('tileFiles', 'tileFile')
		self.outdir = outdir

		if astropy_site_location is not None:
			self.Observatory = astropy_site_location
		
		else:
			site = configParser.get('observation', 'site')
			self.Observatory = EarthLocation.of_site(site)
		
		self.tileData = np.genfromtxt(self.tileCoord, names=True)
		df_ranked_tiles = pd.read_csv(ranked_tiles_csv, header=0, index_col=False)
		self.tileIndices = df_ranked_tiles["tile_index"].values
		self.tileProbs = df_ranked_tiles["tile_prob"].values

		self.tiles = SkyCoord(ra = self.tileData['ra_center']*u.degree, 
					    dec = self.tileData['dec_center']*u.degree, 
					    frame = 'icrs') ### Tile(s) 
		
		# ------------------ FAST VISIBILITY PRECOMPUTE (cached once) ------------------
		# Use the same min-alt cut as tileVisibility (30 deg) for consistency
		self._min_alt_deg = 30.0
		self._min_alt_rad = np.deg2rad(self._min_alt_deg)

		# Site latitude (radians) and its sin/cos
		self._phi = self.Observatory.lat.radian
		self._sphi = np.sin(self._phi)
		self._cphi = np.cos(self._phi)

		# Ranked tile coordinates (radians)
		ranked_ids = self.tileIndices.astype(int)
		self._alpha = np.deg2rad(self.tileData['ra_center'][ranked_ids])    # RA
		self._delta = np.deg2rad(self.tileData['dec_center'][ranked_ids])   # Dec
		sdel = np.sin(self._delta)
		cdel = np.cos(self._delta)

		# Threshold hour-angle H0 per tile solving sin h0 ≤ sφ sδ + cφ cδ cos H
		num = np.sin(self._min_alt_rad) - self._sphi * sdel
		den = self._cphi * cdel
		# Where den == 0 (at poles), mark as invalid to avoid division warning
		cosH0 = np.full_like(num, np.nan, dtype=float)
		mask_den = den != 0.0
		cosH0[mask_den] = num[mask_den] / den[mask_den]
		# Tiles that can ever reach min altitude (clip to [-1,1], rest remain NaN)
		cosH0 = np.clip(cosH0, -1.0, 1.0, where=mask_den, out=cosH0)
		self._H0 = np.arccos(cosH0)                # NaN where never reaches min-alt
		self._visible_ever_mask = ~np.isnan(self._H0)
		# ------------------------------------------------------------------------------

	# ---------------------------- PRIVATE FAST HELPERS ----------------------------
	def _lst_rad(self, t: Time):
		# Apparent sidereal angle at site longitude (radians)
		return t.sidereal_time('apparent', longitude=self.Observatory.lon).radian

	@staticmethod
	def _wrap_pi(x):
		# wrap to (-pi, +pi]
		return (x + np.pi) % (2*np.pi) - np.pi

	@staticmethod
	def _wrap_2pi_pos(x):
		# wrap to [0, 2pi)
		return x % (2*np.pi)

	def _jump_to_first_visible_tile(self, t_start: Time):
		"""
		Analytic, O(N) jump: from t_start, find the earliest time when ANY ranked tile
		crosses the 30° altitude boundary. Returns a Time (or the same t_start if none).
		"""
		v = self._visible_ever_mask
		if not np.any(v):
			return t_start

		theta = self._lst_rad(t_start)              # LST (rad)
		H = self._wrap_pi(theta - self._alpha)      # hour angle now
		H0 = self._H0                                # boundary per tile

		# Already above?
		up_now = np.abs(H[v]) <= H0[v]
		if np.any(up_now):
			return t_start

		# Smallest positive ΔH to reach ±H0
		dH_plus  = self._wrap_2pi_pos(+H0[v] - H[v])
		dH_minus = self._wrap_2pi_pos(-H0[v] - H[v])
		dH = np.minimum(dH_plus, dH_minus)
		dH_min = float(np.min(dH))

		# Convert ΔH to sidereal seconds; add tiny epsilon to clear boundary
		dt_sec = dH_min * (86164.0905 / (2*np.pi)) + 1.0
		return t_start + dt_sec * u.s
	# -----------------------------------------------------------------------------

	def tileVisibility(self, time):
		'''
		METHOD	:: This method takes as input the time (gps or mjd) of observation
				   and the observatory site name, and returns the alt and az of the 
				   ranked tiles. It also returns the alt and az of the sun.
		time	:: The time at which observation is made. Assumes astropy Time object is passed.

		'''
		altAz_tile = self.tiles[self.tileIndices.astype(int)].transform_to(AltAz(obstime=time, location=self.Observatory))
		altAz_sun = get_sun(time).transform_to(AltAz(obstime=time, location=self.Observatory))
		whichTilesUp = altAz_tile.alt.value > 30.0  ### Checks which tiles are up		

        #return [altAz_tile, self.tileProbs, altAz_sun]
		return [self.tileIndices[whichTilesUp], self.tileProbs[whichTilesUp], altAz_tile[whichTilesUp], altAz_sun]
		

	def advanceToSunset(self, eventTime, intTime):
		'''
		This method is called when the observation scheduler determines that the sun is 
		above horizon. It finds the nearest time prior to the next sunset within +/- 
		integration time and then advances the scheduler code to that point. 
		This speeds up the code by refraining from computing pointings during the daytime.
		Currently only works with GPS time.
		
		eventTime	:: The GPS time for which the advancement is to be computed.
		intTim		:: The integration time for the obsevation
		returns     :: astropy Time object
		'''
		
		dt = np.arange(0, 24*3600 + intTime, intTime)
		time = Time(eventTime + dt, format='gps')
		altAz_sun = get_sun(time).transform_to(AltAz(obstime=time, location=self.Observatory))
		timeBeforeSunset = (eventTime + dt)[altAz_sun.alt.value < -18.0][0]
		return Time(timeBeforeSunset, format='gps')


	############### NOT TESTED AND NOT USED CURRENTLY ############
	# def whenThisTileSets(self, index, currentTime, duration):
	# 	'''
	# 	This method approximately computes the amount of time left in seconds for a tile
	# 	to set below 20 degrees.
		
	# 	index		::	The index of the tile for which setting tile is to be found
	# 	currentTime	::	The current time when this tile is scheduled
	# 	'''
	# 	thisTile = SkyCoord(ra = self.tileData['ra_center'][index]*u.degree, 
	# 				    dec = self.tileData['dec_center'][index]*u.degree, 
	# 				    frame = 'icrs') ### Tile(s)
	# 	dt = np.arange(0, duration + 1.0, 1.0)
	# 	times = Time(currentTime + dt, format='gps')
	# 	altAz_tile = thisTile.transform_to(AltAz(obstime=times,
	# 											location=self.Observatory))
		
	# 	setTime = None
	# 	if altAz_tile.alt.value[-1] < 20.0:
	# 		s = interpolate.UnivariateSpline(altAz_tile.alt.value, times.value, k=3)
	# 		setTime = s(20.0)
			
	# 	return setTime

	def observationSchedule(self, duration, eventTime, integrationTime=120, CI=0.9, latency=900,
							observedTiles=None, save_schedule=False, tag=None):
		'''
		METHOD	:: This method takes the duration of observation, time of the GW trigger
				   integration time per tile as input and outputs the observation
				   schedule.
				   
		duration   		 :: Total duration of the observation in seconds.
		eventTime  		 :: The gps time of the time of the GW trigger.
		latency          :: Time between eventTime and start of observations (default=900s)
		integrationTime  :: Time spent per tile in seconds (default == 120 seconds)
		observedTiles	 :: (Future development) Array of tile indices that has been 
							observed in an earlier epoch
				   
		'''

		# includeTiles = np.cumsum(self.tileProbs) < CI
		# includeTiles[0] = True ## Always include the first tile ##
		
		# thresholdTileProb = self.tileProbs[includeTiles][-1]

		observedTime = 0 ## Initiating the observed times ##
		scheduled = np.array([]) ## tile indices scheduled for observation ##
		obs_tile_altAz = []
		ObsTimes = []
		pVal_observed = []
		sun_ra = []
		sun_dec = []
		moon_ra = []
		moon_dec = []
		lunar_illumination = []
		moon_altitude = []
		
		## time clock initialization ##
		time_clock_astropy = Time(eventTime + latency, format='gps')
		[_, _, _, altAz_sun] = self.tileVisibility(time_clock_astropy)

		## Checking and logging if sun is up; advancing to sunset ##
		sun_alt_deg = float(altAz_sun.alt.value)
		if sun_alt_deg >= -18.0: 
			logging.info(f'Event time (UTC): {time_clock_astropy.utc.datetime}; Sun altitude = {sun_alt_deg:.1f}° (twilight/day)')
			time_clock_astropy = self.advanceToSunset(time_clock_astropy.to_value('gps'), integrationTime)
			# FAST: jump to first time any tile crosses 30° (no time scan)
			time_clock_astropy = self._jump_to_first_visible_tile(time_clock_astropy)
			logging.info('Scheduling observations starting (UTC): ' + str(time_clock_astropy.utc.datetime))
		## Logging when sun is down ##
		else: logging.info(f'Event time (UTC): {time_clock_astropy.utc.datetime}; Sun altitude = {sun_alt_deg:.1f}° (astronomical night) — scheduling now!')
		
		## Start scheduling observations ##
		while observedTime <= duration: 
			[tileIndices, tileProbs, altAz_tile, altAz_sun] = self.tileVisibility(time_clock_astropy)
			sun_alt_now = float(altAz_sun.alt.value)

			## If night (not so relevant for the very first observation), try to schedule; if nothing is up, jump once to first visibility within the night
			if sun_alt_now < -18.0:
				if len(tileIndices) == 0:
					# FAST analytic jump (no loops): move to first time any tile crosses 30°
					t_candidate = self._jump_to_first_visible_tile(time_clock_astropy)
					# Only jump if we really moved forward
					if t_candidate > time_clock_astropy:
						time_clock_astropy = t_candidate
						tileIndices, tileProbs, altAz_tile, altAz_sun = self.tileVisibility(time_clock_astropy)

				else:
					for jj in np.arange(len(tileIndices)):
    				## out of all the visible tiles find the one that is above the probability threshold and not scheduled yet ##
						if tileIndices[jj] not in scheduled:
							# if tileProbs[jj] >= thresholdTileProb:
								scheduled = np.append(scheduled, tileIndices[jj])
								obs_tile_altAz.append(altAz_tile[jj])
								ObsTimes.append(time_clock_astropy)
								pVal_observed.append(tileProbs[jj])
								Sun = get_sun(time_clock_astropy)
								sun_ra.append(Sun.ra.value)
								sun_dec.append(Sun.dec.value)
								Moon = get_body("moon", time_clock_astropy)
								sunMoonAngle = Sun.separation(Moon)
								phaseAngle = np.arctan2(Sun.distance*np.sin(sunMoonAngle), Moon.distance - Sun.distance * np.cos(sunMoonAngle))
								illumination = 0.5*(1.0 + np.cos(phaseAngle))
								moon_altAz = get_body("moon", time_clock_astropy).transform_to(AltAz(obstime=time_clock_astropy, location=self.Observatory))
								lunar_illumination.append(illumination)
								moon_altitude.append(moon_altAz.alt.value)
								moon_ra.append(Moon.ra.value)
								moon_dec.append(Moon.dec.value)
								break
							## break soon as you find the desired tile ##
			
			else:
				## this is relevant only at the end of an epoch ##
				logging.info("Epoch completed!")
				logging.info(str(time_clock_astropy.utc.datetime) + f': Sun altitude = {sun_alt_now:.1f}° (twilight/day)')
				time_clock_astropy = self.advanceToSunset(time_clock_astropy.to_value('gps'), integrationTime)
				# FAST: jump to first visibility after sunset
				time_clock_astropy = self._jump_to_first_visible_tile(time_clock_astropy)
				logging.info('Advancing time (UTC) to ' + str(time_clock_astropy.utc.datetime))

			## continue in the loop ##
			observedTime += integrationTime ## Tracking observations ##
			time_clock_astropy += integrationTime * u.s
				
		if np.any(scheduled) == False: 
			logging.info("The input tiles are not visible from the given site")
			return None

		else: 
			tile_obs_times = []
			airmass = []
			alttiles = []
			for ii in np.arange(len(scheduled)):
				tile_obs_times.append(ObsTimes[ii].utc.datetime)
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
				moonTile.append(self.sourceTile(moon_ra, moon_dec))


			alttiles = np.array(alttiles)
			moonTile = np.array(moonTile)
			
			## Formula: Angular separation = arccos(sin(dec1)*sin(dec2) + (cos(dec1)*cos(dec2)*cos(ra1 - ra2)) ##
			
			RA_scheduled_tile = np.deg2rad(self.tileData['ra_center']\
								[np.isin(self.tileData['ID'].astype(int), scheduled.astype('int'))])
			Dec_scheduled_tile = np.deg2rad(self.tileData['dec_center']\
								[np.isin(self.tileData['ID'].astype(int), scheduled.astype('int'))])
			RA_Moontile = np.deg2rad(self.tileData['ra_center'][moonTile])
			Dec_Moontile = np.deg2rad(self.tileData['dec_center'][moonTile])

			moonTileDist = np.rad2deg(np.arccos(np.sin(Dec_scheduled_tile)*np.sin(Dec_Moontile) +\
						  (np.cos(Dec_scheduled_tile)*np.cos(Dec_Moontile)*\
						  np.cos(RA_scheduled_tile - RA_Moontile))))

			moonDist = np.rad2deg(np.arccos(np.sin(Dec_scheduled_tile)*np.sin(np.deg2rad(moon_decs)) +\
						  (np.cos(Dec_scheduled_tile)*np.cos(np.deg2rad(moon_decs))*\
						  np.cos(RA_scheduled_tile - np.deg2rad(moon_ras)))))

			## Slewing angle computation ##
			slewDist = [0.0]
			for ii in range(1, len(tile_obs_times)):
				slewDist.append(np.rad2deg(np.arccos(np.sin(Dec_scheduled_tile[ii])*np.sin(Dec_scheduled_tile[ii-1]) +\
							(np.cos(Dec_scheduled_tile[ii])*np.cos(Dec_scheduled_tile[ii-1])*\
							np.cos(RA_scheduled_tile[ii] - RA_scheduled_tile[ii-1])))))


			slewDist = np.array(slewDist)
			df = pd.DataFrame(np.vstack((tile_obs_times, scheduled.astype('int'), self.tileData['ra_center'][scheduled.astype('int')], self.tileData['dec_center'][scheduled.astype('int')], pVal_observed, slewDist,\
											airmass, moonTile, moonTileDist, moonDist, lunar_illumination, np.array(moon_altitude))).T, columns=['Observation_Time', 'Tile_Index', 'RA', 'Dec', 'Tile_Probs', 'Slew Angle (deg)','Air_Mass', 
										    'Lunar-tile', 'Lunar-tile separation (deg)', 'Lunar separation (deg)', 'Lunar_Illumination', 'Lunar_altitude'])

			if save_schedule:
				if tag is None: 
					tag = self.configParser.get('plot', 'filenametag')
				df.to_csv(self.outdir+tag+"_schedule.csv", index=False)

			return df
