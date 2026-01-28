# Copyright 2019-2026 AstroLab Software
# Author: Julien Peloton and Anais Moller
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Blocks used to build filters"""

import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from fink_utils.xmatch.simbad import return_list_of_eg_host
from fink_utils.xmatch.vsx import return_list_of_stellar, return_list_of_nonstellar


def b_is_solar_system(is_sso: pd.Series) -> pd.Series:
    """Return alerts that are asteroids according to Rubin

    Parameters
    ----------
    is_sso: pd.Series of booleans
        `pred.is_sso`
    """
    return is_sso


def b_outside_galactic_plane(ra: pd.Series, dec: pd.Series) -> pd.Series:
    """Return alerts outside the galactic plane (+/- |20| deg)

    Parameters
    ----------
    ra: pd.Series of float
        RA in degree
    dec: pd.Series of float
        DEC in degree

    Returns
    -------
    out: pd.Series of booleans
        True if outside the plane. False otherwise
    """
    coords = SkyCoord(ra.astype(float), dec.astype(float), unit="deg")
    b = coords.galactic.b.deg
    mask_away_from_galactic_plane = np.abs(b) > 20
    return pd.Series(mask_away_from_galactic_plane)

def b_xmatched_simbad_galaxy(cdsxmatch: pd.Series) -> pd.Series:
    """Return alerts xmatched to a galaxy with SIMBAD.
    
    Parameters
    ----------
    cdsxmatch : pd.Series
        Series of cross-matched SIMBAD types
        
    Returns
    -------
    out: pd.Series of bool
        Boolean series indicating galaxy or failed matches
    """
    f_galaxy = (cdsxmatch.isin(return_list_of_eg_host))
    return f_galaxy

def b_xmatched_simbad_unknown(cdsxmatch: pd.Series) -> pd.Series:
    """Return alerts xmatched as unknown or failed with SIMBAD.
    
    Parameters
    ----------
    cdsxmatch : pd.Series
        Series of cross-matched SIMBAD types
        
    Returns
    -------
    out: pd.Series of bool
        Boolean series indicating unknown or failed matches
    """
    f_unknown = (cdsxmatch.isin(["Unknown","Fail","Fail 504",None, np.nan]))
    return f_unknown


def b_xmatched_mangrove(lum_dist_values: pd.Series) -> pd.Series:
    """Return alerts xmatched with a Mangrove galaxy.
    
    Parameters
    ----------
    lum_dist_values : pd.Series
        Luminosity distance values from Mangrove/Glade catalog
        
    Returns
    -------
    pd.Series of bool
        Boolean series indicating extragalactic sources with lum_dist > 0
    """
    f_mangrove = lum_dist_values > 0
    return f_mangrove

def b_xmatched_gaia_star(gaiaxmatch_DR3Name: pd.Series, gaiaxmatch_Plx: pd.Series, gaiaxmatch_e_Plx: pd.Series) -> pd.Series:
    """Return alerts xmatched to stars with good parallax with Gaia.
    
    Parameters
    ----------
    gaiaxmatch_DR3Name : pd.Series
        Gaia DR3 source names from cross-match
    gaiaxmatch_Plx : pd.Series
        Parallax values from Gaia cross-match
    gaiaxmatch_e_Plx : pd.Series
        Parallax errors from Gaia cross-match
        
    Returns
    -------
    pd.Series of bool
        Boolean series indicating stellar sources with good parallax (Plx/e_Plx > 5)
    """
    f_xmatched_star = gaiaxmatch_DR3Name != "nan"
    f_plx = gaiaxmatch_Plx/gaiaxmatch_e_Plx>5 #select good parallaxes

    f_gaia = f_xmatched_star & f_plx
    return f_gaia

def b_xmatched_vsx_star(vsxxmatch: pd.Series) -> pd.Series:
    """Return alerts xmatched with stellar sources from VSX catalogue.
    
    Parameters
    ----------
    vsxxmatch : pd.Series
        VSX cross-match results
        
    Returns
    -------
    pd.Series of bool
        Boolean series indicating stellar variable sources
    """
    # All known tags ecept AGN
    f_vsx = vsxxmatch.isin(return_list_of_stellar)
    return f_vsx

def b_xmatched_vsx(vsxxmatch: pd.Series) -> pd.Series:
    """Return alerts xmatched with stellar and non-stellar sources from VSX catalogue.
    
    Parameters
    ----------
    vsxxmatch : pd.Series
        VSX cross-match results
        
    Returns
    -------
    pd.Series of bool
        Boolean series indicating successful VSX matches
    """
    f_vsx = vsxxmatch.isin(return_list_of_nonstellar + return_list_of_stellar)
    return f_vsx


def b_is_rising(psfFlux: pd.Series, band_psfFluxMean: pd.Series, band_psfFluxMeanErr: pd.Series) -> bool:
    """Return True if rising light curve in one filter.
    
    Uses any one flux measurement compared to its mean object
    measurement, taking into account errors.

    Parameters
    ----------
    psfFlux : pd.Series of float
        DiffImage flux in nJy
    band_psfFluxMean : pd.Series of float
        Mean flux in nJy for a given band
    band_psfFluxMeanErr : pd.Series of float
        Error of mean flux in nJy for a given band

    Returns
    -------
    bool
        True if rising, False otherwise
    """
    diff = psfFlux - band_psfFluxMean
    f_rising = False
    if np.abs(diff) > band_psfFluxMeanErr:
        if diff >0:
            f_rising = True
    return f_rising

def b_is_fading(psfFlux: pd.Series, band_psfFluxMean: pd.Series, band_psfFluxMeanErr: pd.Series) -> bool:
    """Return True if fading light curve in one filter.
    
    Uses any one flux measurement compared to its mean object
    measurement, taking into account errors.

    Parameters
    ----------
    psfFlux : pd.Series of float
        DiffImage flux in nJy
    band_psfFluxMean : pd.Series of float
        Mean flux in nJy for a given band
    band_psfFluxMeanErr : pd.Series of float
        Error of mean flux in nJy for a given band

    Returns
    -------
    bool
        True if fading, False otherwise
    """
    f_fading = False
    diff = psfFlux - band_psfFluxMean
    if np.abs(diff) > band_psfFluxMeanErr:
        if diff < 0:
            f_fading = True
    return f_fading

