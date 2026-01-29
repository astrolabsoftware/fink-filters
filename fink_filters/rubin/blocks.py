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

BAD_VALUES = ["Unknown", "Fail", "Fail 504", None, np.nan]


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


def b_xmatched_simbad_galaxy(simbad_otype: pd.Series) -> pd.Series:
    """Return alerts xmatched to a galaxy with SIMBAD.

    Parameters
    ----------
    simbad_otype : pd.Series
        Series of cross-matched SIMBAD types

    Returns
    -------
    out: pd.Series of bool
        Boolean series indicating galaxy or failed matches
    """
    f_galaxy = simbad_otype.isin(return_list_of_eg_host())
    return f_galaxy


def b_xmatched_simbad_unknown(simbad_otype: pd.Series) -> pd.Series:
    """Return alerts xmatched as unknown or failed with SIMBAD.

    Parameters
    ----------
    simbad_otype : pd.Series
        Series of cross-matched SIMBAD types

    Returns
    -------
    out: pd.Series of bool
        Boolean series indicating unknown or failed matches
    """
    f_unknown = simbad_otype.isin(BAD_VALUES)
    return f_unknown


def b_xmatched_mangrove(mangrove_lum_dist: pd.Series) -> pd.Series:
    """Return alerts xmatched with a Mangrove galaxy.

    Parameters
    ----------
    mangrove_lum_dist : pd.Series
        Luminosity distance values from Mangrove/Glade catalog

    Returns
    -------
    out: pd.Series of booleans
        Boolean series indicating extragalactic sources with mangrove_lum_dist > 0
    """
    f_mangrove = mangrove_lum_dist > 0
    return f_mangrove


def b_xmatched_gaia_star(
    gaiadr3_DR3Name: pd.Series,
    gaiadr3_Plx: pd.Series,
    gaiadr3_e_Plx: pd.Series,
) -> pd.Series:
    """Return alerts xmatched to stars with good parallax with Gaia.

    Parameters
    ----------
    gaiadr3_DR3Name : pd.Series
        Gaia DR3 source names from cross-match
    gaiadr3_Plx : pd.Series
        Parallax values from Gaia cross-match
    gaiadr3_e_Plx : pd.Series
        Parallax errors from Gaia cross-match

    Returns
    -------
    out: pd.Series of booleans
        Boolean series indicating stellar sources with good parallax (Plx/e_Plx > 5)
    """
    f_xmatched_star = ~gaiadr3_DR3Name.isin(BAD_VALUES)
    f_plx = gaiadr3_Plx / gaiadr3_e_Plx > 5  # select good parallaxes

    f_gaia = f_xmatched_star & f_plx
    return f_gaia


def b_xmatched_vsx_star(vsx_Type: pd.Series) -> pd.Series:
    """Return alerts xmatched with stellar sources from VSX catalogue.

    Parameters
    ----------
    vsx_Type: pd.Series
        VSX cross-match results

    Returns
    -------
    out: pd.Series of booleans
        Boolean series indicating stellar variable sources
    """
    # All known tags except AGN
    f_vsx = vsx_Type.isin(return_list_of_stellar())
    return f_vsx


def b_xmatched_vsx(vsx_Type: pd.Series) -> pd.Series:
    """Return alerts xmatched with stellar and non-stellar sources from VSX catalogue.

    Parameters
    ----------
    vsx_Type: pd.Series
        VSX cross-match results

    Returns
    -------
    out: pd.Series of booleans
        Boolean series indicating successful VSX matches
    """
    f_vsx = vsx_Type.isin(return_list_of_nonstellar() + return_list_of_stellar())
    return f_vsx


def b_is_rising(
    psfFlux: pd.Series, band_psfFluxMean: pd.Series, band_psfFluxMeanErr: pd.Series
) -> pd.Series:
    """Return alerts with rising lightcurve in one filter.

    Uses any one flux measurement compared to its mean object
    measurement, taking into account errors.

    Parameters
    ----------
    psfFlux : pd.Series
        DiffImage flux in nJy
    band_psfFluxMean : pd.Series
        Mean flux in nJy for a given band
    band_psfFluxMeanErr : pd.Series
        Error of mean flux in nJy for a given band

    Returns
    -------
    out: pd.Series of booleans
        True if rising, False otherwise
    """
    diff = psfFlux - band_psfFluxMean
    is_significant = np.abs(diff) > band_psfFluxMeanErr
    is_rising = is_significant & (diff > 0)

    return is_rising


def b_is_fading(
    psfFlux: pd.Series, band_psfFluxMean: pd.Series, band_psfFluxMeanErr: pd.Series
) -> pd.Series:
    """Return alerts with fading lightcurve in one filter.

    Uses any one flux measurement compared to its mean object
    measurement, taking into account errors.

    Parameters
    ----------
    psfFlux : pd.Series
        DiffImage flux in nJy
    band_psfFluxMean : pd.Series
        Mean flux in nJy for a given band
    band_psfFluxMeanErr : pd.Series
        Error of mean flux in nJy for a given band

    Returns
    -------
    out: pd.Series of booleans
        True if fading, False otherwise
    """
    diff = psfFlux - band_psfFluxMean
    is_significant = np.abs(diff) > band_psfFluxMeanErr
    is_fading = is_significant & (diff < 0)

    return is_fading


def b_is_new(
    midpointMjdTai: pd.Series, firstDiaSourceMjdTaiFink: pd.Series
) -> pd.Series:
    """Return alerts for which the underlying object is seen for the first time by Rubin

    Parameters
    ----------
    midpointMjdTai: pd.Series
        Alert emission date
    firstDiaSourceMjdTaiFink: pd.Series
        MJD for the first detection by Rubin. Temporary
        replacement for diaObject.firstDiaSourceMjdTai
        which is not yet populated by the project.

    Returns
    -------
    out: pd.Series of booleans
        True if new. False otherwise
    """
    is_new = (midpointMjdTai - firstDiaSourceMjdTaiFink) == 0
    return is_new
