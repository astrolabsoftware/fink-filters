# Copyright 2026 AstroLab Software
# Author: Riley Clarke
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
"""Select alerts that are likely to be M dwarf flares, in bluer bands, and high airmass for DCR-based temperature inference"""

import pandas as pd
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import fink_filters.rubin.utils as fu
import fink_filters.rubin.blocks as fb

DESCRIPTION = "Select alerts that are likely to be M dwarf flares, in bluer bands, and high airmass for DCR-based temperature inference"
HBASE_SUPPORT = True


def isolated_source(current_mjd, prv_sources, lo_hours=4.0, hi_hours=48.0):
    """
    Check if a source is isolated in time, i.e. no other sources for the same object
    between lo_hours and hi_hours of the current source.

    Parameters
    ----------
    current_mjd: float
        MJD of the current source
    prv_sources: list of dict
        List of previous sources for the same object, each source is a dictionary with keys "midpointMjdTai" and "band"
    lo_hours: float
        Lower bound of the time window in hours
    hi_hours: float
        Upper bound of the time window in hours

    Returns
    -------
    bool
        True if the source is isolated, False otherwise
    """
    if prv_sources is None or len(prv_sources) == 0:
        return True
    prev_mjds = np.array([s["midpointMjdTai"] for s in prv_sources])
    dt_hours = (current_mjd - prev_mjds) * 24.0
    return not np.any((dt_hours >= lo_hours) & (dt_hours <= hi_hours))


def quiescent_colors(prv_sources, mag_fn, bands=("r", "i", "z")):
    """
    Calculate the quiescent colors of a source from its previous sources.

    Parameters
    ----------
    prv_sources: list of dict
        List of previous sources for the same object, each source is a dictionary with keys "band" and "templateFlux"
    mag_fn: function
        Function to convert flux to magnitude
    bands: tuple of str
        Bands to calculate the colors for

    Returns
    -------
    dict
        Dictionary with keys as bands and values as the median magnitude in that band, or NaN if no previous sources in that band
    """
    out = {b: np.nan for b in bands}
    if prv_sources is None or len(prv_sources) == 0:
        return out
    for b in bands:
        fluxes = [
            s["templateFlux"]
            for s in prv_sources
            if (s.get("band") == b) and (s["templateFlux"] is not None)
        ]
        if fluxes:
            out[b] = mag_fn(np.median(fluxes))
    return out


def coord_to_airmass(
    ra: np.ndarray, dec: np.ndarray, midpointTai: np.ndarray
) -> np.ndarray:
    """
    Calculate the airmass for an object given its RA,
    DEC, and observation time.

    Parameters
    ----------
    ra: float
        Right Ascension of the object in degrees.
    dec: float
        Declination of the object in degrees.
    time: astropy.time.Time
        Observation time as an `astropy.time.Time` object.

    Returns
    -------
    float
        Airmass, i.e. secant of zenith angle
    """
    time = Time(midpointTai, format="mjd", scale="tai")
    coord = SkyCoord(
        ra=ra,
        dec=dec,
        unit="deg",
        obstime=time,
        location=EarthLocation.of_site("Cerro Pachon"),
    )
    airmass = coord.transform_to("altaz").secz.value

    return airmass


def mdwarf_flare(
    diaSource: pd.DataFrame,
    diaObject: pd.DataFrame,
    prvDiaSources: pd.Series,
    firstDiaSourceMjdTaiFink: pd.Series,
    gaiadr3_DR3Name: pd.Series,
    gaiadr3_Plx: pd.Series,
    gaiadr3_e_Plx: pd.Series,
) -> pd.Series:
    """
    Returns true for alerts that are likely to be M dwarf flares,
        in bluer bands, and high airmass for DCR-based temperature inference.

    Parameters
    ----------
    diaSource: pd.DataFrame
        Full diaSource section of an alert (dictionary exploded)
    diaObject: pd.DataFrame
        Full diaObject section of an alert (dictionary exploded)
    prvDiaSources: pd.Series
        Series containing previous diaSources for the same object
    firstDiaSourceMjdTaiFink: pd.Series
        Series containing the first diaSource MJD TAI from `xm.firstDiaSourceMjdTaiFink`
    gaiadr3_DR3Name: pd.Series
        Series containing Gaia DR3 names from `xm.gaiadr3_DR3Name`
    gaiadr3_Plx: pd.Series
        Series containing Gaia DR3 parallaxes from `xm.gaiadr3_Plx`
    gaiadr3_e_Plx: pd.Series
        Series containing Gaia DR3 parallax errors from `xm.gaiadr3_e_Plx`

    Returns
    -------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    --------
    >>> from fink_filters.rubin.utils import apply_block
    >>> df2 = apply_block(df, "fink_filters.rubin.livestream.filter_mdwarf_flare.filter.mdwarf_flare")
    >>> df2.count()
    0
    """
    # Require xmatch to Gaia DR3 star if no previous diaSource
    f_new = fb.b_is_new(diaSource.midpointMjdTai, firstDiaSourceMjdTaiFink)
    f_gaia_star = fb.b_xmatched_gaia_star(
        gaiadr3_DR3Name,
        gaiadr3_Plx,
        gaiadr3_e_Plx,
    )
    f_star = ~f_new | (f_new & f_gaia_star)

    # Quality block
    f_quality = fb.b_good_quality(diaSource)

    # Require diaSource detected in u or g band
    f_band = diaSource.band.isin(["u", "g"])

    # Cut on r-i, i-z color
    colors = [quiescent_colors(prv, fu.flux_to_apparent_mag) for prv in prvDiaSources]
    rMag = pd.Series([c["r"] for c in colors], index=diaSource.index)
    iMag = pd.Series([c["i"] for c in colors], index=diaSource.index)
    zMag = pd.Series([c["z"] for c in colors], index=diaSource.index)
    color_available = rMag.notna() & iMag.notna() & zMag.notna()
    f_color = ~color_available | (((rMag - iMag) > 0.53) & ((iMag - zMag) > 0.3))

    # Remove objects with low airmass
    airmass = coord_to_airmass(diaSource.ra, diaSource.dec, diaSource.midpointMjdTai)
    f_airmass = airmass > 1.3

    # Require the star to have brightened by at least 10% relative to its template flux
    valid_baseline = diaSource.templateFlux > 0
    fractional_excursion = diaSource.psfFlux / diaSource.templateFlux
    f_excursion = valid_baseline & (fractional_excursion >= 0.10)

    # Require no other diaSource for the same object between 4 and 48h of the trigger
    f_time_separation = pd.Series(
        [
            isolated_source(mjd, prv)
            for mjd, prv in zip(diaSource.midpointMjdTai, prvDiaSources)
        ],
        index=diaSource.index,
    )

    f_mdwarf_flare = (
        f_star
        & f_band
        & f_color
        & f_airmass
        & f_excursion
        & f_time_separation
        & f_quality
    )

    return f_mdwarf_flare


if __name__ == "__main__":
    """ Execute the test suite """
    # Run the test suite

    from fink_filters.tester import spark_unit_tests

    globs = globals()
    spark_unit_tests(globs, load_rubin_df=True)
