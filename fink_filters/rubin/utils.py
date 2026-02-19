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
"""Utilities for blocks and filters"""

from pyspark.sql.types import BooleanType

import pandas as pd
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from fink_utils.spark.utils import (
    expand_function_from_string,
    FinkUDF,
)

# AB magnitude zero-point for flux in nJy
# m_AB = -2.5 * log10(flux_nJy) + ZP_NJY
ZP_NJY = 31.4


def safe_diaobject_extract(container: dict, key: str):
    """Extract value from dictionary, returning nan if the key does not exist

    Parameters
    ----------
    container: dict
    key: str

    Returns
    -------
    out: Any
        container[key] or NaN if key does not exist in container.
    """
    return container.get(key, np.nan)


def extract_flux_information_static(
    diaSource: pd.DataFrame, diaObject: pd.DataFrame
) -> pd.Series:
    """Extract flux, mean flux, and error on mean flux for the current observation

    Notes
    -----
    Objects with no diaObject, such as SSO, return NaN

    Parameters
    ----------
    diaSource: pd.DataFrame
        Full diaSource section of an alert
    diaObject: pd.DataFrame
        Full diaObject section of an alert

    Returns
    -------
    out: pd.Series
        Series with psfFlux and Mean fluxes per band with error
    """
    # psfFlux = diaSource.apply(lambda x: x["psfFlux"])
    psfFlux = diaSource["psfFlux"]

    # df = pd.DataFrame({"diaSource": diaSource, "diaObject": diaObject})
    df = pd.concat([diaSource, diaObject], axis=1)
    band_psfFluxMean = df.apply(
        lambda row: safe_diaobject_extract(row, "{}_psfFluxMean".format(row["band"])),
        axis=1,
    )
    band_psfFluxMeanErr = df.apply(
        lambda row: safe_diaobject_extract(
            row, "{}_psfFluxMeanErr".format(row["band"])
        ),
        axis=1,
    )

    return psfFlux, band_psfFluxMean, band_psfFluxMeanErr


def compute_diff_flux_from_mean(
    psfFlux: pd.Series, band_psfFluxMean: pd.Series, band_psfFluxMeanErr: pd.Series
) -> pd.Series:
    """Compute the difference between a measurement and mean flux in the same band

    Parameters
    ----------
    psfFlux: pd.Series
        Flux in band b in nJy
    band_psfFluxMean: pd.Series
        Mean flux in band b in nJy
    band_psfFluxMeanErr: pd.Series
        Error on mean flux in band b in nJy

    Returns
    -------
    diff: pd.Series
        Difference in flux
    is_significant: pd.Series
        Series of Booleans based on SNR. True if |diff|/err > 1
    """
    diff = psfFlux - band_psfFluxMean
    is_significant = np.abs(diff) > band_psfFluxMeanErr
    return diff, is_significant


def apply_block(df, function_name):
    """Wrapper around FinkUDF

    Notes
    -----
    This is a convenient wrapper for tests

    Parameters
    ----------
    df: Spark DataFrame
    function_name: str
        Path to the function module.module.function
    """
    filter_func, colnames = expand_function_from_string(df, function_name)
    fink_filter = FinkUDF(
        filter_func,
        BooleanType(),
        "",
    )
    return df.filter(fink_filter.for_spark(*colnames))


def extract_max_flux(diaObject: pd.DataFrame) -> pd.Series:
    """Extract the maximum psfFluxMax across all bands for each diaObject

    Parameters
    ----------
    diaObject: pd.DataFrame
        Full diaObject section of alerts. Must contain columns
        {band}_psfFluxMax for bands

    Returns
    -------
    max_flux: pd.Series
        Maximum psfFluxMax in nJy across all bands per row.
        Returns NaN for rows where all bands are missing.
    """
    BANDS = ["g", "i", "r", "u", "z", "y"]

    flux_cols = ["{}_psfFluxMax".format(band) for band in BANDS]

    available = [col for col in flux_cols if col in diaObject.columns]
    if not available:
        return pd.Series(np.nan, index=diaObject.index)

    max_flux = diaObject[available].max(axis=1)

    return max_flux


def extract_min_flux(diaObject: pd.DataFrame) -> pd.Series:
    """Extract the min psfFluxMax across all bands for each diaObject

    Parameters
    ----------
    diaObject: pd.DataFrame
        Full diaObject section of alerts. Must contain columns
        {band}_psfFluxMin for bands

    Returns
    -------
    min_flux: pd.Series
        Minimum psfFluxMin in nJy across all bands per row.
        Returns NaN for rows where all bands are missing.
    """
    BANDS = ["g", "i", "r", "u", "z", "y"]

    flux_cols = ["{}_psfFluxMin".format(band) for band in BANDS]

    available = [col for col in flux_cols if col in diaObject.columns]
    if not available:
        return pd.Series(np.nan, index=diaObject.index)

    min_flux = diaObject[available].min(axis=1)

    return min_flux


def flux_to_apparent_mag(flux_nJy: np.ndarray) -> np.ndarray:
    """Convert flux in nanoJansky to AB apparent magnitude

    Parameters
    ----------
    flux_nJy: np.ndarray
        Flux in nJy

    Returns
    -------
    mag: np.ndarray
        AB apparent magnitude. NaN where flux <= 0 or non-finite.
    """
    flux = np.atleast_1d(np.asarray(flux_nJy, dtype=float))
    mag = np.full_like(flux, np.nan)

    valid = (flux > 0) & np.isfinite(flux)
    mag[valid] = -2.5 * np.log10(flux[valid]) + ZP_NJY

    return mag


def obs_to_abs_mag(
    m_obs: np.ndarray, z: np.ndarray, H0: float = 70, Om0: float = 0.3
) -> np.ndarray:
    """Convert observed apparent magnitude to absolute magnitude

    Parameters
    ----------
    m_obs: np.ndarray
        Observed apparent magnitude
    z: np.ndarray
        Redshift
    H0: float, optional
        Hubble constant in km/s/Mpc (default: 70)
    Om0: float, optional
        Matter density parameter (default: 0.3)

    Returns
    -------
    M_abs: np.ndarray
        Absolute magnitude. NaN where redshift or magnitude is invalid.
    """
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

    mag = np.atleast_1d(np.asarray(m_obs, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))

    result = np.full_like(z, np.nan, dtype=float)
    valid = (z > 0) & np.isfinite(z) & np.isfinite(mag)

    if np.any(valid):
        dl_pc = cosmo.luminosity_distance(z[valid]).to("pc").value
        mu = 5 * np.log10(dl_pc / 10)
        result[valid] = mag[valid] - mu

    return result


def compute_peak_absolute_magnitude(
    diaObject: pd.DataFrame,
    legacydr8_zphot: pd.Series,
    H0: float = 70,
    Om0: float = 0.3,
) -> pd.Series:
    """Compute peak absolute magnitude from max flux across bands

    Extracts the maximum psfFluxMax across g, i, r, u, z, y bands,
    converts to apparent magnitude, then to absolute magnitude using
    the photo-z from legacydr8_zphot.

    Notes
    -----
    Returns NaN for objects with no valid redshift, no positive flux,
    or missing band data. Output preserves the input row ordering.

    Parameters
    ----------
    diaObject: pd.DataFrame
        Full diaObject section of alerts. Must contain columns
        {band}_psfFluxMax for bands g, i, r, u, z, y.
    legacydr8_zphot: pd.Series
        Photometric redshift from Legacy DR8 cross-match
    H0: float, optional
        Hubble constant in km/s/Mpc (default: 70)
    Om0: float, optional
        Matter density parameter (default: 0.3)

    Returns
    -------
    out: pd.Series
        Peak absolute magnitude per row, ordered as input.
        NaN where conversion is not possible.
    """
    max_flux = extract_max_flux(diaObject)
    apparent_mag = flux_to_apparent_mag(max_flux.values)
    absolute_mag = obs_to_abs_mag(apparent_mag, legacydr8_zphot.values, H0=H0, Om0=Om0)

    return pd.Series(absolute_mag, name="estimated_absoluteMagnitude")
