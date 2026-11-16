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


def flux_to_mag(flux: float) -> float:
    """Convert flux in nJy to AB magnitude.

    Parameters
    ----------
    flux: float
        Flux in nJy.

    Returns
    -------
    out: float
        AB magnitude.
    """
    return ZP_NJY - 2.5 * np.log10(flux)


def mag_to_flux(mag: float) -> float:
    """Convert AB magnitude to flux in nJy.

    Parameters
    ----------
    mag: float
        AB magnitude.

    Returns
    -------
    out: float
        Flux in nJy.
    """
    return 10 ** ((ZP_NJY - mag) / 2.5)


def mag_rate_to_flux_rate(mag_rate: float, mag: float) -> float:
    """Convert a magnitude rate of change to a flux rate of change.

    Notes
    -----
    Derived from the analytical derivative of the magnitude-flux relation:
    dF/dt = -(ln(10) / 2.5) * F * (dm/dt)

    Parameters
    ----------
    mag_rate: float
        Rate of change in magnitude (mag/day).
    mag: float
        Reference magnitude at which the conversion is evaluated.

    Returns
    -------
    out: float
        Rate of change in flux (nJy/day).
    """
    flux = mag_to_flux(mag)
    return -(np.log(10) / 2.5) * flux * mag_rate


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


def compute_mc_sampling_flux_rate(
    diaSource: pd.DataFrame, prvDiaSources: pd.Series, N: int, seed: int = None
):
    """
    Compute Monte Carlo sampling of flux rate of change per band

    For each entry in prvDiaSources, finds the last previous detection
    matching the current observation's band, then estimates the flux rate
    (flux/day) via Monte Carlo sampling to propagate flux uncertainties.

    Notes
    -----
    Only same-band pairs are used. Entries with no previous detection in
    the same band return NaN. Flux uncertainties are propagated independently
    for both current and previous flux using standard normal sampling.
    Output arrays are aligned with diaSource and prvDiaSources, which share
    the same ordering (i-th entry of mag_rate corresponds to the i-th row of
    diaSource and the i-th entry of prvDiaSources).

    Parameters
    ----------
    diaSource: pd.DataFrame
        Current alert sources. Must contain columns midpointMjdTai,
        psfFlux, psfFluxErr, and band.
    prvDiaSources: pd.Series
        Previous detections indexed by alert. Each entry is either None
        (no previous detection) or a list of diaSource-like dicts, each
        containing midpointMjdTai, psfFlux, psfFluxErr and band fields.
        The list is traversed in reverse to find the last same-band detection.
    N: int
        Number of Monte Carlo samples.
    seed: int, optional
        Random seed for reproducibility (default: None).

    Returns
    -------
    mag_rate: np.ndarray, shape (len(prvDiaSources),)
        Mean flux rate of change (flux/day). NaN where no same-band
        previous detection exists.
    mag_std: np.ndarray, shape (len(prvDiaSources),)
        Standard deviation of flux rate across samples (ddof=1). NaN
        where no same-band previous detection exists.
    lower_rate: np.ndarray, shape (len(prvDiaSources),)
        5th percentile of flux rate distribution. NaN where no same-band
        previous detection exists.
    upper_rate: np.ndarray, shape (len(prvDiaSources),)
        95th percentile of flux rate distribution. NaN where no same-band
        previous detection exists.
    """
    rng = np.random.default_rng(seed)

    current_time = diaSource.midpointMjdTai.to_numpy()
    current_flux = diaSource.psfFlux.to_numpy()
    current_flux_err = diaSource.psfFluxErr.to_numpy()
    current_band = diaSource.band

    def get_last_same_band(sub_dict, band):
        if sub_dict is None:
            return [np.nan, np.nan, np.nan]
        for src in reversed(sub_dict):
            if src["band"] == band:
                return [src["midpointMjdTai"], src["psfFlux"], src["psfFluxErr"]]
        return [np.nan, np.nan, np.nan]

    prv_array = np.array(
        [
            get_last_same_band(sub_dict, cb)
            for sub_dict, cb in zip(prvDiaSources, current_band)
        ]
    )

    mask = ~np.isnan(prv_array).any(axis=1)
    len_mask = mask.sum()

    dt = current_time[mask] - prv_array[mask, 0]  # (len_mask,)

    prv_flux = prv_array[mask, 1]  # (len_mask,)
    prv_flux_err = prv_array[mask, 2]  # (len_mask,)
    cur_flux = current_flux[mask]  # (len_mask,)
    cur_flux_err = current_flux_err[mask]  # (len_mask,)

    samples = rng.normal(0.0, 1.0, (N, len_mask, 2))  # (N, len_mask, 2)
    current_flux_sample = cur_flux + samples[..., 0] * cur_flux_err
    last_flux_sample = prv_flux + samples[..., 1] * prv_flux_err

    sample_rate = (current_flux_sample - last_flux_sample) / dt  # (N, len_mask)

    mag_rate = np.full(len(prvDiaSources), np.nan)
    mag_std = np.full(len(prvDiaSources), np.nan)
    lower_rate = np.full(len(prvDiaSources), np.nan)
    upper_rate = np.full(len(prvDiaSources), np.nan)

    mag_rate[mask] = np.mean(sample_rate, axis=0)
    mag_std[mask] = np.std(sample_rate, axis=0, ddof=1)
    lower_rate[mask] = np.percentile(sample_rate, 5.0, axis=0)
    upper_rate[mask] = np.percentile(sample_rate, 95.0, axis=0)

    return mag_rate, mag_std, lower_rate, upper_rate
