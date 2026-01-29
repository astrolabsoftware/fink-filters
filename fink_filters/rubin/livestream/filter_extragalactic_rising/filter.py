# Copyright 2019-2026 AstroLab Software
# Author: Anais Moller
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
"""Return LSST alerts with matches in catalogs to a galaxy"""

import pandas as pd
import fink_filters.rubin.blocks as fb


DESCRIPTION = "Select alerts that are extragalactic candidates, new and rising in at least one filter"


# def processor_risingfading(df):
#     """_summary_
#
#     Parameters
#     ----------
#     df : pd.DataFrame
#         Alert data with columns diaSource and diaObject
#
#     Returns
#     -------
#     out: pd.Series
#         Series with psfFlux and Mean fluxes per band with error
#     """
#     # Extract band from diaSource dictionary
#     df["band"] = df["diaSource"].apply(lambda x: x["band"])
#     df["psfFlux"] = df["diaSource"].apply(lambda x: x["psfFlux"])
#     bands = df.band.unique()
#     conditions = [df["band"] == band for band in bands]
#     # Get mean flux for each band from diaObject dictionary
#     choices_mean = [
#         df["diaObject"].apply(lambda x: x[f"{band}_psfFluxMean"]) for band in bands
#     ]
#     df["band_psfFluxMean"] = np.select(conditions, choices_mean, default=np.nan)
#     # Get mean flux error for each band from diaObject dictionary
#     choices_err = [
#         df["diaObject"].apply(lambda x: x[f"{band}_psfFluxErrMean"]) for band in bands
#     ]
#     df["band_psfFluxErrMean"] = np.select(conditions, choices_err, default=np.nan)
#
#     return df[["psfFlux", "band_psfFluxMean", "band_psfFluxErrMean"]].to_numpy()


def extragalactic_rising_candidate(
    simbad_otype: pd.Series,
    mangrove_lum_dist: pd.Series,
    ra: pd.Series,
    dec: pd.Series,
    is_sso: pd.Series,
    gaiadr3_DR3Name: pd.Series,
    gaiadr3_Plx: pd.Series,
    gaiadr3_e_Plx: pd.Series,
    vsx_Type: pd.Series,
    psfFlux: pd.Series,
    band_psfFluxMean: pd.Series,  # FIXME: does not exist in the alert packet!
    band_psfFluxErrMean: pd.Series,  # FIXME: does not exist in the alert packet!
    nDiaSources: pd.Series,
) -> pd.Series:
    """Flag for alerts in Rubin that are new and rising extragalactic candidates

    Parameters
    ----------
    simbad_otype: pd.Series
        Type xmatched SIMBAD
    mangrove_lum_dist: pd.Series
        Luminosity distance of xmatch with Mangrove
    ra: pd.Series
        Right ascension
    dec: pd.Series
        Declination
    is_sso: pd.Series
        Asteroid tag
    gaiadr3_DR3Name: pd.Series
        Series containing Gaia DR3 names from `xm.gaiadr3_DR3Name`
    gaiadr3_Plx: pd.Series
        Series containing parallax values from `xm.gaiadr3_Plx`
    gaiadr3_e_Plx: pd.Series
        Series containing parallax errors from `xm.gaiadr3_e_Plx`
    vsx_Type: pd.Series
        Series containing VSX variable star catalog matches
    psfFlux: pd.Series
        Alert difference image flux
    band_psfFluxMean: pd.Series
        Alert mean flux in appropiate band
    band_psfFluxErrMean: pd.Series
        Alert mean flux error in appropiate band
    nDiaSources: pd.Series
        Number of alerts per object

    Returns
    -------
    pd.Series
        Alerts that are extragalactic and rising
    """
    # Extragalactic filter
    f_extragalactic = extragalactic_rising_candidate(
        simbad_otype,
        mangrove_lum_dist,
        ra,
        dec,
        is_sso,
        gaiadr3_DR3Name,
        gaiadr3_Plx,
        gaiadr3_e_Plx,
        vsx_Type,
        psfFlux,
    )
    # Rising in at least one band
    f_is_rising = fb.b_is_rising(psfFlux, band_psfFluxMean, band_psfFluxErrMean)

    f_new = nDiaSources < 20  # should be lowered after first alerts

    f_extragalactic_rising = f_extragalactic & f_is_rising & f_new

    return f_extragalactic_rising
