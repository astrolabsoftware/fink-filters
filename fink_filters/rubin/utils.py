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

from fink_utils.spark.utils import (
    expand_function_from_string,
    FinkUDF,
)


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
