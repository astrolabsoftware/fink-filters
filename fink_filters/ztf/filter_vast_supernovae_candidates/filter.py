# Copyright 2025 AstroLab Software
# Author: Julien Peloton
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

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import BooleanType
from fink_filters.tester import spark_unit_tests
from fink_filters import __file__
import os
import pandas as pd


def vast_supernovae_candidates_(lum_dist, dec, snn_sn_vs_all) -> pd.Series:
    """Return a stream of objects passing VAST candidate filter

    Parameters
    ----------
    lum_dist: Pandas series of floats
        Luminosity distance from mangrove
    dec: Pandas series of floats
        Declination of alerts
    snn_sn_vs_all: Pandas series of floats
        Classification scores from SNN

    Returns
    -------
    out: pandas.Series of bool
        Is interesting according to VAST team

    Examples
    --------
    >>> import numpy as np
    >>> lum_dist = pd.Series([100, 50, 10, np.nan, None])
    >>> dec = pd.Series([20, -30, -20, 10, 0])
    >>> snn_sn_vs_all = pd.Series([0.1, 0.7, 0.3, 0.3, 0.9])
    >>> list(vast_supernovae_candidates_(lum_dist, dec, snn_sn_vs_all).values)
    [False, True, False, False, False]
    """
    f1 = lum_dist < 200
    f2 = dec < -10
    f3 = snn_sn_vs_all > 0.5

    return f1 & f2 & f3


@pandas_udf(BooleanType())
def vast_supernovae_candidates(
    lum_dist: pd.Series, dec: pd.Series, snn_sn_vs_all: pd.Series
) -> pd.Series:
    """Pandas UDF version of vast_supernovae_ for Spark

    Parameters
    ----------
    lum_dist: Pandas series of floats
        Luminosity distance from mangrove
    dec: Pandas series of floats
        Declination of alerts
    snn_sn_vs_all: Pandas series of floats
        Classification scores from SNN

    Returns
    -------
    out: pandas.Series of bool
        Is interesting according to VAST team

    Examples
    --------
    >>> import numpy as np
    >>> lum_dist = pd.Series([100, 50, 10, np.nan, None])
    >>> dec = pd.Series([20, -30, -20, 10, 0])
    >>> snn_sn_vs_all = pd.Series([0.1, 0.7, 0.3, 0.3, 0.9])
    >>> pdf = pd.DataFrame({"lum_dist": lum_dist, "dec": dec, "snn_sn_vs_all": snn_sn_vs_all})
    >>> sdf = spark.createDataFrame(pdf)
    >>> sdf = sdf.withColumn('is_vast', vast_supernovae_candidates('lum_dist', "dec", "snn_sn_vs_all"))
    >>> pdf = sdf.toPandas()
    >>> list(pdf['is_vast'].values)
    [False, True, False, False, False]
    """
    f = vast_supernovae_candidates_(lum_dist, dec, snn_sn_vs_all)
    return f


if __name__ == "__main__":
    """Execute the test suite"""
    # Run the test suite
    globs = globals()
    path = os.path.dirname(__file__)
    spark_unit_tests(globs)
