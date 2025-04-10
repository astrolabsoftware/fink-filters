# Copyright 2023 AstroLab Software
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
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import BooleanType

from fink_filters.tester import spark_unit_tests

import pandas as pd


def simbad_grav_candidates_(cdsxmatch) -> pd.Series:
    """Return alerts with Gravitational counterpart in the SIMBAD database

    Parameters
    ----------
    cdsxmatch: Pandas series
        Column containing the SIMBAD cross-match values

    Returns
    -------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    --------
    >>> pdf = pd.read_parquet('datatest/regular')
    >>> classification = simbad_grav_candidates_(pdf['cdsxmatch'])
    >>> nalerts = len(pdf[classification]['objectId'])
    >>> print(nalerts)
    1

    >>> pdf[classification].groupby('cdsxmatch').count().sort_values('objectId', ascending=False)['objectId'].head()
    cdsxmatch
    LensedImage    1
    Name: objectId, dtype: int64
    """
    f_simbad = cdsxmatch.isin([
        "Gravitation",
        "LensingEv",
        "GravLensSystem",
        "GravLens",
        "LensedImage",
        "LensedG",
        "LensedQ",
        "BlackHole",
        "GravWaveEvent",
    ])

    return f_simbad


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def simbad_grav_candidates(cdsxmatch) -> pd.Series:
    """Pandas UDF version of simbad_grav_candidates_ for Spark

    Parameters
    ----------
    cdsxmatch: Spark DataFrame Column
        Column containing the SIMBAD cross-match values

    Returns
    -------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    --------
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> df = spark.read.format('parquet').load('datatest/regular')
    >>> f = 'fink_filters.filter_simbad_grav_candidates.filter.simbad_grav_candidates'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    1

    """
    f_simbad = simbad_grav_candidates_(cdsxmatch)

    return f_simbad


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
