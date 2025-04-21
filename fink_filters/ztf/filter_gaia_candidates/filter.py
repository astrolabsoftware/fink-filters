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
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import BooleanType

from fink_filters.tester import spark_unit_tests

import pandas as pd


def gaia_dr3_candidates_(DR3Name) -> pd.Series:
    """Return alerts with counterpart in the Gaia DR3 catalog

    Parameters
    ----------
    DR3Name: Pandas series
        Column containing the Gaia DR3 cross-match values

    Returns
    -------
    out: pandas.Series of bool
        Return a Pandas Series with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    --------
    >>> pdf = pd.read_parquet('datatest/regular')
    >>> classification = gaia_dr3_candidates_(pdf['DR3Name'])
    >>> nalerts = len(pdf[classification]['objectId'])
    >>> print(nalerts)
    297
    """
    # string nan...
    f_gaia = DR3Name.apply(lambda x: x != "nan")

    return f_gaia


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def gaia_dr3_candidates(DR3Name) -> pd.Series:
    """Pandas UDF version of gaia_dr3_candidates_ for Spark

    Parameters
    ----------
    DR3Name: Spark DataFrame Column
        Column containing the Gaia DR3 cross-match values

    Returns
    -------
    out: pandas.Series of bool
        Return a Pandas Series with the appropriate flag:
        False for bad alert, and True for good alert.

    Examples
    --------
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> df = spark.read.format('parquet').load('datatest/regular')
    >>> f = 'fink_filters.ztf.filter_gaia_candidates.filter.gaia_dr3_candidates'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    297

    """
    f_simbad = gaia_dr3_candidates_(DR3Name)

    return f_simbad


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
