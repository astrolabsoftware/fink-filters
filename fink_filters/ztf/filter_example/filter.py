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
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import BooleanType

from fink_filters.tester import spark_unit_tests

import pandas as pd


def example_filter_(cdsxmatch, magpsf) -> pd.Series:
    """Return alerts with counterpart in the SIMBAD database with difference magnitude above 20

    Parameters
    ----------
    cdsxmatch: Pandas series
        Column containing the SIMBAD cross-match values
    magpsf: Pandas series
        Column containing difference magnitudes

    Returns
    -------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    --------
    >>> pdf = pd.read_parquet('datatest/magnetic_cvs/')
    >>> classification = example_filter_(pdf['cdsxmatch'], pdf['candidate'].apply(lambda x: x['magpsf']))
    >>> nalerts = len(pdf[classification]['objectId'])
    >>> print(nalerts)
    1

    >>> pdf[classification].groupby('cdsxmatch').count().sort_values('objectId', ascending=False)['objectId'].head()
    cdsxmatch
    QSO    1
    Name: objectId, dtype: int64
    """
    myfilter = ~cdsxmatch.isin(["Unknown", "Transient"])

    # mask all kind of failures
    mask = cdsxmatch.apply(lambda x: x.startswith("Fail"))
    myfilter[mask] = False

    # Remove static objects -- https://github.com/astrolabsoftware/fink-filters/issues/120
    mask_gal = cdsxmatch.apply(lambda x: x.startswith("Galaxy"))
    myfilter[mask_gal] = False

    mask_mag = magpsf <= 20.5
    myfilter[mask_mag] = False

    return myfilter


@pandas_udf(BooleanType())
def example_filter(cdsxmatch: pd.Series, magpsf: pd.Series) -> pd.Series:
    """Pandas UDF version of example_filter_ for Spark

    Parameters
    ----------
    cdsxmatch: Spark DataFrame Column
        Column containing the SIMBAD cross-match values
    magpsf: Spark DataFrame Column
        Column containing the difference magnitudes

    Returns
    -------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    --------
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> df = spark.read.format('parquet').load('datatest/magnetic_cvs/')
    >>> f = 'fink_filters.ztf.filter_example.filter.example_filter'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    1
    """
    f_simbad = example_filter_(cdsxmatch, magpsf)

    return f_simbad


if __name__ == "__main__":
    """Execute the test suite"""

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
