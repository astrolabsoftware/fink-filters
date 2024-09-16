# Copyright 2024 AstroLab Software
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


def ztf_quality_cuts_(
    rb,
    nbad
) -> pd.Series:
    """Return alerts considered as scientifically valid for ZTF

    Parameters
    ----------
    rb: Pandas series
        Column containing the Real Bogus score
    nbad: Pandas series
        Column containing number of bad pixels

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> pdf = pd.read_parquet('datatest/regular')
    >>> classification = ztf_quality_cuts_(
    ...     pdf['candidate'].apply(lambda x: x['rb']),
    ...     pdf['candidate'].apply(lambda x: x['nbad']))
    >>> print(len(pdf[classification]['objectId'].values))
    320
    """
    high_rb = rb.astype(float) >= 0.55
    no_nbad = nbad.astype(int) == 0

    return high_rb & no_nbad


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def ztf_quality_cuts(
    rb,
    nbad,
) -> pd.Series:
    """Pandas UDF for ztf_quality_cuts_

    Parameters
    ----------
    rb: Pandas series
        Column containing the Real Bogus score
    nbad: Pandas series
        Column containing the number of bad pixels

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> from fink_utils.spark.utils import concat_col
    >>> df = spark.read.format('parquet').load('datatest/regular')

    >>> f = 'fink_filters.filter_quality_cuts.filter.ztf_quality_cuts'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    320
    """
    series = ztf_quality_cuts_(
        rb,
        nbad,
    )

    return series


if __name__ == "__main__":
    """Execute the test suite"""

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
