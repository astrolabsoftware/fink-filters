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

def example_filter_(magpsf) -> pd.Series:
    """ Return alerts with difference magnitude above 18

    Parameters
    ----------
    magpsf: Pandas series
        Column containing difference magnitudes
        
    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> pdf = pd.read_parquet('datatest/regular/')
    >>> classification = example_filter_(pdf['candidate'].apply(lambda x: x['magpsf']))
    >>> nalerts = len(pdf[classification]['objectId'])
    >>> print(nalerts)
    169

    >>> pdf[classification].groupby('cdsxmatch').count().sort_values('objectId', ascending=False)['objectId'].head()
    cdsxmatch
    LongPeriodV*    16
    EclBin           8
    Unknown          8
    delSctV*         7
    Mira             7
    Name: objectId, dtype: int64
    """
    
    mask_mag = magpsf <= 18
    
    return mask_mag


@pandas_udf(BooleanType())
def example_filter(magpsf: pd.Series) -> pd.Series:
    """ Pandas UDF version of example_filter_ for Spark

    Parameters
    ----------
    magpsf: Spark DataFrame Column
        Column containing the difference magnitudes

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> df = spark.read.format('parquet').load('datatest/regular/')
    >>> f = 'fink_filters.ztf.filter_bright.filter.example_filter'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    169
    """
    f_simbad = example_filter_(magpsf)

    return f_simbad

if __name__ == "__main__":
    """Execute the test suite"""

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)

