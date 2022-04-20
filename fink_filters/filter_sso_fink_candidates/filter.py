# Copyright 2019-2022 AstroLab Software
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

def sso_fink_candidates_(roid) -> pd.Series:
    """ Return alerts considered as Solar System Object candidates by Fink

    Parameters
    ----------
    roid: Pandas series
        Column containing the Solar System label

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> pdf = pd.read_parquet('datatest')
    >>> classification = sso_fink_candidates_(pdf['roid'])
    >>> print(len(pdf[classification]['objectId'].values))
    3

    >>> assert 'ZTF21acqeepb' in pdf[classification]['objectId'].values
    """
    f_roid = roid.astype(int) == 2

    return f_roid

@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def sso_fink_candidates(roid) -> pd.Series:
    """ Pandas UDF version of sso_fink_candidates_ for Spark

    Parameters
    ----------
    roid: Spark DataFrame Column
        Column containing the Solar System label

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> df = spark.read.format('parquet').load('datatest')
    >>> f = 'fink_filters.filter_sso_fink_candidates.filter.sso_fink_candidates'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    3

    """
    f_roid = sso_fink_candidates_(roid)

    return f_roid


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
