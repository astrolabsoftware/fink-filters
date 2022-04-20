# Copyright 2022 AstroLab Software
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

def tracklet_candidates_(tracklet) -> pd.Series:
    """ Return alerts belonging to a tracklet (likely space debris or satellite glint)

    Parameters
    ----------
    tracklet: Pandas series
        Column containing the tracklet label

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> pdf = pd.read_parquet('datatest')
    >>> classification = tracklet_candidates_(pdf['tracklet'])
    >>> print(len(pdf[classification]['objectId'].values))
    2

    >>> assert 'ZTF21acqersq' in pdf[classification]['objectId'].values
    """
    f_tracklet = tracklet.apply(lambda x: str(x).startswith('TRCK_'))

    return f_tracklet

@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def tracklet_candidates(tracklet) -> pd.Series:
    """ Pandas UDF version of tracklet_candidates_ for Spark

    Parameters
    ----------
    tracklet: Spark DataFrame Column
        Column containing the tracklet label

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> df = spark.read.format('parquet').load('datatest')
    >>> f = 'fink_filters.filter_tracklet_candidates.filter.tracklet_candidates'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    2

    """
    f_tracklet = tracklet_candidates_(tracklet)

    return f_tracklet


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
