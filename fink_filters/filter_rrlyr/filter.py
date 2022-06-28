# Copyright 2019 AstroLab Software
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

from typing import Any

@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def rrlyr(cdsxmatch: Any) -> pd.Series:
    """ Return alerts identified as RRLyr by the xmatch module.

    Parameters
    ----------
    cdsxmatch: Spark DataFrame Column
        Column containing the cross-match values

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> df = spark.read.format('parquet').load('datatest')
    >>> f = 'fink_filters.filter_rrlyr.filter.rrlyr'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    3
    """
    # Include new taxonomy
    mask = (cdsxmatch.values == "RRLyr") | (cdsxmatch.values == "RRLyrae")

    return pd.Series(mask)


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
