# Copyright 2025 AstroLab Software
# Author: Julien Peloton, Heloise Stevance
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
"""Return alerts suitable for the Virtual Research Assistant"""

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import BooleanType

from fink_filters.tester import spark_unit_tests

import pandas as pd

from typing import Any


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def vra(magpsf: Any, drb: Any, cdsxmatch: Any, roid: Any) -> pd.Series:
    """Return alerts used to enter the VRA module

    Parameters
    ----------
    magpsf: Spark DataFrame Column
        Magnitude of the alert
    drb: Spark DataFrame Column
        Deep learning real bogus score
    cdsxmatch: Spark DataFrame Column
        Column containing the cross-match values
    roid: Spark DataFrame Column
        Asteroid taxonomy

    Returns
    -------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    --------
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> df = spark.read.format('parquet').load('datatest/regular')
    >>> f = 'fink_filters.ztf.livestream.filter_vra.filter.vra'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    2

    """
    # Include new taxonomy
    f1 = cdsxmatch == "Unknown"
    f2 = roid != 3
    f3 = magpsf > 19.5
    f4 = drb > 0.5

    return pd.Series(f1 & f2 & f3 & f4)


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
