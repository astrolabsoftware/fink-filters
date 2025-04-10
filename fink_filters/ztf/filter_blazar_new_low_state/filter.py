# Copyright 2025 AstroLab Software
# Author: Julian Hamo
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
from line_profiler import profile

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import BooleanType

import pandas as pd

from fink_filters.tester import spark_unit_tests


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
@profile
def new_low_state_filter(m0, m1, m2) -> pd.Series:
    """Returns True the alert is considered a quiescent state, False else.

    Parameters
    ----------
    blazar_stats: Spark DataFrame Column
        Column containing the 3 ratios computed in the blazar_low_state module

    Returns
    -------
    check: pd.Series
        Mask that returns True if the alert is a low state,
        False else

    Examples
    --------
    >>> import pyspark.sql.functions as F
    >>> from fink_utils.spark.utils import apply_user_defined_filter

    # Test
    >>> df = spark.read.parquet(ztf_alert_sample)
    >>> df = df.withColumn("m0", F.col('blazar_stats').getItem('m0').alias("m0"))
    >>> df = df.withColumn("m1", F.col('blazar_stats').getItem('m1').alias("m1"))
    >>> df = df.withColumn("m2", F.col('blazar_stats').getItem('m2').alias("m2"))
    >>> f = 'fink_filters.ztf.filter_blazar_new_low_state'
    >>> f += '.filter.new_low_state_filter'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    1
    """
    f0 = m0 >= 1
    f1 = (m1 < 1) & (m1 >= 0)
    f2 = (m2 < 1) & (m2 >= 0)
    return pd.Series(f0 & f1 & f2)


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    ztf_alert_sample = "datatest/CTAO_blazar/CTAO_blazar_datatest_v20-12-24.parquet"
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
