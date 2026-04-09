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
def high_state_filter(instantness_high, robustness_high) -> pd.Series:
    """Returns True if the alert is considered a high state, False otherwise.

    Parameters
    ----------
    instantness_high: Spark DataFrame Column
        `instantness_high` feature computed in the blazar_extreme_state module.
    robustness_high: Spark DataFrame Column
        `robustness_high` feature computed in the blazar_extreme_state module.

    Returns
    -------
    check: pd.Series
        Mask that returns True if the alert is a high state,
        False else.

    Examples
    --------
    >>> import pyspark.sql.functions as F
    >>> import os
    >>> import numpy as np
    >>> import pandas as pd
    >>> from fink_utils.spark.utils import concat_col
    >>> from fink_science.ztf.standardized_flux.processor import standardized_flux
    >>> from fink_science.ztf.blazar_extreme_state.processor import extreme_state
    >>> from fink_utils.spark.utils import apply_user_defined_filter

    >>> parDF = spark.read.parquet(ztf_alert_sample)
    >>> parDF = parDF.drop("blazar_stats")

    # Required alert columns
    >>> what = [
    ...     "distnr",
    ...     "magpsf",
    ...     "sigmapsf",
    ...     "magnr",
    ...     "sigmagnr",
    ...     "isdiffpos",
    ...     "fid",
    ...     "jd",
    ...     "ra",
    ...     "dec",
    ... ]

    # Concatenation
    >>> prefix = "c"
    >>> for key in what:
    ...     parDF = concat_col(parDF, colname=key, prefix=prefix)

    # Preliminary module run
    >>> args = [
    ...     "candid",
    ...     "objectId",
    ...     "cdistnr",
    ...     "cmagpsf",
    ...     "csigmapsf",
    ...     "cmagnr",
    ...     "csigmagnr",
    ...     "cisdiffpos",
    ...     "cfid",
    ...     "cjd",
    ... ]
    >>> parDF = parDF.withColumn(
    ...     "container",
    ...     standardized_flux(*args)
    ... )
    >>> parDF = parDF.withColumn(
    ...     "cstd_flux",
    ...     parDF["container"].getItem("flux")
    ... )
    >>> parDF = parDF.withColumn(
    ...     "csigma_std_flux",
    ...     parDF["container"].getItem("sigma")
    ... )

    # Drop temporary columns
    >>> what_prefix = [prefix + key for key in what]
    >>> parDF = parDF.drop("container")

    # Test the module
    >>> args = ["candid", "objectId", "cstd_flux", "cjd", "cra", "cdec"]
    >>> parDF = parDF.withColumn("blazar_stats", extreme_state(*args))

    >>> parDF = parDF.withColumn(
    ...     "instantness_high",
    ...     F.col("blazar_stats").getItem("instantness_high").alias("instantness_high")
    ... )
    >>> parDF = parDF.withColumn(
    ...     "robustness_high",
    ...     F.col("blazar_stats").getItem("robustness_high").alias("robustness_high")
    ... )
    >>> f = "fink_filters.ztf.filter_blazar_high_state.filter.high_state_filter"
    >>> parDF = apply_user_defined_filter(parDF, f)
    >>> print(parDF.count())
    24
    """
    f1 = instantness_high > 1
    f2 = robustness_high > 1
    return pd.Series(f1 & f2)


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    ztf_alert_sample = "datatest/CTAO_blazar/CTAO_blazar_datatest_v20-12-24.parquet"
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
