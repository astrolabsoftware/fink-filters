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

import os
import numpy as np
import pandas as pd

from fink_filters.tester import spark_unit_tests


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
@profile
def low_state_filter(blazar_stats) -> pd.Series:
    """Returns True the alert is considered a quiescent state,
       returns False else.

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
    >>> from fink_science.blazar_low_state.processor import quiescent_state
    >>> from fink_utils.spark.utils import apply_user_defined_filter

    # Test
    >>> df = spark.read.parquet(ztf_alert_sample)
    >>> f = 'fink_filters.filter_blazar_low_state.filter.low_state_filter'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    12
    """

    tmp = np.array(blazar_stats.values.tolist())
    tmp = tmp.reshape(tmp.shape[0], tmp.shape[-1]).transpose()
    tmp[pd.isnull(tmp)] = np.nan
    tmp[tmp < 0] = np.nan
    return pd.Series((tmp[1] < 1) & (tmp[2] < 1))


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    path_list = os.path.dirname(__file__).split('/')
    path_parent = '/'.join(path_list[:-2])
    path = os.path.join(path_parent, 'datatest/CTAO_blazar')
    filename = 'CTAO_blazar_datatest_v20-12-24.parquet'
    ztf_alert_sample = "file://{}/{}".format(path, filename)
    globs['parent_path'] = path
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
