# Copyright 2026 AstroLab Software
# Author: Mohammed Chamma
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
"""Select 1% of all live alerts in a uniformly random way"""
import pandas as pd

DESCRIPTION = "Select 1% of all live alerts in a uniformly random way"

def uniform_sample(diaSourceId: pd.Series) -> pd.Series:
    """

    Examples
    ---------
    >>> import numpy as np
    >>> import os
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> # df = pd.read_parquet('dataset/rubin_test_data_10_0.parquet')
    >>> # df = spark.read.format('parquet').load('dataset/rubin_test_data_10_0.parquet')
    >>> totalcount = df.count()
    >>> f = 'fink_filters.rubin.livestream.filter_uniform_sample.uniform_sample'
    >>> df = apply_user_defined_filter(df, f)
    >>> ratio = df.count() / totalcount
    >>> (0.005 <= ratio < 0.02) # check we are between 0.5% and 2%
    True
    """
    # diaSourceId % 113 == 0
    random_alert = diaSourceId.apply(lambda x: x % 113 == 0)
    return random_alert

if __name__ == "__main__":
    from fink_filters.tester import spark_unit_tests
    globs = globals()
    spark_unit_tests(globs, load_rubin_df=True)	
