# Copyright 2019-2026 AstroLab Software
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
"""Select alerts with a known counterpart in the Minor Planet Center at the time of emission by Rubin"""

import pandas as pd

DESCRIPTION = "Select alerts with a known counterpart in the Minor Planet Center at the time of emission by Rubin"
HBASE_SUPPORT = False


def is_solar_system(is_sso: pd.Series) -> pd.Series:
    """Return alerts with a known counterpart in the Minor Planet Center at the time of emission by Rubin

    Parameters
    ----------
    is_sso: pd.Series of booleans
        `pred.is_sso` from the alerts

    Returns
    -------
    out: pd.Series of booleans
        True if in MPC. False otherwise

    Examples
    --------
    >>> s = pd.Series([False, True, True])
    >>> out = is_solar_system(s)
    >>> assert out.sum() == 2, out.sum()

    >>> from fink_filters.rubin.utils import apply_block
    >>> import pyspark.sql.functions as F
    >>> df = apply_block(df, "fink_filters.rubin.livestream.filter_is_solar_system.filter.is_solar_system")
    >>> df.count()
    3
    """
    return is_sso


if __name__ == "__main__":
    """Test suite for filters"""
    # Run the test suite

    from fink_filters.tester import spark_unit_tests

    globs = globals()
    spark_unit_tests(globs, load_rubin_df=True)
