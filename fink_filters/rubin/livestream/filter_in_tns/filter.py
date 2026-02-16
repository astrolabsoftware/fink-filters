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
"""Select alerts with a known counterpart in TNS (AT or confirmed) at the time of emission by Rubin"""

import pandas as pd

DESCRIPTION = "Select alerts with a known counterpart in TNS (AT or confirmed) at the time of emission by Rubin"


def in_tns(tns_fullname: pd.Series) -> pd.Series:
    """Return alerts with a known counterpart in TNS (AT or confirmed) at the time of emission by Rubin

    Parameters
    ----------
    tns_fullname: pd.Series
        Name according to TNS (string or null).

    Returns
    -------
    out: pd.Series of booleans
        True if in TNS. False otherwise

    Examples
    --------
    >>> s = pd.Series(["SN toto", None, "AT titi"])
    >>> out = in_tns(s)
    >>> assert out.sum() == 2, out.sum()

    >>> from fink_filters.rubin.utils import apply_block
    >>> import pyspark.sql.functions as F
    >>> df = df.withColumn("tns_fullname", F.lit(None).astype("string"))
    >>> df2 = apply_block(df, "fink_filters.rubin.livestream.filter_in_tns.filter.in_tns")
    >>> df2.count()
    0
    """
    in_tns = tns_fullname.apply(lambda x: x is not None)
    return in_tns


if __name__ == "__main__":
    """Test suite for filters"""
    # Run the test suite

    from fink_filters.tester import spark_unit_tests

    globs = globals()
    spark_unit_tests(globs, load_rubin_df=True)
