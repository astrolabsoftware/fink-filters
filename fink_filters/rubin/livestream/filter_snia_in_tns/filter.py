# Copyright 2019-2026 AstroLab Software
# Author: Emille E. O. Ishida and Julien Peloton
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
"""Select alerts which have been spectroscopically confirmed as SN Ia in TNS."""

import pandas as pd

DESCRIPTION = "Select alerts which have been spectroscopically confirmed as SN Ia in TNS."
HBASE_SUPPORT = True


def snia_in_tns(tns_type: pd.Series) -> pd.Series:
    """Return alerts associated with objects that have been spectroscopically confirmed as SN Ia in TNS.

    Parameters
    ----------
    tns_type: pd.Series
        Classification according to TNS (string or null).

    Returns
    -------
    out: pd.Series of booleans
        True if classified as SNIa in TNS. False otherwise

    Examples
    --------
    >>> s = pd.Series(["SN Ia", None, "AT titi"])
    >>> out = snia_in_tns(s)
    >>> assert out.sum() == 1, out.sum()

    >>> from fink_filters.rubin.utils import apply_block
    >>> import pyspark.sql.functions as F
    >>> df = df.withColumn("tns_type", F.lit(None).astype("string"))
    >>> df2 = apply_block(df, "fink_filters.rubin.livestream.filter_snia_in_tns.filter.snia_in_tns")
    >>> df2.count()
    0
    """
    snia_list = ['SN Ia', 'SN Ia-91T-like', 'SN Ia-CSM', 'SN Ia-pec']
    snia_in_tns = tns_type.apply(lambda x: x in snia_list)
    
    return snia_in_tns


if __name__ == "__main__":
    """Test suite for filters"""
    # Run the test suite

    from fink_filters.tester import spark_unit_tests

    globs = globals()
    spark_unit_tests(globs, load_rubin_df=True)
