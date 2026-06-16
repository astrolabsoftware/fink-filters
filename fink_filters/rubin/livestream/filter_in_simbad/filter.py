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
"""Select alerts with a known counterpart in SIMBAD at the time of emission by Rubin"""

from fink_filters.rubin.blocks import b_xmatched_simbad_unknown
import pandas as pd

DESCRIPTION = (
    "Select alerts with a known counterpart in SIMBAD at the time of emission by Rubin"
)
HBASE_SUPPORT = True
COLUMNS = "diaObject.diaObjectId,xm"


def in_simbad(simbad_otype: pd.Series) -> pd.Series:
    """Return alerts with a known counterpart in SIMBAD at the time of emission by Rubin

    Parameters
    ----------
    simbad_otype: pd.Series
        Name according to SIMBAD (string or null).

    Returns
    -------
    out: pd.Series of booleans
        True if in SIMBAD. False otherwise

    Examples
    --------
    >>> s = pd.Series(["EB*", None, "SN"])
    >>> out = in_simbad(s)
    >>> assert out.sum() == 2, out.sum()

    >>> from fink_filters.rubin.utils import apply_block
    >>> import pyspark.sql.functions as F
    >>> df = df.withColumn("simbad_otype", F.lit(None).astype("string"))
    >>> df2 = apply_block(df, "fink_filters.rubin.livestream.filter_in_simbad.filter.in_simbad")
    >>> df2.count()
    0
    """
    in_simbad = ~b_xmatched_simbad_unknown(simbad_otype)
    return in_simbad


if __name__ == "__main__":
    """Test suite for filters"""
    # Run the test suite

    from fink_filters.tester import spark_unit_tests

    globs = globals()
    spark_unit_tests(globs, load_rubin_df=True)
