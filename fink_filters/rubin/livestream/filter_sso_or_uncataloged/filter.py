# Copyright 2026 LSST moving-object search contributors
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
"""Select Rubin-associated Solar System objects and uncataloged detections."""

import pandas as pd

DESCRIPTION = "Select Rubin-associated Solar System objects and alerts without a Fink catalog match"
HBASE_SUPPORT = False


def sso_or_uncataloged(is_sso: pd.Series, is_cataloged: pd.Series) -> pd.Series:
    """Keep known Solar System objects and detections without a catalog match.

    Parameters
    ----------
    is_sso : pd.Series of booleans
        Fink's `pred.is_sso` flag.
    is_cataloged : pd.Series of booleans
        Fink's `pred.is_cataloged` flag.

    Returns
    -------
    pd.Series of booleans
        A true Solar System flag wins any overlap. Missing flags count as false.

    Examples
    --------
    >>> from itertools import product
    >>> flags = pd.DataFrame(
    ...     product([True, False, None], repeat=2), columns=["sso", "cataloged"]
    ... )
    >>> sso_or_uncataloged(flags.sso, flags.cataloged).tolist()
    [True, True, True, False, True, True, False, True, True]

    >>> from fink_filters.rubin.utils import apply_block
    >>> name = "fink_filters.rubin.livestream.filter_sso_or_uncataloged.filter.sso_or_uncataloged"
    >>> kept = apply_block(df, name).select("diaSourceId").collect()
    >>> expected = df.where(
    ...     "COALESCE(pred.is_sso, false) OR NOT COALESCE(pred.is_cataloged, false)"
    ... ).select("diaSourceId").collect()
    >>> len(kept), sorted(kept) == sorted(expected)
    (23, True)
    """
    return is_sso.isin([True]) | ~is_cataloged.isin([True])


if __name__ == "__main__":
    from fink_filters.tester import spark_unit_tests

    spark_unit_tests(globals(), load_rubin_df=True)
