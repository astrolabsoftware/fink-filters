# Copyright 2026 AstroLab Software
# Author: Jennifer Scora
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
"""Filters out alerts unlikely to be transients of interest to the DESC community."""

import pandas as pd
from fink_filters.rubin.blocks import b_good_quality

DESCRIPTION = (
    "Filters out alerts unlikely to be transients of interest to the DESC community."
)
HBASE_SUPPORT = False


def remove_unlikely_transients(
    diaSource: pd.DataFrame, nDiaSources: pd.Series, is_sso: pd.Series
) -> pd.Series:
    """Filters out alerts unlikely to be transients of interest to the DESC community.

    Notes
    -----
    This removes any solar system objects, anything with a subset of error flags,
    anything with negative flux or that is a dipole, anything without at least one
    previous source, and cuts SNR <= 10.

    Parameters
    ----------
    diaSource : pd.DataFrame
        Full diaSource section of an alert (dictionary exploded)
    nDiaSources : pd.Series
        Series containing the number of diaSources for this object
    is_sso : pd.Series
        Series containing booleans from solar system object classification

    Returns
    -------
    out: pd.Series
        Booleans: True for candidates of interest, False otherwise.

    Examples
    --------
    >>> from fink_filters.rubin.utils import apply_block
    >>> df2 = apply_block(df, "fink_filters.rubin.livestream.filter_remove_unlikely_transients.filter.remove_unlikely_transients")
    >>> df2.count()
    0
    """
    # set an SNR limit
    f_snr = diaSource.snr > 10

    # set a minimum of at least one previous source (not counting this one)
    f_nsources = nDiaSources >= 2

    # filter out specific flags to get only good quality alerts
    f_good_quality = b_good_quality(diaSource) & ~diaSource.isNegative

    # filter out solar system objects, any alerts with the above flags, any alerts with only one source, or SNR <=10
    f_good_alerts = ~is_sso & f_snr & ~f_good_quality & f_nsources

    return f_good_alerts


if __name__ == "__main__":
    """Test suite for filters"""
    # Run the test suite

    from fink_filters.tester import spark_unit_tests

    globs = globals()

    spark_unit_tests(globs, load_rubin_df=True)
