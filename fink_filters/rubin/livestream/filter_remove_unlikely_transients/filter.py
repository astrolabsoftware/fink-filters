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
"""Filter out alerts unlikely to be transients of interest to the DESC community."""

import pandas as pd

DESCRIPTION = (
    "Filter alerts unlikely to be transients of interest to the DESC community."
)


def remove_unlikely_transients(diaSource: pd.DataFrame, is_sso: pd.Series) -> pd.Series:
    """Filter alerts unlikely to be transients of interest to the DESC community.

    Notes
    -----
    This removes any alerts with an ssObjecdId (solar system objects), anything
    with negative flux or that is a dipole, and cuts SNR <= 5.

    Parameters
    ----------
    diaSource : pd.DataFrame
        Full diaSource section of an alert (dictionary exploded)

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

    # take only objects without a solar system id
    # f_ss_objs = diaSource.ssObjectId == 0

    # set an SNR limit
    f_snr = diaSource.snr > 5

    # filter out all the above and negative and dipole alerts
    f_good_alerts = ~is_sso & f_snr & ~diaSource.isNegative & ~diaSource.isDipole

    return f_good_alerts


if __name__ == "__main__":
    """Test suite for filters"""
    # Run the test suite

    from fink_filters.tester import spark_unit_tests

    globs = globals()

    spark_unit_tests(globs, load_rubin_df=True)
