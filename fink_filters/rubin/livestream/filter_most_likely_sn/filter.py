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
"""Selects alerts that are likely to be SN, based on SuperNNova and CATS classifiers."""

import pandas as pd
from fink_filters.rubin.blocks import b_good_quality

DESCRIPTION = (
    "Selects alerts that are likely to be SN, based on SuperNNova and CATS classifiers."
)


def most_likely_sn(
    cats_class: pd.Series,
    cats_score: pd.Series,
    snnSnVsOthers_score: pd.Series,
    diaSource: pd.DataFrame,
    nDiaSources: pd.Series,
    is_sso: pd.Series,
) -> pd.Series:
    """Selects alerts that are likely to be SN, based on SuperNNova and CATS classifiers.

    Notes
    -----
    Based on the Fink classifiers. Cuts on the classifier scores, as well as
    a number of 'quality' filters (SNR, isn't a solar system object, etc).

    Parameters
    ----------
    cats_class : pd.Series
        CATS classifier broad class prediction with the highest probability.
    cats_score : pd.Series
        CATS classifier highest probability.
    snnSnVsOthers_score : pd.Series
        Score for the SN binary classifier using SuperNNova.
    diaSource : pd.DataFrame
        Full diaSource section of an alert (dictionary exploded)
    nDiaSources : pd.Series
        Series containing the number of diaSources for this object
    is_sso : pd.Series
        Series containing booleans from solar system object classification

    Returns
    -------
    out: pd.Series
        Booleans: True for possible SN, False otherwise.

    Examples
    --------
    >>> from fink_filters.rubin.utils import apply_block
    >>> df2 = apply_block(df, "fink_filters.rubin.livestream.filter_most_likely_sn.filter.most_likely_sn")
    >>> df2.count()
    0
    """
    # set an SNR limit
    f_snr = diaSource.snr > 10

    # set a minimum of at least one previous source (not counting this one)
    f_nsources = nDiaSources >= 2

    # filter out specific flags to get only good quality alerts
    f_good_quality = b_good_quality(diaSource) & ~diaSource.isNegative

    # high probability of SN using SuperNNova
    f_snnSN = snnSnVsOthers_score >= 0.7
    # high probability of SN using CATS
    f_SNlike = (cats_class == 11) & (cats_score >= 0.9)

    # both of the above must be true as well as the above quality flags
    f_likely_sn = f_snnSN & f_SNlike & ~is_sso & f_snr & ~f_good_quality & f_nsources

    return f_likely_sn


if __name__ == "__main__":
    """Test suite for filters"""
    # Run the test suite

    from fink_filters.tester import spark_unit_tests

    globs = globals()

    spark_unit_tests(globs, load_rubin_df=True)
