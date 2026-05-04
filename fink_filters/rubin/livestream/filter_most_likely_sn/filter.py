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
"""Filter for alerts that are likely to be SNIa."""

import pandas as pd

DESCRIPTION = "Filter alerts that are likely to be SNIa."


def most_likely_sn(
    cats_class: pd.Series,
    cats_score: pd.Series,
    earlySNIa_score: pd.Series,
    snnSnVsOthers_score: pd.Series,
) -> pd.Series:
    """Filter alerts that are likely to be SNIa.

    Notes
    -----
    Based on the Fink classifiers. Cuts on the classifier scores.

    Parameters
    ----------
    cats_class : pd.Series
        CATS classifier broad class prediction with the highest probability.
    cats_score : pd.Series
        CATS classifier highest probability.
    earlySNIa_score : pd.Series
        Score for the early SN Ia classifier.
    snnSnVsOthers_score : pd.Series
        Score for teh SN binary classifier using SuperNNova.

    Returns
    -------
    out: pd.Series
        Booleans: True for possible SNIa, False otherwise.

    Examples
    --------
    >>> from fink_filters.rubin.utils import apply_block
    >>> df2 = apply_block(df, "fink_filters.rubin.livestream.filter_most_likely_sn.filter.most_likely_sn")
    >>> df2.count()
    0
    """

    # high probability of early SNIa
    f_earlySN = earlySNIa_score >= 0.5
    # high probability of SN using SNNova
    f_snnSN = snnSnVsOthers_score >= 0.8
    # high probability of SN using CATS
    f_SNlike = (cats_class == 11) & (cats_score >= 0.9)

    # one of the above must be true
    f_likely_sn = f_earlySN | f_snnSN | f_SNlike

    return f_likely_sn


if __name__ == "__main__":
    """Test suite for filters"""
    # Run the test suite

    from fink_filters.tester import spark_unit_tests

    globs = globals()

    spark_unit_tests(globs, load_rubin_df=True)
