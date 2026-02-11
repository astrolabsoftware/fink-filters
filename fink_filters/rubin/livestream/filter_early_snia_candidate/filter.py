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
"""Select alerts with a early SN Ia classifier score above 0.5. See https://arxiv.org/abs/2404.08798."""

import pandas as pd

DESCRIPTION = "Select alerts with a early SN Ia classifier score above 0.5. See https://arxiv.org/abs/2404.08798."


def early_snia_candidate(earlySNIa_score: pd.Series) -> pd.Series:
    """Select alerts with a early SN Ia classifier score above 0.5. See https://arxiv.org/abs/2404.08798.

    Parameters
    ----------
    earlySNIa_score: pd.Series
        Score (0...1) from the early SN Ia classifier

    Returns
    -------
    out: pd.Series of booleans
        True if score above 0.5. False otherwise

    Examples
    --------
    >>> s = pd.Series([0.6, 0.1, None])
    >>> out = early_snia_candidate(s)
    >>> assert out.sum() == 1, out.sum()

    >>> from fink_filters.rubin.utils import apply_block
    >>> df2 = apply_block(df, "fink_filters.rubin.livestream.filter_early_snia_candidate.filter.early_snia_candidate")
    >>> df2.count()
    0
    """
    f_early_snia = earlySNIa_score > 0.5
    return f_early_snia


if __name__ == "__main__":
    """Test suite for filters"""
    # Run the test suite

    from fink_filters.tester import spark_unit_tests

    globs = globals()
    spark_unit_tests(globs, load_rubin_df=True)
