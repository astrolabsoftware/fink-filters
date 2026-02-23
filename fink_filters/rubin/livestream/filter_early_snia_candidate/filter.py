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
"""Select alerts with a early SN Ia classifier score above 0.76 OR those with socre above 0.5 and log10(delta_flux) above 0.5. See https://arxiv.org/abs/2404.08798 for classifier algorithm."""

import numpy as np
import pandas as pd
import fink_filters.rubin.utils as fu

DESCRIPTION = "Select alerts with a early SN Ia classifier score above 0.76 OR those with score above 0.5 and log10(ratio_flux) above 0.5. See https://arxiv.org/abs/2404.08798 for classifier algorithm."


def early_snia_candidate(
    earlySNIa_score: pd.Series, diaObject: pd.DataFrame
) -> pd.Series:
    """Select alerts using the early SN Ia classifier. See https://arxiv.org/abs/2404.08798.

    Parameters
    ----------
    earlySNIa_score: pd.Series
        Score (0...1) from the early SN Ia classifier
    diaObject: pd.DataFrame
        Full diaObject section of alerts. Must contain columns
        {band}_psfFluxMin and {band}_psfFluxMax for bands

    Returns
    -------
    out: pd.Series of booleans
        True if score > 0.76 OR score > 0.5 & log10(flux_ratio) > 0.5.

    Examples
    --------
    >>> from fink_filters.rubin.utils import apply_block
    >>> df2 = apply_block(df, "fink_filters.rubin.livestream.filter_early_snia_candidate.filter.early_snia_candidate")
    >>> df2.count()
    0
    """
    # calculate log flux ratio
    f_min = extract_min_flux(diaObject).apply(lambda x: np.max([10, x]))
    f_max = extract_max_flux(diaObject).apply(lambda x: x if x > 0 else 1e-10)

    flux_ratio = np.log10(f_max / f_min)

    f_flux_ratio = flux_ratio > 0.5
    f_good_early_snia = earlySNIa_score > 0.76
    f_medium_early_snia = np.logical_and(earlySNIa_score > 0.5, f_flux_ratio)

    return np.logical_or(f_good_early_snia, f_medium_early_snia)


if __name__ == "__main__":
    """Test suite for filters"""
    # Run the test suite

    from fink_filters.tester import spark_unit_tests

    globs = globals()
    spark_unit_tests(globs, load_rubin_df=True)
