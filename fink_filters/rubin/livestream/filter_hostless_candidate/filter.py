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
"""Return LSST alerts that are hostless according to ELEPHANT. See https://arxiv.org/abs/2404.18165."""

import pandas as pd
import fink_filters.rubin.blocks as fb


DESCRIPTION = "Select LSST alerts that are hostless according to ELEPHANT. See https://arxiv.org/abs/2404.18165."


def hostless_candidate(
    isDipole: pd.Series,
    shape_flag: pd.Series,
    forced_PsfFlux_flag: pd.Series,
    psfFlux_flag: pd.Series,
    apFlux_flag: pd.Series,
    centroid_flag: pd.Series,
    pixelFlags_interpolated: pd.Series,
    pixelFlags_cr: pd.Series,
    forced_PsfFlux_flag_edge: pd.Series,
    pixelFlags_bad: pd.Series,
    elephant_kstest_template: pd.Series,
) -> pd.Series:
    """Flag for alerts in Rubin that are hostless

    Notes
    -----
    Quality flags are applied

    Parameters
    ----------
    isDipole : pd.Series
        Dipole well fit for source flag
    shape_flag : pd.Series
        Shape photometry flag
    forced_PsfFlux_flag : pd.Series
        Science forced photometry flag
    psfFlux_flag : pd.Series
        Psf model failure flag
    apFlux_flag : pd.Series
        Aperture failure flag
    centroid_flag : pd.Series
        Centroid failure flag
    pixelFlags_interpolated : pd.Series
        Interpolated pixel in footprint
    pixelFlags_cr : pd.Series
        Cosmic ray
    forced_PsfFlux_flag_edge : pd.Series
        Science coordinate too close to edge
    pixelFlags_bad : pd.Series
        Bad pixel in footprint
    elephant_kstest_template: pd.Series
        KS test for the template image

    Returns
    -------
    out: pd.Series
        Booleans: True for good quality & hostless candidates,
        False otherwise.

    Examples
    --------
    >>> from fink_filters.tester import apply_block
    >>> df2 = apply_block(df, "fink_filters.rubin.livestream.filter_hostless_candidate.filter.hostless_candidate")
    >>> df2.count()
    14
    """
    # Good quality
    f_good_quality = fb.b_good_quality(
        isDipole,
        shape_flag,
        forced_PsfFlux_flag,
        psfFlux_flag,
        apFlux_flag,
        centroid_flag,
        pixelFlags_interpolated,
        pixelFlags_cr,
        forced_PsfFlux_flag_edge,
        pixelFlags_bad,
    )

    f_hostless = f_good_quality & (elephant_kstest_template < 3.0)

    return f_hostless


if __name__ == "__main__":
    """Test suite for filters"""
    # Run the test suite

    from fink_filters.tester import spark_unit_tests

    globs = globals()
    spark_unit_tests(globs, load_rubin_df=True)
