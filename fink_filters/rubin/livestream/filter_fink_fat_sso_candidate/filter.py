# Copyright 2019-2026 AstroLab Software
# Author: Roman Le Montagner
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
"""Return LSST alerts potentially unknown solar system, used to feed the Fink Asteroid Tracker (Fink-FAT)"""

import pandas as pd
import fink_filters.rubin.blocks as fb

DESCRIPTION = "Select alerts that are potential unknown solar system moving object for the Fink Asteroid Tracker (Fink-FAT)"
HBASE_SUPPORT = True


def fink_fat_sso_candidate(
    diaSource: pd.DataFrame,
    firstDiaSourceMjdTaiFink: pd.Series,
    is_sso: pd.Series,
    gaiadr3_DR3Name: pd.Series,
    gaiadr3_Plx: pd.Series,
    gaiadr3_e_Plx: pd.Series,
    simbad_otype: pd.Series,
    vsx_Type: pd.Series,
):
    """Flag for alerts in Rubin that are potentially unknown solar system moving object.

    It looks for new single point source alert without any history which could be
    a single asteroid observation before any tracking attempt. The alert outputs by this filter
    serves as inputs for the Fink Asteroid Tracker (https://github.com/FusRoman/fink-fat/tree/release-1.0.0)

    Parameters
    ----------
    diaSource: pd.DataFrame
        Full diaSource section of an alert (dictionary exploded)
    firstDiaSourceMjdTaiFink: pd.Series
            FIrst time the object was seen by Rubin, as derived by Fink.
    is_sso: pd.Series
        Series containing booleans from solar system object classification
    gaiadr3_DR3Name: pd.Series
        Series containing Gaia DR3 names from `xm.gaiadr3_DR3Name`
    gaiadr3_Plx: pd.Series
        Series containing parallax values from `xm.gaiadr3_Plx`
    gaiadr3_e_Plx: pd.Series
        Series containing parallax errors from `xm.gaiadr3_e_Plx`
    simbad_otype: pd.Series
            Series containing labels from `xm.simbad_otype`
    vsx_Type: pd.Series
        Series containing VSX variable star catalog matches

    Returns
    -------
    out: pd.Series
        Booleans: True for unknown single alert unknown solar system candidates,
        False otherwise.

    Examples
    --------
    >>> from fink_filters.rubin.utils import apply_block
    >>> df2 = apply_block(df, "fink_filters.rubin.livestream.filter_fink_fat_sso_candidate.filter.fink_fat_sso_candidate")
    >>> df2.count()
    23
    """
    is_new = fb.b_is_new(diaSource.midpointMjdTai, firstDiaSourceMjdTaiFink)

    not_known_sso = ~fb.b_is_solar_system(is_sso)

    not_gaia_star = ~fb.b_xmatched_gaia_star(
        gaiadr3_DR3Name, gaiadr3_Plx, gaiadr3_e_Plx
    )

    simbad_unknown = fb.b_xmatched_simbad_unknown(simbad_otype)

    not_vsx = ~fb.b_xmatched_vsx(vsx_Type)

    not_vsx_star = ~fb.b_xmatched_vsx_star(vsx_Type)

    return (
        is_new & not_known_sso & not_gaia_star & simbad_unknown & not_vsx & not_vsx_star
    )


if __name__ == "__main__":
    """Test suite for filters"""
    # Run the test suite

    from fink_filters.tester import spark_unit_tests

    globs = globals()
    spark_unit_tests(globs, load_rubin_df=True)
