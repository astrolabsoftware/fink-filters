# Copyright 2019-2026 AstroLab Software
# Author: Anais Moller
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
"""Return LSST alerts with matches in catalogs to a galaxy"""

import pandas as pd
import fink_filters.rubin.blocks as fb
from fink_filters.rubin.livestream.filter_extragalactic_candidate.filter import (
    extragalactic_candidate,
)

DESCRIPTION = "Select alerts that are extragalactic candidates, recent and rising in at least one filter"


def extragalactic_rising_candidate(
    diaSource: pd.DataFrame,
    diaObject: pd.DataFrame,
    simbad_otype: pd.Series,
    mangrove_lum_dist: pd.Series,
    is_sso: pd.Series,
    gaiadr3_DR3Name: pd.Series,
    gaiadr3_Plx: pd.Series,
    gaiadr3_e_Plx: pd.Series,
    vsx_Type: pd.Series,
    legacydr8_zphot: pd.Series,
) -> pd.Series:
    """Flag for alerts in Rubin that are new and rising extragalactic candidates

    Parameters
    ----------
    diaSource: pd.DataFrame
        Full diaSource section of an alert (dictionary exploded)
    diaObject: pd.DataFrame
        Full diaObject section of an alert (dictionary exploded)
    simbad_otype: pd.Series
        Type xmatched SIMBAD
    mangrove_lum_dist: pd.Series
        Luminosity distance of xmatch with Mangrove
    is_sso: pd.Series
        Asteroid tag
    gaiadr3_DR3Name: pd.Series
        Series containing Gaia DR3 names from `xm.gaiadr3_DR3Name`
    gaiadr3_Plx: pd.Series
        Series containing parallax values from `xm.gaiadr3_Plx`
    gaiadr3_e_Plx: pd.Series
        Series containing parallax errors from `xm.gaiadr3_e_Plx`
    vsx_Type: pd.Series
        Series containing VSX variable star catalog matches
    legacydr8_zphot: pd.Series
        Series containing photometric redshift from `xm.legacydr8_zphot` (Duncan 2022)

    Returns
    -------
    pd.Series
        Alerts that are extragalactic and rising

    Examples
    --------
    >>> from fink_filters.rubin.utils import apply_block
    >>> df2 = apply_block(df, "fink_filters.rubin.livestream.filter_extragalactic_rising_candidate.filter.extragalactic_rising_candidate")
    >>> df2.count()
    0
    """
    # Good quality
    f_good_quality = fb.b_good_quality(diaSource)

    # Extragalactic filter
    f_extragalactic = extragalactic_candidate(
        diaSource,
        simbad_otype,
        mangrove_lum_dist,
        is_sso,
        gaiadr3_DR3Name,
        gaiadr3_Plx,
        gaiadr3_e_Plx,
        vsx_Type,
        legacydr8_zphot        
    )

    # Rising in at least one band
    f_is_rising = fb.b_is_rising(diaSource, diaObject)

    f_new = diaObject.nDiaSources < 20  # should be lowered after first alerts

    f_extragalactic_rising = f_good_quality & f_extragalactic & f_is_rising & f_new

    return f_extragalactic_rising


if __name__ == "__main__":
    """Test suite for filters"""
    # Run the test suite

    from fink_filters.tester import spark_unit_tests

    globs = globals()
    spark_unit_tests(globs, load_rubin_df=True)
