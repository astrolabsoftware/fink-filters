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
"""Select LSST alerts new (< 48h first apparition) and potentially extragalactic"""

import pandas as pd
import fink_filters.rubin.blocks as fb


DESCRIPTION = (
    "Select LSST alerts new (< 48h first apparition) and potentially extragalactic"
)


def extragalactic_new_candidate(
    diaSource: pd.DataFrame,
    simbad_otype: pd.Series,
    mangrove_lum_dist: pd.Series,
    is_sso: pd.Series,
    gaiadr3_DR3Name: pd.Series,
    gaiadr3_Plx: pd.Series,
    gaiadr3_e_Plx: pd.Series,
    vsx_Type: pd.Series,
    legacydr8_zphot: pd.Series,
    firstDiaSourceMjdTaiFink: pd.Series,
) -> pd.Series:
    """Select LSST alerts new (< 48h first apparition) and potentially extragalactic

    Notes
    -----
    Based on an extragalactic block and time cut



    Parameters
    ----------
    diaSource: pd.DataFrame
        Full diaSource section of an alert (dictionary exploded)
    diaObject: pd.DataFrame
        Full diaObject section of an alert (dictionary exploded)
    simbad_otype: pd.Series
        Series containing labels from `xm.simbad_otype`
    mangrove_lum_dist: pd.Series
        Series containing floats from `xm.mangrove_lum_dist`
    is_sso: pd.Series
        Series containing booleans from solar system object classification
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
    firstDiaSourceMjdTaiFink: pd.Series
        First time the object emitted an alert. This is currently not set
        by the Rubin project, and we use instead the oldest date in the history.

    Returns
    -------
    out: pd.Series
        Booleans: True for good quality alerts extragalactic candidates,
        False otherwise.

    Examples
    --------
    >>> from fink_filters.rubin.utils import apply_block
    >>> df2 = apply_block(df, "fink_filters.rubin.livestream.filter_extragalactic_new_candidate.filter.extragalactic_new_candidate")
    >>> df2.count()
    0
    """
    # Near galaxy
    f_extragalactic_near_galaxy = fb.b_extragalactic_near_galaxy_candidate(
        diaSource,
        simbad_otype,
        mangrove_lum_dist,
        is_sso,
        gaiadr3_DR3Name,
        gaiadr3_Plx,
        gaiadr3_e_Plx,
        vsx_Type,
        legacydr8_zphot,
    )

    # 48h maximum
    f_new = (diaSource.midpointMjdTai - firstDiaSourceMjdTaiFink) < 2.0

    f_extragalactic_new = f_extragalactic_near_galaxy & f_new

    return f_extragalactic_new


if __name__ == "__main__":
    """Test suite for filters"""
    # Run the test suite

    from fink_filters.tester import spark_unit_tests

    globs = globals()
    spark_unit_tests(globs, load_rubin_df=True)
