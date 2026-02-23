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
"""Return LSST alerts rising, bright (mag < 20) and potentially extragalactic"""

import pandas as pd
import fink_filters.rubin.blocks as fb
import fink_filters.rubin.utils as fu


DESCRIPTION = (
    "Select alerts that are rising, bright (mag < 20), and extragalactic candidates"
)


def extragalactic_lt20mag_candidate(
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
    firstDiaSourceMjdTaiFink: pd.Series,
) -> pd.Series:
    """Flag for alerts in Rubin that are rising, bright (mag < 20), and extragalactic candidates

    Notes
    -----
    based on a loose extragalactic block, rising light-curve and a magnitude cut

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
        FIrst time the object was seen by Rubin, as derived by Fink.

    Returns
    -------
    out: pd.Series
        Booleans: True for good quality alerts extragalactic candidates,
        False otherwise.

    Examples
    --------
    >>> from fink_filters.rubin.utils import apply_block
    >>> df2 = apply_block(df, "fink_filters.rubin.livestream.filter_extragalactic_lt20mag_candidate.filter.extragalactic_lt20mag_candidate")
    >>> df2.count()
    0
    """
    # Loose extragalactic candidate
    f_extragalactic = fb.b_extragalactic_loose_candidate(
        diaSource,
        simbad_otype,
        mangrove_lum_dist,
        is_sso,
        gaiadr3_DR3Name,
        gaiadr3_Plx,
        gaiadr3_e_Plx,
        vsx_Type,
        legacydr8_zphot,
    )  # Xmatch galaxy or Unknown

    f_bright = fu.flux_to_apparent_mag(diaSource.psfFlux) < 20

    f_is_rising = fb.b_is_rising(diaSource, diaObject)

    f_sampling = (diaObject.nDiaSources > 4) & (
        diaSource.midPointMjdTai - firstDiaSourceMjdTaiFink > 1
    )

    f_extragalactic_gt20mag_rising = (
        f_extragalactic & f_bright & f_is_rising & f_sampling
    )

    return f_extragalactic_gt20mag_rising


if __name__ == "__main__":
    """Test suite for filters"""
    # Run the test suite

    from fink_filters.tester import spark_unit_tests

    globs = globals()
    spark_unit_tests(globs, load_rubin_df=True)
