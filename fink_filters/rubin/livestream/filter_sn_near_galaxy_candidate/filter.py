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
"""Return LSST alerts with matches in catalogs to a galaxy and properties consistent with SNe"""

import pandas as pd
import fink_filters.rubin.blocks as fb
import fink_filters.rubin.utils as fu


DESCRIPTION = (
    "Select alerts matching in catalogs to a galaxy and properties consistent with SNe"
)


def sn_near_galaxy_candidate(
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
    """Select alerts matching in catalogs to a galaxy and properties consistent with SNe

    Notes
    -----
    based on a near galaxy extragalactic block and absolute magnitude
    Beware, this filter is not robust for close-by candidates due to xm association radius.

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

    Returns
    -------
    out: pd.Series
        Booleans: True for good quality alerts sn candidates,
        False otherwise.

    Examples
    --------
    >>> from fink_filters.rubin.utils import apply_block
    >>> df2 = apply_block(df, "fink_filters.rubin.livestream.filter_sn_near_galaxy_candidate.filter.sn_near_galaxy_candidate")
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
    )  # Xmatch galaxy

    # Minimum photometric sampling
    f_min_sampling = diaObject.nDiaSources > 5

    # All SNe types and lower Mabs to account not yet at max SNe
    estimated_absoluteMagnitude = fu.compute_peak_absolute_magnitude(
        diaObject, legacydr8_zphot
    )

    f_sn_Mabs = (estimated_absoluteMagnitude > -23) & (
        estimated_absoluteMagnitude < -13
    )

    # TO DO: can improve with (f_sn_Mabs | f_sn_ML) when SNN is robust
    f_sn_near_galaxy = f_extragalactic_near_galaxy & f_min_sampling & f_sn_Mabs

    return f_sn_near_galaxy


if __name__ == "__main__":
    """Test suite for filters"""
    # Run the test suite

    from fink_filters.tester import spark_unit_tests

    globs = globals()
    spark_unit_tests(globs, load_rubin_df=True)
