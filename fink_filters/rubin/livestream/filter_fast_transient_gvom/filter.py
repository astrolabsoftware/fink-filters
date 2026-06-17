# Copyright 2019-2026 AstroLab Software
# Author: Julien Peloton, Roman Le Montagner, Damien Turpin
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

import pandas as pd

import fink_filters.rubin.blocks as fb
import fink_filters.rubin.utils as fu

DESCRIPTION = """
Select LSST alerts that are:
- new (< 5days first apparition)
- bright (mag < 21)
- potentially extragalactic
- behaving like fast transient (magnitude rate < 0.3 mag/day)
"""
HBASE_SUPPORT = False


def fast_transient_gvom(
    diaSource: pd.DataFrame,
    simbad_otype: pd.Series,
    mangrove_lum_dist: pd.Series,
    is_sso: pd.Series,
    gaiadr3_DR3Name: pd.Series,
    gaiadr3_e_Plx: pd.Series,
    vsx_Type: pd.Series,
    legacydr8_zphot: pd.Series,
    firstDiaSourceMjdTaiFink: pd.Series,
    prvDiaSources: pd.Series,
) -> pd.Series:
    """Select LSST alerts that are new, bright, extragalactic, and fast-evolving

    Notes
    -----
    Designed to target fast transients of interest for GVOM follow-up.
    Based on an extragalactic loose candidate block, a age cut (< 5 days
    since first detection), a brightness cut (mag < 21), and a flux rate cut
    requiring a statistically significant brightening of at least 0.3 mag/day
    estimated via Monte Carlo sampling over previous detections.

    Parameters
    ----------
    diaSource: pd.DataFrame
        Full diaSource section of an alert (dictionary exploded)
    simbad_otype: pd.Series
        Series containing labels from `xm.simbad_otype`
    mangrove_lum_dist: pd.Series
        Series containing floats from `xm.mangrove_lum_dist`
    is_sso: pd.Series
        Series containing booleans from solar system object classification
    gaiadr3_DR3Name: pd.Series
        Series containing Gaia DR3 names from `xm.gaiadr3_DR3Name`
    gaiadr3_e_Plx: pd.Series
        Series containing parallax errors from `xm.gaiadr3_e_Plx`
    vsx_Type: pd.Series
        Series containing VSX variable star catalog matches
    legacydr8_zphot: pd.Series
        Series containing photometric redshift from `xm.legacydr8_zphot` (Duncan 2022)
    firstDiaSourceMjdTaiFink: pd.Series
        First time the object emitted an alert. This is currently not set
        by the Rubin project, and we use instead the oldest date in the history.
    prvDiaSources: pd.Series
        Series of lists of previous diaSource dicts for each alert

    Returns
    -------
    out: pd.Series
        Booleans: True for alerts matching GVOM fast transient candidate
        requirements, False otherwise.

    Examples
    --------
    # >>> from fink_filters.rubin.utils import apply_block
    # >>> df2 = apply_block(df, "fink_filters.rubin.livestream.filter_fast_transient_gvom.filter.fast_transient_gvom")
    # >>> df2.count()
    # 0
    """
    print("toto")

    len_data = len(diaSource)

    print(f"nb alerts: {len_data}")

    # Extragalactic loose candidate (no known star, SSO, or galactic contaminant)
    f_extra_gal = fb.b_extragalactic_loose_candidate(
        diaSource,
        simbad_otype,
        mangrove_lum_dist,
        is_sso,
        gaiadr3_DR3Name,
        gaiadr3_e_Plx,
        gaiadr3_e_Plx,
        vsx_Type,
        legacydr8_zphot,
    )

    # age cut: source first appeared less than 5 days ago
    f_new = (diaSource.midpointMjdTai - firstDiaSourceMjdTaiFink) < 5.0

    # Brightness cut: apparent magnitude brighter than 21
    f_bright = fu.flux_to_apparent_mag(diaSource.psfFlux) < 21

    # Flux rate cut: mean flux rate exceeds 0.3 mag/day threshold with SNR > 3
    f_mean_fast = fb.b_mag_rate_mc_sampling(diaSource, prvDiaSources)

    # Combine all criteria
    f_match_gvom_candidate_requirement = f_extra_gal & f_new & f_bright & f_mean_fast

    return f_match_gvom_candidate_requirement


if __name__ == "__main__":
    """Test suite for filters"""
    # Run the test suite

    from fink_filters.tester import spark_unit_tests

    globs = globals()
    spark_unit_tests(globs, load_rubin_df=True)
