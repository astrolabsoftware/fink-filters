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


DESCRIPTION = "Select alerts that are extragalactic candidates"


def extragalactic_candidate(
    simbad_otype: pd.Series,
    mangrove_lum_dist: pd.Series,
    ra: pd.Series,
    dec: pd.Series,
    is_sso: pd.Series,
    gaiadr3_DR3Name: pd.Series,
    gaiadr3_Plx: pd.Series,
    gaiadr3_e_Plx: pd.Series,
    vsx_Type: pd.Series,
) -> pd.Series:
    """Flag for alerts in Rubin that are extragalactic candidates

    Notes
    -----
    based on xmatch with catalogues, galactic coordinates,
    and asteroid veto

    Parameters
    ----------
    simbad_otype: pd.Series
        Series containing labels from `xm.simbad_otype`
    mangrove_lum_dist: pd.Series
        Series containing floats from `xm.mangrove_lum_dist`
    ra: pd.Series
        Series containing floats from `diaObject.ra`
    dec: pd.Series
        Series containing floats from `diaObject.dec`
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

    Returns
    -------
    out: pd.Series
        Booleans: True for alerts extragalactic candidates,
        False otherwise.
    """
    # Xmatch galaxy or Unknown
    f_in_galaxy_simbad = fb.b_xmatched_simbad_galaxy(simbad_otype)
    f_in_galaxy_mangrove = fb.b_xmatched_mangrove(mangrove_lum_dist)
    f_unknown_simbad = fb.b_xmatched_simbad_unknown(simbad_otype)

    # Outside galactic plane
    f_outside_galactic_plane = fb.b_outside_galactic_plane(ra, dec)

    # Not a roid
    f_roid = fb.b_is_solar_system(is_sso)

    # Not a catalogued star
    f_in_gaia = fb.b_xmatched_gaia_star(gaiadr3_DR3Name, gaiadr3_Plx, gaiadr3_e_Plx)
    f_in_vsx_star = fb.b_xmatched_vsx_star(vsx_Type)
    f_not_star = ~f_in_gaia & ~f_in_vsx_star

    f_extragalactic = (
        (f_in_galaxy_simbad | f_in_galaxy_mangrove | f_unknown_simbad)
        & (f_outside_galactic_plane)
        & ~f_roid
        & f_not_star
    )

    return f_extragalactic
