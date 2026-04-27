# Copyright 2019-2026 AstroLab Software
# Author: Julien Peloton, Camille Douzet
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
"""Select LSST alerts new (< 5days first apparition), bright (mag < 24), potentially extragalactic with a fading or rising rate passing the cuts"""

import pandas as pd

import fink_filters.rubin.blocks as fb
import fink_filters.rubin.utils as fu

DESCRIPTION = "Select LSST alerts new (< 5days first apparition), bright (mag < 24), potentially extragalactic with a fading or rising rate passing the cuts"


def has_two_points_same_band(
    diaSource: pd.DataFrame, diaObject: pd.DataFrame
) -> pd.Series:
    """Check if an object has at least 2 detections in the same band as the current alert.

    Parameters
    ----------
    diaSource: pd.DataFrame
        Full diaSource section of an alert (dictionary exploded)
    diaObject: pd.DataFrame
        Full diaObject section of an alert (dictionary exploded)

    Returns
    -------
    out: pd.Series of booleans
        True if at least 2 detections exist in the current alert's band, False otherwise.
    """
    result = pd.Series(False, index=diaSource.index)

    for b in ["u", "g", "r", "i", "z", "y"]:
        mask = diaSource.band == b
        col = f"{b}_psfFluxNdata"
        try:
            result[mask] = diaObject.loc[mask, col] >= 2
        except KeyError:
            continue
    return result


def get_latest_source_same_band(row) -> dict | None:
    """Return the most recent previous detection in the same band as the current alert.

    Parameters
    ----------
    row: pd.Series
        A single row from a DataFrame combining diaSource and prvDiaSources columns.

    Returns
    -------
    out: dict or None
        The most recent previous diaSource dict in the same band, or None if none exists.
    """
    sources = row["prvDiaSources"]
    if sources is None or len(sources) == 0:
        return None
    current_band = row["band"]
    same_band = [s for s in sources if s["band"] == current_band]
    if not same_band:
        return None
    return max(same_band, key=lambda s: s["midpointMjdTai"])


def extragalactic_new_candidate(
    diaSource: pd.DataFrame,
    diaObject: pd.DataFrame,
    prvDiaSources: pd.Series,
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
    """Select LSST alerts new (< 5days first apparition), bright (mag < 24), potentially extragalactic with a fading or rising rate passing the cuts

    Notes
    -----
    Based on an extragalactic block, time cut, sampling cut, and rate cut.
    Rising alerts must have rate < -0.2 mag/day and last less than 3 days.
    Fading alerts must have rate > 0.2 mag/day in r/i bands, or > 0.5 mag/day in g/u bands.

    Parameters
    ----------
    diaSource: pd.DataFrame
        Full diaSource section of an alert (dictionary exploded)
    diaObject: pd.DataFrame
        Full diaObject section of an alert (dictionary exploded)
    prvDiaSources: pd.Series
        Series of lists of previous diaSource dicts for each alert
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

    # 5 days maximum
    f_new = (diaSource.midpointMjdTai - firstDiaSourceMjdTaiFink) < 5.0

    f_bright = fu.flux_to_apparent_mag(diaSource.psfFlux) < 24

    # Minimum 2 points

    f_sampling = has_two_points_same_band(diaSource, diaObject)

    # Check rising and fading rate
    df = pd.concat([diaSource, prvDiaSources.rename("prvDiaSources")], axis=1)
    prev = df.apply(get_latest_source_same_band, axis=1)

    prev_flux = prev.apply(lambda s: s["psfFlux"] if s is not None else float("nan"))
    prev_time = prev.apply(
        lambda s: s["midpointMjdTai"] if s is not None else float("nan")
    )

    delta_mag = fu.flux_to_apparent_mag(diaSource.psfFlux) - fu.flux_to_apparent_mag(
        prev_flux
    )
    delta_time = diaSource.midpointMjdTai - prev_time
    delta_time_rising = diaSource.midpointMjdTai - firstDiaSourceMjdTaiFink

    rate = delta_mag / delta_time

    f_rising = (rate < -0.2) & (delta_time_rising < 3)

    f_fading_ri = ((diaSource.band == "r") | (diaSource.band == "i")) & (rate > 0.2)
    f_fading_gu = ((diaSource.band == "g") | (diaSource.band == "u")) & (rate > 0.5)

    f_rate = f_rising | f_fading_ri | f_fading_gu

    f_extragalactic_new = (
        f_extragalactic_near_galaxy & f_new & f_sampling & f_bright & f_rate
    )

    return f_extragalactic_new


if __name__ == "__main__":
    """Test suite for filters"""
    # Run the test suite

    from fink_filters.tester import spark_unit_tests

    globs = globals()
    spark_unit_tests(globs, load_rubin_df=True)
