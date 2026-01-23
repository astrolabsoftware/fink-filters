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
"""Blocks used to build filters"""

import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord


def b_is_solar_system(is_sso: pd.Series) -> pd.Series:
    """Return alerts that are asteroids according to Rubin

    Parameters
    ----------
    is_sso: pd.Series of booleans
        `pred.is_sso`
    """
    return is_sso


def b_outside_galactic_plane(ra: pd.Series, dec: pd.Series) -> pd.Series:
    """Return alerts outside the galactic plane (+/- |20| deg)

    Parameters
    ----------
    ra: pd.Series of float
        RA in degree
    dec: pd.Series of float
        DEC in degree

    Returns
    -------
    out: pd.Series of booleans
        True if outside the plane. False otherwise
    """
    coords = SkyCoord(ra.astype(float), dec.astype(float), unit="deg")
    b = coords.galactic.b.deg
    mask_away_from_galactic_plane = np.abs(b) > 20
    return pd.Series(mask_away_from_galactic_plane)
