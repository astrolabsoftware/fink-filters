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
"""Return LSST alerts with only quality cuts applied"""

import pandas as pd
import fink_filters.rubin.blocks as fb


DESCRIPTION = "Select alerts that are good quality"


def extragalactic_candidate(
    isDipole: pd.Series,
    shape_flag: pd.Series, 
    forced_PsfFlux_flag: pd.Series, 
    psfFlux_flag: pd.Series, 
    apFlux_flag: pd.Series, 
    centroid_flag: pd.Series, 
    pixelFlags_interpolated: pd.Series, 
    pixelFlags_cr: pd.Series, 
    forced_PsfFlux_flag_edge : pd.Series, 
    pixelFlags_bad : pd.Series,
) -> pd.Series:
    """Flag for alerts in Rubin that are good quality

    Notes
    -----
    based on source quality only

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

    Returns
    -------
    out: pd.Series
        Booleans: True for good quality alerts,
        False otherwise.
    """
    # Good quality
    f_good_quality = fb.b_good_quality(isDipole,shape_flag, forced_PsfFlux_flag, psfFlux_flag, 
                                       apFlux_flag, centroid_flag, pixelFlags_interpolated, pixelFlags_cr, 
                                       forced_PsfFlux_flag_edge, pixelFlags_bad)
    
    return f_good_quality
