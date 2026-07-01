# Copyright 2026 AstroLab Software
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
"""Select faint trails measuring more than 2 arcsec (mag between 18-21)"""

import pandas as pd
import fink_filters.rubin.utils as fu
from fink_filters.rubin.blocks import b_good_quality
from fink_filters.rubin.livestream.filter_faint_trails.utils import compute_elongation_from_image
from astropy.io import fits
import io

DESCRIPTION = "Select faint trails measuring more than 2 arcsec (mag between 18-21)"


def faint_trails(diaSource: pd.DataFrame, cutoutScience: pd.Series) -> pd.Series:
    """Select faint trails measuring more than 2 arcsec (mag between 18-21)

    Parameters
    ----------
    diaSource: pd.DataFrame
        Full diaSource section of an alert (dictionary exploded)

    Returns
    -------
    out: pd.Series
        Booleans: True for faint trail candidate,
        False otherwise.

    Examples
    --------
    # Test data contains the following alerts:
    # 170595945463939107 => FILTERED OUT because not faint enough (high psf flux)
    # 170595940828184636 => FILTERED OUT because trailLength too short
    # 170587117053804617 => FILTERED OUT because cosmic ray
    # 170591547471429724 => FILTERED OUT because not good quality
    # 170600291299229879 => FILTERED OUT because not elongated enough
    # MATCHING: 170591526414450946, 170591533447250193
    >>> from fink_filters.rubin.utils import apply_block
    >>> df = spark.read.format("parquet").load("datatest/faint_trails")
    >>> df2 = apply_block(df, "fink_filters.rubin.livestream.filter_faint_trails.filter.faint_trails")
    >>> df2.count()
    2
    >>> ids = [row["diaSourceId"] for row in df2.selectExpr("diaSourceId").collect()]
    >>> sorted(ids)
    [170591526414450946, 170591533447250193]
    """
    mag = fu.flux_to_apparent_mag(diaSource["psfFlux"])
    f_faint = (mag > 18) & (mag < 21)
    f_long_trail = diaSource["trailLength"] > 2
    f_not_cosmic_ray = ~diaSource["pixelFlags_cr"]
    f_good_quality = b_good_quality(diaSource)

    # Positive flux constraints
    f_flux_pos = diaSource["psfFlux"] > 0
    f_trailflux_pos = diaSource["trailFlux"] > 0

    f_intermediate = f_long_trail & f_faint & f_not_cosmic_ray & f_good_quality & f_flux_pos & f_trailflux_pos

    # Decode FITS image and compute elongation only for alerts which passed the intermediate filter
    f_elong = pd.Series(False, index=cutoutScience.index)
    subset_idx = f_intermediate[f_intermediate].index
    if len(subset_idx) > 0:
      subset_images = cutoutScience.loc[subset_idx].apply(
          lambda x: fits.getdata(io.BytesIO(x)) if isinstance(x, (bytes, bytearray)) else x
      )
      elong_values = subset_images.apply(compute_elongation_from_image)
      f_elong.loc[subset_idx] = (elong_values > 2.0).fillna(False)

    return f_intermediate & f_elong

if __name__ == "__main__":
    from fink_filters.tester import spark_unit_tests

    globs = globals()
    spark_unit_tests(globs, load_rubin_df=False)
