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

DESCRIPTION = "Select faint trails measuring more than 2 arcsec (mag between 18-21)"


def faint_trails(diaSource: pd.DataFrame) -> pd.Series:
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
    >>> from fink_filters.rubin.utils import apply_block
    >>> df = spark.read.format("parquet").load("datatest/faint_trails")
    >>> df2 = apply_block(df, "fink_filters.rubin.livestream.filter_faint_trails.filter.faint_trails")
    >>> df2.count()
    2
    >>> ids = [row["diaSourceId"] for row in df2.selectExpr("diaSourceId").collect()]
    >>> sorted(ids)
    [170046085932777529, 170046093991084072]
    """
    mag = fu.flux_to_apparent_mag(diaSource.psfFlux)
    f_faint = (mag > 18) & (mag < 21)
    f_long_trail = diaSource.trailLength > 2
    f_not_cosmic_ray = ~diaSource.pixelFlags_cr
    
    return f_long_trail & f_faint & f_not_cosmic_ray

if __name__ == "__main__":
    from fink_filters.tester import spark_unit_tests

    globs = globals()
    spark_unit_tests(globs, load_rubin_df=False)
