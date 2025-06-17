# Copyright 2025 AstroLab Software
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
"""Find newly appearing and hostless transients"""

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import BooleanType

import pandas as pd
import numpy as np
from fink_filters.ztf.livestream.filter_new_hostless.utils import is_uncataloged
from fink_filters.ztf.livestream.filter_new_hostless.utils import is_hostless_base
from fink_filters.tester import spark_unit_tests


def new_hostless_(
    cutoutScience, cutoutTemplate, ndethist, distnr, cdsxmatch, DR3Name, roid
):
    """Find newly appearing and hostless transients

    Parameters
    ----------
    cutoutScience: pd.Series of NxN arrays
        Science cutouts
    cutoutTemplate: pd.Series of NxN arrays
        Template cutouts
    ndethist: pd.Series of int
        Number of spatially-coincident detections falling within 1.5 arcsec
        going back to beginning of survey; only detections that fell on the
        same field and readout-channel ID where the input candidate was
        observed are counted. All raw detections down to a photometric S/N
        of ~ 3 are included.
    distnr: pd.Series of float
        Distance to nearest source in reference
        image PSF-catalog within 30 arcsec [pixels]
    cdsxmatch: pd.Series of str
        Object type of the closest source from SIMBAD
        database; if exists within 1 arcsec. 'Unknown' otherwise.
    dr3name: pd.Series of str
        Unique source designation of closest source
        from Gaia catalog; if exists within 1 arcsec. NaN otherwise.
    roid: pd.Series of int
        Determine if the alert is a potential Solar
        System object (experimental):
            0: likely not SSO
            1: first appearance but likely not SSO
            2: candidate SSO
            3: found in MPC.

    Returns
    -------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        False for bad alert, and True for good alert.

    Examples
    --------
    >>> import pandas as pd
    >>> pdf = pd.read_parquet('datatest/regular')
    >>> is_new_hostless = new_hostless_(
    ...     pdf["cutoutScience"].apply(lambda x: x["stampData"]),
    ...     pdf["cutoutTemplate"].apply(lambda x: x["stampData"]),
    ...     pdf["candidate"].apply(lambda x: x["ndethist"]),
    ...     pdf["candidate"].apply(lambda x: x["distnr"]),
    ...     pdf["cdsxmatch"],
    ...     pdf["DR3Name"],
    ...     pdf["roid"])
    >>> is_new_hostless.sum()
    3
    """
    # New and uncatalogued
    is_uncat = is_uncataloged(distnr, cdsxmatch, DR3Name, roid)
    is_new = ndethist.to_numpy() == 1
    is_uncat_and_new = is_uncat & is_new

    # Hostless
    is_host = is_hostless_base(
        cutoutScience[is_uncat_and_new], cutoutTemplate[is_uncat_and_new]
    )

    # Combine results
    mask = np.zeros_like(ndethist, dtype=bool)
    mask[is_uncat_and_new] = is_host

    return pd.Series(mask)


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def new_hostless(
    cutoutScience, cutoutTemplate, ndethist, distnr, cdsxmatch, DR3Name, roid
):
    """Find newly appearing and hostless transients (Spark)

    Parameters
    ----------
    cutoutScience: pd.DataFrame
        Science cutouts struct
    cutoutTemplate: pd.DataFrame
        Template cutouts struct
    ndethist: pd.Series of int
        Number of spatially-coincident detections falling within 1.5 arcsec
        going back to beginning of survey; only detections that fell on the
        same field and readout-channel ID where the input candidate was
        observed are counted. All raw detections down to a photometric S/N
        of ~ 3 are included.
    distnr: pd.Series of float
        Distance to nearest source in reference
        image PSF-catalog within 30 arcsec [pixels]
    cdsxmatch: pd.Series of str
        Object type of the closest source from SIMBAD
        database; if exists within 1 arcsec. 'Unknown' otherwise.
    dr3name: pd.Series of str
        Unique source designation of closest source
        from Gaia catalog; if exists within 1 arcsec. NaN otherwise.
    roid: pd.Series of int
        Determine if the alert is a potential Solar
        System object (experimental):
            0: likely not SSO
            1: first appearance but likely not SSO
            2: candidate SSO
            3: found in MPC.

    Returns
    -------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        False for bad alert, and True for good alert.

    Examples
    --------
    >>> df = spark.read.format('parquet').load('datatest/regular')
    >>> df.filter(new_hostless(
    ...     df["cutoutScience"],
    ...     df["cutoutTemplate"],
    ...     df["candidate.ndethist"],
    ...     df["candidate.distnr"],
    ...     df["cdsxmatch"],
    ...     df["DR3Name"],
    ...     df["roid"])).count()
    3
    """
    return new_hostless_(
        cutoutScience["stampData"],
        cutoutTemplate["stampData"],
        ndethist,
        distnr,
        cdsxmatch,
        DR3Name,
        roid,
    )


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
