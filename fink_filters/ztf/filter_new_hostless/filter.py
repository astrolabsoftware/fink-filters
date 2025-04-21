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
"""Filters for live hostless detections with ZTF"""

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import BooleanType

import pandas as pd
import numpy as np
from fink_filters.ztf.filter_new_hostless.utils import is_uncataloged
from fink_filters.ztf.filter_new_hostless.utils import is_hostless_base
from fink_filters.ztf.filter_new_hostless.utils import intra_night_transients
from fink_filters.ztf.filter_new_hostless.utils import inter_night_transients
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
    >>> df = spark.read.format('parquet').load('datatest/regular')
    >>> df.filter(new_hostless(
    ...     df["cutoutScience.stampData"],
    ...     df["cutoutTemplate.stampData"],
    ...     df["candidate.ndethist"],
    ...     df["candidate.distnr"],
    ...     df["cdsxmatch"],
    ...     df["DR3Name"],
    ...     df["roid"])).count()
    3
    """
    return new_hostless_(
        cutoutScience, cutoutTemplate, ndethist, distnr, cdsxmatch, DR3Name, roid
    )


def intra_night_hostless_(
    cjdc,
    cmagpsfc,
    cutoutScience,
    cutoutTemplate,
    ndethist,
    distnr,
    cdsxmatch,
    DR3Name,
    roid,
):
    """Find new hostless transients that repeat over the same night

    Notes
    -----
    We request 2 observations from the same night.

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
    >>> import pyspark.sql.functions as F
    >>> from fink_utils.spark.utils import concat_col
    >>> df = spark.read.format('parquet').load('datatest/regular')

    >>> to_expand = ['jd', 'magpsf']

    >>> prefix = 'c'
    >>> for colname in to_expand:
    ...    df = concat_col(df, colname, prefix=prefix)

    # quick fix for https://github.com/astrolabsoftware/fink-broker/issues/457
    >>> for colname in to_expand:
    ...    df = df.withColumnRenamed('c' + colname, 'c' + colname + 'c')

    >>> pdf = df.select(["cjdc", "cmagpsfc", F.col("cutoutScience.stampData").alias("cutoutScience"), F.col("cutoutTemplate.stampData").alias("cutoutTemplate"), "candidate.ndethist", "candidate.distnr", "cdsxmatch", "DR3Name", "roid"]).toPandas()

    >>> is_intra_night_hostless = intra_night_hostless_(
    ...     pdf["cjdc"],
    ...     pdf["cmagpsfc"],
    ...     pdf["cutoutScience"],
    ...     pdf["cutoutTemplate"],
    ...     pdf["ndethist"],
    ...     pdf["distnr"],
    ...     pdf["cdsxmatch"],
    ...     pdf["DR3Name"],
    ...     pdf["roid"])
    >>> is_intra_night_hostless.sum()
    0
    """
    # Uncatalogued
    is_uncat = is_uncataloged(distnr, cdsxmatch, DR3Name, roid)
    is_second = ndethist.to_numpy() == 2

    # Intra Night
    is_intra_night = intra_night_transients(cjdc, cmagpsfc, nobs=2, lapse_hour=12)

    # Hostless
    is_candidate = is_uncat & is_second & is_intra_night
    is_host = is_hostless_base(
        cutoutScience[is_candidate], cutoutTemplate[is_candidate]
    )

    # Combine results
    mask = np.zeros_like(ndethist, dtype=bool)
    mask[is_candidate] = is_host

    return pd.Series(mask)


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def intra_night_hostless(
    cjdc,
    cmagpsfc,
    cutoutScience,
    cutoutTemplate,
    ndethist,
    distnr,
    cdsxmatch,
    DR3Name,
    roid,
):
    """Find new hostless transients that repeat over the same night (Spark)

    Notes
    -----
    We request 2 observations from the same night.

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
    >>> import pyspark.sql.functions as F
    >>> from fink_utils.spark.utils import concat_col
    >>> df = spark.read.format('parquet').load('datatest/regular')

    >>> to_expand = ['jd', 'magpsf']

    >>> prefix = 'c'
    >>> for colname in to_expand:
    ...    df = concat_col(df, colname, prefix=prefix)

    # quick fix for https://github.com/astrolabsoftware/fink-broker/issues/457
    >>> for colname in to_expand:
    ...    df = df.withColumnRenamed('c' + colname, 'c' + colname + 'c')

    >>> df.filter(intra_night_hostless(
    ...     df["cjdc"],
    ...     df["cmagpsfc"],
    ...     df["cutoutScience.stampData"].alias("cutoutScience"),
    ...     df["cutoutTemplate.stampData"].alias("cutoutTemplate"),
    ...     df["candidate.ndethist"],
    ...     df["candidate.distnr"],
    ...     df["cdsxmatch"],
    ...     df["DR3Name"],
    ...     df["roid"])).count()
    0
    """
    return intra_night_hostless_(
        cjdc,
        cmagpsfc,
        cutoutScience,
        cutoutTemplate,
        ndethist,
        distnr,
        cdsxmatch,
        DR3Name,
        roid,
    )


def inter_night_hostless_(
    cjdc,
    cmagpsfc,
    cutoutScience,
    cutoutTemplate,
    ndethist,
    distnr,
    cdsxmatch,
    DR3Name,
    roid,
):
    """Find new hostless transients that repeat over nights

    Notes
    -----
    We request 3 observations, 2 the first night, and 1 a subsequent night.

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
    >>> import pyspark.sql.functions as F
    >>> from fink_utils.spark.utils import concat_col
    >>> df = spark.read.format('parquet').load('datatest/regular')

    >>> to_expand = ['jd', 'magpsf']

    >>> prefix = 'c'
    >>> for colname in to_expand:
    ...    df = concat_col(df, colname, prefix=prefix)

    # quick fix for https://github.com/astrolabsoftware/fink-broker/issues/457
    >>> for colname in to_expand:
    ...    df = df.withColumnRenamed('c' + colname, 'c' + colname + 'c')

    >>> pdf = df.select(["cjdc", "cmagpsfc", F.col("cutoutScience.stampData").alias("cutoutScience"), F.col("cutoutTemplate.stampData").alias("cutoutTemplate"), "candidate.ndethist", "candidate.distnr", "cdsxmatch", "DR3Name", "roid"]).toPandas()

    >>> is_inter_night_hostless = inter_night_hostless_(
    ...     pdf["cjdc"],
    ...     pdf["cmagpsfc"],
    ...     pdf["cutoutScience"],
    ...     pdf["cutoutTemplate"],
    ...     pdf["ndethist"],
    ...     pdf["distnr"],
    ...     pdf["cdsxmatch"],
    ...     pdf["DR3Name"],
    ...     pdf["roid"])
    >>> is_inter_night_hostless.sum()
    0
    """
    # Uncatalogued
    is_uncat = is_uncataloged(distnr, cdsxmatch, DR3Name, roid)
    is_third = ndethist.to_numpy() == 3

    # Inter Night
    is_inter_night = inter_night_transients(cjdc, cmagpsfc, nobs=3, lapse_hour=12)

    # Hostless
    is_candidate = is_uncat & is_third & is_inter_night
    is_host = is_hostless_base(
        cutoutScience[is_candidate], cutoutTemplate[is_candidate]
    )

    # Combine results
    mask = np.zeros_like(ndethist, dtype=bool)
    mask[is_candidate] = is_host

    return pd.Series(mask)


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def inter_night_hostless(
    cjdc,
    cmagpsfc,
    cutoutScience,
    cutoutTemplate,
    ndethist,
    distnr,
    cdsxmatch,
    DR3Name,
    roid,
):
    """Find new hostless transients that repeat over nights (Spark)

    Notes
    -----
    We request 3 observations, 2 the first night, and 1 a subsequent night.

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
    >>> import pyspark.sql.functions as F
    >>> from fink_utils.spark.utils import concat_col
    >>> df = spark.read.format('parquet').load('datatest/regular')

    >>> to_expand = ['jd', 'magpsf']

    >>> prefix = 'c'
    >>> for colname in to_expand:
    ...    df = concat_col(df, colname, prefix=prefix)

    # quick fix for https://github.com/astrolabsoftware/fink-broker/issues/457
    >>> for colname in to_expand:
    ...    df = df.withColumnRenamed('c' + colname, 'c' + colname + 'c')

    >>> df.filter(inter_night_hostless(
    ...     df["cjdc"],
    ...     df["cmagpsfc"],
    ...     df["cutoutScience.stampData"].alias("cutoutScience"),
    ...     df["cutoutTemplate.stampData"].alias("cutoutTemplate"),
    ...     df["candidate.ndethist"],
    ...     df["candidate.distnr"],
    ...     df["cdsxmatch"],
    ...     df["DR3Name"],
    ...     df["roid"])).count()
    0
    """
    return inter_night_hostless_(
        cjdc,
        cmagpsfc,
        cutoutScience,
        cutoutTemplate,
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
