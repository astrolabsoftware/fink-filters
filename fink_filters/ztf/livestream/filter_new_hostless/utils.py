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
"""Utilities for live hostless detections with ZTF"""

import os
import numpy as np

from fink_filters.ztf.filter_simbad_candidates.filter import simbad_candidates_
from fink_filters.ztf.filter_gaia_candidates.filter import gaia_dr3_candidates_

from fink_science.ztf.hostless_detection.run_pipeline import HostLessExtragalactic
from fink_science.ztf.hostless_detection.pipeline_utils import load_json
from fink_science.ztf.hostless_detection import __file__ as hostpath

from fink_filters.tester import spark_unit_tests


hostdir = os.path.dirname(os.path.abspath(hostpath))
CONFIGS_BASE = load_json("{}/config_base.json".format(hostdir))


def is_uncataloged(distnr, cdsxmatch, dr3name, roid):
    """Check if an object does not appear in catalogs

    Parameters
    ----------
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
    out: bool
       Return a Pandas Series with the appropriate flag:
       False for bad alert, and True for good alert.

    Examples
    --------
    >>> import pandas as pd
    >>> pdf = pd.read_parquet('datatest/regular')
    >>> is_uncat = is_uncataloged(
    ...     pdf["candidate"].apply(lambda x: x["distnr"]),
    ...     pdf["cdsxmatch"],
    ...     pdf["DR3Name"],
    ...     pdf["roid"])
    >>> print(is_uncat.sum())
    20
    """
    # Not in ZTF internal
    f1 = distnr > 1.5

    # not in SIMBAD & not in Gaia
    f2 = ~simbad_candidates_(cdsxmatch)
    f3 = ~gaia_dr3_candidates_(dr3name)

    # Not in Minor planet center
    f4 = roid != 3

    return f1 & f2 & f3 & f4


def is_hostless_base(cutoutScience, cutoutTemplate):
    """Check if an alert is hostless

    Parameters
    ----------
    cutoutScience: pd.Series
        science stamp images
    cutoutTemplate: pd.Series
        template stamp images

    Returns
    -------
    pd.Series
        Scores (array of 2 floats) for being hostless

    References
    ----------
    1. ELEPHANT: ExtragaLactic alErt Pipeline for Hostless AstroNomical
    Transients (https://arxiv.org/abs/2404.18165)

    Examples
    --------
    >>> import pandas as pd
    >>> pdf = pd.read_parquet('datatest/regular')
    >>> is_uncat = is_uncataloged(
    ...     pdf["candidate"].apply(lambda x: x["distnr"]),
    ...     pdf["cdsxmatch"],
    ...     pdf["DR3Name"],
    ...     pdf["roid"])

    >>> pdf_uncat = pdf[is_uncat]

    >>> is_host = is_hostless_base(
    ...     pdf_uncat["cutoutScience"].apply(lambda x: x["stampData"]),
    ...     pdf_uncat["cutoutTemplate"].apply(lambda x: x["stampData"]))
    >>> print(is_host.sum())
    3
    """
    # load the configuration file
    hostless_science_class = HostLessExtragalactic(CONFIGS_BASE)

    # Init values
    science_all, template_all = [], []
    for index in range(cutoutScience.shape[0]):
        science_stamp = cutoutScience.to_numpy()[index]
        template_stamp = cutoutTemplate.to_numpy()[index]
        kstest_science, kstest_template = hostless_science_class.process_candidate_fink(
            science_stamp, template_stamp
        )
        science_all.append(kstest_science)
        template_all.append(kstest_template)

    f1 = (np.array(science_all) >= 0) * (np.array(science_all) <= 0.5)
    f2 = (np.array(template_all) >= 0) * (np.array(template_all) <= 0.85)

    return f1 * f2


def intra_night_transients(cjdc, cmagpsfc, nobs=2, lapse_hour=12):
    """Find objects that vary within a night with `nobs` observations

    Parameters
    ----------
    cjdc: pd.Series
        Series of JD vectors
    cmagpsfc: pd.Series
        Series of magpsf vectors
    nobs: int
        Number of observations. Default is 2.
    lapse_hour: float
        Maximum time for variation, in hours. Default is 12

    Returns
    -------
    out: np.array
        Array of booleans. True is intra-night, False otherwise.

    Examples
    --------
    >>> from fink_utils.spark.utils import concat_col
    >>> df = spark.read.format('parquet').load('datatest/regular')

    >>> to_expand = ['jd', 'magpsf']

    >>> prefix = 'c'
    >>> for colname in to_expand:
    ...    df = concat_col(df, colname, prefix=prefix)

    # quick fix for https://github.com/astrolabsoftware/fink-broker/issues/457
    >>> for colname in to_expand:
    ...    df = df.withColumnRenamed('c' + colname, 'c' + colname + 'c')

    >>> pdf = df.select(["cjdc", "cmagpsfc"]).toPandas()
    >>> mask = intra_night_transients(pdf["cjdc"], pdf["cmagpsfc"], nobs=2, lapse_hour=12)
    >>> np.sum(mask)
    117
    """
    to_return = np.zeros_like(cjdc, dtype=bool)
    # last nobs are not None
    mask = cmagpsfc.apply(lambda x: np.sum([i is not None for i in x[-nobs:]])) == nobs

    # difference between last and first JD
    to_return[mask] = (
        cjdc[mask].apply(lambda x: x[-nobs:][-1] - x[-nobs:][0]) < lapse_hour / 24.0
    )
    return to_return


def inter_night_transients(cjdc, cmagpsfc, nobs=3, lapse_hour=12):
    """Find new objects that vary between 2 observing night with `nobs` observations

    Parameters
    ----------
    cjdc: pd.Series
        Series of JD vectors
    cmagpsfc: pd.Series
        Series of magpsf vectors
    nobs: int
        Number of observations. Default is 3.
    lapse_hour: float
        Minimum time for variation, in hours. Default is 12

    Examples
    --------
    >>> from fink_utils.spark.utils import concat_col
    >>> df = spark.read.format('parquet').load('datatest/regular')

    >>> to_expand = ['jd', 'magpsf']

    >>> prefix = 'c'
    >>> for colname in to_expand:
    ...    df = concat_col(df, colname, prefix=prefix)

    # quick fix for https://github.com/astrolabsoftware/fink-broker/issues/457
    >>> for colname in to_expand:
    ...    df = df.withColumnRenamed('c' + colname, 'c' + colname + 'c')

    >>> pdf = df.select(["cjdc", "cmagpsfc"]).toPandas()
    >>> mask = inter_night_transients(pdf["cjdc"], pdf["cmagpsfc"], nobs=3, lapse_hour=12)
    >>> np.sum(mask)
    30
    """
    to_return = np.zeros_like(cjdc, dtype=bool)
    # last nobs are not None
    mask = cmagpsfc.apply(lambda x: np.sum([i is not None for i in x[-nobs:]])) == nobs

    # difference between last and first JD
    cond1 = cjdc[mask].apply(lambda x: x[-nobs:][2] - x[-nobs:][1]) > lapse_hour / 24.0
    cond2 = cjdc[mask].apply(lambda x: x[-nobs:][1] - x[-nobs:][0]) < 12 / 24.0
    to_return[mask] = cond1 * cond2
    return to_return


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
