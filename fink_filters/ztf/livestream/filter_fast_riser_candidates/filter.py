# Copyright 2026 AstroLab Software
# Author: Edel Moreno Lemus
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
"""Return alerts considered as young and fast-rising transient candidates"""

import numpy as np
import pandas as pd
from fink_utils.xmatch.simbad import return_list_of_eg_host
from pyspark.sql.functions import PandasUDFType, pandas_udf
from pyspark.sql.types import BooleanType

from fink_filters.tester import spark_unit_tests

# Maximum age of the variability, in days, for an object to count as "young".
MAX_AGE_DAYS = 60.0
# Minimum brightening rate, in mag/day, to count as "fast rising".
MIN_RISE_RATE = 0.25
# Pairs closer in time than this are ignored: mag/dt explodes for intra-night
# pairs and produces meaningless rates.
MIN_DT_DAYS = 0.02
# Physical guard on the brightening rate (mag/day).
MAX_RISE_RATE = 6.0


def compute_rise_rate(cjd, cmagpsf, cfid) -> float:
    """Return the fastest brightening rate of a light curve, in mag/day.

    Brightening means the magnitude decreases, so the rate is positive when the
    object gets brighter. Same-band pairs are preferred to avoid colour offsets,
    and non-detections (NaN) are ignored.

    Parameters
    ----------
    cjd: list
        Concatenated Julian dates (history + current measurement)
    cmagpsf: list
        Concatenated magnitudes from PSF-fit photometry. Non-detections are NaN.
    cfid: list
        Concatenated filter IDs

    Returns
    -------
    out: float
        Fastest brightening rate in mag/day, 0.0 if it cannot be computed.

    Examples
    --------
    >>> compute_rise_rate([0.0, 1.0], [20.0, 18.0], [1, 1])
    2.0

    >>> compute_rise_rate([0.0, 1.0], [18.0, 20.0], [1, 1])
    0.0

    >>> compute_rise_rate([0.0, 1.0], [np.nan, 18.0], [1, 1])
    0.0
    """
    jd = np.array(cjd, dtype=float)
    mag = np.array(cmagpsf, dtype=float)
    fid = np.array(cfid, dtype=float)

    mask = ~np.isnan(mag) & ~np.isnan(jd)
    jd, mag, fid = jd[mask], mag[mask], fid[mask]
    if len(mag) < 2:
        return 0.0

    order = np.argsort(jd)
    jd, mag, fid = jd[order], mag[order], fid[order]

    rates = []
    for band in np.unique(fid):
        idx = np.where(fid == band)[0]
        for i, j in zip(idx, idx[1:]):
            dt = jd[j] - jd[i]
            if dt >= MIN_DT_DAYS:
                rates.append((mag[i] - mag[j]) / dt)

    positive = [rate for rate in rates if rate > 0]
    if not positive:
        return 0.0

    return float(min(max(positive), MAX_RISE_RATE))


def fast_riser_candidates_(
    cdsxmatch, roid, drb, jd, jdstarthist, cjd, cmagpsf, cfid
) -> pd.Series:
    """Return alerts considered as young and fast-rising transient candidates

    The selection targets the corner of the parameter space where young
    supernovae and fast blue optical transients live: an object whose
    variability started recently, that is brightening quickly, and that is not
    already known as a variable star or a Solar System object.

    Three conditions are combined:

    1. the object is not a known variable star (`cdsxmatch` must be an
       extragalactic host or unknown) and is not a Solar System object (`roid`),
    2. the variability started less than `MAX_AGE_DAYS` days ago,
    3. the light curve brightens faster than `MIN_RISE_RATE` mag/day.

    Parameters
    ----------
    cdsxmatch: Pandas series
        Column containing the cross-match values
    roid: Pandas series
        Column containing the Solar System label
    drb: Pandas series
        Column containing the Deep-Learning Real Bogus score
    jd: Pandas series
        Column containing the Julian date of the current measurement
    jdstarthist: Pandas series
        Column containing the earliest Julian date of the variability
    cjd: Pandas series
        Column containing the concatenated Julian dates
    cmagpsf: Pandas series
        Column containing the concatenated magnitudes from PSF-fit photometry
    cfid: Pandas series
        Column containing the concatenated filter IDs

    Returns
    -------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    --------
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F
    >>> df = spark.read.format('parquet').load('datatest/regular')

    # Append temp columns with historical + current measurements
    >>> what = ['jd', 'magpsf', 'fid']
    >>> prefix = 'c'
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    >>> pdf = df.select(
    ...     'objectId', 'cdsxmatch', 'roid', 'cjd', 'cmagpsf', 'cfid',
    ...     F.col('candidate.drb').alias('drb'),
    ...     F.col('candidate.jd').alias('jd'),
    ...     F.col('candidate.jdstarthist').alias('jdstarthist')).toPandas()

    >>> classification = fast_riser_candidates_(
    ...     pdf['cdsxmatch'], pdf['roid'], pdf['drb'], pdf['jd'],
    ...     pdf['jdstarthist'], pdf['cjd'], pdf['cmagpsf'], pdf['cfid'])
    >>> print(len(pdf[classification]['objectId'].to_numpy()))
    8

    >>> assert 'ZTF21acobels' in pdf[classification]['objectId'].to_numpy()
    """
    # 1. not a known variable star, and not a Solar System object
    keep_cds = return_list_of_eg_host()
    not_known = cdsxmatch.isin(keep_cds)
    not_sso = ~roid.astype(int).isin([2, 3])
    high_drb = drb.astype(float) > 0.5

    # 2. the variability started recently
    young = (jd.astype(float) - jdstarthist.astype(float)) <= MAX_AGE_DAYS

    # 3. the light curve is brightening quickly
    rise_rate = pd.Series(
        [
            compute_rise_rate(jd_, mag_, fid_)
            for jd_, mag_, fid_ in zip(cjd, cmagpsf, cfid)
        ],
        index=cjd.index,
    )
    fast = rise_rate >= MIN_RISE_RATE

    return not_known & not_sso & high_drb & young & fast


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def fast_riser_candidates(
    cdsxmatch, roid, drb, jd, jdstarthist, cjd, cmagpsf, cfid
) -> pd.Series:
    """Pandas UDF for fast_riser_candidates_

    Parameters
    ----------
    cdsxmatch: Spark DataFrame Column
        Column containing the cross-match values
    roid: Spark DataFrame Column
        Column containing the Solar System label
    drb: Spark DataFrame Column
        Column containing the Deep-Learning Real Bogus score
    jd: Spark DataFrame Column
        Column containing the Julian date of the current measurement
    jdstarthist: Spark DataFrame Column
        Column containing the earliest Julian date of the variability
    cjd: Spark DataFrame Column
        Column containing the concatenated Julian dates
    cmagpsf: Spark DataFrame Column
        Column containing the concatenated magnitudes from PSF-fit photometry
    cfid: Spark DataFrame Column
        Column containing the concatenated filter IDs

    Returns
    -------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    --------
    >>> from fink_utils.spark.utils import concat_col
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> df = spark.read.format('parquet').load('datatest/regular')

    # Append temp columns with historical + current measurements
    >>> what = ['jd', 'magpsf', 'fid']
    >>> prefix = 'c'
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    >>> f = 'fink_filters.ztf.livestream.filter_fast_riser_candidates.filter.fast_riser_candidates'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    8
    """
    series = fast_riser_candidates_(
        cdsxmatch, roid, drb, jd, jdstarthist, cjd, cmagpsf, cfid
    )

    return series


if __name__ == "__main__":
    """Execute the test suite"""
    globs = globals()
    spark_unit_tests(globs)
