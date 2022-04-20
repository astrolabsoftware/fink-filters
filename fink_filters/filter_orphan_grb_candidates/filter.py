# Copyright 2021 AstroLab Software
# Author: Julien Peloton, Johan Bregeon
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
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import BooleanType

from fink_filters.tester import spark_unit_tests

import pandas as pd
import numpy as np

def get_valid_rate(mag, filt):
    """ Try to constrain the rate between the 2nd and 3rd measurements

    case 1: the measurements are taken with the same filter
        - mag[2] - mag[1] > 0.0 (becomes fainter)
    case 2: filt(1)=g, filt(2)=r
        - mag[1] - mag[2] <= 0.3 (the difference is smaller than the baseline g-r = 0.3)
    case 1: filt(1)=r, filt(2)=g
        - mag[2] - mag[1] > 0.0 (no real constraints...)

    """
    v = lambda val, mag: val[~np.isnan(mag)]
    filt2nd = v(filt, mag)[1]
    filt3rd = v(filt, mag)[2]

    if filt2nd == filt3rd:
        cond = (v(mag, mag)[2] - v(mag, mag)[1]) > 0.0
    elif filt3rd > filt2nd:
        # g puis r
        cond = (v(mag, mag)[1] - v(mag, mag)[2]) <= 0.3
    else:
        cond = (v(mag, mag)[2] - v(mag, mag)[1]) > 0.0
    return cond

@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def orphan_grb(jd, jdstarthist, cjdc, cfidc, cssnamenrc, cmagpsfc):
    """ Simple filter to extract orphan GRB candidates.

    The filter has 6 steps:
    1. No more than a month between first and last detection
    2. Max magnitude at 18 (faint object)
    3. At least 3  detections in 10 days (slow transient)
    4. The last measurement must be lower (increase in mag) than the previous one
    5. The difference between the g-band and r-band must be almost constant and positive
    6. The alert should not be an identified Solar System Objects

    Based on current ZTF data, it yields about 50-100 candidates per month (out
    of >2,000,000 incoming alerts).

    Parameters
    ----------
    jd: pandas.Series of float
        JD for the emission of the alert
    jdstarthist: pandas.Series of float
        JD for the first detection of the object to which the alert belongs to
    cjdc: pandas.Series of list of float
        Concatenated jd for the object
    cfidc: pandas.Series of list of int
        Concatenated filter ID for the object
    cssnamenrc: pandas.Series of list of str, or NaN
        Concatenated SSO name for the object
    cmagpsfc: pandas.Series of list of float, or Nan
        Concatenated mag for the object

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> from fink_utils.spark.utils import concat_col
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> df = spark.read.format('parquet').load('datatest')

    >>> to_expand = ['jd', 'fid', 'ssnamenr', 'magpsf']

    >>> prefix = 'c'
    >>> for colname in to_expand:
    ...    df = concat_col(df, colname, prefix=prefix)

    # quick fix for https://github.com/astrolabsoftware/fink-broker/issues/457
    >>> for colname in to_expand:
    ...    df = df.withColumnRenamed('c' + colname, 'c' + colname + 'c')

    >>> f = 'fink_filters.filter_orphan_grb_candidates.filter.orphan_grb'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    0
    """
    # 1 - No more than a month between first and last detection
    at_most_a_month = (jd - jdstarthist) <= 30

    # 2 - Max magnitude at 18 (faint object)
    above_18 = cmagpsfc.apply(lambda lc: np.all(lc[~np.isnan(lc)] > 18))

    # 3 - At least 3 detections in 10 days (slow transient)
    at_least_3_det = cmagpsfc.apply(lambda lc: len(lc[~np.isnan(lc)]) == 3)

    valid_times = lambda mag, time: time[~np.isnan(mag)]
    tmp1 = np.array(
        [
            False if not at_least_3_det.values[n] else (valid_times(i, j)[2] - valid_times(i, j)[0]) < 10.0 for n, i, j in zip(range(len(at_least_3_det)), cmagpsfc.values, cjdc.values)
        ]
    )

    # 4 - The last measurement must be lower (increase in mag) than
    # the previous one. /!\ no band info currently
    tmp2 = np.array(
        [
            False if not at_least_3_det.values[n] else get_valid_rate(i, j) for n, i, j in zip(range(len(at_least_3_det)), cmagpsfc.values, cfidc.values)
        ]
    )

    # 5 - The difference between the g-band and
    # r-band must be almost constant and positive
    condg = lambda mag, filt: mag[~np.isnan(mag) & (filt.astype(int) == 1)]
    condr = lambda mag, filt: mag[~np.isnan(mag) & (filt.astype(int) == 2)]

    meang = np.array(
        [
            np.mean(condg(i, j)) for i, j in zip(cmagpsfc.values, cfidc.values)
        ]
    )
    meanr = np.array(
        [
            np.mean(condr(i, j)) for i, j in zip(cmagpsfc.values, cfidc.values)
        ]
    )
    tmp3 = (meang - meanr) >= 0

    # 6 - The alert should not be an identified Solar System Objects
    v = lambda val, mag: val[~np.isnan(mag)]
    tmp4 = [np.all([k in [None, 'null'] for k in v(i, j)]) for i, j in zip(cssnamenrc.values, cmagpsfc.values)]

    # Final
    tmp = at_most_a_month & above_18 & at_least_3_det & tmp1 & tmp2 & tmp3 & tmp4

    return pd.Series(tmp, dtype=bool)


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
