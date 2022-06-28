# Copyright 2019-2022 AstroLab Software
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
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import BooleanType

from fink_utils.xmatch.simbad import return_list_of_eg_host

from fink_filters.tester import spark_unit_tests

import pandas as pd

def sn_candidates_(
        cdsxmatch, snn_snia_vs_nonia, snn_sn_vs_all,
        drb, classtar, jd, jdstarthist, roid, ndethist) -> pd.Series:
    """ Return alerts considered as SN-Ia candidates

    Parameters
    ----------
    cdsxmatch: Pandas series
        Column containing the cross-match values
    snn_snia_vs_nonia: Pandas series
        Column containing the probability to be a SN Ia from SuperNNova.
    snn_sn_vs_all: Pandas series
        Column containing the probability to be a SNe from SuperNNova.
    drb: Pandas series
        Column containing the Deep-Learning Real Bogus score
    classtar: Pandas series
        Column containing the sextractor score
    jd: Pandas series
        Column containing the JD of the _alert_
    jdstarthist: Pandas series
        Column containing the starting JD of the _object_
    roid: Pandas series
        Column containing the SSO label
    ndethist: Pandas series
        Column containing the number of detection at 3 sigma since the
        beginning of the survey

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> pdf = pd.read_parquet('datatest')
    >>> classification = sn_candidates_(
    ...     pdf['cdsxmatch'],
    ...     pdf['snn_snia_vs_nonia'],
    ...     pdf['snn_sn_vs_all'],
    ...     pdf['candidate'].apply(lambda x: x['drb']),
    ...     pdf['candidate'].apply(lambda x: x['classtar']),
    ...     pdf['candidate'].apply(lambda x: x['jd']),
    ...     pdf['candidate'].apply(lambda x: x['jdstarthist']),
    ...     pdf['roid'],
    ...     pdf['candidate'].apply(lambda x: x['ndethist']))
    >>> print(len(pdf[classification]['objectId'].values))
    9

    >>> assert 'ZTF21acoqgmy' in pdf[classification]['objectId'].values
    """
    snn1 = snn_snia_vs_nonia.astype(float) > 0.5
    snn2 = snn_sn_vs_all.astype(float) > 0.5
    sn_history = jd.astype(float) - jdstarthist.astype(float) <= 90
    high_drb = drb.astype(float) > 0.5
    high_classtar = classtar.astype(float) > 0.4
    no_mpc = roid.astype(int) != 3
    no_first_det = ndethist.astype(int) > 1

    keep_cds = return_list_of_eg_host()

    f_sn = (snn1 | snn2) & cdsxmatch.isin(keep_cds) & sn_history & high_drb & high_classtar & no_first_det & no_mpc

    return f_sn

@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def sn_candidates(
        cdsxmatch, snn_snia_vs_nonia, snn_sn_vs_all,
        drb, classtar, jd, jdstarthist, roid, ndethist) -> pd.Series:
    """ Pandas UDF for sn_candidates_

    Examples
    ----------
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> df = spark.read.format('parquet').load('datatest')
    >>> f = 'fink_filters.filter_sn_candidates.filter.sn_candidates'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    9
    """
    series = sn_candidates_(
        cdsxmatch, snn_snia_vs_nonia, snn_sn_vs_all,
        drb, classtar, jd, jdstarthist, roid, ndethist
    )
    return series


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
