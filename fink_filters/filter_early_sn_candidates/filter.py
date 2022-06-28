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

def early_sn_candidates_(
        cdsxmatch, snn_snia_vs_nonia, snn_sn_vs_all, rf_snia_vs_nonia,
        ndethist, drb, classtar) -> pd.Series:
    """ Return alerts considered as Early SN-Ia candidates

    Parameters
    ----------
    cdsxmatch: Pandas series
        Column containing the cross-match values
    snn_snia_vs_nonia: Pandas series
        Column containing the probability to be a SN Ia from SuperNNova.
    snn_sn_vs_all: Pandas series
        Column containing the probability to be a SNe from SuperNNova.
    rf_snia_vs_nonia: Pandas series
        Column containing the probability to be a SN Ia from RandomForestClassifier.
    ndethist: Pandas series
        Column containing the number of detection by ZTF
    drb: Pandas series
        Column containing the Deep-Learning Real Bogus score
    classtar: Pandas series
        Column containing the sextractor score

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> pdf = pd.read_parquet('datatest')
    >>> classification = early_sn_candidates_(
    ...     pdf['cdsxmatch'],
    ...     pdf['snn_snia_vs_nonia'],
    ...     pdf['snn_sn_vs_all'],
    ...     pdf['rf_snia_vs_nonia'],
    ...     pdf['candidate'].apply(lambda x: x['ndethist']),
    ...     pdf['candidate'].apply(lambda x: x['drb']),
    ...     pdf['candidate'].apply(lambda x: x['classtar']))
    >>> print(len(pdf[classification]['objectId'].values))
    5

    >>> assert 'ZTF21acobels' in pdf[classification]['objectId'].values
    """
    snn1 = snn_snia_vs_nonia.astype(float) > 0.5
    snn2 = snn_sn_vs_all.astype(float) > 0.5
    active_learn = rf_snia_vs_nonia.astype(float) > 0.5
    early_ndethist = ndethist.astype(int) <= 20
    high_drb = drb.astype(float) > 0.5
    high_classtar = classtar.astype(float) > 0.4

    keep_cds = return_list_of_eg_host()

    f_sn = (snn1 | snn2) & cdsxmatch.isin(keep_cds) & high_drb & high_classtar
    f_sn_early = early_ndethist & active_learn & f_sn

    return f_sn_early


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def early_sn_candidates(
        cdsxmatch, snn_snia_vs_nonia, snn_sn_vs_all, rf_snia_vs_nonia,
        ndethist, drb, classtar) -> pd.Series:
    """ Pandas UDF for early_sn_candidates_

    Parameters
    ----------
    cdsxmatch: Pandas series
        Column containing the cross-match values
    snn_snia_vs_nonia: Pandas series
        Column containing the probability to be a SN Ia from SuperNNova.
    snn_sn_vs_all: Pandas series
        Column containing the probability to be a SNe from SuperNNova.
    rf_snia_vs_nonia: Pandas series
        Column containing the probability to be a SN Ia from RandomForestClassifier.
    ndethist: Pandas series
        Column containing the number of detection by ZTF
    drb: Pandas series
        Column containing the Deep-Learning Real Bogus score
    classtar: Pandas series
        Column containing the sextractor score

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> df = spark.read.format('parquet').load('datatest')
    >>> f = 'fink_filters.filter_early_sn_candidates.filter.early_sn_candidates'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    5
    """
    series = early_sn_candidates_(
        cdsxmatch, snn_snia_vs_nonia, snn_sn_vs_all, rf_snia_vs_nonia,
        ndethist, drb, classtar
    )
    return series


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
