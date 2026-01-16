# Copyright 2026 AstroLab Software
# Author: Julien Peloton, Christian Heissenbuettel
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
"""Return unclassified alerts from the stream"""

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import BooleanType

from fink_filters.ztf.classification import extract_fink_classification_

from fink_filters.tester import spark_unit_tests

import pandas as pd


def unknowns_(
    cdsxmatch,
    roid,
    mulens,
    snn_snia_vs_nonia,
    snn_sn_vs_all,
    rf_snia_vs_nonia,
    ndethist,
    drb,
    classtar,
    jd,
    jdstarthist,
    rf_kn_vs_nonkn,
) -> pd.Series:
    """Return unclassified alerts in the stream

    Notes
    -----
    This filter re-applies the classification. In the near
    future, this will be simpler as the classification tag will
    be provided in the alert packet.

    Notes
    -----
    `tracklet` is not available in real-time, therefore it is not set.


    Parameters
    ----------
    cdsxmatch: Pandas series
        Column containing the cross-match values
    roid: Pandas series
        Column containing the Solar System label
    mulens: Pandas series
        Probability of an event to be a microlensing event from LIA.
        The number is the mean of the per-band probabilities, and it is
        non-zero only for events favoured as microlensing by both bands.
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
    jd: Pandas series
        Column containing observation Julian dates at start of exposure [days]
    jdstarthist: Pandas series
        Column containing earliest Julian dates corresponding to ndethist
    rf_kn_vs_nonkn: Pandas series
        Column containing the probability to be a Kilonova

    Returns
    -------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    --------
    >>> pdf = pd.read_parquet('datatest/regular')
    >>> unknown_alerts_mask = unknowns_(
    ...     pdf['cdsxmatch'],
    ...     pdf['roid'],
    ...     pdf['mulens'],
    ...     pdf['snn_snia_vs_nonia'],
    ...     pdf['snn_sn_vs_all'],
    ...     pdf['rf_snia_vs_nonia'],
    ...     pdf['candidate'].apply(lambda x: x['ndethist']),
    ...     pdf['candidate'].apply(lambda x: x['drb']),
    ...     pdf['candidate'].apply(lambda x: x['classtar']),
    ...     pdf['candidate'].apply(lambda x: x['jd']),
    ...     pdf['candidate'].apply(lambda x: x['jdstarthist']),
    ...     pdf['rf_kn_vs_nonkn'])
    >>> print(unknown_alerts_mask.sum())
    13
    """
    classification = extract_fink_classification_(
        cdsxmatch,
        roid,
        mulens,
        snn_snia_vs_nonia,
        snn_sn_vs_all,
        rf_snia_vs_nonia,
        ndethist,
        drb,
        classtar,
        jd,
        jdstarthist,
        rf_kn_vs_nonkn,
        pd.Series([""] * len(cdsxmatch)),
    )

    is_unknown = classification == "Unknown"

    # TODO: are there any other criteria?

    return is_unknown


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def unknowns(
    cdsxmatch,
    roid,
    mulens,
    snn_snia_vs_nonia,
    snn_sn_vs_all,
    rf_snia_vs_nonia,
    ndethist,
    drb,
    classtar,
    jd,
    jdstarthist,
    rf_kn_vs_nonkn,
) -> pd.Series:
    """Return unclassified alerts in the stream

    Notes
    -----
    Apache Spark version of unknowns_


    Parameters
    ----------
    cdsxmatch: Pandas series
        Column containing the cross-match values
    roid: Pandas series
        Column containing the Solar System label
    mulens: Pandas series
        Probability of an event to be a microlensing event from LIA.
        The number is the mean of the per-band probabilities, and it is
        non-zero only for events favoured as microlensing by both bands.
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
    jd: Pandas series
        Column containing observation Julian dates at start of exposure [days]
    jdstarthist: Pandas series
        Column containing earliest Julian dates corresponding to ndethist
    rf_kn_vs_nonkn: Pandas series
        Column containing the probability to be a Kilonova

    Returns
    -------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    --------
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> import pyspark.sql.functions as F
    >>> df = spark.read.format('parquet').load('datatest/regular')

    >>> f = 'fink_filters.ztf.livestream.filter_unknowns.filter.unknowns'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    13
    """
    is_unknown = unknowns_(
        cdsxmatch,
        roid,
        mulens,
        snn_snia_vs_nonia,
        snn_sn_vs_all,
        rf_snia_vs_nonia,
        ndethist,
        drb,
        classtar,
        jd,
        jdstarthist,
        rf_kn_vs_nonkn,
    )

    return is_unknown


if __name__ == "__main__":
    """Execute the test suite"""

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
