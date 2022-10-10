#!/usr/bin/env python
# Copyright 2022 AstroLab Software
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
"""Derive alert classification from alert fields"""

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StringType

from fink_filters.filter_early_sn_candidates.filter import early_sn_candidates_
from fink_filters.filter_sn_candidates.filter import sn_candidates_
from fink_filters.filter_kn_candidates.filter import kn_candidates_
from fink_filters.filter_tracklet_candidates.filter import tracklet_candidates_
from fink_filters.filter_microlensing_candidates.filter import microlensing_candidates_
from fink_filters.filter_simbad_candidates.filter import simbad_candidates_
from fink_filters.filter_sso_ztf_candidates.filter import sso_ztf_candidates_
from fink_filters.filter_sso_fink_candidates.filter import sso_fink_candidates_

import numpy as np
import pandas as pd

import sys
import doctest

def extract_fink_classification_(
        cdsxmatch, roid, mulens,
        snn_snia_vs_nonia, snn_sn_vs_all, rf_snia_vs_nonia,
        ndethist, drb, classtar, jd, jdstarthist, rf_kn_vs_nonkn, tracklet) -> pd.Series:
    """ Extract the classification of an alert based on module outputs

    Rule of thumb:

    1. if an alert has not been flagged by any of the filters, it is tagged as `Unknown`
    2. if an alert has a counterpart in the SIMBAD database, its classification is the one from SIMBAD.
    3. if an alert has been flagged by one filter, its classification is given by the filter (`Early SN Ia candidate`, `KN candidate`, `SSO candidate`, etc.)
    4. if an alert has been flagged by more than one filter (except the SIMBAD one), it is tagged as `Ambiguous`.

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
    tracklet: Pandas series
        Column containing the tracklet label

    Returns
    ----------
    out: pandas.Series of string
        Return a Pandas series with the classification tag

    Examples
    ---------
    >>> pdf = pd.read_parquet('datatest')
    >>> classification = extract_fink_classification_(
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
    ...     pdf['rf_kn_vs_nonkn'],
    ...     pdf['tracklet'])
    >>> pdf['class'] = classification
    >>> pdf.groupby('class').count().sort_values('objectId', ascending=False)['objectId'].head(10)
    class
    Unknown                  14
    QSO                       8
    Blue                      7
    HotSubdwarf               6
    Symbiotic*                5
    Early SN Ia candidate     5
    CataclyV*                 5
    Candidate_YSO             5
    TTau*                     5
    Candidate_CV*             4
    Name: objectId, dtype: int64
    """
    classification = pd.Series(['Unknown'] * len(cdsxmatch))
    ambiguity = pd.Series([0] * len(cdsxmatch))

    # Tracklet ID
    f_tracklet = tracklet_candidates_(tracklet)

    # Simbad xmatch
    f_simbad = simbad_candidates_(cdsxmatch)

    # Early SN Ia
    f_sn_early = early_sn_candidates_(
        cdsxmatch,
        snn_snia_vs_nonia, snn_sn_vs_all, rf_snia_vs_nonia,
        ndethist, drb, classtar
    )

    # SN
    f_sn = sn_candidates_(
        cdsxmatch, snn_snia_vs_nonia, snn_sn_vs_all,
        drb, classtar, jd, jdstarthist, roid, ndethist
    )

    # Microlensing
    f_mulens = microlensing_candidates_(mulens)

    # Kilonova (ML)
    f_kn = kn_candidates_(
        rf_kn_vs_nonkn, rf_snia_vs_nonia, snn_snia_vs_nonia, snn_sn_vs_all, drb,
        classtar, jd, jdstarthist, ndethist, cdsxmatch
    )

    # SSO (MPC)
    f_roid_3 = sso_ztf_candidates_(roid)

    # SSO (candidates)
    f_roid_2 = sso_fink_candidates_(roid)

    classification.mask(f_mulens.values, 'Microlensing candidate', inplace=True)
    classification.mask(f_sn.values, 'SN candidate', inplace=True)
    classification.mask(f_sn_early.values, 'Early SN Ia candidate', inplace=True)
    classification.mask(f_kn.values, 'Kilonova candidate', inplace=True)
    classification.mask(f_roid_2.values, 'Solar System candidate', inplace=True)
    classification.mask(f_tracklet.values, 'Tracklet', inplace=True)
    classification.mask(f_roid_3.values, 'Solar System MPC', inplace=True)

    # If several flags are up, we cannot rely on the classification
    ambiguity[f_mulens.values] += 1
    ambiguity[f_sn.values] += 1
    ambiguity[f_roid_2.values] += 1
    ambiguity[f_roid_3.values] += 1
    f_ambiguity = ambiguity > 1
    classification.mask(f_ambiguity.values, 'Ambiguous', inplace=True)

    classification = np.where(f_simbad, cdsxmatch, classification)

    return pd.Series(classification)

def extract_fink_classification_from_pdf(pdf):
    """ Extract classification from a DataFrame made of alerts

    Parameters
    ----------
    pdf: Pandas DataFrame
        DataFrame containing alert values (with Fink schema)

    Returns
    ----------
    out: pandas.Series of string
        Return a Pandas series with the classification tag

    >>> pdf = pd.read_parquet('datatest')
    >>> classification = extract_fink_classification_from_pdf(pdf)
    >>> pdf['class'] = classification
    >>> pdf.groupby('class').count().sort_values('objectId', ascending=False)['objectId'].head(10)
    class
    Unknown                  14
    QSO                       8
    Blue                      7
    HotSubdwarf               6
    Symbiotic*                5
    Early SN Ia candidate     5
    CataclyV*                 5
    Candidate_YSO             5
    TTau*                     5
    Candidate_CV*             4
    Name: objectId, dtype: int64
    """
    classification = extract_fink_classification_(
        pdf['cdsxmatch'],
        pdf['roid'],
        pdf['mulens'],
        pdf['snn_snia_vs_nonia'],
        pdf['snn_sn_vs_all'],
        pdf['rf_snia_vs_nonia'],
        pdf['candidate'].apply(lambda x: x['ndethist']),
        pdf['candidate'].apply(lambda x: x['drb']),
        pdf['candidate'].apply(lambda x: x['classtar']),
        pdf['candidate'].apply(lambda x: x['jd']),
        pdf['candidate'].apply(lambda x: x['jdstarthist']),
        pdf['rf_kn_vs_nonkn'],
        pdf['tracklet']
    )

    return classification

@pandas_udf(StringType(), PandasUDFType.SCALAR)
def extract_fink_classification(
        cdsxmatch, roid, mulens,
        snn_snia_vs_nonia, snn_sn_vs_all, rf_snia_vs_nonia,
        ndethist, drb, classtar, jd, jdstarthist, rf_kn_vs_nonkn, tracklet) -> pd.Series:
    """ Pandas UDF version of extract_fink_classification_ for Apache Spark

    Parameters
    ----------
    cdsxmatch: Spark DataFrame Column
        Column containing the cross-match values
    roid: Spark DataFrame Column
        Column containing the Solar System label
    mulens: Spark DataFrame Column
        Probability of an event to be a microlensing event from LIA.
        The number is the mean of the per-band probabilities, and it is
        non-zero only for events favoured as microlensing by both bands.
    snn_snia_vs_nonia: Spark DataFrame Column
        Column containing the probability to be a SN Ia from SuperNNova.
    snn_sn_vs_all: Spark DataFrame Column
        Column containing the probability to be a SNe from SuperNNova.
    rf_snia_vs_nonia: Spark DataFrame Column
        Column containing the probability to be a SN Ia from RandomForestClassifier.
    ndethist: Spark DataFrame Column
        Column containing the number of detection by ZTF
    drb: Spark DataFrame Column
        Column containing the Deep-Learning Real Bogus score
    classtar: Spark DataFrame Column
        Column containing the sextractor score
    jd: Spark DataFrame Column
        Column containing observation Julian dates at start of exposure [days]
    jdstarthist: Spark DataFrame Column
        Column containing earliest Julian dates corresponding to ndethist
    rf_kn_vs_nonkn: Spark DataFrame Column
        Column containing the probability to be a Kilonova
    tracklet: Spark DataFrame Column
        Column containing the tracklet label

    Returns
    ----------
    out: pandas.Series of string
        Return a Pandas series with the classification tag

    See https://arxiv.org/abs/2009.10185 for more information
    """
    series = extract_fink_classification_(
        cdsxmatch, roid, mulens,
        snn_snia_vs_nonia, snn_sn_vs_all, rf_snia_vs_nonia,
        ndethist, drb, classtar, jd, jdstarthist, rf_kn_vs_nonkn, tracklet
    )

    return series


if __name__ == "__main__":
    """ Execute the test suite """
    # Numpy introduced non-backward compatible change from v1.14.
    if np.__version__ >= "1.14.0":
        np.set_printoptions(legacy="1.13")

    sys.exit(doctest.testmod()[0])
