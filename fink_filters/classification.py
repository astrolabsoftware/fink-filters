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
"""Derive Fink classification from alert fields"""
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
        cdsxmatch, roid, mulens_class_1, mulens_class_2,
        snn_snia_vs_nonia, snn_sn_vs_all, rfscore,
        ndethist, drb, classtar, jd, jdstarthist, knscore_, tracklet) -> pd.Series:
    """ Extract the classification of an alert based on module outputs

    See https://arxiv.org/abs/2009.10185 for more information

    Examples
    ---------
    >>> pdf = pd.read_parquet('datatest')
    >>> classification = extract_fink_classification_(
    ...     pdf['cdsxmatch'],
    ...     pdf['roid'],
    ...     pdf['mulens'].apply(lambda x: x['class_1']),
    ...     pdf['mulens'].apply(lambda x: x['class_2']),
    ...     pdf['snn_snia_vs_nonia'],
    ...     pdf['snn_sn_vs_all'],
    ...     pdf['rfscore'],
    ...     pdf['candidate'].apply(lambda x: x['ndethist']),
    ...     pdf['candidate'].apply(lambda x: x['drb']),
    ...     pdf['candidate'].apply(lambda x: x['classtar']),
    ...     pdf['candidate'].apply(lambda x: x['jd']),
    ...     pdf['candidate'].apply(lambda x: x['jdstarthist']),
    ...     pdf['knscore'],
    ...     pdf['tracklet'])
    >>> pdf['class'] = classification
    >>> pdf.groupby('class').count().sort_values('objectId', ascending=False)['objectId'].head(10)
    class
    Unknown               11
    QSO                    8
    Blue                   7
    HotSubdwarf            6
    Candidate_YSO          5
    TTau*                  5
    CataclyV*              5
    Symbiotic*             5
    Early SN candidate     5
    V*                     4
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
        snn_snia_vs_nonia, snn_sn_vs_all, rfscore,
        ndethist, drb, classtar
    )

    # SN
    f_sn = sn_candidates_(
        cdsxmatch, snn_snia_vs_nonia, snn_sn_vs_all,
        drb, classtar, jd, jdstarthist, roid, ndethist
    )

    # Microlensing
    f_mulens = microlensing_candidates_(ndethist, mulens_class_1, mulens_class_2)

    # Kilonova (ML)
    f_kn = kn_candidates_(
        knscore_, rfscore, snn_snia_vs_nonia, snn_sn_vs_all, drb,
        classtar, jd, jdstarthist, ndethist, cdsxmatch
    )

    # SSO (MPC)
    f_roid_3 = sso_ztf_candidates_(roid)

    # SSO (candidates)
    f_roid_2 = sso_fink_candidates_(roid)

    classification.mask(f_mulens.values, 'Microlensing candidate', inplace=True)
    classification.mask(f_sn.values, 'SN candidate', inplace=True)
    classification.mask(f_sn_early.values, 'Early SN candidate', inplace=True)
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
    """ Extract lazily classification from a DataFrame made of alerts

    >>> pdf = pd.read_parquet('datatest')
    >>> classification = extract_fink_classification_from_pdf(pdf)
    >>> pdf['class'] = classification
    >>> pdf.groupby('class').count().sort_values('objectId', ascending=False)['objectId'].head(10)
    class
    Unknown               11
    QSO                    8
    Blue                   7
    HotSubdwarf            6
    Candidate_YSO          5
    TTau*                  5
    CataclyV*              5
    Symbiotic*             5
    Early SN candidate     5
    V*                     4
    Name: objectId, dtype: int64
    """
    classification = extract_fink_classification_(
        pdf['cdsxmatch'],
        pdf['roid'],
        pdf['mulens'].apply(lambda x: x['class_1']),
        pdf['mulens'].apply(lambda x: x['class_2']),
        pdf['snn_snia_vs_nonia'],
        pdf['snn_sn_vs_all'],
        pdf['rfscore'],
        pdf['candidate'].apply(lambda x: x['ndethist']),
        pdf['candidate'].apply(lambda x: x['drb']),
        pdf['candidate'].apply(lambda x: x['classtar']),
        pdf['candidate'].apply(lambda x: x['jd']),
        pdf['candidate'].apply(lambda x: x['jdstarthist']),
        pdf['knscore'],
        pdf['tracklet']
    )

    return classification

@pandas_udf(StringType(), PandasUDFType.SCALAR)
def extract_fink_classification(
        cdsxmatch, roid, mulens_class_1, mulens_class_2,
        snn_snia_vs_nonia, snn_sn_vs_all, rfscore,
        ndethist, drb, classtar, jd, jdstarthist, knscore_, tracklet) -> pd.Series:
    """ Extract the classification of an alert based on module outputs

    For Spark usage

    See https://arxiv.org/abs/2009.10185 for more information
    """
    series = extract_fink_classification_(
        cdsxmatch, roid, mulens_class_1, mulens_class_2,
        snn_snia_vs_nonia, snn_sn_vs_all, rfscore,
        ndethist, drb, classtar, jd, jdstarthist, knscore_, tracklet
    )

    return series


if __name__ == "__main__":
    """ Execute the test suite """
    # Numpy introduced non-backward compatible change from v1.14.
    if np.__version__ >= "1.14.0":
        np.set_printoptions(legacy="1.13")

    sys.exit(doctest.testmod()[0])
