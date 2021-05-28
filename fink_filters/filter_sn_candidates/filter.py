# Copyright 2019-2021 AstroLab Software
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

import pandas as pd

@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def sn_candidates(cdsxmatch, snn_snia_vs_nonia, snn_sn_vs_all,
        drb, classtar, jd, jdstarthist, roid, ndethist) -> pd.Series:
    """ Return alerts considered as SN-Ia candidates

    Parameters
    ----------
    cdsxmatch: Spark DataFrame Column
        Column containing the cross-match values
    snn_snia_vs_nonia: Spark DataFrame Column
        Column containing the probability to be a SN Ia from SuperNNova.
    snn_sn_vs_all: Spark DataFrame Column
        Column containing the probability to be a SNe from SuperNNova.
    drb: Spark DataFrame Column
        Column containing the Deep-Learning Real Bogus score
    classtar: Spark DataFrame Column
        Column containing the sextractor score
    jd: Spark DataFrame Column
        Column containing the JD of the _alert_
    jdstarthist: Spark DataFrame Column
        Column containing the starting JD of the _object_
    roid: Spark DataFrame Column
        Column containing the SSO label
    ndethist: Spark DataFrame Column
        Column containing the number of detection at 3 sigma since the
        beginning of the survey

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    """
    snn1 = snn_snia_vs_nonia.astype(float) > 0.5
    snn2 = snn_sn_vs_all.astype(float) > 0.5
    sn_history = jd.astype(float) - jdstarthist.astype(float) <= 90
    high_drb = drb.astype(float) > 0.5
    high_classtar = classtar.astype(float) > 0.4
    no_mpc = roid.astype(int) != 3
    no_first_det = ndethist.astype(int) > 1

    list_simbad_galaxies = [
        "galaxy",
        "Galaxy",
        "EmG",
        "Seyfert",
        "Seyfert_1",
        "Seyfert_2",
        "BlueCompG",
        "StarburstG",
        "LSB_G",
        "HII_G",
        "High_z_G",
        "GinPair",
        "GinGroup",
        "BClG",
        "GinCl",
        "PartofG",
    ]
    keep_cds = \
        ["Unknown", "Candidate_SN*", "SN", "Transient", "Fail"] + list_simbad_galaxies

    f_sn = (snn1 | snn2) & cdsxmatch.isin(keep_cds) & sn_history & high_drb & high_classtar & no_first_det & no_mpc

    return f_sn
