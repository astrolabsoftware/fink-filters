# Copyright 2023 AstroLab Software
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

from astropy.coordinates import SkyCoord
from astropy import units as u

from fink_science.xmatch.utils import cross_match_astropy

from fink_filters.filter_anomaly_notification.filter_utils import msg_handler_slack
from fink_filters.tester import spark_unit_tests

import pandas as pd
import numpy as np
import os

def known_tde_(objectId, ra, dec, radius_arcsec=pd.Series([5])) -> pd.Series:
    """ Return alerts matching with known TDEs

    Parameters
    ----------
    objectId: Pandas series
        Colujmn containing ZTF object ID
    ra: Pandas series
        Column containing the RA values of alerts
    dec: Pandas series
        Column containing the DEC values of alerts
    radius_arcsec: series
        Radius for crossmatch, in arcsecond

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> pdf = pd.read_parquet('datatest_tde')
    >>> classification = known_tde_(
    ...     pdf['objectId'],
    ...     pdf['candidate'].apply(lambda x: x['ra']),
    ...     pdf['candidate'].apply(lambda x: x['dec']))
    >>> print(np.sum(classification))
    1

    """
    curdir = os.path.dirname(os.path.abspath(__file__))
    tdes = pd.read_parquet(curdir + '/data/tde.parquet')

    catalog_tde = SkyCoord(
        ra=np.array(tdes.ra, dtype=float) * u.degree,
        dec=np.array(tdes.dec, dtype=float) * u.degree
    )

    pdf = pd.DataFrame(
        {
            'ra': ra,
            'dec': dec,
            'candid': range(len(ra)),
            'objectId': objectId
        }
    )

    catalog_ztf = SkyCoord(
        ra=np.array(ra.values, dtype=float) * u.degree,
        dec=np.array(dec.values, dtype=float) * u.degree
    )

    pdf_merge, mask, idx2 = cross_match_astropy(
        pdf, catalog_ztf, catalog_tde, radius_arcsec=radius_arcsec
    )

    pdf_merge['match'] = False
    pdf_merge.loc[mask, 'match'] = True

    pdf_merge['intname'] = 'Unknown'
    pdf_merge.loc[mask, 'intname'] = [
        str(i).strip() for i in tdes['name'].astype(str).values[idx2]
    ]

    if ('ANOMALY_SLACK_TOKEN' in os.environ) and ('GITHUB_ENV' not in os.environ):
        # send to Slack recursively
        for _, row in pdf_merge[mask].iterrows():
            slack_data = []
            t1 = f'<https://fink-portal.org/{row.objectId}|{row.objectId}> associated with {row.intname}'
            slack_data.append(f'''{t1}''')

            msg_handler_slack(slack_data, "known_tde_follow_up", init_msg='New TDE association!')

    return pdf_merge['match']


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def known_tde(objectId, ra, dec) -> pd.Series:
    """ Pandas UDF for early_sn_candidates_

    Parameters
    ----------
    objectId: Pandas series
        Column containing ZTF object ID
    ra: Pandas series
        Column containing the RA values of alerts
    dec: Pandas series
        Column containing the DEC values of alerts

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> df = spark.read.format('parquet').load('datatest_tde')
    >>> f = 'fink_filters.filter_known_tde.filter.known_tde'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    1
    """
    series = known_tde_(objectId, ra, dec)
    return series


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
