# Copyright 2023-2024 AstroLab Software
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
from pyspark.sql.types import StringType

from astropy.coordinates import SkyCoord
from astropy import units as u

from fink_science.xmatch.utils import cross_match_astropy
from fink_filters import __file__

from fink_filters.tester import spark_unit_tests

import pandas as pd
import numpy as np
import os

def known_tde_(ra, dec, radius_arcsec=pd.Series([5])) -> pd.Series:
    """ Return labels for alerts matching with known TDEs

    Parameters
    ----------
    ra: Pandas series
        Column containing the RA values of alerts
    dec: Pandas series
        Column containing the DEC values of alerts
    radius_arcsec: series
        Radius for crossmatch, in arcsecond

    Returns
    ----------
    out: pandas.Series of str
        Return a Pandas DataFrame with the appropriate label:
        Unknown if no match, the name of the TDE otherwise.

    Examples
    ----------
    >>> pdf = pd.read_parquet('datatest/tde')
    >>> classification = known_tde_(
    ...     pdf['candidate'].apply(lambda x: x['ra']),
    ...     pdf['candidate'].apply(lambda x: x['dec']))
    >>> print(np.sum([i != "Unknown" for i in classification]))
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
        }
    )

    catalog_ztf = SkyCoord(
        ra=np.array(ra.values, dtype=float) * u.degree,
        dec=np.array(dec.values, dtype=float) * u.degree
    )

    pdf_merge, mask, idx2 = cross_match_astropy(
        pdf, catalog_ztf, catalog_tde, radius_arcsec=radius_arcsec
    )

    pdf_merge['intname'] = 'Unknown'
    pdf_merge.loc[mask, 'intname'] = [
        str(i).strip() for i in tdes['name'].astype(str).values[idx2]
    ]

    return pdf_merge['intname']


@pandas_udf(StringType(), PandasUDFType.SCALAR)
def known_tde(isdiffpos, ra, dec) -> pd.Series:
    """ Pandas UDF for known_tde_

    Parameters
    ----------
    isdiffpos: Pandas series of str
        Column containing positiveness flag
    ra: Pandas series of float
        Column containing the RA values of alerts
    dec: Pandas series of float
        Column containing the DEC values of alerts

    Returns
    ----------
    out: pandas.Series of str
        Return a Pandas DataFrame with the appropriate label:
        Unknown if no match, the name of the TDE otherwise.

    Examples
    ----------
    >>> df = spark.read.format('parquet').load('datatest/tde')
    >>> df = df.withColumn("tde", known_tde("candidate.isdiffpos", "candidate.ra", "candidate.dec"))
    >>> print(df.filter(df["tde"] != "Unknown").count())
    1
    """
    # Keep only positive alerts
    valid = isdiffpos.apply(lambda x: (x == 't') or (x == '1'))

    # perform crossmatch
    series = known_tde_(ra[valid], dec[valid])

    # Default values are Unknown
    to_return = pd.Series(["Unknown"] * len(ra))
    to_return[valid] = series.to_numpy()

    return to_return


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
