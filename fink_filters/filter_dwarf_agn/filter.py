# Copyright 2024 AstroLab Software
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
"""Crossmatch utilities for Dwarf AGN"""

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StringType

from fink_science.xmatch.utils import cross_match_astropy
from fink_filters import __file__

from astropy.coordinates import SkyCoord
from astropy import units as u

import os
import numpy as np
import pandas as pd

from fink_filters.tester import spark_unit_tests


@pandas_udf(StringType(), PandasUDFType.SCALAR)
def crossmatch_dwarf_agn(candid, ra, dec):
    """Crossmatch ZTF alert with dwarfs AGN

    Notes
    -----
    This is not a filter -- it only performs the crossmach.
    The reason we do not use normal xmatch method is that
    each source has its own crossmatching radius.

    Parameters
    ----------
    candid: long
        Candidate ID
    ra: float
        RA coordinates
    dec: float
        DEC coordinates

    Returns
    -------
    out: str
        Results of the xmatch: either `Unknown` or
        the name of the source in Manga.

    Examples
    --------
    >>> import pyspark.sql.functions as F
    >>> df = spark.read.format('parquet').load('datatest/dwarf_agn')
    >>> args = ['candidate.candid', 'candidate.ra', 'candidate.dec']
    >>> pdf = df.withColumn('manga', crossmatch_dwarf_agn(*args)).filter(F.col('manga') != 'Unknown').select(['objectId', 'manga'] + args).toPandas()
    >>> assert len(pdf) == 1, len(pdf)
    """
    curdir = os.path.dirname(os.path.abspath(__file__))
    pdf_lsb = pd.read_parquet(curdir + "/data/list_dwarfs_AGN_RADEC.parquet")

    pdf = pd.DataFrame(
        {
            "ra": ra.to_numpy(),
            "dec": dec.to_numpy(),
            "candid": candid.to_numpy(),
        }
    )

    # create catalogs
    catalog_ztf = SkyCoord(
        ra=np.array(ra.to_numpy(), dtype=float) * u.degree,
        dec=np.array(dec.to_numpy(), dtype=float) * u.degree,
    )

    out = np.array(["Unknown"] * len(pdf), dtype=object)
    for _, source in pdf_lsb.iterrows():
        catalog_other = SkyCoord(
            ra=np.array([source["RA"]], dtype=float) * u.degree,
            dec=np.array([source["DEC"]], dtype=float) * u.degree,
        )

        pdf_merge, mask, idx2 = cross_match_astropy(
            pdf, catalog_ztf, catalog_other, radius_arcsec=pd.Series([source["Re_arc"]])
        )

        out[mask] = source["MaNGAID"]

    return pd.Series(out)


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
