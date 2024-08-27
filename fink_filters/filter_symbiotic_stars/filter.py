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
"""Crossmatch utilities for symbiotic and cataclysmic stars"""

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
def crossmatch_symbiotic(ra, dec):
    """Crossmatch ZTF alert with symbiotic and cataclysmic star catalogs

    Notes
    -----
    This is not a filter -- it only performs the crossmach.

    Parameters
    ----------
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
    --------)
    >>> import pyspark.sql.functions as F
    >>> df = spark.read.format('parquet').load('datatest/symbiotic')
    >>> args = ['candidate.ra', 'candidate.dec']
    >>> pdf = df\
    ...     .withColumn('symbiotic', crossmatch_symbiotic(*args))\
    ...     .filter(F.col('symbiotic') != 'Unknown')\
    ...     .select(['objectId', 'symbiotic'] + args)\
    ...     .toPandas()
    >>> assert len(pdf) == 20, len(pdf)
    """
    curdir = os.path.dirname(os.path.abspath(__file__))
    pdf_sym = pd.read_parquet(curdir + "/data/symbiotic_and_cataclysmic.parquet")

    pdf = pd.DataFrame(
        {
            "ra": ra.to_numpy(),
            "dec": dec.to_numpy(),
            "candid": range(len(ra))
        }
    )

    # create catalogs
    catalog_ztf = SkyCoord(
        ra=np.array(ra.to_numpy(), dtype=float) * u.degree,
        dec=np.array(dec.to_numpy(), dtype=float) * u.degree,
    )

    catalog_other = SkyCoord(
        ra=pdf_sym["RA(J2000)"].to_numpy(),
        dec=pdf_sym["DEC(J2000)"].to_numpy(),
        unit=(u.hourangle, u.deg)
    )

    pdf_merge, mask, idx2 = cross_match_astropy(
        pdf, catalog_ztf, catalog_other, radius_arcsec=pdf_sym["Radius"].astype(float)
    )

    pdf_sym_conc = np.array(["{},{}".format(i, j) for i, j in zip(pdf_sym["Name"], pdf_sym["source"])])
    pdf_merge['Type'] = 'Unknown'
    pdf_merge.loc[mask, 'Type'] = pdf_sym_conc[idx2]

    return pdf_merge['Type']


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
