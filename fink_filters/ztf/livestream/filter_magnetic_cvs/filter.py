# Copyright 2025 AstroLab Software
# Author: Emille Ishida
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
"""Return alerts with a match in a magnetic cataclysmic variables catalog"""

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StringType

from fink_science.ztf.xmatch.utils import cross_match_astropy
from fink_filters import __file__

from fink_utils.tg_bot.utils import get_curve
from fink_utils.tg_bot.utils import msg_handler_tg

from astropy.coordinates import SkyCoord
from astropy import units as u

import os
import numpy as np
import pandas as pd

from fink_filters.tester import spark_unit_tests


def send_to_telegram(pdf, channel):
    """ """
    # Loop over matches
    if ("FINK_TG_TOKEN" in os.environ) and os.environ["FINK_TG_TOKEN"] != "":
        payloads = []
        for _, alert in pdf.iterrows():
            curve_png = get_curve(
                objectId=alert["objectId"],
                origin="API",
            )

            text = """
*Object ID*: [{}](https://fink-portal.org/{})
*Name*: {}
*RA/Dec coordinates*: {} {}
            """.format(
                alert["objectId"],
                alert["objectId"],
                alert["name"],
                alert["ra"],
                alert["dec"],
            )

            payloads.append((text, None, curve_png))

        if len(payloads) > 0:
            # Send to tg
            msg_handler_tg(payloads, channel_id=channel, init_msg="")
    else:
        print("Telegram token FINK_TG_TOKEN is not set")


def magnetic_cvs_(ra, dec):
    """Return labels for alerts matching the magnetic CVs catalog.

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
    --------
    >>> pdf = pd.read_parquet('datatest/magnetic_cvs/')
    >>> classification = magnetic_cvs_(
    ...     pdf['candidate'].apply(lambda x: x['ra']),
    ...     pdf['candidate'].apply(lambda x: x['dec']))
    >>> print(np.sum([i != "Unknown" for i in classification]))
    10
    """
    curdir = os.path.dirname(os.path.abspath(__file__))
    pdf_mcvs = pd.read_csv(curdir + "/data/magnetic_cataclysmic_variables.csv")

    pdf = pd.DataFrame({
        "ra": ra.to_numpy(),
        "dec": dec.to_numpy(),
        "candid": range(len(ra)),
    })

    # create catalogs
    catalog_ztf = SkyCoord(
        ra=np.array(ra.to_numpy(), dtype=float) * u.degree,
        dec=np.array(dec.to_numpy(), dtype=float) * u.degree,
    )

    catalog_other = SkyCoord(
        ra=pdf_mcvs["RA(J2000)"].to_numpy(),
        dec=pdf_mcvs["DEC(J2000)"].to_numpy(),
        unit=(u.hourangle, u.deg),
    )

    pdf_merge, mask, idx2 = cross_match_astropy(
        pdf, catalog_ztf, catalog_other, radius_arcsec=pdf_mcvs["Radius"].astype(float)
    )

    pdf_merge["intname"] = "Unknown"
    pdf_merge.loc[mask, "intname"] = [
        str(i).strip() for i in pdf_mcvs["Name"].astype(str).to_numpy()[idx2]
    ]

    return pdf_merge["intname"]


@pandas_udf(StringType(), PandasUDFType.SCALAR)
def magnetic_cvs(objectId, isdiffpos, ra, dec) -> pd.Series:
    """Pandas UDF for magnetic_cvs_

    Parameters
    ----------
    objectId: Pandas series of str
        Column containing ZTF objectId
    isdiffpos: Pandas series of str
        Column containing positiveness flag
    ra: Pandas series of float
        Column containing the RA values of alerts
    dec: Pandas series of float
        Column containing the DEC values of alerts

    Returns
    -------
    out: pandas.Series of str
        Return a Pandas DataFrame with the appropriate label:
        Unknown if no match, the name of the magnetic CV otherwise.

    Examples
    --------
    >>> df = spark.read.format('parquet').load('datatest/magnetic_cvs/')
    >>> df = df.withColumn("mcvs", magnetic_cvs("objectId", "candidate.isdiffpos", "candidate.ra", "candidate.dec"))
    >>> print(df.filter(df["mcvs"] != "Unknown").count())
    10
    """
    # Keep only positive alerts
    valid = isdiffpos.apply(lambda x: (x == "t") or (x == "1"))

    if len(ra[valid]) == 0:
        return pd.Series(["Unknown"] * len(ra))

    # perform crossmatch
    out = magnetic_cvs_(ra[valid], dec[valid])

    # Default values are Unknown
    names = pd.Series(["Unknown"] * len(ra))
    names[valid] = out.to_numpy()

    mask = names != "Unknown"
    if len(names[mask]) > 0:
        pdf = pd.DataFrame({
            "objectId": objectId[mask],
            "ra": ra[mask],
            "dec": dec[mask],
            "name": names[mask],
        })
        send_to_telegram(pdf, channel="@fink_magnetic_cv_stars")

    return names


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
