# Copyright 2019-2024 AstroLab Software
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

from fink_utils.tg_bot.utils import get_curve
from fink_utils.tg_bot.utils import get_cutout
from fink_utils.tg_bot.utils import msg_handler_tg

from fink_filters.tester import spark_unit_tests

from astropy.coordinates import SkyCoord, get_constellation
import pandas as pd
import os


def extract_url_from_class(tns: str) -> str:
    """ Wikipedia link based on the TNS tag

    Parameters
    ----------
    tns: str
        TNS tag

    Returns
    -------
    out: str
        Wikipedia URL
    """
    if tns.startswith("SN Ia"):
        return "https://en.wikipedia.org/wiki/Type_Ia_supernova"
    elif tns.startswith("SN II"):
        return "https://en.wikipedia.org/wiki/Type_II_supernova"
    elif tns == "Impostor-SN":
        return "https://en.wikipedia.org/wiki/Supernova_impostor"
    elif tns.startswith("TDE"):
        return "https://en.wikipedia.org/wiki/Tidal_disruption_event"
    elif tns == "Varstar":
        return "https://en.wikipedia.org/wiki/Variable_star"
    elif tns.startswith("SN Ib"):
        return "https://en.wikipedia.org/wiki/Type_Ib_and_Ic_supernovae"
    elif tns.startswith("SN Ic"):
        return "https://en.wikipedia.org/wiki/Type_Ib_and_Ic_supernovae"
    elif tns == "Nova":
        return "https://en.wikipedia.org/wiki/Nova"
    elif tns == "Kilonova":
        return "https://en.wikipedia.org/wiki/Kilonova"
    elif tns == "LBV":
        return "https://en.wikipedia.org/wiki/Luminous_blue_variable"
    elif tns == "AGN":
        return "https://en.wikipedia.org/wiki/Active_galactic_nucleus"
    elif tns == "CV":
        return "https://en.wikipedia.org/wiki/Cataclysmic_variable_star"
    elif tns == "FRB":
        return "https://en.wikipedia.org/wiki/Fast_radio_burst"
    elif tns == "M dwarf":
        return "https://en.wikipedia.org/wiki/Red_dwarf"
    else:
        return "https://en.wikipedia.org/wiki/Time-domain_astronomy"


def tns_match_(
    tns,
    jd,
    jdstarthist,
) -> pd.Series:
    """Return alerts with a counterpart in TNS

    Parameters
    ----------
    tns: Pandas series
        Column containing the TNS cross-match values
    jd: Pandas series
        Column containing observation Julian dates at start of exposure [days]
    jdstarthist: Pandas series
        Column containing earliest Julian dates corresponding to ndethist

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> pdf = pd.read_parquet('datatest/regular')
    >>> fake_tns = ["SN Ia" for i in range(len(pdf))]
    >>> pdf["tns"] = fake_tns
    >>> classification = tns_match_(
    ...     pdf['tns'],
    ...     pdf['candidate'].apply(lambda x: x['jd']),
    ...     pdf['candidate'].apply(lambda x: x['jdstarthist']))
    >>> print(classification.sum())
    16
    """
    is_in_tns = tns != ""
    is_young = jd.astype(float) - jdstarthist.astype(float) <= 30

    return is_in_tns & is_young


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def tns_match(
    objectId,
    ra,
    dec,
    jd,
    jdstarthist,
    tns,
) -> pd.Series:
    """Pandas UDF for tns_match_

    Parameters
    ----------
    objectId: Pandas series
        Column with ZTF objectId
    ra: Pandas series
        Column with RA coordinate
    dec: Pandas series
        Column with Dec coordinate
    jd: Pandas series
        Column containing observation Julian dates at start of exposure [days]
    jdstarthist: Pandas series
        Column containing earliest Julian dates corresponding to ndethist
    tns: Pandas series
        Column containing the TNS cross-match values

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> import pyspark.sql.functions as F
    >>> df = spark.read.format('parquet').load('datatest/regular')
    >>> df = df.filter(df["candidate.jd"] - df["candidate.jdstarthist"] <= 30).limit(2)

    # Add a fake column
    >>> df = df.withColumn("tns", F.lit("SN Ia"))

    >>> f = 'fink_filters.filter_tns_match.filter.tns_match'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    2
    """
    series = tns_match_(tns, jd, jdstarthist)

    pdf = pd.DataFrame(
        {
            "objectId": objectId,
            "ra": ra,
            "dec": dec,
            "tns": tns,
            "dt": jd - jdstarthist,
        }
    )

    # Loop over matches
    if ("FINK_TG_TOKEN" in os.environ) and os.environ["FINK_TG_TOKEN"] != "":
        payloads = []
        for _, alert in pdf[series.values].iterrows():
            curve_png = get_curve(
                objectId=alert["objectId"],
                origin="API",
            )

            cutout = get_cutout(ztf_id=alert["objectId"], kind="Science", origin="API")

            constellation = get_constellation(SkyCoord(alert["ra"], alert["dec"], unit="deg"))
            text = """
🔭 Appeared {:.0f} days ago

*Object name*: {} (inspect it on the [portal](https://fink-portal.org/{}))
*Classification*: [{}]({})
*Constellation*: {}
            """.format(
                alert["dt"],
                alert["objectId"],
                alert["objectId"],
                alert["tns"].replace("SN", "Supernova"),
                extract_url_from_class(alert["tns"]),
                constellation
            )

            payloads.append((text, curve_png, cutout))

        if len(payloads) > 0:
            msg_handler_tg(payloads, channel_id="@fink_tns", init_msg="")
    return series


if __name__ == "__main__":
    """Execute the test suite"""

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
