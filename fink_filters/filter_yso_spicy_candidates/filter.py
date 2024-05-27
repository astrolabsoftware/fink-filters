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
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import BooleanType

from fink_utils.tg_bot.utils import get_curve, msg_handler_tg

from fink_filters.tester import spark_unit_tests

import pandas as pd
import os


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def yso_spicy_candidates(
    spicy_id, spicy_class, objectId, cjdc, cmagpsfc, csigmapsfc, cdiffmaglimc, cfidc
) -> pd.Series:
    """Return alerts with a match in the SPICY catalog

    Parameters
    ----------
    spicy_id: Spark DataFrame Column
        Column containing the ID of the SPICY catalog
        -1 if no match, otherwise > 0

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> from fink_utils.spark.utils import concat_col
    >>> df = spark.read.format('parquet').load('datatest/spicy_yso')

    >>> to_expand = ['jd', 'fid', 'magpsf', 'sigmapsf', 'diffmaglim']

    >>> prefix = 'c'
    >>> for colname in to_expand:
    ...    df = concat_col(df, colname, prefix=prefix)

    # quick fix for https://github.com/astrolabsoftware/fink-broker/issues/457
    >>> for colname in to_expand:
    ...    df = df.withColumnRenamed('c' + colname, 'c' + colname + 'c')

    >>> f = 'fink_filters.filter_yso_spicy_candidates.filter.yso_spicy_candidates'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    10
    """
    mask = spicy_id.values != -1

    pdf = pd.DataFrame(
        {
            "objectId": objectId,
            "magpsf": cmagpsfc,
            "sigmapsf": csigmapsfc,
            "diffmaglim": cdiffmaglimc,
            "fid": cfidc,
            "jd": cjdc,
            "spicy_id": spicy_id,
            "spicy_class": spicy_class,
        }
    )

    # Loop over matches
    if ("FINK_TG_TOKEN" in os.environ) and os.environ["FINK_TG_TOKEN"] != "":
        payloads = []
        for _, alert in pdf[mask].iterrows():
            curve_png = get_curve(
                jd=alert["jd"],
                magpsf=alert["magpsf"],
                sigmapsf=alert["sigmapsf"],
                diffmaglim=alert["diffmaglim"],
                fid=alert["fid"],
                objectId=alert["objectId"],
                origin="fields",
            )

            hyperlink = "[{}](https://fink-portal.org/{}): ID {} ({})".format(
                alert["objectId"],
                alert["objectId"],
                alert["spicy_id"],
                alert["spicy_class"],
            )
            payloads.append((hyperlink, None, curve_png))

        if len(payloads) > 0:
            msg_handler_tg(payloads, channel_id="@spicy_fink", init_msg="")

    return pd.Series(mask)


if __name__ == "__main__":
    """Execute the test suite"""

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
