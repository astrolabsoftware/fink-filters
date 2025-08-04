# Copyright 2024-2025 AstroLab Software
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
"""Return alerts with a match in the SPICY catalog"""

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import BooleanType

from fink_utils.tg_bot.utils import get_curve, msg_handler_tg

from fink_filters.tester import spark_unit_tests

import numpy as np
import pandas as pd
import os

def r2_score(x, y):
    """
    Compute the coefficient of determination (R²) for two numeric arrays.

    Parameters
    ----------
    x : array-like
        Independent variable (predictor values).
    y : array-like
        Dependent variable (observed values).

    Returns
    -------
    float
        R² value.
    """
    
    x_mean = x.mean()
    y_mean = y.mean()

    sxx = np.sum((x - x_mean) ** 2)
    if sxx == 0:
        return np.nan

    sxy = np.sum((x - x_mean) * (y - y_mean))
    beta1 = sxy / sxx
    beta0 = y_mean - beta1 * x_mean

    y_hat = beta0 + beta1 * x
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)

    res = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
    
    return res


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def yso_spicy_candidates(
    spicy_id, spicy_class, objectId, cjdc, cmagpsfc, csigmapsfc, cdiffmaglimc, cfidc, linear_fit_slope
) -> pd.Series:
    """Return alerts with a match in the SPICY catalog

    Parameters
    ----------
    spicy_id: Spark DataFrame Column
        Column containing the ID of the SPICY catalog
        -1 if no match, otherwise > 0

    Returns
    -------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    --------
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

    >>> f = 'fink_filters.ztf.livestream.filter_yso_spicy_candidates.filter.yso_spicy_candidates'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    6
    """
    
    slope_lim = 0.025   # minimum slope threshold
    npoints = 5         # minimum required number of points
    r2_lim = 0.6        # minimum required r2
    
    # select spicy objecs
    mask_spicy = (spicy_class != "Unknown")

    # select spicy objects which respect the slope threshold
    mask_slope = mask_spicy & (linear_fit_slope.abs() > slope_lim)
    
    # convert to pandas
    pdf = pd.DataFrame({"cmagpsfc": cmagpsfc[mask_slope], 
                        "cfidc": cfidc[mask_slope], 
                        "cjdc": cjdc[mask_slope]})
                        
    pdf['sum_rband'] = pd.Series([sum(item == 2) for item in pdf['cfidc'].values])

    # select objects with minimum number of points
    mask_points = mask_slope & (pdf['sum_rband'] >= npoints)
    
    use_jd = pdf['cjdc'][mask_points]
    use_magpsf = pdf['cmagpsfc'][mask_points]

    # calculate r2 statistics
    r2_values = r2_score(use_jd, use_magpsf)
    
    mask = r2_values > r2_lim 

    pdf = pd.DataFrame({
        "objectId": objectId,
        "magpsf": cmagpsfc,
        "sigmapsf": csigmapsfc,
        "diffmaglim": cdiffmaglimc,
        "fid": cfidc,
        "jd": cjdc,
        "spicy_id": spicy_id,
        "spicy_class": spicy_class,
    })

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

    return mask


if __name__ == "__main__":
    """Execute the test suite"""

    # Run the test suite
    globs = globals()
    path = os.path.dirname(__file__)
    sample_file = (
        "./fink-filters/datatest/spicy_yso/"
    )
    globs["test_yso_cuts"] = sample_file
    spark_unit_tests(globs)
