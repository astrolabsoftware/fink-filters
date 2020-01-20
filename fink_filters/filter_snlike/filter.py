# Copyright 2019-2020 AstroLab Software
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
def snialike(rfscore, cdsxmatch, neargaia, distpsnr1) -> pd.Series:
    """ Return alerts considered as SN-Ia like

    Parameters
    ----------
    rfscore: Spark DataFrame Column
        Column containing the probability to be a SN Ia.
    cdsxmatch: Spark DataFrame Column
        Column containing the cross-match values
    neargaia: Spark DataFrame Column
        Column containing the distance to closest Gaia object
    distpsnr1: Spark DataFrame Column
        Column containing the distance to closest PS1 object

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    """
    mask = rfscore.values > 0.0
    mask *= cdsxmatch.values == "Unknown"
    mask *= neargaia.values > 5.0
    mask *= distpsnr1.values > 5.0

    return pd.Series(mask)
