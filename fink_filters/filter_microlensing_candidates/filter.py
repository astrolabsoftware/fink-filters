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
def microlensing_candidates(mulens_class_1, mulens_class_2) -> pd.Series:
    """ Return alerts considered as microlensing candidates

    Parameters
    ----------
    mulens_class_1: Spark DataFrame Column
        Column containing the LIA results for band g
    mulens_class_2: Spark DataFrame Column
        Column containing the LIA results for band r

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    """
    f_mulens = (mulens_class_1 == 'ML') & (mulens_class_2 == 'ML')

    return f_mulens
