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

from fink_filters.tester import spark_unit_tests

import pandas as pd

def example_filter_(magpsf) -> pd.Series:
    """ Return alerts with difference magnitude above 18

    Parameters
    ----------
    magpsf: Pandas series
        Column containing difference magnitudes
        
    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.
    """
    
    mask_mag = magpsf <= 18
    
    return mask_mag


@pandas_udf(BooleanType())
def example_filter(magpsf: pd.Series) -> pd.Series:
    """ Pandas UDF version of example_filter_ for Spark

    Parameters
    ----------
    magpsf: Spark DataFrame Column
        Column containing the difference magnitudes

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.
    """
    survived = example_filter_(magpsf)

    return survived

