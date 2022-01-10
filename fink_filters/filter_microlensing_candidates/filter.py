# Copyright 2019-2022 AstroLab Software
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

def microlensing_candidates_(mulens) -> pd.Series:
    """ Return alerts considered as microlensing candidates

    Parameters
    ----------
    mulens: Pandas series
        Probability of an event to be a microlensing event from LIA.
        The number is the mean of the per-band probabilities, and it is
        non-zero only for events favoured as microlensing by both bands.

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> pdf = pd.read_parquet('datatest')
    >>> classification = microlensing_candidates_(pdf['mulens'])
    >>> print(pdf[classification]['objectId'].values)
    []
    """
    f_mulens = mulens > 0.0

    return f_mulens


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def microlensing_candidates(mulens) -> pd.Series:
    """ Return alerts considered as microlensing candidates

    Parameters
    ----------
    mulens: Spark DataFrame Column
        Probability of an event to be a microlensing event from LIA.
        The number is the mean of the per-band probabilities, and it is
        non-zero only for events favoured as microlensing by both bands.

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    """
    f_mulens = microlensing_candidates_(
        mulens
    )

    return f_mulens


if __name__ == "__main__":
    """ Execute the test suite """
    import sys
    import doctest
    import numpy as np

    # Numpy introduced non-backward compatible change from v1.14.
    if np.__version__ >= "1.14.0":
        np.set_printoptions(legacy="1.13")

    sys.exit(doctest.testmod()[0])
