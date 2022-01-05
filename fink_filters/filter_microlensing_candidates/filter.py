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

def microlensing_candidates_(ndethist, mulens_class_1, mulens_class_2) -> pd.Series:
    """ Return alerts considered as microlensing candidates

    Parameters
    ----------
    ndethist: Pandas series
        Column containing the number of prior detections (theshold of 3 sigma)
    mulens_class_1: Pandas series
        Column containing the LIA results for band g
    mulens_class_2: Pandas series
        Column containing the LIA results for band r

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> pdf = pd.read_parquet('datatest')
    >>> classification = microlensing_candidates_(
    ...     pdf['candidate'].apply(lambda x: x['ndethist']),
    ...     pdf['mulens'].apply(lambda x: x['class_1']),
    ...     pdf['mulens'].apply(lambda x: x['class_2']))
    >>> print(pdf[classification]['objectId'].values)
    []
    """
    medium_ndethist = ndethist.astype(int) < 100
    f_mulens = (mulens_class_1 == 'ML') & (mulens_class_2 == 'ML') & medium_ndethist

    return f_mulens


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def microlensing_candidates(ndethist, mulens_class_1, mulens_class_2) -> pd.Series:
    """ Return alerts considered as microlensing candidates

    Parameters
    ----------
    ndethist: Spark DataFrame Column
        Column containing the number of prior detections (theshold of 3 sigma)
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
    f_mulens = microlensing_candidates_(
        ndethist, mulens_class_1, mulens_class_2
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
