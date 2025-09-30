# Copyright 2023 AstroLab Software
# Author: Etienne Russeil, Julien Peloton
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

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import BooleanType
from fink_filters.tester import spark_unit_tests
from fink_filters import __file__
import os
import pandas as pd


def slsn_filter_(slsn_probas) -> pd.Series:
    """Return a stream of objects classified as superluminous supernovae.

    Parameters
    ----------
    slsn_probas: Pandas series of floats
        Probability of being a slsn.

    Returns
    -------
    out: pandas.Series of bool
        Is classified as slsn
    """
    threshold = 0.5
    slsn_mask = slsn_probas.apply(lambda x: x >= threshold)

    return slsn_mask


@pandas_udf(BooleanType())
def slsn_filter(slsn_probas: pd.Series) -> pd.Series:
    """Pandas UDF version of slsn_filter_ for Spark

    Parameters
    ----------
    slsn_probas: Spark DataFrame Column of floats
        Probability of being a slsn.

    Returns
    -------
    out: pandas.Series of bool
        Meet the transient criteria.
    """
    f = slsn_filter_(slsn_probas)

    return f


if __name__ == "__main__":
    """Execute the test suite"""

    # Run the test suite
    globs = globals()
    path = os.path.dirname(__file__)
    spark_unit_tests(globs)
