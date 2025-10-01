# Copyright 2025 AstroLab Software
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


def slsn_filter_(slsn_score, threshold) -> pd.Series:
    """Return a stream of objects classified as superluminous supernovae.

    Parameters
    ----------
    slsn_score: Pandas series of floats
        Probability of being a slsn.
    threshold: Pandas series of floats
        Score value above which the the alerts are
        classified as SLSN.

    Returns
    -------
    out: pandas.Series of bool
        Is classified as slsn

    Examples
    --------
    >>> scores = pd.Series([0.1, -1.0, 0.0, 0.5, 1.0])
    >>> threshold = pd.Series([0.5] * len(scores))
    >>> list(slsn_filter_(scores, threshold).values)
    [False, False, False, True, True]
    """
    slsn_mask = slsn_score >= threshold

    return slsn_mask


@pandas_udf(BooleanType())
def slsn_filter(slsn_score: pd.Series, threshold: pd.Series) -> pd.Series:
    """Pandas UDF version of slsn_filter_ for Spark

    Parameters
    ----------
    slsn_score: Spark DataFrame Column of floats
        Probability of being a slsn.

    Returns
    -------
    out: pandas.Series of bool
        Is classified as slsn
    threshold: pandas.Series of floats
        Score value above which the the alerts are
        classified as SLSN.

    Examples
    --------
    >>> scores = pd.DataFrame(data = {'slsn_score':[0.1, -1.0, 0.0, 0.5, 1.0]})
    >>> scores['threshold'] = [0.5] * len(scores)
    >>> sdf = spark.createDataFrame(scores)
    >>> sdf = sdf.withColumn('is_slsn', slsn_filter('slsn_score', 'threshold'))
    >>> pdf = sdf.toPandas()
    >>> list(pdf['is_slsn'].values)
    [False, False, False, True, True]
    """
    f = slsn_filter_(slsn_score, threshold)
    return f


if __name__ == "__main__":
    """Execute the test suite"""
    # Run the test suite
    globs = globals()
    path = os.path.dirname(__file__)
    spark_unit_tests(globs)
