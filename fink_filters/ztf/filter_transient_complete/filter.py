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


def transient_complete_filter_(
    faint,
    positivesubtraction,
    real,
    pointunderneath,
    brightstar,
    variablesource,
    stationary,
    roid,
) -> pd.Series:
    """Return a relatively complete stream of transient alerts.

    Notes
    -----
    Should keep good quality transients,
    and remove a significant part of the contamination.

    Parameters
    ----------
    faint: Pandas series of bool
        Is currently fainter than 19.8,
        or the source had a very recent detection fainter than 19.
    positivesubtraction: Pandas series
        Is brighter than the template image.
    real: Pandas series of bool
        Is likely a genuine astrophysical transient and not an artifact.
    pointunderneath: Pandas series of bool
        Is likely sitting on top of or blended with a star in Pan-STARRS.
    brightstar: Pandas series of bool
        Is likely contaminated by a nearby bright star.
    variablesource: Pandas series of bool
        Is likely a variable star
    stationary: Pandas series of bool
        Is not a moving source.
    roid: Pandas series
        Is an asteroid.

    Returns
    -------
    out: pandas.Series of bool
        Meet the transient criteria

    Examples
    --------
    >>> from pyspark.sql import SparkSession
    >>> from pyspark.sql.functions import col
    >>> from fink_science.ztf.transient_features.processor import extract_transient_features
    >>> sdf = spark.read.load(ztf_alert_sample)
    >>> sdf = extract_transient_features(sdf)
    >>> pdf = sdf.toPandas()
    >>> is_transient = transient_complete_filter_(pdf["faint"],
    ...     pdf["positivesubtraction"], pdf["real"], pdf["pointunderneath"],
    ...     pdf["brightstar"], pdf["variablesource"], pdf["stationary"],
    ...     pdf["roid"])
    >>> sum(is_transient)
    29
    """
    roid_mask = roid.apply(lambda x: x == 0)

    # Remove alerts that are too faint
    faint_mask = faint.apply(lambda x: x is False)

    # Remove alerts with flux below reference image
    positivesubtraction_mask = positivesubtraction.apply(lambda x: x is True)

    # Remove alerts that are likely artifacts
    real_mask = real.apply(lambda x: x is True)

    # Remove alerts that are likely blended with a star
    pointunderneath_mask = pointunderneath.apply(lambda x: x is False)

    # Remove alerts that are likely too close to a bright star
    brightstar_mask = brightstar.apply(lambda x: x is False)

    # Remove probable variable sources
    variablesource_mask = variablesource.apply(lambda x: x is False)

    # Remove non-stationnary alerts
    stationary_mask = stationary.apply(lambda x: x is True)

    # Remove asteroid alerts
    roid_mask = roid.apply(lambda x: x == 0)

    final_mask = (
        faint_mask
        & positivesubtraction_mask
        & real_mask
        & pointunderneath_mask
        & brightstar_mask
        & variablesource_mask
        & stationary_mask
        & roid_mask
    )

    return final_mask


@pandas_udf(BooleanType())
def transient_complete_filter(
    faint: pd.Series,
    positivesubtraction: pd.Series,
    real: pd.Series,
    pointunderneath: pd.Series,
    brightstar: pd.Series,
    variablesource: pd.Series,
    stationary: pd.Series,
    roid: pd.Series,
) -> pd.Series:
    """Pandas UDF version of transient_complete_filter_ for Spark

    Parameters
    ----------
    faint: Spark DataFrame Column of bool
        Is currently fainter than 19.8, or the source had
        a very recent detection fainter than 19.
    positivesubtraction: Spark DataFrame Column of bool
        Is brighter than the template image.
    real: Spark DataFrame Column of bool
        Is likely a genuine astrophysical transient and not an artifact.
    pointunderneath: Spark DataFrame Column of bool
        Is likely sitting on top of or blended with a star in Pan-STARRS.
    brightstar: Spark DataFrame Column of bool
        Is likely contaminated by a nearby bright star.
    variablesource: Spark DataFrame Column of bool
        Is likely a variable star
    stationary: Spark DataFrame Column of bool
        Is not a moving source.
    roid: Spark DataFrame Column
        Is an asteroid.

    Returns
    -------
    out: pandas.Series of bool
        Meet the transient criteria.

    Examples
    --------
    >>> from pyspark.sql import SparkSession
    >>> from pyspark.sql.functions import col
    >>> from fink_science.ztf.transient_features.processor import extract_transient_features
    >>> sdf = spark.read.load(ztf_alert_sample)
    >>> sdf = extract_transient_features(sdf)
    >>> sdf = sdf.withColumn(
    ...     "is_transient",
    ...     transient_complete_filter(
    ...     "faint", "positivesubtraction", "real", "pointunderneath",
    ...     "brightstar", "variablesource", "stationary", "roid"))
    >>> pdf = sdf.filter(sdf["is_transient"]).toPandas()
    >>> len(pdf)
    29
    """
    f = transient_complete_filter_(
        faint,
        positivesubtraction,
        real,
        pointunderneath,
        brightstar,
        variablesource,
        stationary,
        roid,
    )

    return f


if __name__ == "__main__":
    """Execute the test suite"""

    # Run the test suite
    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = "file://{}/data/few_SLSN_alerts.parquet".format(path)

    globs["ztf_alert_sample"] = ztf_alert_sample
    spark_unit_tests(globs)
