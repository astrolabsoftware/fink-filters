# Copyright 2019-2023 AstroLab Software
# Author: Roman Le Montagner
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

from scipy import special
from math import sqrt

from fink_filters.tester import spark_unit_tests


def bronze_events(fink_class, realbogus_score):
    """
    Return alerts spatially and temporally consistent with a gcn alerts
    Keep alerts with real bogus score higher than 0.9
    and the alerts classified as "SN candidates", "Unknown", "Ambiguous"

    Parameters
    ----------
    fink_class : pd.Series
        Fink classification
    realbogus_score : pd.Series

    Return
    ------
    f_bronze : pd.Series
        alerts falling in the bronze filters

    Examples
    --------
    >>> df = pd.read_parquet(grb_output_data)
    >>> df["f_bronze"] = bronze_events(df["fink_class"], df["rb"])
    >>> len(df[df["f_bronze"]])
    25
    """
    f_bogus = realbogus_score >= 0.5

    f_class = fink_class.isin(["SN candidate", "Unknown", "Ambiguous"])

    f_fail = fink_class.str.startswith("Fail")

    f_bronze = f_bogus & (f_class | f_fail)
    return f_bronze


@pandas_udf(BooleanType())
def f_bronze_events(fink_class, realbogus_score):
    """
    see bronze_events documentation

    Examples
    --------
    >>> df = spark.read.format('parquet').load(grb_output_data)
    >>> df = df.withColumn("f_bronze", f_bronze_events(df["fink_class"], df["rb"])).filter("f_bronze == True").drop("f_bronze")
    >>> df.count()
    25
    """
    f_bronze = bronze_events(fink_class, realbogus_score)
    return f_bronze


def silver_events(fink_class, realbogus_score, grb_proba):
    """
    Return alerts spatially and temporally consistent with a gcn alerts
    Keep alerts with real bogus score higher than 0.9
    and the alerts classified as "SN candidates", "Unknown", "Ambiguous"
    and the alerts with a proba > 5 sigma

    Parameters
    ----------
    fink_class : pd.Series
        Fink classification
    realbogus_score : pd.Series
        real bogus score
    grb_proba : pd.Series
        serendipitous probilities to associates a ZTF alerts with a GCN,
        computed by the grb module

    Return
    ------
    f_silver : pd.Series
        alerts falling in the silver filters

    Examples
    --------
    >>> df = pd.read_parquet(grb_output_data)
    >>> df = df[df["grb_proba"] != 1.0]
    >>> df["f_silver"] = silver_events(df["fink_class"], df["rb"], df["grb_proba"])
    >>> len(df[df["f_silver"]])
    9
    """
    f_bronze = bronze_events(fink_class, realbogus_score)
    grb_ser_assoc = (1 - grb_proba) > special.erf(5 / sqrt(2))
    f_silver = f_bronze & grb_ser_assoc
    return f_silver


@pandas_udf(BooleanType())
def f_silver_events(fink_class, realbogus_score, grb_proba):
    """
    see silver_events documentation

    Examples
    --------
    >>> df = spark.read.format('parquet').load(grb_output_data)
    >>> df = df.withColumn("f_silver", f_silver_events(df["fink_class"], df["rb"], df["grb_proba"])).filter("f_silver == True").drop("f_silver")
    >>> df.count()
    9
    """
    f_silver = silver_events(fink_class, realbogus_score, grb_proba)
    return f_silver


def gold_events(fink_class, realbogus_score, grb_proba, rate):
    """
    Return alerts spatially and temporally consistent with a gcn alerts
    Keep alerts with real bogus score higher than 0.9
    and the alerts classified as "SN candidates", "Unknown", "Ambiguous"
    and the alerts with a proba > 5 sigma
    and a rate > 0.3 mag / day

    Parameters
    ----------
    fink_class : pd.Series
        Fink classification
    realbogus_score : pd.Series
        real bogus score
    grb_proba : pd.Series
        serendipitous probilities to associates a ZTF alerts with a GCN,
        computed by the grb module
    rate : pd.Series
        magnitude rate in mag/day computed by the grb module.

    Return
    ------
    f_gold : pd.Series
        alerts falling in the gold filters

    Examples
    --------
    >>> df = pd.read_parquet(grb_output_data)
    >>> df = df[df["grb_proba"] != 1.0]
    >>> df["f_gold"] = gold_events(df["fink_class"], df["rb"], df["grb_proba"], df["rate"])
    >>> len(df[df["f_gold"]])
    7
    """
    f_silver = silver_events(fink_class, realbogus_score, grb_proba)
    f_rate = rate.abs() > 0.3
    f_gold = f_silver & f_rate
    return f_gold


@pandas_udf(BooleanType())
def f_gold_events(fink_class, realbogus_score, grb_proba, rate):
    """
    see gold_events documentation

    Examples
    --------
    >>> df = spark.read.format('parquet').load(grb_output_data)
    >>> df = df.withColumn("f_gold", f_gold_events(df["fink_class"], df["rb"], df["grb_proba"], df["rate"])).filter("f_gold == True").drop("f_gold")
    >>> df.count()
    7
    """
    f_gold = gold_events(fink_class, realbogus_score, grb_proba, rate)
    return f_gold


if __name__ == "__main__":
    """Execute the test suite"""

    import pandas as pd  # noqa: F401

    # Run the test suite
    globs = globals()
    globs["grb_output_data"] = "datatest_grb/grb_join_output.parquet"
    spark_unit_tests(globs)
