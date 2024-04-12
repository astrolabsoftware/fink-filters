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
from fink_utils.xmatch.simbad import return_list_of_eg_host

# ------ GRB filters ------
GRB_OBSERVATORY = ["Fermi", "SWIFT", "INTEGRAL"]


def generic_bronze_filter(fink_class, observatory, rb, obs_filter):
    """
    Generic bronze filter

    Parameters
    ----------
    fink_class : pd.Series
        Fink classification
    observatory : pd.Series
        GCN observatory emitter
    rb : pd.Series
        real bogus column

    Returns
    -------
    boolean series
        if True, is a bronze event
    """
    f_obs = observatory.isin(obs_filter)  # select only the GRB observatories

    f_bogus = rb >= 0.7

    base_extragalactic = return_list_of_eg_host()  # include Unknown and Fail as well
    fink_extragalactic = ["Kilonova candidate", "SN candidate", "Early SN Ia candidate", "Ambiguous"]
    extragalactic = list(base_extragalactic) + list(fink_extragalactic)
    f_class = fink_class.isin(extragalactic)

    f_bronze = f_bogus & f_obs & f_class
    return f_bronze


def grb_bronze_events(fink_class, observatory, rb):
    """
    Return alerts spatially and temporally consistent with a gcn alerts
    Keep alerts with real bogus score higher than 0.9
    and the alerts classified as "SN candidates", "Unknown", "Ambiguous"

    Parameters
    ----------
    fink_class : pd.Series
        Fink classification
    observatory : pd.Series
        GCN observatory emitter
    rb : pd.Series

    Return
    ------
    f_bronze : pd.Series
        alerts falling in the bronze filters

    Examples
    --------
    >>> df = pd.read_parquet(grb_output_data)
    >>> df["f_bronze"] = grb_bronze_events(df["fink_class"], df["observatory"], df["rb"])
    >>> len(df[df["f_bronze"]])
    4
    """
    return generic_bronze_filter(fink_class, observatory, rb, GRB_OBSERVATORY)


@pandas_udf(BooleanType())
def f_grb_bronze_events(fink_class, observatory, rb):
    """
    see bronze_events documentation

    Examples
    --------
    >>> df = spark.read.format('parquet').load(grb_output_data)
    >>> df = df.withColumn("f_bronze", f_grb_bronze_events(df["fink_class"], df["observatory"], df["rb"])).filter("f_bronze == True").drop("f_bronze")
    >>> df.count()
    4
    """
    f_bronze = grb_bronze_events(fink_class, observatory, rb)
    return f_bronze


def grb_silver_events(fink_class, observatory, rb, grb_proba):
    """
    Return alerts spatially and temporally consistent with a gcn alerts
    Keep alerts with real bogus score higher than 0.9
    and the alerts classified as "SN candidates", "Unknown", "Ambiguous"
    and the alerts with a proba > 5 sigma

    Parameters
    ----------
    fink_class : pd.Series
        Fink classification
    rb : pd.Series
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
    >>> df["f_silver"] = grb_silver_events(df["fink_class"], df["observatory"], df["rb"], df["grb_proba"])
    >>> len(df[df["f_silver"]])
    2
    """
    f_bronze = grb_bronze_events(fink_class, observatory, rb)
    grb_ser_assoc = (1 - grb_proba) > special.erf(5 / sqrt(2))
    f_silver = f_bronze & grb_ser_assoc
    return f_silver


@pandas_udf(BooleanType())
def f_grb_silver_events(fink_class, observatory, rb, grb_proba):
    """
    see silver_events documentation

    Examples
    --------
    >>> df = spark.read.format('parquet').load(grb_output_data)
    >>> df = df.withColumn("f_silver", f_grb_silver_events(df["fink_class"], df["observatory"], df["rb"], df["grb_proba"])).filter("f_silver == True").drop("f_silver")
    >>> df.count()
    2
    """
    f_silver = grb_silver_events(fink_class, observatory, rb, grb_proba)
    return f_silver


def grb_gold_events(fink_class, observatory, rb, grb_loc_error, grb_proba, rate):
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
    rb : pd.Series
        real bogus score
    grb_loc_error : pd.Series
        the sky localization error of the grb events
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
    >>> df["f_gold"] = grb_gold_events(df["fink_class"], df["observatory"], df["rb"], df["grb_loc_error"], df["grb_proba"], df["rate"])
    >>> len(df[df["f_gold"]])
    1
    """
    f_silver = grb_silver_events(fink_class, observatory, rb, grb_proba)
    f_bogus = rb >= 0.9
    f_sky_loc = (grb_loc_error / 60) <= 5  # grb_loc_error is given in arcminute
    f_rate = rate.abs() > 0.3
    f_gold = f_silver & f_rate & f_bogus & f_sky_loc
    return f_gold


@pandas_udf(BooleanType())
def f_grb_gold_events(fink_class, observatory, rb, grb_loc_error, grb_proba, rate):
    """
    see gold_events documentation

    Examples
    --------
    >>> df = spark.read.format('parquet').load(grb_output_data)
    >>> df = df.withColumn("f_gold", f_grb_gold_events(df["fink_class"], df["observatory"], df["rb"], df["grb_loc_error"], df["grb_proba"], df["rate"])).filter("f_gold == True").drop("f_gold")
    >>> df.count()
    1
    """
    f_gold = grb_gold_events(fink_class, observatory, rb, grb_loc_error, grb_proba, rate)
    return f_gold


# ------ Gravitational Waves (GW) filters ------
GW_OBSERVATORY = ["LVK"]


def gw_bronze_events(fink_class, observatory, rb):
    """
    Return alerts spatially and temporally consistent with a gcn alerts
    Keep alerts with real bogus score higher than 0.9
    and the alerts classified as "SN candidates", "Unknown", "Ambiguous"

    Parameters
    ----------
    fink_class : pd.Series
        Fink classification
    observatory : pd.Series
        GCN observatory emitter
    rb : pd.Series

    Return
    ------
    f_bronze : pd.Series
        alerts falling in the bronze filters

    Examples
    --------
    >>> df = pd.read_parquet(grb_output_data)
    >>> df["f_bronze"] = gw_bronze_events(df["fink_class"], df["observatory"], df["rb"])
    >>> len(df[df["f_bronze"]])
    0
    """
    return generic_bronze_filter(fink_class, observatory, rb, GW_OBSERVATORY)


@pandas_udf(BooleanType())
def f_gw_bronze_events(fink_class, observatory, rb):
    """
    see bronze_events documentation

    Examples
    --------
    >>> df = spark.read.format('parquet').load(grb_output_data)
    >>> df = df.withColumn("f_bronze", f_gw_bronze_events(df["fink_class"], df["observatory"], df["rb"])).filter("f_bronze == True").drop("f_bronze")
    >>> df.count()
    0
    """
    f_bronze = gw_bronze_events(fink_class, observatory, rb)
    return f_bronze


if __name__ == "__main__":
    """Execute the test suite"""

    import pandas as pd  # noqa: F401

    # Run the test suite
    globs = globals()
    globs["grb_output_data"] = "datatest/grb/grb_test_data.parquet"
    spark_unit_tests(globs)
