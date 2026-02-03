# Copyright 2022 AstroLab Software
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
import sys
import doctest
import numpy as np


def spark_unit_tests(
    global_args: dict = None, load_rubin_df: bool = False, verbose: bool = False
):
    """Base commands for the Spark unit test suite

    Include this routine in the main of a module, and execute:
    python3 mymodule.py
    to run the tests.

    It should exit gracefully if no error (exit code 0),
    otherwise it will print on the screen the failure.

    Parameters
    ----------
    global_args: dict, optional
        Dictionary containing user-defined variables to
        be passed to the test suite. Default is None.
    load_rubin_df: bool, optional
        If True, load science data for Rubin in globals
    verbose: bool
        If True, print useful debug messages.
        Default is False.

    """
    if global_args is None:
        global_args = globals()

    from pyspark.sql import SparkSession
    from pyspark import SparkConf

    spark = SparkSession.builder.getOrCreate()

    conf = SparkConf()
    confdic = {"spark.python.daemon.module": "coverage_daemon"}

    if spark.version.startswith("2"):
        confdic.update({
            "spark.jars.packages": "org.apache.spark:spark-avro_2.11:{}".format(
                spark.version
            )
        })
    elif spark.version.startswith("3"):
        confdic.update({
            "spark.jars.packages": "org.apache.spark:spark-avro_2.12:{}".format(
                spark.version
            )
        })
    conf.setMaster("local[2]")
    conf.setAppName("fink_filters_test")
    for k, v in confdic.items():
        conf.set(key=k, value=v)
    spark = (
        SparkSession.builder
        .appName("fink_filters_test")
        .config(conf=conf)
        .getOrCreate()
    )

    global_args["spark"] = spark

    if load_rubin_df:
        df = spark.read.format("parquet").load("datatest/rubin_test_data_10_0.parquet")
        global_args["df"] = df

    # Numpy introduced non-backward compatible change from v1.14.
    if np.__version__ >= "1.14.0":
        np.set_printoptions(legacy="1.13")

    sys.exit(doctest.testmod(globs=global_args, verbose=verbose)[0])
