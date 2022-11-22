# Copyright 2019-2022 AstroLab Software
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

@pandas_udf(BooleanType())
def bronze_events(fink_class, realbogus_score):

    f_bogus = realbogus_score >= 0.9
    f_class = fink_class.isin(["SN candidates", "Unknown", "Ambiguous"])

    return (f_bogus & f_class)

@pandas_udf(BooleanType())
def silver_events(fink_class, realbogus_score, grb_proba):

    f_bronze = bronze_events(fink_class, realbogus_score)
    grb_ser_assoc = (1 - grb_proba) > special.erf(5 / sqrt(2))

    return f_bronze & grb_ser_assoc

@pandas_udf(BooleanType())
def gold_events(fink_class, realbogus_score, grb_proba, rate):

    f_silver = silver_events(fink_class, realbogus_score, grb_proba)
    f_rate = rate.abs() > 0.3

    return f_silver & f_rate