"Select 1% of all live alerts in a uniformly random way"
import pandas as pd

DESCRIPTION = "Select 1% of all live alerts in a uniformly random way"

def get_random_alert(diaSourceId: pd.Series) -> pd.Series:
    """

    Examples
    ---------
    >>> import numpy as np
    >>> import os
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> # df = pd.read_parquet('dataset/rubin_test_data_10_0.parquet')
    >>> # df = spark.read.format('parquet').load('dataset/rubin_test_data_10_0.parquet')
    >>> totalcount = df.count()
    >>> f = 'fink_filters.rubin.livestream.filter_uniform_sample.get_random_alert'
    >>> df = apply_user_defined_filter(df, f)
    >>> ratio = df.count() / totalcount
    >>> (0.005 <= ratio < 0.02) # check we are between 0.5% and 2%
    True
    """
    # diaSourceId % 113 == 0
    random_alert = diaSourceId.apply(lambda x: x % 113 == 0)
    return random_alert

if __name__ == "__main__":
    from fink_filters.tester import spark_unit_tests
    globs = globals()
    spark_unit_tests(globs, load_rubin_df=True)	
