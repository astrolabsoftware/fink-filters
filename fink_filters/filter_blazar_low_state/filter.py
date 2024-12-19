from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import BooleanType

import numpy as np
import pandas as pd

from fink_filters.tester import spark_unit_tests


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def low_state_filter(flux_state: Any) -> pd.Series:
    # CHange argument to only flux_state column ?
    """Returns True the alert is considered a low state,
       returns False else.

    Parameters
    ----------
    flux_state: Spark DataFrame Column
        Column containing the 3 ratios computed in the blazar_low_state module

    Returns
    -------
    check: np.ndarray
        Mask that returns True if the alert is a low state, 
        False else
    """

    tmp = np.array(flux_state.toPandas().values.tolist())
    tmp = tmp.reshape(tmp.shape[0], tmp.shape[-1]).transpose()
    tmp[pd.isnull(tmp)] = np.nan
    return pd.Series((tmp[1] < 1) & (tmp[2] < 1))


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
