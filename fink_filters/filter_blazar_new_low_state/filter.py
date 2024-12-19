from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import BooleanType

import numpy as np
import pandas as pd

from fink_filters.tester import spark_unit_tests


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def low_state_filter(alerts: pd.core.frame.DataFrame) -> np.ndarray:
    # CHange argument to only flux_state column ?
    """Returns True the alert is considered a low state,
       returns False else.

    Parameters
    ----------
    alerts: pd.core.frame.DataFrame
        DataFrame batch of alerts received by Fink with history available

    Returns
    -------
    check: np.ndarray
        Mask that returns True if the alert is a low state, 
        False else
    """

    tmp = np.array(alerts.select('flux_state').toPandas().values.tolist())
    tmp = tmp.reshape(tmp.shape[0], tmp.shape[-1]).transpose()
    tmp[pd.isnull(tmp)] = np.nan
    return (tmp[1] < 1) & (tmp[2] < 1) & (tmp[0] >= 1)


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
