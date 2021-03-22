from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import BooleanType
#from astropy.time import Time

import pandas as pd

@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def kn_candidates(#jd, 
                  #cdsxmatch, 
                  drb, classtar) -> pd.Series:
    """ Return alerts considered as KN candidates
    
    Parameters
    ----------
    cdsxmatch: Spark DataFrame Column
        Column containing the cross-match values
    drb: Spark DataFrame Column
        Column containing the Deep-Learning Real Bogus score
    classtar: Spark DataFrame Column
        Column containing the sextractor score
    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.
    """
    high_drb = drb.astype(float) > 0.5
    high_classtar = classtar.astype(float) > 0.4
    #recent_data = Time.now().jd-jd.astype(float) <0.25
    
    # keep_cds = \
    #     ["Unknown", "Transient"]

    f_kn = high_drb & high_classtar #& recent_data #& cdsxmatch.isin(keep_cds)

    return f_kn