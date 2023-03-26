from pyspark.sql.functions import when, lit
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd
import filter_utils



def anomaly_notification_(df, threshold=10) -> pd.Series:
    """
    Create event notifications with a high anomaly_score value
    Parameters
    ----------
    df : Spark DataFrame with column :
        objectId : unique identifier for this object
        lc_features: Dict of dicts of floats.
            Keys of first dict - filters (fid),
            keys of inner dicts - names of features
        rb: RealBogus quality score
        anomaly_score: Anomaly score
        timestamp : UTC time
    threshold : Number of notifications (10 by default)

    Returns
    ----------
    out: pandas.Series of bool
    Return a Pandas DataFrame with the appropriate flag:
    false for bad alert, and true for good alert.
    Examples
    ----------
    >>> from fink_utils.spark.utils import concat_col
    >>> from fink_science.ad_features.processor import extract_features_ad
    >>> from fink_science.anomaly_detection.processor import anomalys_score
    >>> df = spark.read.format('parquet').load('datatest')
    >>> what = ['magpsf', 'jd', 'sigmapsf', 'fid']
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)
    >>> df = df.withColumn('lc_features', extract_features_ad(*what_prefix, 'objectId'))
    >>> df = df.withColumn("anomaly_score", anomaly_score("lc_features"))
    >>> df_proc = df.select('objectId', 'candidate.ra',
    >>>                     'candidate.dec', 'candidate.rb',
    >>>                     'anomaly_score', 'timestamp')
    >>> mask = anomaly_notification_(df_proc)
    >>> print(sum(mask))
    """
    med = df.select('anomaly_score').approxQuantile('anomaly_score', [0.5], 0.25)
    df_filtred = df.sort(['anomaly_score'], ascending=True).limit(min(df.count(), threshold))
    df_min = df_filtred.toPandas() #Only 10-20 objects fall into the Pandas dataframe
    filtred_id = set(df_min['objectId'].values)
    df = df.withColumn('flag',
                       when((df.objectId.isin(filtred_id)), lit(True))
                       .otherwise(lit(False)))
    result = np.array(df.select('flag').collect()).reshape(1,-1)[0]
    send_data, slack_data = [], []
    for _, row in df_min.iterrows():
        gal = SkyCoord(ra=row.ra*u.degree, dec=row.dec*u.degree, frame='icrs').galactic
        send_data.append(f'''ID: [{row.objectId}](https://fink-portal.org/{row.objectId})
GAL coordinates: {round(gal.l.deg, 6)},   {round(gal.b.deg, 6)}
UTC: {str(row.timestamp)[:-3]}
Real bogus: {round(row.rb, 2)}
Anomaly score: {round(row.anomaly_score, 2)}''')
        slack_data.append(f'''ID: <https://fink-portal.org/{row.objectId}|{row.objectId}>
GAL coordinates: {round(gal.l.deg, 6)},   {round(gal.b.deg, 6)}
UTC: {str(row.timestamp)[:-3]}
Real bogus: {round(row.rb, 2)}
Anomaly score: {round(row.anomaly_score, 2)}''')
    filter_utils.msg_handler(send_data, slack_data, med)
    med = round(med, 2)
    return pd.Series(result)
