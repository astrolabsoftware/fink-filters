from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import BooleanType
from pyspark.sql.functions import when, lit
import numpy as np
import filter_utils
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd


def anomaly_notification_(df, threshold=10):
    med = df.select('anomaly_score').approxQuantile('anomaly_score', [0.5], 0.25)
    df_filtred = df.sort(['anomaly_score'], ascending=True).limit(min(df.count(), threshold))
    print(df_filtred)
    df_min = df_filtred.toPandas()
    filtred_ID = set(df_min['objectId'].values)
    df = df.withColumn('flag', 
                       when((df.objectId.isin(filtred_ID)), lit(True))
                       .otherwise(lit(False)))
    result = np.array(df.select('flag').collect()).reshape(1,-1)[0]
    send_data, slack_data = [], []
    for i, row in df_min.iterrows():
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
    filter_utils.tg_handler(send_data, med)
    filter_utils.send_slack(slack_data, med)
    return pd.Series(result)
    
    
    

                               


