from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import BooleanType
import utilities
import threading
from astropy import units as u
from astropy.coordinates import SkyCoord
from fink_science import __file__
from fink_science.tester import spark_unit_tests
import tokens
import pandas as pd
import os
import json
import requests
import time
from slackclient import SlackClient
import seaborn as sns
import tokens


def tg_handler(data, graph_data) -> None:
    tg_sendgraph(graph_data)
    url = "https://api.telegram.org/bot"
    channel_id = "@ZTF_anomaly_bot"
    url += tokens.tg_token
    method = url + "/sendMessage"
    for obj in data:
        r = requests.post(method, data={
             "chat_id": channel_id,
             "text": obj,
             "parse_mode": "markdown"
              })
        if r.status_code != 200:
            r = requests.post(method, data={
                "chat_id": "@fink_test",
                "text": str(r.status_code)
            })
        time.sleep(3)


def send_slack(data) -> None:
    slack_client = SlackClient(tokens.slack_token)
    channels = slack_client.api_call("conversations.list")['channels']
    for channel in channels:
        if channel['name'] == 'fink_alert':
            for obj in data:
                slack_client.api_call(
                    "chat.postMessage",
                    channel=channel['id'],
                    text=obj,
                    username='fink-bot'
                )
                time.sleep(3)


def tg_sendgraph(data) -> None:
    url = "https://api.telegram.org/bot"
    channel_id = "@ZTF_anomaly_bot"
    url += tokens.tg_token
    method = url + "/sendPhoto"
    graph_methods = (sns.histplot,)
    for method_v in graph_methods:
        graph = method_v(data)
        graph = graph.get_figure()
        graph.savefig('anomaly_graph.png')
        graph_file = open('anomaly_graph.png', 'rb')
        r = requests.post(method, data={
            "chat_id": channel_id,
            "text": 'Distribution of anomaly score in the last upload'
        },
            files={"photo": graph_file})
        if r.status_code != 200:
            r = requests.post(method, data={
                "chat_id": "@fink_test",
                "text": str(r.status_code)
            })
        graph_file.close()
        time.sleep(3)

@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def anomaly_notification(objectId, candidate, rb, anomaly_score, timestamp) -> pd.Series:
    """
    Create event notifications with a high anomaly_score value

    Parameters
    ----------
    objectId : Spark DataFrame Column
        unique identifier for this object
    lc_features: Spark Map
        Dict of dicts of floats. Keys of first dict - filters (fid), keys of inner dicts - names of features
    rb: Spark DataFrame Columns
        RealBogus quality score
    anomaly_score: Spark DataFrame Column
        Anomaly score
    timestamp : Spark DataFrame Column
        UTC time

    Returns
    ----------
    out: pandas.Series of bool
    Return a Pandas DataFrame with the appropriate flag:
    false for bad alert, and true for good alert.

    Examples
    ----------
     >>> from fink_utils.spark.utils import concat_col
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> from fink_science.ad_features.processor import extract_features_ad
    >>> from fink_science.anomaly_detection.processor import anomaly_score
    >>> df = spark.read.format('parquet').load('datatest')


    # Required alert columns, concatenated with historical data
    >>> what = ['magpsf', 'jd', 'sigmapsf', 'fid']
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    >>> df = df.withColumn('lc_features', extract_features_ad(*what_prefix, 'objectId'))
    >>> df = df.withColumn("anomaly_score", anomaly_score("lc_features"))
    >>> f = 'fink_filters.filter_anomaly_notification.filter.anomaly_notification'
    >>> df = apply_user_defined_filter(df, f)
    """
    df_pand = pd.DataFrame(data=zip(objectId, candidate, rb, anomaly_score, timestamp), columns=['objectId', 'candidate', 'anomaly_score', 'timestamp'])
    df_pand = df_pand.sort_values('anomaly_score', ascending=True)
    df_min = df_pand.head(min(10, df_pand.shape[0]))
    mask = df_pand['anomaly_score'] <= min(df_min['anomaly_score'].values)
    send_data, slack_data = [], []
    for i, row in df_min.iterrows():
        gal = SkyCoord(ra=row.candidate.ra*u.degree, dec=row.candidate.dec*u.degree, frame='icrs').galactic
        send_data.append(f'''ID: [{row.objectId}](https://fink-portal.org/{row.objectId})
GAL coordinates: {round(gal.l.deg, 6)},   {round(gal.b.deg, 6)}
UTC: {str(row.timestamp)[:-3]}
Real bogus: {round(row.candidate.rb, 2)}
Anomaly score: {round(row.anomaly_score, 2)}''')
        slack_data.append(f'''ID: <https://fink-portal.org/{row.objectId}|{row.objectId}>
GAL coordinates: {round(gal.l.deg, 6)},   {round(gal.b.deg, 6)}
UTC: {str(row.timestamp)[:-3]}
Real bogus: {round(row.candidate.rb, 2)}
Anomaly score: {round(row.anomaly_score, 2)}''')
    tg_handler(send_data, df_pand['anomaly_score'])
    send_slack(slack_data)
    return mask

if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
