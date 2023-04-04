from pyspark.sql.functions import when, lit
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd
import filter_utils


def anomaly_notification_(
        df, threshold=10,
        send_to_tg=False, channel_id=None,
        send_to_slack=False, channel_name=None):
    """ Create event notifications with a high anomaly_score value

    Notes
    ----------
    Notifications can be sent to a Slack or Telegram channels.

    Parameters
    ----------
    df : Spark DataFrame
        Mandatory columns are:
            objectId : unique identifier for this object
            ra: Right Ascension of candidate; J2000 [deg]
            dec: Declination of candidate; J2000 [deg]
            rb: RealBogus quality score
            anomaly_score: Anomaly score
            timestamp : UTC time
    threshold: optional, int
        Number of notifications. Default is 10
    send_to_tg: optional, boolean
        If true, send message to Telegram. Default is False
    channel_id: str
        If `send_to_tg` is True, `channel_id` is the name of the
        Telegram channel.
    send_to_slack: optional, boolean
        If true, send message to Slack. Default is False
    channel_id: str
        If `send_to_slack` is True, `channel_name` is the name of the
        Slack channel.

    Returns
    ----------
    out: DataFrame
        Return the input Spark DataFrame with a new column `flag`
        for locating anomalous alerts
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
    >>> df_proc = df.select(
    ...     'objectId', 'candidate.ra',
    ...     'candidate.dec', 'candidate.rb',
    ...     'anomaly_score', 'timestamp')
    >>> df_out = anomaly_notification_(df_proc)
    >>> #For sending to messengers:
    >>> #df_out = anomaly_notification_(df_proc, threshold=10,
        send_to_tg=True, channel_id="@ZTF_anomaly_bot",
        send_to_slack=True, channel_name='fink_alert')
    >>> print(df_out.filter(df_out['flag']).count())
    """
    # Compute the median for the night
    med = df.select('anomaly_score').approxQuantile('anomaly_score', [0.5], 0.05)
    med = round(med[0], 2)

    # Extract anomalous objects
    pdf_anomalies = df.sort(['anomaly_score'], ascending=True).limit(threshold).toPandas()
    upper_bound = np.max(pdf_anomalies['anomaly_score'].values)

    tg_data, slack_data = [], []
    for _, row in pdf_anomalies.iterrows():
        gal = SkyCoord(ra=row.ra * u.degree, dec=row.dec * u.degree, frame='icrs').galactic
        t1a = f'ID: [{row.objectId}](https://fink-portal.org/{row.objectId})'
        t1b = f'ID: <https://fink-portal.org/{row.objectId}|{row.objectId}>'
        t2 = f'GAL coordinates: {round(gal.l.deg, 6)},   {round(gal.b.deg, 6)}'
        t3 = f'UTC: {str(row.timestamp)[:-3]}'
        t4 = f'Real bogus: {round(row.rb, 2)}'
        t5 = f'Anomaly score: {round(row.anomaly_score, 2)}'
        tg_data.append(f'''{t1a}
{t2}
{t3}
{t4}
{t5}''')
        slack_data.append(f'''{t1b}
{t2}
{t3}
{t4}
{t5}''')
    if send_to_slack:
        filter_utils.msg_handler_slack(slack_data, channel_name, med)
    if send_to_tg:
        filter_utils.msg_handler_tg(tg_data, channel_id, med)

    df = df.withColumn('flag', df['anomaly_score'] <= upper_bound)
    return df