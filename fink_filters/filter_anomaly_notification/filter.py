# Copyright 2023 AstroLab Software
# Author: Тимофей Пшеничный
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
import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord

from fink_filters.filter_anomaly_notification import filter_utils

from fink_filters.tester import spark_unit_tests


def anomaly_notification_(
        df_proc, threshold=10,
        send_to_tg=False, channel_id=None,
        send_to_slack=False, channel_name=None,
        trick_par=10):
    """ Create event notifications with a high `anomaly_score` value

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
    channel_name: str
        If `send_to_slack` is True, `channel_name` is the name of the
        Slack channel.
    trick_par: int, optional
        Internal buffer to reduce the number of candidates while giving
        space to reach the `threshold`. Defaut is 10.

    Returns
    ----------
    out: Pandas DataFrame
        Pandas DataFrame with anomaly information for the selected candidates

    Examples
    ----------
    >>> import pyspark.sql.functions as F
    >>> from fink_utils.spark.utils import concat_col
    >>> from fink_science.ad_features.processor import extract_features_ad
    >>> from fink_science.anomaly_detection.processor import anomaly_score

    >>> df = spark.read.format('parquet').load('datatest')

    >>> what = [
    ...     'jd', 'fid', 'magpsf', 'sigmapsf',
    ...     'magnr', 'sigmagnr', 'isdiffpos', 'distnr']

    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    >>> # Add a fake distnr to skip dcmag computation
    >>> df = df\
            .withColumn('tmp', F.expr('TRANSFORM(cdistnr, el -> el + 100)'))\
            .drop('cdistnr').withColumnRenamed('tmp', 'cdistnr')

    >>> ad_args = [
    ...     'cmagpsf', 'cjd', 'csigmapsf', 'cfid', 'objectId',
    ...     'cdistnr', 'cmagnr', 'csigmagnr', 'cisdiffpos']

    >>> df = df.withColumn('lc_features', extract_features_ad(*ad_args))
    >>> df = df.withColumn("anomaly_score", anomaly_score("lc_features"))

    >>> df_proc = df.select(
    ...     'objectId', 'candidate.ra',
    ...     'candidate.dec', 'candidate.rb',
    ...     'anomaly_score', 'timestamp')
    >>> df_out = anomaly_notification_(df_proc)

    # Disable communication
    >>> pdf_anomalies = anomaly_notification_(df_proc, threshold=10,
    ...     send_to_tg=False, channel_id=None,
    ...     send_to_slack=False, channel_name=None)
    >>> print(pdf_anomalies['objectId'].values)
    ['ZTF21acoshvy' 'ZTF18aapgymv' 'ZTF19aboujyi' 'ZTF18abgjtxx' 'ZTF18aaypnnd'
     'ZTF18abbtxsx' 'ZTF18aaakhsv' 'ZTF18actxdmj' 'ZTF18aapoack' 'ZTF18abzvnya']
    """
    # Compute the median for the night
    med = df_proc.select('anomaly_score').approxQuantile('anomaly_score', [0.5], 0.05)
    med = round(med[0], 2)

    # Extract anomalous objects
    pdf_anomalies_ext = df_proc.sort(['anomaly_score'], ascending=True).limit(trick_par * threshold).toPandas()
    pdf_anomalies_ext = pdf_anomalies_ext.drop_duplicates(['objectId'])
    upper_bound = np.max(pdf_anomalies_ext['anomaly_score'].values[:threshold])
    pdf_anomalies = pdf_anomalies_ext[pdf_anomalies_ext['anomaly_score'] <= upper_bound]

    tg_data, slack_data = [], []
    for _, row in pdf_anomalies.iterrows():
        gal = SkyCoord(ra=row.ra * u.degree, dec=row.dec * u.degree, frame='icrs').galactic
        oid = filter_utils.get_OID(row.ra, row.dec)
        t1a = f'ID: [{row.objectId}](https://fink-portal.org/{row.objectId})'
        t1b = f'ID: <https://fink-portal.org/{row.objectId}|{row.objectId}>'
        t_oid_1a = f'DR OID (<1"): [{oid}](https://ztf.snad.space/view/{oid})'
        t_oid_1b = f'DR OID (<1"): <https://ztf.snad.space/view/{oid}|{oid}>'
        t2_ = f'GAL coordinates: {round(gal.l.deg, 6)},   {round(gal.b.deg, 6)}'
        t3_ = f'UTC: {str(row.timestamp)[:-3]}'
        t4_ = f'Real bogus: {round(row.rb, 2)}'
        t5_ = f'Anomaly score: {round(row.anomaly_score, 2)}'
        tg_data.append(f'''{t1a}
{t_oid_1a}
{t2_}
{t3_}
{t4_}
{t5_}''')
        slack_data.append(f'''{t1b}
{t_oid_1b}
{t2_}
{t3_}
{t4_}
{t5_}''')
    if send_to_slack:
        filter_utils.msg_handler_slack(slack_data, channel_name, med)
    if send_to_tg:
        filter_utils.msg_handler_tg(tg_data, channel_id, med)

    return pdf_anomalies


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
