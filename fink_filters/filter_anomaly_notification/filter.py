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
import pandas as pd
import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord


from fink_filters.filter_anomaly_notification import filter_utils

from fink_filters.tester import spark_unit_tests


def anomaly_notification_(
        df_proc, threshold=10,
        send_to_tg=False, channel_id=None,
        send_to_slack=False, channel_name=None,
        trick_par=10, cut_coords=False, history_period=90, send_to_anomaly_base=False, model=''):
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
    cut_coords: bool
        If this is True, only objects from the area bounded
        by the following coordinates are considered:
            1) delta <= 20°
            2) alpha ∈ (160°, 240°)
    history_period: int
            Time period in days for which the number
            of references is calculated
    model: str
        Name of the model used.
        Name must start with a ‘_’ and be ‘_{user_name}’,
        where user_name is the user name of the model at https://anomaly.fink-portal.org/.
    send_to_anomaly_base: bool
        If True, notifications are uploaded to
        https://anomaly.fink-portal.org/ in the selected model's
        account. Only works for model != ‘’


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

    >>> df = spark.read.format('parquet').load('datatest/regular')
    >>> MODELS = ['_beta', ''] # '' corresponds to the model for a telegram channel
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
    >>> for model in MODELS:
    ...     df = df.withColumn(f'anomaly_score{model}', anomaly_score("lc_features", F.lit(model)))

    >>> for model in MODELS:
    ...     df_proc = df.select(
    ...         'objectId', 'candidate.ra',
    ...         'candidate.dec', 'candidate.rb',
    ...         f'anomaly_score{model}', 'timestamp')
    ...     df_out = anomaly_notification_(df_proc, send_to_tg=False,
    ...     send_to_slack=False, send_to_anomaly_base=True, model=model)

    # Disable communication
    >>> df_proc = df.select(
    ...         'objectId', 'candidate.ra',
    ...         'candidate.dec', 'candidate.rb',
    ...         'anomaly_score', 'timestamp')
    >>> pdf_anomalies = anomaly_notification_(df_proc, threshold=10,
    ...     send_to_tg=False, channel_id=None,
    ...     send_to_slack=False, channel_name=None)
    >>> print(sorted(pdf_anomalies['objectId'].values))
    ['ZTF17aaabbbp', 'ZTF18aaakhsv', 'ZTF18aabeyfi', 'ZTF18aaypnnd', 'ZTF18abgjtxx', 'ZTF18abhxigz', 'ZTF18abjuixy', 'ZTF19aboujyi', 'ZTF21acobels', 'ZTF21acoshvy']

    # Check cut_coords
    >>> pdf_anomalies = anomaly_notification_(df_proc, threshold=10,
    ...     send_to_tg=False, channel_id=None,
    ...     send_to_slack=False, channel_name=None, cut_coords=True)

    # Not empty in this case
    >>> assert not pdf_anomalies.empty
    """
    # Filtering by coordinates
    if cut_coords:
        df_proc = df_proc.filter("dec <= 20 AND (ra >= 160 AND ra <= 240)")
        # We need to know the total number of objects per night which satisfy the condition on coordinates
        cut_count = df_proc.count()
        if cut_count == 0:
            return pd.DataFrame()

    # Compute the median for the night
    buf_df = df_proc.select(f'anomaly_score{model}')
    med = buf_df.approxQuantile(f'anomaly_score{model}', [0.5], 0.05)
    med = round(med[0], 2)

    # Extract anomalous objects

    pdf_anomalies_ext = df_proc.sort([f'anomaly_score{model}'], ascending=True).limit(trick_par * threshold).toPandas()
    pdf_anomalies_ext = pdf_anomalies_ext.drop_duplicates(['objectId'])
    upper_bound = np.max(pdf_anomalies_ext[f'anomaly_score{model}'].values[:threshold])
    pdf_anomalies = pdf_anomalies_ext[pdf_anomalies_ext[f'anomaly_score{model}'] <= upper_bound].head(10)

    history_objects = filter_utils.get_an_history(history_period)

    tg_data, slack_data, base_data = [], [], []

    for _, row in pdf_anomalies.iterrows():
        gal = SkyCoord(
            ra=row.ra * u.degree, dec=row.dec * u.degree, frame="icrs"
        ).galactic
        oid = filter_utils.get_OID(row.ra, row.dec)
        t1a = f'**ID**: [{row.objectId}](https://fink-portal.org/{row.objectId})'
        t1b = f'ID: <https://fink-portal.org/{row.objectId}|{row.objectId}>'
        t_oid_1a = f"**DR OID (<1'')**: [{oid}](https://ztf.snad.space/view/{oid})"
        t_oid_1b = f"DR OID (<1''): <https://ztf.snad.space/view/{oid}|{oid}>"
        t2_ = f'**GAL coordinates**: {round(gal.l.deg, 6)},   {round(gal.b.deg, 6)}'
        t_ = f'''
**EQU**: {row.ra},   {row.dec}'''
        t2_ += t_
        t3_ = f'**UTC**: {str(row.timestamp)[:-3]}'
        t4_ = f'**Real bogus**: {round(row.rb, 2)}'
        t5_ = f'**Anomaly score**: {round(row[f"anomaly_score{model}"], 2)}'
        if row.objectId in history_objects:
            t5_ += f"""
Detected as top-{threshold} in the last {history_period} days: {history_objects[row.objectId]} {'times' if history_objects[row.objectId] > 1 else 'time'}."""
        cutout, curve, cutout_perml, curve_perml = (
            filter_utils.get_data_permalink_slack(row.objectId)
        )
        curve.seek(0)
        cutout.seek(0)
        cutout_perml = f"<{cutout_perml}|{' '}>"
        curve_perml = f"<{curve_perml}|{' '}>"
        if model == '':
            tg_data.append((f'''{t1a}
{t_oid_1a}
{t2_}
{t3_}
{t4_}
{t5_}''', cutout, curve))
            slack_data.append(f'''==========================
{t1b}
{t_oid_1b}
{t2_}
{t3_}
{t4_}
{t5_}
{cutout_perml}{curve_perml}''')
        base_data.append((row.objectId, f'''{t1a}
{t_oid_1a}
{t2_}
{t3_}
{t4_}
{t5_}'''.replace('\n', '  \n'), cutout, curve))

    init_msg = f'Median anomaly score overnight: {med}.'
    if cut_coords and model == '':
        init_msg += f"""
(of the objects in the sky area)
Sky area:
1) delta <= 20°
2) alpha ∈ (160°, 240°)
Total number of objects per night in the area: {cut_count}.
"""
    if send_to_slack:
        filter_utils.msg_handler_slack(slack_data, channel_name, init_msg)
    if send_to_tg:
        filter_utils.msg_handler_tg(tg_data, channel_id, init_msg)
    if model != '':
        filter_utils.load_to_anomaly_base(base_data, model)
    return pdf_anomalies


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
