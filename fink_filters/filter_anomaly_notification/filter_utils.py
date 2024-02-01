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
import os
import io
import time
import requests
from datetime import datetime, timedelta
from collections import Counter
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


def get_an_history(delta_date=90):
    '''

    Parameters
    ----------
    delta_date : int
        Time period in days for which objects are considered

    Returns
    -------
    res_obj : Counter
        object Counter of the following content:
            key : object ID
            value : number of top-10 hits for the period
    '''
    history_data = requests.post(
        'https://fink-portal.org/api/v1/anomaly',
        json={
            'n': 100000000,
            'columns': 'i:objectId',
            'start_date': str((datetime.now() - timedelta(days=delta_date)).date())
        }
    )

    if status_check(history_data):
        res_obj = Counter(pd.read_json(io.BytesIO(history_data.content))['i:objectId'].values)
        return res_obj
    else:
        return Counter()


def get_data_permalink_slack(ztf_id):
    '''

    Loads cutout and light curve via the Fink API and copies them to the Slack server

    Parameters
    ----------
    ztf_id : str
        unique identifier for this object

    Returns
    -------
    cutout : BytesIO stream
        cutout image in png format
    curve : BytesIO stream
        light curve picture
    cutout_perml : str
        Link to the cutout image uploaded to the Slack server
    curve_perml : str
        Link to the light curve image uploaded to the Slack server

    '''
    cutout = get_cutout(ztf_id)
    curve = get_curve(ztf_id)
    if 'ANOMALY_SLACK_TOKEN' in os.environ:
        slack_client = WebClient(os.environ['ANOMALY_SLACK_TOKEN'])
    else:
        raise KeyError("You need to set up ANOMALY_SLACK_TOKEN in your .bashrc")
    try:
        curve.seek(0)
        cutout.seek(0)
        result = slack_client.files_upload_v2(
            file_uploads=[
                {
                    "file": cutout,
                    "title": "cutout"
                },
                {
                    "file": curve,
                    "title": "light curve"
                }
            ]
        )
        time.sleep(3)
    except SlackApiError as e:
        if e.response["ok"] is False:
            requests.post(
                "https://api.telegram.org/bot" + os.environ['ANOMALY_TG_TOKEN'] + "/sendMessage",
                data={
                    "chat_id": "@fink_test",
                    "text": e.response["error"]
                },
                timeout=25
            )
            return cutout, curve, None, None
    return cutout, curve, result['files'][0]['permalink'], result['files'][1]['permalink']


def status_check(res):
    '''
    Checks whether the request was successful.
    In case of an error, sends information about the error to the @fink_test telegram channel

    Parameters
    ----------
    res : Response object

    Returns
    -------
        result : bool
            True : The request was successful
            False: The request was executed with an error
    '''
    if res.status_code != 200:
        url = "https://api.telegram.org/bot"
        url += os.environ['ANOMALY_TG_TOKEN']
        method = url + "/sendMessage"
        time.sleep(8)
        requests.post(
            method,
            data={
                "chat_id": "@fink_test",
                "text": str(res.status_code)
            },
            timeout=25
        )
        return False
    return True

def msg_handler_slack(slack_data, channel_name, init_msg):
    '''
    Notes
    ----------
    The function sends notifications to the "channel_name" channel of Slack.

    Parameters
    ----------
    slack_data: list
        List of lines. Each item is a separate notification
    channel_name: string
        Channel name in Slack
    init_msg: str
        Initial message

    Returns
    -------
        None
    '''
    slack_client = WebClient(os.environ['ANOMALY_SLACK_TOKEN'])
    slack_data = [init_msg] + slack_data
    try:
        for slack_obj in slack_data:
            slack_client.chat_postMessage(
                channel=channel_name,
                text=slack_obj,
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": slack_obj
                        }
                    }
                ]
            )
            time.sleep(3)
    except SlackApiError as e:
        if e.response["ok"] is False:
            requests.post(
                "https://api.telegram.org/bot" + os.environ['ANOMALY_TG_TOKEN'] + "/sendMessage",
                data={
                    "chat_id": "@fink_test",
                    "text": e.response["error"]
                },
                timeout=25
            )

def msg_handler_tg(tg_data, channel_id, init_msg):
    '''
    Notes
    ----------
    The function sends notifications to the "channel_id" channel of Telegram.

    Parameters
    ----------
    tg_data: list
        List of tuples. Each item is a separate notification.
        Content of the tuple:
            text_data : str
                Notification text
            cutout : BytesIO stream
                cutout image in png format
            curve : BytesIO stream
                light curve picture
    channel_id: string
        Channel id in Telegram
    init_msg: str
        Initial message

    Returns
    -------
        None
    '''
    url = "https://api.telegram.org/bot"
    url += os.environ['ANOMALY_TG_TOKEN']
    method = url + "/sendMediaGroup"
    res = requests.post(
        url + '/sendMessage',
        data={
            "chat_id": channel_id,
            "text": init_msg,
            "parse_mode": "markdown"
        },
        timeout=25
    )
    status_check(res)
    time.sleep(10)
    for text_data, cutout, curve in tg_data:
        res = requests.post(
            method,
            params={
                "chat_id": channel_id,
                "media": f'''[
                    {{
                        "type" : "photo",
                        "media": "attach://second",
                        "caption" : "{text_data}",
                        "parse_mode": "markdown"
                    }},
                    {{
                        "type" : "photo",
                        "media": "attach://first"
                    }}
                ]'''
            },
            files={
                "second": cutout,
                "first": curve,
            },
            timeout=25
        )
        status_check(res)
        time.sleep(10)


def get_OID(ra, dec):
    '''
    Notes
    ----------
    The function determines the nearest ZTF DR OID by the given ra and dec.

    Parameters
    ----------
    dec: float
        Declination of candidate; J2000 [deg]
    ra: float
        Right Ascension of candidate; J2000 [deg]

    Returns
    -------
        out: str
            ZTF DR OID
    '''
    r = requests.get(
        url=f'http://db.ztf.snad.space/api/v3/data/latest/circle/full/json?ra={ra}&dec={dec}&radius_arcsec=1')
    if not status_check(r):
        return None
    oids = [key for key, _ in r.json().items()]
    if oids:
        return oids[0]
    return None


def get_cutout(ztf_id):
    '''
    The function loads cutout image via Fink API

    Parameters
    ----------
        ztf_id : str
            unique identifier for this object

    Returns
    -------
        out : BytesIO stream
            cutout image in png format

    '''
    r = requests.post(
        'https://fink-portal.org/api/v1/cutouts',
        json={
            'objectId': ztf_id,
            'kind': 'Difference',
        },
        timeout=25
    )
    status_check(r)
    return io.BytesIO(r.content)

def get_curve(ztf_id):
    '''
    The function loads light curve image via Fink API
    Parameters
    ----------
        ztf_id : str
            unique identifier for this object

    Returns
    -------
            out : BytesIO stream
                light curve picture
    '''
    r = requests.post(
        'https://fink-portal.org/api/v1/objects',
        json={
            'objectId': ztf_id,
            'withupperlim': 'True'
        }
    )
    if not status_check(r):
        return None

    # Format output in a DataFrame
    pdf = pd.read_json(io.BytesIO(r.content))

    plt.figure(figsize=(15, 6))

    colordic = {1: 'C0', 2: 'C1'}
    filter_dict = {1: 'g band', 2: 'r band'}

    for filt in np.unique(pdf['i:fid']):
        maskFilt = pdf['i:fid'] == filt

        # The column `d:tag` is used to check data type
        maskValid = pdf['d:tag'] == 'valid'
        plt.errorbar(
            pdf[maskValid & maskFilt]['i:jd'].apply(lambda x: x - 2400000.5),
            pdf[maskValid & maskFilt]['i:magpsf'],
            pdf[maskValid & maskFilt]['i:sigmapsf'],
            ls='', marker='o', color=colordic[filt], label=filter_dict[filt]
        )

        maskUpper = pdf['d:tag'] == 'upperlim'
        plt.plot(
            pdf[maskUpper & maskFilt]['i:jd'].apply(lambda x: x - 2400000.5),
            pdf[maskUpper & maskFilt]['i:diffmaglim'],
            ls='', marker='^', color=colordic[filt], markerfacecolor='none'
        )

        maskBadquality = pdf['d:tag'] == 'badquality'
        plt.errorbar(
            pdf[maskBadquality & maskFilt]['i:jd'].apply(lambda x: x - 2400000.5),
            pdf[maskBadquality & maskFilt]['i:magpsf'],
            pdf[maskBadquality & maskFilt]['i:sigmapsf'],
            ls='', marker='v', color=colordic[filt]
        )

    plt.gca().invert_yaxis()
    plt.legend()
    plt.xlabel('Modified Julian Date')
    plt.ylabel('Difference magnitude')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf
