import os
import time
import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


def msg_handler_slack(slack_data, channel_name, med):
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
    med: float
        Median anomaly score overnight

    Returns
    -------
        None
    '''
    slack_client = WebClient(os.environ['ANOMALY_SLACK_TOKEN'])
    slack_data = [f'Median anomaly score overnight: {med}'] + slack_data
    try:
        for slack_obj in slack_data:
            slack_client.chat_postMessage(channel=channel_name, text=slack_obj)
            time.sleep(3)
    except SlackApiError as e:
        if e.response["ok"] is False:
            requests.post("https://api.telegram.org/bot" + os.environ['ANOMALY_TG_TOKEN'] + "/sendMessage", data={
                "chat_id": "@fink_test",
                "text": e.response["error"]
            }, timeout=8)

def msg_handler_tg(tg_data, channel_id, med):
    '''
    Notes
    ----------
    The function sends notifications to the "channel_id" channel of Telegram.

    Parameters
    ----------
    tg_data: list
        List of lines. Each item is a separate notification
    channel_id: string
        Channel id in Telegram
    med: float
        Median anomaly score overnight

    Returns
    -------
        None
    '''
    url = "https://api.telegram.org/bot"
    url += os.environ['ANOMALY_TG_TOKEN']
    method = url + "/sendMessage"
    tg_data = [f'Median anomaly score overnight: {med}'] + tg_data
    for tg_obj in tg_data:
        res = requests.post(method, data={
            "chat_id": channel_id,
            "text": tg_obj,
            "parse_mode": "markdown"
        }, timeout=8
        )
        if res.status_code != 200:
            res = requests.post(method, data={
                "chat_id": "@fink_test",
                "text": str(res.status_code)
            }, timeout=8)
        time.sleep(3)
