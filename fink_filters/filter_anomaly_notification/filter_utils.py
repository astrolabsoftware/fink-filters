import os
import json
import requests
import time
from slackclient import SlackClient
import seaborn as sns
import tokens

                
def msg_handler(tg_data, slack_data, med):
    url = "https://api.telegram.org/bot"
    channel_id = "@ZTF_anomaly_bot"
    url += tokens.tg_token
    method = url + "/sendMessage"
    tg_data = [f'Median anomaly score overnight: {med}'] + tg_data
    slack_client = SlackClient(tokens.slack_token)
    try:
        channels = slack_client.api_call("conversations.list")['channels']
        for channel in channels:
            if channel['name'] == 'fink_alert':
                channel_buf = channel['id']
                break
    except KeyError:
        r = requests.post(method, data={
            "chat_id": "@fink_test",
            "text": 'Slack API error'
        })
        channel_buf = None
    slack_data = [f'Median anomaly score overnight: {med}'] + slack_data
    for tg_obj, slack_obj in zip(tg_data, slack_data):
        r = requests.post(method, data={
             "chat_id": channel_id,
             "text": tg_obj,
             "parse_mode": "markdown"
              })
        if r.status_code != 200:
            r = requests.post(method, data={
                "chat_id": "@fink_test",
                "text": str(r.status_code)
            })
        if channel_buf:
            slack_client.api_call(
                "chat.postMessage",
                channel=channel_buf,
                text=slack_obj,
                username='fink-bot'
            )
        time.sleep(3)
    
