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
    data = [f'Median anomaly score overnight: {med}'] + data
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


def send_slack(data, med) -> None:
    slack_client = SlackClient(tokens.slack_token)
    channels = slack_client.api_call("conversations.list")['channels']
    data = [f'Median anomaly score overnight: {med}'] + data
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
