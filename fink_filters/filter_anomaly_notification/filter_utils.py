import requests
import time
from slackclient import SlackClient
import tokens


def msg_handler_slack(slack_data, channel_name, med):
    slack_client = SlackClient(tokens.slack_token)
    try:
        channels = slack_client.api_call("conversations.list")['channels']
        for channel in channels:
            if channel['name'] == channel_name:
                channel_buf = channel['id']
                break
    except KeyError:
        r = requests.post("https://api.telegram.org/bot" + "/sendMessage", data={
            "chat_id": "@fink_test",
            "text": 'Slack API error'
        })
        channel_buf = None
    slack_data = [f'Median anomaly score overnight: {med}'] + slack_data
    for slack_obj in slack_data:
        if channel_buf:
            slack_client.api_call(
                "chat.postMessage",
                channel=channel_buf,
                text=slack_obj,
                username='fink-bot'
            )
        time.sleep(3)

def msg_handler_tg(tg_data, channel_id, med):
    url = "https://api.telegram.org/bot"
    url += tokens.tg_token
    method = url + "/sendMessage"
    tg_data = [f'Median anomaly score overnight: {med}'] + tg_data
    for tg_obj in tg_data:
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
        time.sleep(3)