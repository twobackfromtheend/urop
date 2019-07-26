import os
import requests

IFTTT_KEY = os.getenv("IFTTT_KEY")


def trigger_event(event: str, value1=None, value2=None, value3=None):
    url = f"https://maker.ifttt.com/trigger/{event}/with/key/{IFTTT_KEY}"

    r = requests.post(url, data={"value1": value1, "value2": value2, "value3": value3})
    print(f"Triggered event. Response: {r.content}")
