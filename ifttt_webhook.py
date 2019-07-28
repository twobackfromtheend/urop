import os
import requests

IFTTT_KEY = os.getenv("IFTTT_KEY", "d_Cg1I9DPuemzCHobG7cnz")

if IFTTT_KEY is None:
    print("IFTTT_KEY env var not provided. Not using IFTTT webhooks.")


def trigger_event(event: str, value1=None, value2=None, value3=None):
    if IFTTT_KEY is None:
        return

    url = f"https://maker.ifttt.com/trigger/{event}/with/key/{IFTTT_KEY}"

    r = requests.post(url, data={"value1": value1, "value2": value2, "value3": value3})
    print(f"Triggered event. Response: {r.content}")
