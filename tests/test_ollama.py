import base64
import json
from pathlib import Path

import requests


image_path = Path(r"input_images\cc_sc_test_1.jpg")

image_base64 = base64.b64encode(
    image_path.read_bytes()
).decode("ascii")

payload = {
    "model": "qwen3-vl:2b",
    "stream": False,
    "think": False,
    "messages": [
        {
            "role": "user",
            "content": (
                "/no_think\n"
                "Read this engineering table. "
                "Return JSON with an array called results. "
                "For every DIM NO beginning with No., return "
                "dimension_number and whether the final SC/CC "
                "column contains SC or CC."
            ),
            "images": [image_base64],
        }
    ],
    "format": "json",
    "options": {
        "temperature": 0,
        "num_ctx": 4096,
        "num_predict": 2048,
    },
}

response = requests.post(
    "http://localhost:11434/api/chat",
    json=payload,
    timeout=600,
)

response.raise_for_status()

body = response.json()

print("DONE REASON:")
print(body.get("done_reason"))

print("\nCONTENT:")
print(repr(body.get("message", {}).get("content")))

print("\nTHINKING CHARACTERS:")
thinking = body.get("message", {}).get("thinking") or ""
print(len(thinking))

print("\nCOMPLETE RESPONSE:")
print(json.dumps(body, indent=2))