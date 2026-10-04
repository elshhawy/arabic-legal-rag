"""Quick manual test for the /ask endpoint. Run with the server already up:
    python test_ask.py
"""

import requests

response = requests.post(
    "http://127.0.0.1:8000/ask",
    json={"question": "هل يجوز فسخ العقد؟"},
)
data = response.json()

print("Status:", response.status_code)
print("Answer:", data["answer"])
for source in data["sources"]:
    print(f"\n- Article {source['article_number']} ({source['citation']})")
    print(" ", source["text"][:150])