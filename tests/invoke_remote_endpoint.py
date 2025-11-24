import os 
import json
import requests

ENDPOINT_URI = os.getenv("CODE_QUALITY_ENDPOINT_URI")
ENDPOINT_KEY = os.getenv("CODE_QUALITY_ENDPOINT_KEY")

if not ENDPOINT_URI or not ENDPOINT_KEY:
    raise RuntimeError(
        "Please set CODE_QUALITY_ENDPOINT)URI and CODE_QUALITY_ENDPOINT_KEY"
        "environment variables before running this script."
    )

sample_request = {
    "data": [
        {
            "lines_of_code__scaled": 0.1,
            "number_of_classes__scaled": 0.2,
            "number_of_packages__scaled": 0.1,
            "number_of_problematic_classes__scaled": 0.05,
            "number_of_highly_problematic_classes__scaled": 0.01,
            "commits_repo__scaled": 0.3,
            "branches__scaled": 0.1,
            "contributors__scaled": 0.2,
            "stars__scaled": 0.15,
            "forks__scaled": 0.1,
        }
    ]
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {ENDPOINT_KEY}",
}

print(f"Calling endpoint: {ENDPOINT_URI}")

response = requests.post(
    ENDPOINT_URI,
    headers=headers,
    data=json.dumps(sample_request),
    timeout=30,
)

print("Status code:", response.status_code)
print("Raw response text:", response.text)

try: 
    data = response.json()
    print("Parsed JSON:", data)
except json.JSONDecodeError:
    print("Response was not valid JSON")