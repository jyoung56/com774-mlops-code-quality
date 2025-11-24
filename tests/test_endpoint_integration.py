import json
import os

import pytest 
import requests

ENDPOINT_URI = os.getenv("CODE_QUALITY_ENDPOINT_URI")
ENDPOINT_KEY = os.getenv("CODE_QUALITY_ENDPOINT_KEY")

should_run_remote_tests = os.getenv("TEST_REMOTE_ENDPOINT") == "1"


@pytest.mark.skipif(
    not should_run_remote_tests or not (ENDPOINT_URI and ENDPOINT_KEY),
    reason="Remote endpoint integration tests disabled or env vars not set",
)
def test_remote_endpoint_returns_predictions():
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

    resp = requests.post(ENDPOINT_URI, headers=headers, data=json.dumps(sample_request), timeout=30)

    assert resp.status_code == 200

    data = resp.json()
    assert "predictions" in data
    assert "probabilities" in data
    assert isinstance(data["predictions"], list)
    assert isinstance(data["probabilities"], list)
    assert len(data["predictions"]) == 1
    # each row should have probability for each class → len(probabilities) == n_rows or n_rows x n_classes
    # for now just assert non-empty
    assert len(data["probabilities"]) >= 1
