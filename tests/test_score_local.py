import json
from endpoints import score


class DummyModel:
    def predict(self, X):
        # One prediction per row
        n_samples = len(X)
        return [0] * n_samples

    def predict_proba(self, X):
        # One [p0, p1] pair per row (binary classifier-style)
        n_samples = len(X)
        return [[0.7, 0.3]] * n_samples


def _inject_dummy_model():
    """
    Inject a dummy model into the score module so that
    tests don't depend on loading the real joblib file.
    """
    score.model = DummyModel()


def test_score_output_schema():
    _inject_dummy_model()

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

    # score.run expects a JSON string, so we dump the dict
    raw_response = score.run(json.dumps(sample_request))

    # score.run already returns a Python dict, not a JSON string
    parsed = raw_response

    assert isinstance(parsed, dict)
    assert "predictions" in parsed
    assert "probabilities" in parsed
    assert isinstance(parsed["predictions"], list)
    assert isinstance(parsed["probabilities"], list)
    assert len(parsed["predictions"]) == 1
    assert len(parsed["probabilities"]) == 1


def test_score_accepts_multiple_rows():
    _inject_dummy_model()

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
            },
            {
                "lines_of_code__scaled": 0.2,
                "number_of_classes__scaled": 0.3,
                "number_of_packages__scaled": 0.2,
                "number_of_problematic_classes__scaled": 0.06,
                "number_of_highly_problematic_classes__scaled": 0.02,
                "commits_repo__scaled": 0.4,
                "branches__scaled": 0.2,
                "contributors__scaled": 0.3,
                "stars__scaled": 0.25,
                "forks__scaled": 0.2,
            },
        ]
    }

    raw_response = score.run(json.dumps(sample_request))
    parsed = raw_response

    assert isinstance(parsed, dict)
    assert "predictions" in parsed
    assert "probabilities" in parsed
    assert len(parsed["predictions"]) == 2
    assert len(parsed["probabilities"]) == 2
