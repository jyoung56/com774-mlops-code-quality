# COM774 CW2 – MLOps Code Quality Project

![CI](https://github.com/jyoung56/com774-mlops-code-quality/actions/workflows/ci.yml/badge.svg)

## MLOps Features Implemented

This project demonstrates an end-to-end MLOps workflow for a code quality prediction model:

- **Experiment tracking & training**
  - Baseline training submitted via `submit_baseline.py` to Azure ML.
  - Metrics logged: `accuracy`, `f1`, `precision`, `recall`, `roc_auc`.
  - A second iteration trained using *scaled SonarQube features only* to reduce data leakage and simplify the feature space.

- **Pipeline training**
  - `pipeline_train.py` builds and submits an Azure ML Pipeline with a `PythonScriptStep` running `src/train.py`.
  - The pipeline registers the trained model as `code_quality_rf_scaled` in the workspace.

- **Model registry & versioning**
  - Best-performing model is registered in the Azure ML model registry under the name `code_quality_rf_scaled`.
  - Future iterations can register new versions while keeping lineage and metrics.

- **Online endpoint deployment**
  - Managed online endpoint: `code-quality-endpoint`.
  - Deployment created via Azure ML CLI v2 using `endpoints/endpoint.yml` and `endpoints/deployment.yml`.
  - Inference container loads the registered model (`model.joblib`) and exposes `/score`.

- **Local + remote testing of the endpoint**
  - `endpoints/score.py` can be exercised locally using:
    ```bash
    pytest -q
    ```
  - Local tests use a **dummy model** injected into the module so tests don’t depend on Azure.
  - Remote endpoint tested via:
    ```bash
    az ml online-endpoint invoke \
      --name code-quality-endpoint \
      --resource-group COM774-CW2 \
      --workspace-name COM774-CW2 \
      --request-file endpoints/sample_request.json
    ```

## CI / CD and Testing

- 🧪 **Unit-style tests with pytest**
  - Tests are defined in `tests/test_score_local.py`.
  - They verify:
    - Input schema handling for the `/score` request payload.
    - Output contains `predictions` and `probabilities`.
    - Multiple rows are handled correctly.

- 🔁 **GitHub Actions CI**
  - Workflow runs on every `push` and `pull_request` to `main`.
  - Steps:
    - Set up Python 3.10.
    - Install dependencies from `requirements-ci.txt`.
    - Run `pytest`.
  - CI ensures that:
    - The scoring script stays compatible with the expected request/response schema.
    - Changes that break the scoring logic are caught before merging.

### Running tests locally

From the project root:

```bash
python -m venv .venv
.\.venv\Scripts\activate 
pip install -r requirements.txt
pytest -q
