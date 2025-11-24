from azureml.core import Workspace, Experiment, Environment, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep

EXPERIMENT_NAME = "code_quality_pipeline"
COMPUTE_NAME = "cw2compute"
ENVIRONMENT_NAME = "code-quality-env"

def get_workspace():
    print("Loading workspace from config.json")
    ws = Workspace.from_config()
    print(f"Connected to workspace: {ws.name}")
    return ws

def get_compute(ws, compute_name):
    print(f"Looking for compute target: {compute_name}")
    try:
        compute_target = ComputeTarget(workspace=ws, name=compute_name)
        print(f"Found compute targetL {compute_target.name}")
        return compute_target
    except ComputeTargetException:
        raise RuntimeError(
            f"Compute target '{compute_name}' not found. "
            f"Check the name in Azure ML Studio Compute."
        )

def get_environment(ws):
    print(f"Creating Azure ML environment '{ENVIRONMENT_NAME}' from environment.yml")
    env = Environment.from_conda_specification(
        name=ENVIRONMENT_NAME,
        file_path="environment.yml"
    )
    return env

def build_pipeline(ws, compute_target, env):
    print("Configuring run environment")
    run_config = RunConfiguration()
    run_config.environment = env

    print("Creating PythonScriptStep for training")
    train_step = PythonScriptStep(
        name="TrainRandomForestModel",
        source_directory=".",
        script_name="src/train.py",
        arguments=["--n_estimators", "100"],
        compute_target=compute_target,
        runconfig=run_config,
        allow_reuse=True
    )

    pipeline = Pipeline(workspace=ws, steps=[train_step])
    print("Pipeline object created.")
    return pipeline

def main():
    ws = get_workspace()
    compute_target = get_compute(ws, COMPUTE_NAME)
    env = get_environment(ws)

    pipeline = build_pipeline(ws, compute_target, env)

    print(f"Submitting pipeline as experiment: {EXPERIMENT_NAME}")
    experiment = Experiment(workspace=ws, name=EXPERIMENT_NAME)
    pipeline_run=experiment.submit(pipeline)
    print(f"Submitted PipelineRun: {pipeline_run.id}")

    print("Waiting for pipeline to complete")
    pipeline_run.wait_for_completion(show_output=True)

    print(f"Pipeline finished with status: {pipeline_run.get_status()}")
    print("View Pipeline in Azure ML Studio")

if __name__ == "__main__":
    main()