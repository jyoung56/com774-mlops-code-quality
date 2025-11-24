from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, ComputeTarget

ws = Workspace.from_config()

experiment_name = "code_quality_baseline"
experiment = Experiment(workspace = ws, name = experiment_name)

# Use my compute
compute_name = "cw2compute"
compute_target = ComputeTarget(workspace=ws, name= compute_name)


# Define the Azure ML env
env = Environment.from_conda_specification(
    name = "code-quality-env",
    file_path = "environment.yml"
)

src = ScriptRunConfig(
    source_directory = ".",
    script = "src/train.py",
    arguments = ["--n_estimators", 100],
    compute_target=compute_target,
    environment=env,
)

run = experiment.submit(src)
print(f"Submitted run: {run.id}")
run.wait_for_completion(show_output=True)