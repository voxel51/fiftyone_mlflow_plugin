import mlflow


def _format_run_name(run_name):
    return run_name.replace("-", "_")


def _initialize_fiftyone_run_for_mlflow_experiment(
    dataset, experiment_name, tracking_uri=None
):
    """
    Initialize a new FiftyOne custom run given an MLflow experiment.

    Args:
    - dataset: The FiftyOne `Dataset` used for the experiment
    - experiment_name: The name of the MLflow experiment to create the run for
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    # tracking_uri = mlflow.get_tracking_uri()
    tracking_uri = tracking_uri or "http://localhost:8080"

    config = dataset.init_run()

    config.method = "mlflow_experiment"
    config.artifact_location = experiment.artifact_location
    config.created_at = experiment.creation_time
    config.experiment_name = experiment_name
    config.experiment_id = experiment.experiment_id
    config.tracking_uri = tracking_uri
    config.tags = experiment.tags
    config.runs = []
    dataset.register_run(experiment_name, config)


def _fiftyone_experiment_run_exists(dataset, experiment_name):
    return experiment_name in dataset.list_runs()


def _add_fiftyone_run_for_mlflow_run(dataset, experiment_name, run_id):
    """
    Add an MLflow run to a FiftyOne custom run.

    Args:
    - dataset: The FiftyOne `Dataset` used for the experiment
    - run_id: The MLflow run_id to add
    """
    run = mlflow.get_run(run_id)
    run_name = run.data.tags["mlflow.runName"]

    config = dataset.init_run()
    config.method = "mlflow_run"
    config.run_name = run_name
    config.run_id = run_id
    config.run_uuid = run.info.run_uuid
    config.experiment_id = run.info.experiment_id
    config.artifact_uri = run.info.artifact_uri
    config.metrics = run.data.metrics
    config.tags = run.data.tags

    dataset.register_run(_format_run_name(run_name), config)

    ## add run to experiment
    experiment_run_info = dataset.get_run_info(experiment_name)
    experiment_run_info.config.runs.append(run_name)
    dataset.update_run_config(experiment_name, experiment_run_info.config)


def log_mlflow_run_to_fiftyone_dataset(
    sample_collection, experiment_name, run_id=None
):
    """
    Log an MLflow run to a FiftyOne custom run.

    Args:
    - sample_collection: The FiftyOne `Dataset` or `DatasetView` used for the experiment
    - experiment_name: The name of the MLflow experiment to create the run for
    - run_id: The MLflow run_id to add
    """
    dataset = sample_collection._dataset

    if not _fiftyone_experiment_run_exists(dataset, experiment_name):
        _initialize_fiftyone_run_for_mlflow_experiment(
            dataset, experiment_name
        )
    if run_id:
        _add_fiftyone_run_for_mlflow_run(dataset, experiment_name, run_id)
