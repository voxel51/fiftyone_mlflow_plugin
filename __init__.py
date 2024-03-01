"""MLflow Experiment Tracking plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import fiftyone.operators as foo
import fiftyone.operators.types as types

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


def get_candidate_experiments(dataset):
    urls = []
    name = dataset.name
    mlflow_experiment_runs = [
        dataset.get_run_info(r)
        for r in dataset.list_runs()
        if dataset.get_run_info(r).config.method == "mlflow_experiment"
    ]

    for mer in mlflow_experiment_runs:
        cfg = mer.config
        name = cfg.experiment_name
        try:
            uri = cfg.tracking_uri
        except:
            uri = "http://localhost:8080"
        id = cfg.experiment_id
        urls.append({"url": f"{uri}/#/experiments/{id}", "name": name})

    return {"urls": urls}


class OpenMLflowPanel(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="open_mlflow_panel",
            label="Open MLflow Panel",
            unlisted=False,
        )
        _config.icon = "/assets/mlflow.svg"
        return _config

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_SECONDARY_ACTIONS,
            types.Button(
                label="Open MLFlow Panel",
                prompt=False,
                icon="/assets/mlflow.svg",
            ),
        )

    def execute(self, ctx):
        ctx.trigger(
            "open_panel",
            params=dict(
                name="MLflowPanel", isActive=True, layout="horizontal"
            ),
        )


class GetMLflowExperimentURLs(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="get_mlflow_experiment_urls",
            label="MLflow: Get experiment URLs",
            unlisted=True,
        )

    def execute(self, ctx):
        return get_candidate_experiments(ctx.dataset)


def _initialize_run_output():
    outputs = types.Object()
    outputs.str("run_key", label="Run key")
    outputs.str("timestamp", label="Creation time")
    outputs.str("version", label="FiftyOne version")
    outputs.obj("config", label="Config", view=types.JSONView())
    return outputs


def _execute_run_info(ctx, run_key):
    info = ctx.dataset.get_run_info(run_key)

    timestamp = info.timestamp.strftime("%Y-%M-%d %H:%M:%S")
    version = info.version
    config = info.config.serialize()
    config = {k: v for k, v in config.items() if v is not None}

    return {
        "run_key": run_key,
        "timestamp": timestamp,
        "version": version,
        "config": config,
    }


class GetMLflowExperimentInfo(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="get_mlflow_experiment_info",
            label="MLflow: get experiment info",
            dynamic=True,
        )
        _config.icon = "/assets/mlflow.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="MLflow: choose experiment",
            description="Get information about an MLflow experiment",
        )

        dataset = ctx.dataset
        run_keys = [
            r
            for r in dataset.list_runs()
            if dataset.get_run_info(r).config.method == "mlflow_experiment"
        ]

        run_choices = types.DropdownView()
        for run_key in run_keys:
            run_choices.add_choice(run_key, label=run_key)

        inputs.enum(
            "run_key",
            run_choices.values(),
            label="Run key",
            description="The experiment to retrieve information for",
            required=True,
            view=types.DropdownView(),
        )

        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        run_key = ctx.params.get("run_key", None)
        return _execute_run_info(ctx, run_key)

    def resolve_output(self, ctx):
        outputs = _initialize_run_output()
        view = types.View(label="MLflow experiment info")
        return types.Property(outputs, view=view)


def register(p):
    p.register(OpenMLflowPanel)
    p.register(GetMLflowExperimentURLs)
    p.register(GetMLflowExperimentInfo)
