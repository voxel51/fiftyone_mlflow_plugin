"""MLflow Experiment Tracking plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import json
from bson import json_util

import fiftyone.operators as foo
import fiftyone.operators.types as types

import mlflow

DEFAULT_TRACKING_URI = "http://localhost:5000"


def _get_tracking_uri(ctx):
    for key, value in getattr(ctx, "secrets", {}).items():
        if key == "MLFLOW_TRACKING_URI":
            return value
    else:
        return DEFAULT_TRACKING_URI


def _get_client(ctx):
    uri = _get_tracking_uri(ctx)
    client = mlflow.MlflowClient(tracking_uri=uri)
    return client


def _get_experiment_id_by_name(experiment_name, client):
    return client.get_experiment_by_name(experiment_name).experiment_id


def _get_run(ctx, experiment_name, client):
    experiment_id = _get_experiment_id_by_name(experiment_name, client)
    run_name = ctx.params.get("run_name", None)
    if run_name:
        run_id = client.search_runs(
            [experiment_id], filter_string=f"run_name='{run_name}'"
        )[0].info.run_id
        return client.get_run(run_id)
    else:
        return mlflow.last_active_run()


def _get_experiment_uri(ctx, experiment_name, client):
    experiment_id = _get_experiment_id_by_name(experiment_name, client)
    return f"{_get_tracking_uri(ctx)}/#/experiments/{experiment_id}"


def _get_run_uri(ctx, experiment_name, run_id, client):
    experiment_uri = _get_experiment_uri(ctx, experiment_name, client)
    return f"{experiment_uri}/runs/{run_id}"


def _format_run_name(run_name):
    return run_name.replace("-", "_")


def serialize_view(view):
    return json.loads(json_util.dumps(view._serialize()))


def _get_gt_field(ctx, dataset):
    if "gt_field" in ctx.params and ctx.params["gt_field"] is not None:
        return ctx.params["gt_field"]
    elif "ground_truth" in dataset.get_field_schema():
        return "ground_truth"
    else:
        return None


def _connect_predictions_to_run(
    ctx, dataset, predictions_field, experiment_name, run_id, run_name, client
):
    ## Add run info to predictions field
    field = dataset.get_field(predictions_field)
    run_uri = _get_run_uri(ctx, experiment_name, run_id, client)
    field.info = {
        "experiment_name": experiment_name,
        "run_name": run_name,
        "url": run_uri,
    }
    field.save()

    ## Add label_field to mlflow run tags
    client.set_tag(run_id, "predictions_field", predictions_field)

    ## Add ground truth field to mlflow run tags
    gt_field = _get_gt_field(ctx, dataset)
    if gt_field is not None:
        client.set_tag(run_id, "gt_field", gt_field)


def _initialize_fiftyone_run_for_mlflow_experiment(
    dataset, experiment_name, client
):
    """
    Initialize a new FiftyOne custom run given an MLflow experiment.

    Args:
    - dataset: The FiftyOne `Dataset` used for the experiment
    - experiment_name: The name of the MLflow experiment to create the run for
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    tracking_uri = client.tracking_uri
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


def _add_fiftyone_run_for_mlflow_run(
    dataset, experiment_name, run_id, client, **kwargs
):
    """
    Add an MLflow run to a FiftyOne custom run.

    Args:
    - dataset: The FiftyOne `Dataset` used for the experiment
    - run_id: The MLflow run_id to add
    """
    run = mlflow.get_run(run_id)
    run_name = run.info.run_name

    config = dataset.init_run()
    config.method = "mlflow_run"
    config.run_name = run_name
    config.run_id = run_id
    config.run_uuid = run.info.run_uuid
    config.experiment_id = run.info.experiment_id
    config.artifact_uri = run.info.artifact_uri
    config.metrics = run.data.metrics
    config.tags = run.data.tags
    config.tracking_uri = client.tracking_uri

    if "predictions_field" in kwargs:
        config.predictions_field = kwargs["predictions_field"]
    if "gt_field" in kwargs:
        config.gt_field = kwargs["gt_field"]

    fmt_run_name = _format_run_name(run_name)

    dataset.register_run(fmt_run_name, config)

    if "view" in kwargs:
        results = dataset.init_run_results(fmt_run_name)
        results.target_view = kwargs["view"]._serialize()
        dataset.save_run_results(fmt_run_name, results, overwrite=True)

    ## add run to experiment
    experiment_run_info = dataset.get_run_info(experiment_name)
    experiment_run_info.config.runs.append(run_name)
    dataset.update_run_config(experiment_name, experiment_run_info.config)


def _is_subset_view(sample_collection):
    """Checks if the sample collection is the entire dataset or a view"""
    return sample_collection.view() != sample_collection._dataset.view()


def _connect_dataset_to_experiment_if_necessary(
    dataset, experiment_name, client
):
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_tags = experiment.tags
    if "dataset_name" not in experiment_tags:
        experiment_id = experiment.experiment_id
        client.set_experiment_tag(experiment_id, "dataset_name", dataset.name)

    # Create FiftyOne Custom Run for the experiment
    if experiment_name not in dataset.list_runs():
        _initialize_fiftyone_run_for_mlflow_experiment(
            dataset, experiment_name, client
        )


def log_mlflow_run(ctx):
    client = _get_client(ctx)
    dataset = ctx.dataset
    view = ctx.view
    predictions_field = ctx.params.get("predictions_field", None)
    gt_field = ctx.params.get("gt_field", None)
    experiment_name = ctx.params.get("experiment", None)
    run = _get_run(ctx, experiment_name, client)
    run_name, run_id = run.info.run_name, run.info.run_id

    _connect_dataset_to_experiment_if_necessary(
        dataset, experiment_name, client
    )

    add_run_kwargs = {}

    if (
        predictions_field is not None
        and predictions_field in dataset.get_field_schema()
    ):
        _connect_predictions_to_run(
            ctx,
            dataset,
            predictions_field,
            experiment_name,
            run_id,
            run_name,
            client,
        )
        add_run_kwargs["predictions_field"] = predictions_field

    if gt_field is not None and gt_field in dataset.get_field_schema():
        add_run_kwargs["gt_field"] = gt_field

    is_subset = _is_subset_view(view)
    if is_subset:
        serial_view = serialize_view(view)
        client.set_tag(run_id, "view", serial_view)

    ## Add run to FiftyOne custom run
    _add_fiftyone_run_for_mlflow_run(
        dataset, experiment_name, run_id, client, **add_run_kwargs
    )


class LogMLflowRun(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="log_mlflow_run",
            label="MLflow: Log MLflow run to the FiftyOne dataset",
            dynamic=True,
            unlisted=True,
        )
        return _config

    def __call__(
        self,
        sample_collection,
        experiment_name,
        run_name=None,
        predictions_field=None,
        gt_field=None,
    ):
        dataset = sample_collection._dataset
        view = sample_collection.view()
        ctx = dict(view=view, dataset=dataset)
        params = dict(
            experiment=experiment_name,
            run_name=run_name,
            predictions_field=predictions_field,
            gt_field=gt_field,
        )
        return foo.execute_operator(self.uri, ctx, params=params)

    def execute(self, ctx):
        log_mlflow_run(ctx)


def get_candidate_experiment_names(ctx):
    experiment_names = [
        r
        for r in ctx.dataset.list_runs()
        if ctx.dataset.get_run_info(r).config.method == "mlflow_experiment"
    ]
    return experiment_names


def get_candidate_run_names(ctx, experiment_name):
    experiment_info = ctx.dataset.get_run_info(experiment_name)
    experiment_runs = experiment_info.config.runs
    return experiment_runs


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
                label="Open MLflow Panel",
                prompt=False,
                icon="/assets/mlflow.svg",
            ),
        )

    def execute(self, ctx):
        ctx.trigger(
            "open_panel",
            params=dict(
                name="MLFlowPanel", isActive=True, layout="horizontal"
            ),
        )


def _get_mlflow_url_input(ctx, inputs):
    dataset = ctx.dataset


class ShowMLflowRun(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="show_mlflow_run",
            label="Show MLflow run",
            dynamic=True,
            description=(
                "View the data and metrics for an MLflow experiment/run"
                ", all in one place!"
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        experiments = get_candidate_experiment_names(ctx)
        if len(experiments) == 0:
            inputs.view(
                "warning",
                types.Warning(
                    label="No experiments",
                    description="Tracking Server home page will be opened instead.",
                ),
            )
            return types.Property(inputs)

        exp_choices = types.DropdownView()
        for experiment in experiments:
            exp_choices.add_choice(experiment, label=experiment)
        inputs.enum(
            "experiment_name",
            exp_choices.values(),
            label="Experiment name",
            description="The name of the MLflow experiment to display",
            required=True,
            view=types.DropdownView(),
        )

        experiment_name = ctx.params.get("experiment_name", None)
        if experiment_name is not None:
            runs = get_candidate_run_names(ctx, experiment_name)
            run_choices = types.DropdownView()
            for run in runs:
                run_choices.add_choice(run, label=run)
            inputs.enum(
                "run_name",
                run_choices.values(),
                label="Run name",
                description="The name of the MLflow run to display",
                required=False,
                view=types.DropdownView(),
            )

        return types.Property(inputs)

    def execute(self, ctx):
        client = _get_client(ctx)
        experiment_name = ctx.params.get("experiment_name", None)
        run_name = ctx.params.get("run_name", None)
        run = None
        if experiment_name is None:
            url = _get_tracking_uri(ctx)
        elif run_name is None:
            url = _get_experiment_uri(ctx, experiment_name, client)
        else:
            run = _get_run(ctx, experiment_name, client)
            url = _get_run_uri(ctx, experiment_name, run.info.run_id, client)

        if run is not None:
            fmt_run_name = _format_run_name(run_name)
            run_info = ctx.dataset.get_run_info(fmt_run_name)

            keep_fields = []
            if hasattr(run_info.config, "gt_field"):
                keep_fields.append(run_info.config.gt_field)

            if hasattr(run_info.config, "predictions_field"):
                keep_fields.append(run_info.config.predictions_field)

                view = ctx.dataset.select_fields(keep_fields)
                ctx.trigger(
                    "set_view",
                    params=dict(view=serialize_view(view)),
                )

        ctx.trigger(
            "@voxel51/mlflow/set_iframe_url",
            params=dict(url=url),
        )
        ctx.trigger(
            "open_panel",
            params=dict(name="MLFlowPanel", layout="horizontal"),
        )


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
    p.register(GetMLflowExperimentInfo)
    p.register(LogMLflowRun)
    p.register(ShowMLflowRun)
