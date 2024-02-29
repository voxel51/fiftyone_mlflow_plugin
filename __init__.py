"""YouTube Player plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import fiftyone.operators as foo
import fiftyone.operators.types as types

from mlflow import MlflowClient


def get_candidate_experiments(dataset):
    name = dataset.name
    client = MlflowClient(tracking_uri="http://127.0.0.1:8080")
    all_experiments = client.search_experiments()
    experiment_names_and_ids = []
    for exp in all_experiments:
        if "dataset" not in exp.tags:
            continue
        if exp.tags['dataset'] == name:
            experiment_names_and_ids.append((exp.experiment_id, exp.name))
    return experiment_names_and_ids


class OpenMLFlowPanel(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="open_mlflow_panel",
            label="Open MLFlow Panel",
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
                name="MLFlowPanel", isActive=True, layout="horizontal"
            ),
        )




def register(p):
    p.register(OpenMLFlowPanel)
