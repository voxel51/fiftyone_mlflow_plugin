# MLflow Experiment Tracking Plugin for FiftyOne

Training models is hard, and bridging the divide between data and models is even harder.
Fortunately, the right tooling can make data-model co-development a whole lot easier.

This plugin integrates [FiftyOne](https://docs.voxel51.com/) with [MLflow](https://mlflow.org/) to provide a seamless experience for tracking, visualizing, and comparing your datasets and models.

## What is FiftyOne?

[FiftyOne](https://docs.voxel51.com/) is an open-source tool for data exploration and debugging in computer vision. It provides a powerful Python API for working with datasets, and a web-based UI for visualizing and interacting with your data.

## What is MLflow?

[MLflow](https://mlflow.org/) is an open-source platform for the complete machine learning lifecycle. It provides tools for tracking experiments, packaging code into reproducible runs, and sharing and deploying models.

## What does this plugin do?

This plugin helps you to connect your MLflow model training experiments (and runs) to your FiftyOne datasets for enhanced tracking, visualization, model comparison, and debugging!

You can use this plugin to:

- Visualize the MLflow dashboard right beside your FiftyOne dataset in the FiftyOne App
- Get helpful information about your MLflow runs and experiments in the FiftyOne App

## Installation

First, install the dependencies:

```bash
pip install fiftyone mlflow
```

Then, download the plugin:

```bash
fiftyone plugins download https://github.com/jacobmarks/fiftyone_mlflow_plugin
```

## Usage

Here is a basic template for using the plugin.

First, set up MLflow experiment tracking, and load a FiftyOne dataset. Then import the `log_mlflow_run_to_fiftyone_dataset` function to log MLflow runs to your FiftyOne dataset.

```python
import json
from bson import json_util
import sys
import os


import mlflow
from mlflow import MlflowClient

client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.operators as foo
import fiftyone.plugins as fop
import fiftyone.brain as fob
from fiftyone import ViewField as F

dataset = foz.load_zoo_dataset("quickstart")

package_directory = os.path.dirname(
    fop.find_plugin("@jacobmarks/mlflow_tracking")
)
if package_directory not in sys.path:
    sys.path.append(package_directory)
from fiftyone_mlflow_plugin import log_mlflow_run_to_fiftyone_dataset
```

Here are some functions that you may find useful:

```python
def serialize_view(view):
    return json.loads(json_util.dumps(view._serialize()))


def experiment_exists(experiment_name):
    return mlflow.get_experiment_by_name(experiment_name) is not None


def create_fiftyone_mlflow_experiment(
    experiment_name, sample_collection, experiment_description=None
):
    """
    Create a new MLflow experiment for a FiftyOne sample collection.

    Args:
    - experiment_name: The name of the MLflow experiment to create
    - sample_collection: A FiftyOne sample collection to use as the dataset for the experiment
    - experiment_description: An optional description for the MLflow experiment
    """

    tags = {
        "mlflow.note.content": experiment_description,
        "dataset": sample_collection._dataset.name,
    }
    client.create_experiment(name=experiment_name, tags=tags)
```

And here is an example of a high-level loop (pseudocode) for logging MLflow runs to your FiftyOne dataset:

```python
def run_fiftyone_mlflow_experiment(
    sample_collection,
    model,
    training_func,
    experiment_name,
    experiment_description=None,
    apply_model_func=None,
    add_predictions=True,
):
    """
    Run an MLFlow experiment on a FiftyOne sample collection using the provided model and training function.

    Args:
    - sample_collection: A FiftyOne sample collection to use as the dataset for the experiment
    - model: A model to train and log metrics for
    - training_func: A function that trains the model and returns it
    - experiment_name: The name of the MLflow experiment to create
    - experiment_description: An optional description for the MLflow experiment
    - apply_model_func: An optional function that applies the model to the sample collection
    - add_predictions: Whether to add the model's predictions for the sample collection
    """

    if not experiment_exists(experiment_name):
        create_fiftyone_mlflow_experiment(
            experiment_name, sample_collection, experiment_description
        )

    mlflow.set_experiment(experiment_name)

    # Train the model
    with mlflow.start_run() as run:
        if sample_collection._dataset != sample_collection:
            ## log the serialized `DatasetView`
            mlflow.log_param("dataset_view", serialize_view(sample_collection))

        model = training_func(dataset)
        mlflow.log_params(model.hyperparameters)
        mlflow.log_metrics(model.metrics)

        signature = infer_signature(X.numpy(), net(X).detach().numpy())
        model_info = mlflow.pytorch.log_model(
            model, "model", signature=signature
        )

        pytorch_pyfunc = mlflow.pyfunc.load_model(
            model_uri=model_info.model_uri
        )

        dataset = sample_collection._dataset
        if "runs" not in dataset.info["mlflow"]:
            dataset.info["mlflow"]["runs"] = []
            dataset.save()
        dataset.info["mlflow"]["runs"].append(run.info.run_id)
        dataset.save()

        log_mlflow_run_to_fiftyone_dataset(
            sample_collection, experiment_name, run_id=run.info.run_id
        )

    if add_predictions:
        predictions_field = f"{run.info.run_id}_predictions"
        apply_model_func(sample_collection, pytorch_pyfunc, predictions_field)
```

You can then run the experiment like this:

```python
model = "<insert model here>"
training_func = "<insert training function here>"
experiment_name = "test_experiment"
experiment_description = "test experiment description"
run_fiftyone_mlflow_experiment(
    dataset, model, training_func, experiment_name, experiment_description
)
```
