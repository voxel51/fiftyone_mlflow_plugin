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

- Connect your MLflow experiments and runs to your FiftyOne datasets
- Visualize the MLflow dashboard right beside your FiftyOne dataset in the FiftyOne App
- Get helpful information about your MLflow runs and experiments in the FiftyOne App

## Installation

First, install the dependencies:

```bash
pip install -U fiftyone mlflow
```

Then, download the plugin:

```bash
fiftyone plugins download https://github.com/jacobmarks/fiftyone_mlflow_plugin
```

## Usage

Here is a basic template for using the plugin.

First, set your tracking URI as an environment variable:

```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

Next, start the MLflow server:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Load a FiftyOne dataset, and the `log_mlflow_run` operator:

```python
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")

log_mlflow_run = foo.get_operator("@jacobmarks/mlflow_tracking/log_mlflow_run")
```

Run your experiment, and log the MLflow run to your FiftyOne dataset:

```python
experiment_name = "<your-experiment-name-here>"
run_name = "<your-run-name-here>"
label_field = "<your-label-field-here>"  ## if you have predictions associated with your run

log_mlflow_run(
    dataset, experiment_name, run_name=run_name, predictions_field=label_field
)
```

In the FiftyOne App, you can now visualize your MLflow runs and experiments right beside your dataset
using the `show_mlflow_run` operator, which will open the MLflow dashboard within the app
(or change the state of the tab if it is already open), opening an iframe directly to the
chosen experiment (and optionally run)!

You can also get summary information about your MLflow runs and experiments using the `get_mlflow_experiment_info` operator.
