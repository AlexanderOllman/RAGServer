import mlflow
import boto3
import os
import shutil
from langchain.document_loaders import PyPDFLoader

def pdf_into_documents(file_path):
    """
    Convert a PDF file into LangChain-compatible documents.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        List[Document]: A list of LangChain Document objects.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def get_artifact_uri_from_experiment(experiment_name):
    """
    Retrieve the artifact URI from the latest run of a given MLflow experiment.

    Args:
        experiment_name (str): The name of the MLflow experiment.

    Returns:
        str: The artifact URI.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:/tmp/mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow.")

    # Get the latest run for the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    if not runs:
        raise ValueError(f"No runs found for experiment '{experiment_name}'.")

    # Retrieve the artifact URI
    latest_run = runs[0]
    return latest_run.info.artifact_uri

def download_artifact(artifact_uri, local_dir):
    """
    Download artifacts from an MLflow artifact URI to a local directory.

    Args:
        artifact_uri (str): The MLflow artifact URI.
        local_dir (str): Path to the local directory to save artifacts.
    """
    if artifact_uri.startswith("s3://"):
        s3 = boto3.client("s3")
        bucket, key = artifact_uri[5:].split("/", 1)

        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=key):
            for obj in page.get("Contents", []):
                local_file_path = os.path.join(local_dir, obj["Key"].replace(key, "").lstrip("/"))
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                s3.download_file(bucket, obj["Key"], local_file_path)
    else:
        raise ValueError("Unsupported URI scheme. Only 's3://' is supported for now.")
