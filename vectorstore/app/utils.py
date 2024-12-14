import os
import boto3
import shutil
from langchain.document_loaders import PyPDFLoader

def download_artifact(artifact_uri, local_dir):
    """Download artifacts from a URI to a local directory."""
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

def pdf_into_documents(file_path):
    """Convert a PDF file into LangChain-compatible documents."""
    loader = PyPDFLoader(file_path)
    return loader.load()
