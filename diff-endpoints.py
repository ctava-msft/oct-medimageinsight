from azure.ai.ml import MLClient
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential
)
import base64
import json
import numpy as np
import os
from dotenv import load_dotenv
import logging
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the environment variables
load_dotenv()

required_vars = [
    "SUBSCRIPTION_ID",
    "RESOURCE_GROUP",
    "WORKSPACE_NAME",
    "ENDPOINT_NAME",
    "DEPLOYMENT_NAME"
]

for var in required_vars:
    if not os.getenv(var):
        logger.error(f"Missing required environment variable: {var}")
        raise ValueError(f"Missing required environment variable: {var}")


subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WORKSPACE_NAME")
API_VERSION = "2023-07-01-Preview"
endpoint_name=os.getenv("ENDPOINT_NAME")
deployment_name=os.getenv("DEPLOYMENT_NAME")

# Setup the request
_REQUEST_FILE_NAME = "request.json"

def read_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()

def make_request_images(image_path, text):
    request_json = {
        "input_data": {
            "columns": ["image", "text"],
            "index": [0],
            "data": [
                {
                    "image": base64.encodebytes(read_image(image_path)).decode("utf-8"),
                    "text": text
                }
            ],
        },
        "params": {
            "image_standardization_jpeg_compression_ratio": 75,
            "image_standardization_image_size": 512,
            "get_scaling_factor": True
        }
    }
    with open(_REQUEST_FILE_NAME, "wt") as f:
        json.dump(request_json, f)

# Initialize the ML client
try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    credential = InteractiveBrowserCredential()

workspace_ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

# Get the embeddings
def get_embeddings(image_type, image_paths):
    embeddings = []
    for image_path in tqdm(image_paths, desc="Calculating embeddings"):
        make_request_images(image_path, image_type)
        MAX_RETRIES = 3
        IMAGE_EMBEDDING = None
        for r in range(MAX_RETRIES):
            try:
                response = workspace_ml_client.online_endpoints.invoke(
                    endpoint_name=endpoint_name,
                    deployment_name=deployment_name,
                    request_file=_REQUEST_FILE_NAME,
                )
                response = json.loads(response)
                IMAGE_EMBEDDING = response[0]["image_features"][0]
                embeddings.append(IMAGE_EMBEDDING)
                break
            except Exception as e:
                print(f"Unable to get embeddings for image {image_path}: {e}")
                if r == MAX_RETRIES - 1:
                    print(f"Attempt {r} failed, reached retry limit")
                else:
                    print(f"Attempt {r} failed, retrying")
    return embeddings

# Compute the difference
def compute_diff(x, y):
    x = np.array(x)
    y = np.array(y)
    # Compute the dot product of x with its transpose
    xx = np.dot(x, x.T)
    # Compute the dot product of y with its transpose
    yy = np.dot(y, y.T)
    # Compute the dot product of x with y's transpose
    xy = np.dot(x, y.T)
    # Extract the diagonal elements of xx
    rx = np.diag(xx)
    # Extract the diagonal elements of yy
    ry = np.diag(yy)
    # Calculate the kernel matrix for x
    k = np.exp(-0.5 * (rx[:, None] + rx[None, :] - 2 * xx))
    # Calculate the kernel matrix for y
    l = np.exp(-0.5 * (ry[:, None] + ry[None, :] - 2 * yy))
    # Calculate the cross-kernel matrix between x and y
    m = np.exp(-0.5 * (rx[:, None] + ry[None, :] - 2 * xy))
    # Compute and return the mean difference
    return np.mean(k) + np.mean(l) - 2 * np.mean(m)

# Calculate the difference
def calculate_diff(image_type, image_paths1, image_paths2):
    embeddings1 = get_embeddings(image_type, image_paths1)
    embeddings2 = get_embeddings(image_type, image_paths2)
    return compute_diff(embeddings1, embeddings2)

# Execute the function
image_type = "drusen"
image_paths1 = ["./data/samples/normal.jpg"]
image_paths2 = ["./data/samples/normal2.jpg"]
value = calculate_diff(image_type, image_paths1, image_paths2)
print(f"Diff: {value}")