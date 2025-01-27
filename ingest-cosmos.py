from azure.ai.ml import MLClient
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential
)
import base64
from azure.cosmos import CosmosClient, exceptions
import json
import numpy as np
import os
from dotenv import load_dotenv
import logging
from tqdm.auto import tqdm
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the environment variables
load_dotenv()

required_vars = [
    "SUBSCRIPTION_ID",
    "RESOURCE_GROUP",
    "WORKSPACE_NAME",
    "ENDPOINT_NAME",
    "DEPLOYMENT_NAME",
    "COSMOSDB_ENDPOINT",
    # "COSMOSDB_KEY",  # Removed COSMOSDB_KEY as AAD token will be used
]

for var in required_vars:
    if not os.getenv(var):
        logger.error(f"Missing required environment variable: {var}")
        raise ValueError(f"Missing required environment variable: {var}")
    else:
        logger.info(f"Environment variable {var} is set.")

subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WORKSPACE_NAME")
API_VERSION = "2023-07-01-Preview"
endpoint_name=os.getenv("ENDPOINT_NAME")
deployment_name=os.getenv("DEPLOYMENT_NAME")
cosmosdb_endpoint = os.getenv("COSMOSDB_ENDPOINT")
# cosmosdb_key = os.getenv("COSMOSDB_KEY")  # Removed COSMOSDB_KEY retrieval

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

try:
    logger.info("Initializing MLClient with the provided credentials and workspace details.")
    workspace_ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )
    logger.info("MLClient initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize MLClient: {e}")
    raise

def validate_resources():
    # Validate ML Workspace
    try:
        workspace_ml_client.workspaces.get(workspace_name)
        logger.info(f"ML Workspace '{workspace_name}' exists.")
    except Exception as e:
        if "Owner resource does not exist" in str(e):
            logger.error("Owner resource does not exist. Please verify your Azure resources.")
        else:
            logger.error(f"ML Workspace '{workspace_name}' does not exist: {e}")
        raise

    # Validate Cosmos DB Database
    try:
        client = CosmosClient(cosmosdb_endpoint, credential=credential)
        client.get_database_client("cosmos-apcwtmd6jvrko")
        logger.info("Cosmos DB database 'cosmos-apcwtmd6jvrko' exists.")
    except exceptions.CosmosHttpResponseError as e:
        logger.error(f"Cosmos DB database does not exist: {e.message}")
        raise

# Initialize Cosmos DB client
def initialize_cosmos(database_name):
    try:
        logger.info(f"Initializing CosmosClient with endpoint: {cosmosdb_endpoint}")
        client = CosmosClient(cosmosdb_endpoint, credential=credential, consistency_level="Session")
        database = client.create_database_if_not_exists(id=database_name)
        logger.info(f"Accessed or created database: {database_name}")
        container_names = ['cache', 'chat', 'products']
        containers = {}
        for name in container_names:
            containers[name] = database.create_container_if_not_exists(id=name, partition_key={'kind': 'Hash', 'paths': ['/id']})
        logger.info(f"Containers initialized or created: {list(containers.keys())}")
        return containers
    except exceptions.CosmosHttpResponseError as e:
        logger.error(f"Failed to initialize CosmosClient: {e.message}")
        logger.debug(f"Exception details: {e}")
        raise

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

def upsert_item_sync(container, item):
    try:
        container.upsert_item(body=item)
        logger.info(f"Successfully upserted item with id: {item['id']}")
    except exceptions.CosmosHttpResponseError as e:
        print(f"Failed to insert document: {e.message}")

# process
def process(containers, image_type, image_paths1):
    embeddings1 = get_embeddings(image_type, image_paths1)
    item = {}
    item['id'] = str(uuid.uuid4())
    item['embedding'] = embeddings1
    item['text'] = image_type
    item['filename'] = image_paths1[0]
    upsert_item_sync(containers["products"], item)

# Execute the function
validate_resources()
image_type = "normal"
image_paths1 = ["./data/samples/normal.jpg"]
containers = initialize_cosmos("cosmosoct")
process(containers, image_type, image_paths1)