from azure.ai.ml import MLClient
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential
)
import json
import os
from dotenv import load_dotenv
import requests
import logging
from tqdm.auto import tqdm
import base64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

required_vars = [
    "SEARCH_KEY",
    "SEARCH_SERVICE_NAME",
    "SEARCH_INDEX_NAME",
    "SUBSCRIPTION_ID",
    "RESOURCE_GROUP",
    "WORKSPACE_NAME",
    "ENDPOINT_NAME",
    "DEPLOYMENT_NAME",
    "SEARCH_INDEX_VECTOR_DIMENSION"
]

for var in required_vars:
    if not os.getenv(var):
        logger.error(f"Missing required environment variable: {var}")
        raise ValueError(f"Missing required environment variable: {var}")

SEARCH_KEY = os.getenv("SEARCH_KEY")
SEARCH_SERVICE_NAME = os.getenv("SEARCH_SERVICE_NAME")
SEARCH_INDEX_NAME = os.getenv("SEARCH_INDEX_NAME")
SEARCH_INDEX_VECTOR_DIMENSION=os.getenv("SEARCH_INDEX_VECTOR_DIMENSION")
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WORKSPACE_NAME")
API_VERSION = "2023-07-01-Preview"
endpoint_name=os.getenv("ENDPOINT_NAME")
deployment_name=os.getenv("DEPLOYMENT_NAME")

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

ADD_DATA_REQUEST_URL = "https://{search_service_name}.search.windows.net/indexes/{index_name}/docs/index?api-version={api_version}".format(
    search_service_name=SEARCH_SERVICE_NAME,
    index_name=SEARCH_INDEX_NAME,
    api_version=API_VERSION,
)

# Get dataset
idm=100
dataset_dir = './data/OCT-5/DRUSEN'
image_type = "drusen"

image_paths = [
    os.path.join(dp, f)
    for dp, dn, filenames in os.walk(dataset_dir)
    for f in filenames
    if os.path.splitext(f)[1] == ".jpeg"
]

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

for idx, image_path in enumerate(tqdm(image_paths)):
    ID = (idx+1) * idm
    FILENAME = image_path
    print(f"Processing image {FILENAME}")
    MAX_RETRIES = 3

    # get embedding from endpoint
    embedding_request = make_request_images(image_path, image_type)

    response = None
    request_failed = False
    IMAGE_EMBEDDING = None
    for r in range(MAX_RETRIES):
        try:
            response = workspace_ml_client.online_endpoints.invoke(
                endpoint_name=endpoint_name,
                deployment_name=deployment_name,
                request_file=_REQUEST_FILE_NAME,
            )
            response = json.loads(response)
            print(response)
            IMAGE_EMBEDDING = response[0]["image_features"]
            IMAGE_EMBEDDING = IMAGE_EMBEDDING[0]
            break
        except Exception as e:
            print(f"Unable to get embeddings for image {FILENAME}: {e}")
            print(response)
            if r == MAX_RETRIES - 1:
                print(f"attempt {r} failed, reached retry limit")
                request_failed = True
            else:
                print(f"attempt {r} failed, retrying")

    # add embedding to index
    if IMAGE_EMBEDDING:
        add_data_request = {
            "value": [
                {
                    "id": str(ID),
                    "filename": FILENAME,
                    "imagetype": image_type,  # Added imagetype field
                    "imageEmbeddings": IMAGE_EMBEDDING,
                    "@search.action": "upload",
                }
            ]
        }
        response = requests.post(
            ADD_DATA_REQUEST_URL,
            json=add_data_request,
            headers={"api-key": SEARCH_KEY},
        )
        print(response.json())