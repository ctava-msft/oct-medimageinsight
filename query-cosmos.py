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
import matplotlib.pyplot as plt
import numpy as np
from azure.cosmos import CosmosClient, exceptions
from sklearn.metrics.pairwise import cosine_similarity

from PIL import Image

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
    "COSMOSDB_ENDPOINT",
]

for var in required_vars:
    if not os.getenv(var):
        logger.error(f"Missing required environment variable: {var}")
        raise ValueError(f"Missing required environment variable: {var}")

cosmosdb_endpoint = os.getenv("COSMOSDB_ENDPOINT")
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WORKSPACE_NAME")
endpoint_name=os.getenv("ENDPOINT_NAME")
deployment_name=os.getenv("DEPLOYMENT_NAME")

def run_query(query):
    logger.info(f"Running query: {query}")

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


    TEXT_QUERY = query
    K = 4  # number of results to retrieve
    _REQUEST_FILE_NAME = "request.json"

    def make_request_text(text_sample):
        request_json = {
            "input_data": {
                "columns": ["image", "text"],
                "data": [["", text_sample]],
            }
        }

        with open(_REQUEST_FILE_NAME, "wt") as f:
            json.dump(request_json, f)


    make_request_text(TEXT_QUERY)
    response = workspace_ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        request_file=_REQUEST_FILE_NAME,
    )
    response = json.loads(response)
    QUERY_TEXT_EMBEDDING = response[0]["text_features"]
    QUERY_TEXT_EMBEDDING = np.squeeze(QUERY_TEXT_EMBEDDING)  # Ensure it's 1D

    # Enhanced logging for embedding
    logger.debug(f"Query Text Embedding: {QUERY_TEXT_EMBEDDING}")

    # Initialize Cosmos DB client
    client = CosmosClient(cosmosdb_endpoint, credential=credential, consistency_level="Session")
    database = client.get_database_client('cosmosoct')  # Use your database name
    container = database.get_container_client('products')  # Use your container name

    # Fetch items from Cosmos DB
    items = list(container.read_all_items())
    stored_embeddings = [np.squeeze(item['embedding']) for item in items]  # Ensure each embedding is 1D
    filenames = [item.get('filename', 'unknown') for item in items]  # Adjust as needed

    # #neighbors = response_json["value"]
    # K1, K2 = 3, 4

    # def make_pil_image(image_path):
    #     pil_image = Image.open(image_path)
    #     return pil_image

    # _, axes = plt.subplots(nrows=K1 + 1, ncols=K2, figsize=(64, 64))
    # for i in range(K1 + 1):
    #     for j in range(K2):
    #         axes[i, j].axis("off")

    # i, j = 0, 0

    # for neighbor in items:
    #     pil_image = make_pil_image(neighbor["filename"])
    #     axes[i, j].imshow(np.asarray(pil_image), aspect="auto")
    #     axes[i, j].text(1, 1, "{:.4f}".format(neighbor["@search.score"]), fontsize=32)

    #     j += 1
    #     if j == K2:
    #         i += 1
    #         j = 0

    # plt.show()


    # Compute cosine similarity
    similarities = cosine_similarity([QUERY_TEXT_EMBEDDING], stored_embeddings)[0]

    # Get top K results
    K = 4
    top_k_indices = similarities.argsort()[-K:][::-1]
    top_k_filenames = [filenames[i] for i in top_k_indices]
    top_k_scores = [similarities[i] for i in top_k_indices]

    # Plot results
    _, axes = plt.subplots(nrows=1, ncols=K, figsize=(16, 4))
    for ax in axes:
        ax.axis("off")

    for idx, (filename, score) in enumerate(zip(top_k_filenames, top_k_scores)):
        pil_image = Image.open(filename)
        axes[idx].imshow(np.asarray(pil_image), aspect="auto")
        axes[idx].set_title(f"Score: {score:.4f}", fontsize=12)

    plt.show()

def main():
    logger.info("Starting query script execution.")
    print("Ready to accept your question.")
    query = input("What type of images: ")
    run_query(query)

if __name__ == "__main__":
    main()