# Introduction

This repo holds data, infrastructure and python scripts to ingest and query for images using the medimageinsight embeddings model.

# Motivation

Embeddings models provide a robust way to capture the semantic information of images, enabling efficient and meaningful indexing. By converting images into high-dimensional vectors, these models facilitate similarity searches, clustering, and classification tasks. The potential benefits include improved retrieval accuracy, reduced computational overhead, and the ability to uncover hidden patterns within image datasets.

# MedImageInsight

<<<<<<< HEAD
[MedicalImageParse Model](https://ai.azure.com/explore/models/MedImageInsight/version/5/registry/azureml)
  
[MedicalImageParse Paper](https://arxiv.org/pdf/2410.06542)
=======
The specific embeddings model used in this repo. can be found here:
![MedImageInsight](https://aka.ms/medimageinsightpaper)
>>>>>>> 37f4bc0 (chg: updated readme.md and scripts)

# Setup Azure Infrastructure

Issue the following commands: 
```
azd login
azd up
```

Then issue the following command to grant a role:
az cosmosdb sql role assignment create --account-name "cosmos-<your_cosmos_account_name>" --resource-group "<your_resource_group>" --scope "/" --principal-id $(az ad signed-in-user show --query id -o tsv) --role-definition-id "00000000-0000-0000-0000-000000000002"

# Setup local Pyhton Environment

Run the following commands to setup a python virtual env.

```
python -m venv .venv
[windows].venv\Scripts\activate
[linux]source .venv/bin/activate
pip install -r requirements.txt
```

# Ingestion and Querying

to ingest files into cosmosdb using embeddings model execute the following command:

<<<<<<< HEAD
az cosmosdb sql role assignment create --account-name "cosmos-khifsnz7gfujg" --resource-group "cosmodb-lab" --scope "/" --principal-id $(az ad signed-in-user show --query id -o tsv) --role-definition-id "00000000-0000-0000-0000-000000000002"
=======
`python ingest.py`

to query files execute the following command:

`python ingest.py`
>>>>>>> 37f4bc0 (chg: updated readme.md and scripts)
