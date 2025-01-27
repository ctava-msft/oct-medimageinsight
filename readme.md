# Introduction

This repo holds data and AI scripts to index images using the medimageinsight embeddings model
and to create segmentation masks and store them too.

# Motivation

AI engineers could be motivated to use embeddings models.

Embeddings models provide a robust way to capture the semantic information of images, enabling efficient and meaningful indexing. By converting images into high-dimensional vectors, these models facilitate similarity searches, clustering, and classification tasks. The potential benefits include improved retrieval accuracy, reduced computational overhead, and the ability to uncover hidden patterns within large image datasets.

# Differences

The compute_diff function calculates the squared Maximum Mean Discrepancy (MMD) between two datasets, x and y, using the Gaussian (RBF) kernel. This metric measures the difference between the distributions of x and y.
Here's how it works:
Convert inputs to NumPy arrays:
x = np.array(x)
y = np.array(y)

Compute Gram matrices (dot products):
For x:

xx = np.dot(x, x.T)

For y:

yy = np.dot(y, y.T)

Between x and y:

xy = np.dot(x, y.T)

Extract squared norms (diagonal elements):
For x:

rx = np.diag(xx)

For y:

ry = np.diag(yy)
Compute kernel matrices using the Gaussian kernel:
Within x:
k = np.exp(-0.5 * (rx[:, None] + rx[None, :] - 2 * xx))
Within y:
l = np.exp(-0.5 * (ry[:, None] + ry[None, :] - 2 * yy))
Between x and y:
m = np.exp(-0.5 * (rx[:, None] + ry[None, :] - 2 * xy))

Compute the squared MMD (difference score):
diff_score = np.mean(k) + np.mean(l) - 2 * np.mean(m)
Impactful Operations:
Kernel Computation: The most significant impact on the difference score comes from the computation of the kernel matrices k, l, and m. Specifically:
The exponentials of the negative squared Euclidean distances amplify differences between data points.
When data points in x and y are similar, the kernel values are close to 1.
When data points are different, the kernel values decrease exponentially toward 0.
Averaging: The means of the kernel matrices (np.mean(k), np.mean(l), np.mean(m)) aggregate these differences across all pairs of points.
Range of Possible Values:
Minimum Value: 0

Occurs when x and y are identical, resulting in np.mean(k) + np.mean(l) = 2 * np.mean(m).
Maximum Value: Approaches 2

Occurs when x and y are completely different.
In this case, np.mean(k) and np.mean(l) are close to 1 (since data points within x and within y are compared with themselves), and np.mean(m) approaches 0 (since x and y are dissimilar).
Overall Range: The difference score ranges from 0 to 2.
Interpretation:
Difference Score ≈ 0: x and y have very similar distributions.
Higher Difference Score: Indicates greater disparity between the distributions of x and y.
Maximum Difference Score: Theoretically 2, signifying completely different distributions.

# Setup environment

Run the following commands to setup a python virtual env.

```
python -m venv .venv
pip install virtualenv
.venv\Scripts\activate
[linux]source .venv/bin/activate
pip install -r requirements.txt
```

# Setup Infra

Issue the following commands to 
azd login
azd up

az cosmosdb sql role assignment create --account-name "cosmos-khifsnz7gfujg" --resource-group "cosmodb-lab" --scope "/" --principal-id $(az ad signed-in-user show --query id -o tsv) --role-definition-id "00000000-0000-0000-0000-000000000002"