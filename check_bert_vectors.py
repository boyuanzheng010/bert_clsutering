import json
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans


def read_embeddings(embeds_file):
    with open(embeds_file) as fp:
        embeds_list = json.loads(fp.read())
    arr = np.array(embeds_list)
    return arr

# Load BERT Vectors
print("Start Loading Vectors")
vectors = read_embeddings('./bert_vectors.txt')
print("Finish Vector Loading")

# Perform K-Means Clustering
print("Build KMeans Clustering Model")
model = KMeans(n_clusters=4, max_iter=100, init="k-means++")
print("Fiting vectors to KMeans Model")
model.fit(vectors)

