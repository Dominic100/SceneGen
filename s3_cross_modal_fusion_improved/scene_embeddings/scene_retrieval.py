# scene_retrieval.py

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# CONFIG
EMBEDDINGS_CSV = "scene_embeddings.csv"
OUTPUT_CSV = "retrieval_results.csv"
TOP_K = 5  # Number of neighbors to retrieve

# Load scene embeddings
df = pd.read_csv(EMBEDDINGS_CSV)
scene_ids = df["scene_id"].tolist()
embeddings = df.drop(columns=["scene_id"]).values

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Collect top-K results per scene
results = []
for i, scene_id in tqdm(enumerate(scene_ids), total=len(scene_ids), desc="Retrieving top-K"):
    sims = similarity_matrix[i]
    top_k_indices = np.argsort(sims)[::-1][1:TOP_K+1]  # Skip self-match at index 0

    result = {"query_scene_id": scene_id}
    for rank, idx in enumerate(top_k_indices, start=1):
        result[f"top{rank}_id"] = scene_ids[idx]
        result[f"top{rank}_score"] = float(sims[idx])
    results.append(result)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Top-{TOP_K} retrieval results saved to {OUTPUT_CSV}")
