# plot_scene_embeddings.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# CONFIG
EMBEDDINGS_CSV = "scene_embeddings.csv"
OUTPUT_PNG = "scene_embedding_tsne.png"
PERPLEXITY = 30  # You can tune this based on number of samples

# Load embeddings
df = pd.read_csv(EMBEDDINGS_CSV)

# Extract scene IDs and high-D embeddings
scene_ids = df["scene_id"].tolist()
X = df.drop(columns=["scene_id"]).values

# Optional: extract video_id from scene_id
video_ids = [sid.split("_scene_")[0] for sid in scene_ids]

# Encode video_id as color label
label_encoder = LabelEncoder()
color_labels = label_encoder.fit_transform(video_ids)

# Apply t-SNE
print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=42)
X_2d = tsne.fit_transform(X)

# Plot
plt.figure(figsize=(10, 8))
palette = sns.color_palette("husl", len(set(color_labels)))
sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=color_labels, palette=palette, legend=False, s=30)
plt.title("t-SNE Visualization of Scene Embeddings")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.tight_layout()
plt.savefig(OUTPUT_PNG)
plt.close()
print(f"Saved t-SNE plot to {OUTPUT_PNG}")
