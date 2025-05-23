import matplotlib.pyplot as plt
import numpy as np
import json

# Load captions data
with open(r"C:\Aneesh\EDI VI\data\processed\cleaned_captions.json", "r") as f:
    cleaned_captions = json.load(f)

# Count captions per video
video_counts = {}
for entry in cleaned_captions:
    vid = entry["video_id"]
    video_counts[vid] = video_counts.get(vid, 0) + 1

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(list(video_counts.values()), bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Captions per Video')
plt.xlabel('Number of Captions')
plt.ylabel('Number of Videos')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig1.png')
