# Code to generate modality distribution visualization
import matplotlib.pyplot as plt
import json
import numpy as np

# Load multimodal triplets
with open(r"C:\Aneesh\EDI VI\data\processed\multimodal_triplets.json", "r") as f:
    mm_triplets = json.load(f)

# Count modality successes
total = len(mm_triplets)
with_image = sum(1 for t in mm_triplets if 'image_path' in t)
with_audio = sum(1 for t in mm_triplets if 'audio_path' in t)
with_text = sum(1 for t in mm_triplets if 'text_path' in t)
complete = sum(1 for t in mm_triplets if 'image_path' in t and 'audio_path' in t and 'text_path' in t)

# Plot
labels = ['With Image', 'With Audio', 'With Text', 'Complete Triplets']
values = [with_image/total*100, with_audio/total*100, with_text/total*100, complete/total*100]

plt.figure(figsize=(10, 6))
plt.bar(labels, values, color=['skyblue', 'lightgreen', 'salmon', 'purple'])
plt.ylabel('Percentage of Dataset')

# Add title with more space above
plt.title('Modality Availability in Multimodal Dataset', pad=15)

plt.ylim(0, 100)
for i, v in enumerate(values):
    plt.text(i, v+1, f"{v:.1f}%", ha='center')
plt.grid(axis='y', alpha=0.3)
plt.savefig('figure3.png')