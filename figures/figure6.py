# Code to generate comparison visualization between branches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample data - replace with actual results
methods = ['Concat+MLP', 'Attention', 'Cross-Attention', 'Tensor Fusion', 'Scene Transformer']
precision = [0.71, 0.75, 0.77, 0.79, 0.83]
recall = [0.85, 0.87, 0.88, 0.90, 0.92]
mrr = [0.73, 0.76, 0.78, 0.80, 0.83]
inference_time = [0.84, 1.17, 1.23, 3.45, 2.78]  # ms

# Normalize inference time for visualization
max_time = max(inference_time)
norm_time = [t/max_time for t in inference_time]

# Create figure with two y-axes
fig, ax1 = plt.figure(figsize=(12, 6)), plt.gca()
ax2 = ax1.twinx()

# Plot bars for precision and recall
x = np.arange(len(methods))
width = 0.25
ax1.bar(x - width, precision, width, label='Precision@3', color='skyblue', edgecolor='navy')
ax1.bar(x, recall, width, label='Recall@3', color='lightgreen', edgecolor='darkgreen')
ax1.bar(x + width, mrr, width, label='MRR', color='salmon', edgecolor='darkred')

# Plot line for inference time
ax2.plot(x, norm_time, 'o-', color='purple', linewidth=2, label='Relative Inference Time')

# Add labels and legend
ax1.set_xlabel('Fusion Method')
ax1.set_ylabel('Performance Metric')
ax2.set_ylabel('Relative Inference Time')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=45, ha='right')
ax1.set_ylim(0, 1)
ax2.set_ylim(0, 1.2)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('Comparison of Fusion Methods: Performance vs. Efficiency')
plt.tight_layout()
plt.savefig('fig6.png')
