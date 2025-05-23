# Code to generate triplet alignment visualization
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import json
import cv2
import librosa
import librosa.display
from PIL import Image
import os

# Create output directory if it doesn't exist
os.makedirs('fig2', exist_ok=True)

# Increase font size globally
plt.rcParams.update({'font.size': 14})

# Load triplet data
with open(r"C:\Aneesh\EDI VI\data\processed\triplets.json", "r") as f:
    triplets = json.load(f)

# Function to plot a single triplet
def plot_triplet(triplet, idx):
    fig = plt.figure(figsize=(10, 12))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 0.7, 1.3])
    
    # Load image
    img = cv2.imread(triplet["frame_path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load audio
    audio, sr = librosa.load(triplet["audio_path"])
    
    # Plot image
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img)
    ax1.set_title("Video Frame", fontsize=16)
    ax1.axis('off')
    
    # Plot audio waveform
    ax2 = fig.add_subplot(gs[1])
    librosa.display.waveshow(audio, sr=sr, ax=ax2)
    ax2.set_title("Audio Waveform", fontsize=16)
    ax2.set_xlabel("Time (s)", fontsize=14)
    ax2.set_ylabel("Amplitude", fontsize=14)
    
    # Plot audio spectrogram
    ax3 = fig.add_subplot(gs[2])
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax3)
    ax3.set_title(f"Caption: {triplet['caption']}", fontsize=16)
    ax3.set_xlabel("Time (s)", fontsize=14)
    ax3.set_ylabel("Frequency (Hz)", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'fig2/triplet_example_{idx}.png', dpi=300)
    plt.close()

# Plot a few examples
for i in range(min(3, len(triplets))):
    plot_triplet(triplets[i], i)

print("Visualization complete. Images saved to fig2/ directory.")