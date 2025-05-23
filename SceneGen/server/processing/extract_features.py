import os
import numpy as np
import hashlib
import librosa
import torch
import open_clip
from PIL import Image
from panns_inference import AudioTagging

def extract_all_features(triplets, session_dir):
    """
    Extract features from all modalities in the triplets
    
    Args:
        triplets (list): List of triplet dictionaries
        session_dir (str): Session directory path
        
    Returns:
        list: List of feature dictionaries
    """
    features_dir = os.path.join(session_dir, 'features')
    image_feature_dir = os.path.join(features_dir, 'image')
    audio_feature_dir = os.path.join(features_dir, 'audio')
    text_feature_dir = os.path.join(features_dir, 'text')
    
    # Create directories
    for dir_path in [image_feature_dir, audio_feature_dir, text_feature_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Initialize models
    # Check if CUDA is available for PyTorch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize CLIP model for image and text
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', 
        pretrained='laion2b_s34b_b79k'
    )
    clip_model = clip_model.to(device).eval()
    
    # Initialize PANNs model for audio
    try:
        audio_model = AudioTagging(checkpoint_path=None, device=device)
    except Exception as e:
        print(f"Warning: Could not initialize PANNs model with CUDA: {e}")
        audio_model = AudioTagging(checkpoint_path=None, device='cpu')
    
    feature_results = []
    
    for index, triplet in enumerate(triplets):
        # Generate unique ID for this triplet
        triplet_id = f"triplet_{str(index + 1).zfill(3)}"
        
        # Extract image features
        image_features = extract_image_features(
            triplet['framePath'], 
            clip_model, 
            clip_preprocess, 
            device
        )
        image_feature_path = os.path.join(image_feature_dir, f"{triplet_id}_image.npy")
        np.save(image_feature_path, image_features)
        
        # Extract audio features
        audio_features = extract_audio_features(
            triplet['audioPath'], 
            audio_model
        )
        audio_feature_path = os.path.join(audio_feature_dir, f"{triplet_id}_audio.npy")
        np.save(audio_feature_path, audio_features)
        
        # Extract text features
        text_features = extract_text_features(
            triplet['text'], 
            clip_model, 
            device
        )
        text_feature_path = os.path.join(text_feature_dir, f"{triplet_id}_text.npy")
        np.save(text_feature_path, text_features)
        
        feature_results.append({
            'id': triplet_id,
            'imageFeaturePath': image_feature_path,
            'audioFeaturePath': audio_feature_path,
            'textFeaturePath': text_feature_path,
            'timestamp': triplet['timestamp']
        })
    
    return feature_results

def extract_image_features(image_path, model, preprocess, device):
    """
    CLIP image feature extraction
    
    Args:
        image_path (str): Path to the image file
        model: CLIP model
        preprocess: CLIP preprocessing function
        device: Torch device
        
    Returns:
        ndarray: Feature vector
    """
    try:
        # Load and preprocess image
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = model.encode_image(image)
            # Normalize features
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy().squeeze()
    except Exception as e:
        print(f"Error extracting image features from {image_path}: {e}")
        # Fallback to random features if extraction fails
        seed = hash_string(image_path)
        return create_seeded_random_features(512, seed)

def extract_audio_features(audio_path, model):
    """
    PANNs audio feature extraction
    
    Args:
        audio_path (str): Path to the audio file
        model: PANNs AudioTagging model
        
    Returns:
        ndarray: Feature vector
    """
    try:
        # Load audio file with librosa
        waveform, sr = librosa.load(audio_path, sr=32000, mono=True)
        audio_tensor = waveform[None, :]  # shape: (1, num_samples)
        
        # Extract features
        _, embedding = model.inference(audio_tensor)
        
        return embedding.squeeze()
    except Exception as e:
        print(f"Error extracting audio features from {audio_path}: {e}")
        # Fallback to random features if extraction fails
        seed = hash_string(audio_path)
        return create_seeded_random_features(2048, seed)

def extract_text_features(text, model, device):
    """
    CLIP text feature extraction
    
    Args:
        text (str): Text to extract features from
        model: CLIP model
        device: Torch device
        
    Returns:
        ndarray: Feature vector
    """
    try:
        # Skip if text is empty
        if not text.strip():
            raise ValueError("Empty text")
            
        # Tokenize and extract features
        tokenized_text = open_clip.tokenize([text]).to(device)
        with torch.no_grad():
            features = model.encode_text(tokenized_text)
            # Normalize features
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy().squeeze()
    except Exception as e:
        print(f"Error extracting text features from '{text}': {e}")
        # Fallback to random features if extraction fails
        seed = hash_string(text)
        return create_seeded_random_features(512, seed)

# Helper to create seeded random features (consistent for same input)
def create_seeded_random_features(dim, seed):
    """
    Create random features with a consistent seed
    
    Args:
        dim (int): Dimension of feature vector
        seed (int): Random seed
        
    Returns:
        ndarray: Feature vector
    """
    np.random.seed(seed)
    return np.random.uniform(-1, 1, dim).astype(np.float32)

# Simple string hash function for seeding
def hash_string(s):
    """
    Create a hash from a string
    
    Args:
        s (str): String to hash
        
    Returns:
        int: Hash value
    """
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (10 ** 8)