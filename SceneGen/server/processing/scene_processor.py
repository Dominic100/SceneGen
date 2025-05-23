import os
import json
import numpy as np
import hashlib

def generate_triplets(frames, audio_clips, text, session_dir):
    """
    Generate triplets from extracted frames and audio
    
    Args:
        frames (list): List of frame dictionaries
        audio_clips (list): List of audio clip dictionaries
        text (str): Text description
        session_dir (str): Session directory path
        
    Returns:
        list: List of triplet dictionaries
    """
    triplets = []
    
    # Match frames with closest audio clip by timestamp
    for frame in frames:
        # Find closest audio clip
        closest_audio = audio_clips[0]
        min_diff = abs(frame['timestamp'] - audio_clips[0]['startTime'])
        
        for i in range(1, len(audio_clips)):
            diff = abs(frame['timestamp'] - audio_clips[i]['startTime'])
            if diff < min_diff:
                min_diff = diff
                closest_audio = audio_clips[i]
        
        # Create triplet
        triplets.append({
            'id': f"triplet_{frame['id']}",
            'framePath': frame['path'],
            'audioPath': closest_audio['path'],
            'text': text,
            'timestamp': frame['timestamp']
        })
    
    # Save triplets
    triplets_path = os.path.join(session_dir, 'triplets.json')
    with open(triplets_path, 'w') as f:
        json.dump(triplets, f, indent=2)
    
    return triplets

def build_temporal_windows(features, session_dir, window_size=5, stride=1):
    """
    Build temporal windows from features
    
    Args:
        features (list): List of feature dictionaries
        session_dir (str): Session directory path
        window_size (int): Size of each window
        stride (int): Stride for window creation
        
    Returns:
        list: List of window dictionaries
    """
    # Sort features by timestamp
    features.sort(key=lambda x: x['timestamp'])
    
    windows = []
    window_id = 1
    
    # Create sliding windows
    for i in range(0, len(features) - window_size + 1, stride):
        window_features = features[i:i + window_size]
        
        windows.append({
            'scene_id': f"scene_{str(window_id).zfill(3)}",
            'entries': window_features
        })
        
        window_id += 1
    
    # Save windows
    windows_path = os.path.join(session_dir, 'scene_windows.json')
    with open(windows_path, 'w') as f:
        json.dump(windows, f, indent=2)
    
    return windows

def generate_scene_triplets(windows, session_dir):
    """
    Generate scene triplets for contrastive learning
    
    Args:
        windows (list): List of window dictionaries
        session_dir (str): Session directory path
        
    Returns:
        list: List of scene triplet dictionaries
    """
    triplets = []
    
    # For each window, create a triplet
    for idx, window in enumerate(windows):
        # Anchor is the current window
        anchor = window
        
        # Positive is an adjacent window (if available)
        pos_options = []
        if idx > 0:
            pos_options.append(windows[idx - 1])
        if idx < len(windows) - 1:
            pos_options.append(windows[idx + 1])
        
        if pos_options:
            positive = np.random.choice(pos_options)
        else:
            positive = anchor  # Fallback to self
        
        # Negative is a random window far from anchor
        far_indices = [i for i in range(len(windows)) if abs(i - idx) > 3]
        
        if far_indices:
            neg_idx = np.random.choice(far_indices)
            negative = windows[neg_idx]
        else:
            # If not enough windows, just pick random one
            negative = windows[np.random.randint(0, len(windows))]
        
        triplets.append({
            'id': f"scene_triplet_{str(idx + 1).zfill(3)}",
            'anchor': anchor,
            'positive': positive,
            'negative': negative
        })
    
    # Save scene triplets
    triplets_path = os.path.join(session_dir, 'scene_triplets.json')
    with open(triplets_path, 'w') as f:
        json.dump(triplets, f, indent=2)
    
    return triplets

def generate_scene_embeddings(scene_triplets, session_dir):
    """
    Generate fused scene embeddings
    
    Args:
        scene_triplets (list): List of scene triplet dictionaries
        session_dir (str): Session directory path
        
    Returns:
        list: List of scene embedding dictionaries
    """
    embeddings = []
    
    # For each triplet's anchor, generate an embedding
    for idx, triplet in enumerate(scene_triplets):
        scene = triplet['anchor']
        
        # Here we create consistent random embeddings based on scene ID
        seed = hash_string(scene['scene_id'])
        embedding_features = create_seeded_random_features(512, seed)
        
        embeddings.append({
            'scene_id': scene['scene_id'],
            'features': embedding_features.tolist()
        })
    
    # Save embeddings
    embeddings_path = os.path.join(session_dir, 'scene_embeddings.json')
    with open(embeddings_path, 'w') as f:
        json.dump(embeddings, f, indent=2)
    
    return embeddings

# Helper functions
def hash_string(s):
    """Hash a string to an integer"""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (10 ** 8)

def create_seeded_random_features(dim, seed):
    """Create random features with a seed"""
    np.random.seed(seed)
    return np.random.uniform(-1, 1, dim).astype(np.float32)