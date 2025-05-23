import os
import json
import uuid
import time
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import processing modules
from processing.extract_frames import extract_random_frames
from processing.extract_audio import extract_random_audio_clips
from processing.extract_features import extract_all_features
from processing.scene_processor import (
    generate_triplets, 
    build_temporal_windows,
    generate_scene_triplets, 
    generate_scene_embeddings
)
from processing.runtime_utils import process_session_data

app = Flask(__name__)
CORS(app)

# Configure upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
SESSION_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sessions')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mp3', 'wav'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SESSION_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_files():
    try:
        # Check if the post request has a video file
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'Video file is required'}), 400
        
        video_file = request.files['video']
        audio_file = request.files.get('audio')  # Use get() so it can be None
        text = request.form.get('text', 'Default description')
        
        # Check if video filename is valid
        if video_file.filename == '':
            return jsonify({'success': False, 'error': 'No video file selected'}), 400
        
        # Check if files are allowed types
        if not allowed_file(video_file.filename):
            return jsonify({'success': False, 'error': 'Video file type not allowed'}), 400
        
        if audio_file and audio_file.filename != '' and not allowed_file(audio_file.filename):
            return jsonify({'success': False, 'error': 'Audio file type not allowed'}), 400
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(SESSION_FOLDER, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save text description
        with open(os.path.join(session_dir, 'description.txt'), 'w') as f:
            f.write(text)
        
        # Save video file with secure filename
        video_filename = secure_filename(f"{int(time.time())}-{video_file.filename}")
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        video_file.save(video_path)
        
        # Save audio file if provided
        audio_path = None
        if audio_file and audio_file.filename != '':
            audio_filename = secure_filename(f"{int(time.time())}-{audio_file.filename}")
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
            audio_file.save(audio_path)
        
        # Start processing in a background thread
        import threading
        thread = threading.Thread(
            target=process_session, 
            args=(session_id, video_path, audio_path, text)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'sessionId': session_id,
            'message': 'Upload successful, processing will begin shortly'
        })
        
    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status/<session_id>', methods=['GET'])
def get_status(session_id):
    status_path = os.path.join(SESSION_FOLDER, session_id, 'status.json')
    
    if os.path.exists(status_path):
        with open(status_path, 'r') as f:
            status = json.load(f)
        return jsonify(status)
    else:
        return jsonify({'error': 'Session not found'}), 404

@app.route('/api/result/<session_id>', methods=['GET'])
def get_result(session_id):
    result_path = os.path.join(SESSION_FOLDER, session_id, 'result.ply')
    
    if os.path.exists(result_path):
        return send_file(result_path, as_attachment=True)
    else:
        return jsonify({'error': 'Result not found'}), 404

def process_session(session_id, video_path, audio_path, text):
    """Process files asynchronously and update status"""
    session_dir = os.path.join(SESSION_FOLDER, session_id)
    status_path = os.path.join(session_dir, 'status.json')
    
    # Create subdirectories
    dirs = ['frames', 'audio', 'features']
    for dir_name in dirs:
        os.makedirs(os.path.join(session_dir, dir_name), exist_ok=True)
    
    # Initialize status
    update_status(status_path, 'initializing', 0, 'Setting up processing environment')
    
    try:
        # Step 1: Extract frames
        update_status(status_path, 'extracting_frames', 5, 'Extracting video frames...')
        frames = extract_random_frames(
            video_path, 
            os.path.join(session_dir, 'frames')
        )
        
        # Step 2: Extract audio clips
        update_status(status_path, 'extracting_audio', 15, 'Extracting audio segments...')
        audio_clips = extract_random_audio_clips(
            audio_path, 
            os.path.join(session_dir, 'audio'),
            video_path=video_path if audio_path is None else None
        )
                
        # Step 3: Generate triplets
        update_status(status_path, 'generating_triplets', 25, 'Creating multimodal triplets...')
        triplets = generate_triplets(
            frames, 
            audio_clips, 
            text, 
            session_dir
        )
        
        # Step 4: Extract features
        update_status(status_path, 'extracting_features', 35, 'Extracting deep features from all modalities...')
        features = extract_all_features(triplets, session_dir)
        
        # Step 5: Build temporal windows
        update_status(status_path, 'building_windows', 50, 'Building temporal scene windows...')
        windows = build_temporal_windows(features, session_dir)
        
        # Step 6: Generate scene triplets
        update_status(status_path, 'scene_triplets', 65, 'Generating scene-level contrastive triplets...')
        scene_triplets = generate_scene_triplets(windows, session_dir)
        
        # Step 7: Generate fused embeddings
        update_status(status_path, 'fusing_embeddings', 75, 'Creating fused scene embeddings...')
        scene_embeddings = generate_scene_embeddings(scene_triplets, session_dir)
        
        # Step 8: Simulate 3D model construction (long process)
        update_status(status_path, 'constructing_3d', 85, 'Constructing 3D scene from multimodal embeddings...')
        process_session_data(session_dir)
        
        # Done!
        update_status(status_path, 'completed', 100, 'Processing complete!', True)
        
    except Exception as e:
        app.logger.error(f"Processing error for session {session_id}: {str(e)}")
        update_status(status_path, 'error', 0, f"Error: {str(e)}", False, True)

def update_status(status_path, stage, progress, message, completed=False, error=False):
    """Update the status file for the current session"""
    status = {
        'stage': stage,
        'progress': progress,
        'message': message,
        'completed': completed,
        'error': error
    }
    with open(status_path, 'w') as f:
        json.dump(status, f)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)