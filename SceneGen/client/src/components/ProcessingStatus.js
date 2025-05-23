import React from 'react';
import './ProcessingStatus.css';

function ProcessingStatus({ status }) {
  if (!status) {
    return (
      <div className="processing-status">
        <h2>Initializing Processing...</h2>
        <div className="loading-spinner"></div>
      </div>
    );
  }

  const stageInfo = {
    initializing: {
      title: 'Initializing',
      description: 'Setting up processing environment...'
    },
    extracting_frames: {
      title: 'Extracting Video Frames',
      description: 'Analyzing video and extracting key frames...'
    },
    extracting_audio: {
      title: 'Extracting Audio Segments',
      description: 'Processing audio and extracting relevant clips...'
    },
    generating_triplets: {
      title: 'Creating Multimodal Triplets',
      description: 'Aligning video frames with audio and text...'
    },
    extracting_features: {
      title: 'Extracting Deep Features',
      description: 'Applying neural networks to extract meaningful features from all modalities...'
    },
    building_windows: {
      title: 'Building Temporal Windows',
      description: 'Creating scene-level representations across time...'
    },
    scene_triplets: {
      title: 'Generating Scene Triplets',
      description: 'Creating contrastive examples for scene understanding...'
    },
    fusing_embeddings: {
      title: 'Fusing Modalities',
      description: 'Combining visual, audio, and textual information into unified scene embeddings...'
    },
    constructing_3d: {
      title: 'Constructing 3D Scene',
      description: 'Transforming multimodal embeddings into 3D point cloud representation...'
    },
    completed: {
      title: 'Processing Complete',
      description: 'Your 3D scene is ready to view and download!'
    },
    error: {
      title: 'Processing Error',
      description: 'An error occurred during processing.'
    }
  };

  const currentStage = stageInfo[status.stage] || {
    title: 'Processing',
    description: 'Working on your request...'
  };

  return (
    <div className="processing-status">
      <h2>{currentStage.title}</h2>
      <p>{status.message || currentStage.description}</p>

      <div className="progress-container">
        <div 
          className="progress-bar" 
          style={{ width: `${status.progress}%` }}
        ></div>
        <span className="progress-text">{status.progress}%</span>
      </div>

      {status.stage === 'constructing_3d' && (
        <div className="time-estimate">
          <p>This step takes approximately 15 minutes to complete.</p>
          <p>Our advanced neural network is generating a detailed 3D model from your inputs.</p>
        </div>
      )}

      {status.error && (
        <div className="error-message">
          <p>Error: {status.message}</p>
        </div>
      )}
    </div>
  );
}

export default ProcessingStatus;