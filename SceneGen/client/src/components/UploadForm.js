import React, { useState } from 'react';
import './UploadForm.css';

function UploadForm({ onSubmit }) {
  const [videoFile, setVideoFile] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [text, setText] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleVideoChange = (e) => {
    if (e.target.files[0]) {
      setVideoFile(e.target.files[0]);
    }
  };

  const handleAudioChange = (e) => {
    if (e.target.files[0]) {
      setAudioFile(e.target.files[0]);
    }
  };

  const handleTextChange = (e) => {
    setText(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!videoFile || !text.trim()) {
      alert('Please provide a video and text description.');
      return;
    }
    
    const formData = new FormData();
    formData.append('video', videoFile);
    if (audioFile) {
      formData.append('audio', audioFile);
    }
    formData.append('text', text);
    
    setIsSubmitting(true);
    
    try {
      await onSubmit(formData);
    } catch (error) {
      console.error('Submission error:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="upload-form-container">
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="video-upload">Video File:</label>
          <input
            type="file"
            id="video-upload"
            accept="video/*"
            onChange={handleVideoChange}
            required
          />
          {videoFile && (
            <div className="file-info">
              <span>Selected: {videoFile.name}</span>
            </div>
          )}
        </div>

        <div className="form-group">
          <label htmlFor="audio-upload">Audio File (optional):</label>
          <input
            type="file"
            id="audio-upload"
            accept="audio/*"
            onChange={handleAudioChange}
          />
          {audioFile && (
            <div className="file-info">
              <span>Selected: {audioFile.name}</span>
            </div>
          )}
        </div>

        <div className="form-group">
          <label htmlFor="text-description">Scene Description:</label>
          <textarea
            id="text-description"
            value={text}
            onChange={handleTextChange}
            placeholder="Describe the scene..."
            rows={4}
            required
          />
        </div>

        <button 
          type="submit" 
          className="submit-button"
          disabled={isSubmitting || !videoFile || !text.trim()}
        >
          {isSubmitting ? 'Processing...' : 'Generate 3D Scene'}
        </button>
      </form>
    </div>
  );
}

export default UploadForm;