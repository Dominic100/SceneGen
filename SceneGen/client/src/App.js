import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

// Components
import UploadForm from './components/UploadForm';
import ProcessingStatus from './components/ProcessingStatus';
import ResultViewer from './components/ResultViewer';

const API_URL = 'http://localhost:5000/api';

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const statusInterval = useRef(null);

  // Submit handler for the upload form
  const handleSubmit = async (formData) => {
    try {
      setError(null);
      const response = await axios.post(`${API_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      if (response.data.success) {
        setSessionId(response.data.sessionId);
        // Start polling for status
        startStatusPolling(response.data.sessionId);
      }
    } catch (err) {
      console.error('Upload error:', err);
      setError(err.response?.data?.error || 'Upload failed. Please try again.');
    }
  };

  // Start polling for status updates
  const startStatusPolling = (id) => {
    if (statusInterval.current) {
      clearInterval(statusInterval.current);
    }
    
    // Poll every 3 seconds
    statusInterval.current = setInterval(() => {
      fetchStatus(id);
    }, 3000);
  };

  // Fetch current status
  const fetchStatus = async (id) => {
    try {
      const response = await axios.get(`${API_URL}/status/${id}`);
      setStatus(response.data);
      
      // If processing is complete or errored, stop polling
      if (response.data.completed || response.data.error) {
        clearInterval(statusInterval.current);
      }
    } catch (err) {
      console.error('Status fetch error:', err);
      setError('Failed to fetch processing status.');
      clearInterval(statusInterval.current);
    }
  };

  // Download result
  const handleDownload = async () => {
    try {
      window.open(`${API_URL}/result/${sessionId}`, '_blank');
    } catch (err) {
      setError('Failed to download result.');
    }
  };

  // Clean up interval on unmount
  useEffect(() => {
    return () => {
      if (statusInterval.current) {
        clearInterval(statusInterval.current);
      }
    };
  }, []);

  return (
    <div className="app-container">
      <header>
        <h1>Video to 3D Scene Reconstruction</h1>
        <p>Upload a video, audio file, and description to generate a 3D scene</p>
      </header>

      <main>
        {!sessionId ? (
          <UploadForm onSubmit={handleSubmit} />
        ) : (
          <>
            <ProcessingStatus status={status} />
            
            {status?.completed && !status?.error && (
              <ResultViewer 
                sessionId={sessionId} 
                onDownload={handleDownload} 
              />
            )}
          </>
        )}

        {error && (
          <div className="error-message">
            <p>{error}</p>
            <button onClick={() => window.location.reload()}>Start Over</button>
          </div>
        )}
      </main>

      <footer>
        <p>Advanced Multimodal Scene Reconstruction System</p>
      </footer>
    </div>
  );
}

export default App;