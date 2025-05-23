import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import './ResultViewer.css';

function ResultViewer({ sessionId, onDownload }) {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Initialize THREE.js scene
    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.z = 5;
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);

    // Load PLY model - this is fake since we don't actually have the model yet
    // We'll just create a placeholder mesh for visualization
    createPlaceholderMesh();

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Handle resize
    const handleResize = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      container.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  const createPlaceholderMesh = () => {
    const scene = sceneRef.current;
    
    // Create a point cloud-like geometry
    const geometry = new THREE.BufferGeometry();
    const numPoints = 5000;
    const positions = new Float32Array(numPoints * 3);
    const colors = new Float32Array(numPoints * 3);
    
    for (let i = 0; i < numPoints; i++) {
      const i3 = i * 3;
      // Position in a sphere
      const radius = 2 + Math.random() * 1;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      
      positions[i3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i3 + 2] = radius * Math.cos(phi);
      
      // Color based on position
      colors[i3] = 0.5 + positions[i3] / 5;
      colors[i3 + 1] = 0.5 + positions[i3 + 1] / 5;
      colors[i3 + 2] = 0.5 + positions[i3 + 2] / 5;
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const material = new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true
    });
    
    const points = new THREE.Points(geometry, material);
    scene.add(points);
  };

  return (
    <div className="result-viewer-container">
      <h2>3D Scene Generated Successfully!</h2>
      <div className="result-viewer" ref={containerRef}></div>
      <div className="result-controls">
        <button onClick={onDownload} className="download-button">
          Download 3D Model
        </button>
        <p className="result-info">
          You can rotate the model using your mouse and zoom with the scroll wheel.
        </p>
      </div>
    </div>
  );
}

export default ResultViewer;