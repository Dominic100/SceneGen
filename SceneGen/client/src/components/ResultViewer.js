import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader'; // Make sure this is imported

function ResultViewer({ resultUrl }) {
  const mountRef = useRef(null);
  
  useEffect(() => {
    if (!resultUrl) return;
    
    // Scene setup
    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;
    
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);
    
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.z = 5;
    
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    mountRef.current.appendChild(renderer.domElement);
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(0, 1, 1).normalize();
    scene.add(directionalLight);
    
    // Add orbit controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    
    // Add loading indicator
    console.log('Loading 3D model from:', resultUrl);
    const loadingElem = document.createElement('div');
    loadingElem.style.position = 'absolute';
    loadingElem.style.top = '50%';
    loadingElem.style.left = '50%';
    loadingElem.style.transform = 'translate(-50%, -50%)';
    loadingElem.style.color = 'white';
    loadingElem.style.fontSize = '24px';
    loadingElem.textContent = 'Loading 3D Model...';
    mountRef.current.appendChild(loadingElem);
    
    // Load PLY file
    const loader = new PLYLoader();
    loader.load(
      resultUrl,
      (geometry) => {
        console.log('PLY loaded successfully:', geometry);
        // Remove loading indicator
        if (mountRef.current.contains(loadingElem)) {
          mountRef.current.removeChild(loadingElem);
        }
        
        // Center the geometry
        geometry.computeBoundingBox();
        const center = new THREE.Vector3();
        geometry.boundingBox.getCenter(center);
        geometry.center();
        
        // Scale the geometry to fit the view
        geometry.computeBoundingSphere();
        const radius = geometry.boundingSphere.radius;
        const scale = 2.0 / radius;
        geometry.scale(scale, scale, scale);
        
        // Create material with vertex colors
        const material = new THREE.PointsMaterial({
          size: 0.01,
          vertexColors: true
        });
        
        // Create and add point cloud to scene
        const pointCloud = new THREE.Points(geometry, material);
        scene.add(pointCloud);
        
        // Adjust camera position
        camera.position.z = 3;
        
        // Initial render
        renderer.render(scene, camera);
      },
      (xhr) => {
        const percent = (xhr.loaded / xhr.total * 100).toFixed(0);
        loadingElem.textContent = `Loading 3D Model... ${percent}%`;
      },
      (error) => {
        console.error('Error loading PLY:', error);
        loadingElem.textContent = 'Error loading 3D model';
      }
    );
    
    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();
    
    // Handle window resize
    const handleResize = () => {
      const width = mountRef.current.clientWidth;
      const height = mountRef.current.clientHeight;
      
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    window.addEventListener('resize', handleResize);
    
    // Cleanup on unmount
    return () => {
      window.removeEventListener('resize', handleResize);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      if (mountRef.current && mountRef.current.contains(loadingElem)) {
        mountRef.current.removeChild(loadingElem);
      }
    };
  }, [resultUrl]);
  
  return (
    <div 
      ref={mountRef} 
      style={{ 
        width: '100%', 
        height: '500px', 
        position: 'relative',
        backgroundColor: '#222'
      }}
    />
  );
}

export default ResultViewer;