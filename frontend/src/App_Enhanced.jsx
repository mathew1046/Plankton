/**
 * Enhanced Main Application - Visually Stunning Design
 * Marine Organism Identification System - SIH 2025 PS25043
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Microscope, Play, Square, Camera as CameraIcon, Sparkles, Waves } from 'lucide-react';
import { useWebSocket } from './hooks/useWebSocket';
import { useCamera } from './hooks/useCamera';
import CameraCapture from './components/CameraCapture';
import PredictionOverlay from './components/PredictionOverlay';
import ControlPanel from './components/ControlPanel';
import CountDisplay from './components/CountDisplay';
import LabelCorrection from './components/LabelCorrection';
import ModelStatus from './components/ModelStatus';
import ConnectionStatus from './components/ConnectionStatus';
import config from './config';

function App() {
  // WebSocket state
  const { isConnected, lastPrediction, error: wsError, connect, disconnect, sendFrame, reconnectAttempt } = useWebSocket();
  
  // Camera state
  const { videoRef, isActive: cameraActive, captureFrame } = useCamera();
  
  // Capture state
  const [isCapturing, setIsCapturing] = useState(false);
  const [fps, setFps] = useState(config.defaultFps);
  const [quality, setQuality] = useState(config.defaultQuality);
  const [confidenceThreshold, setConfidenceThreshold] = useState(config.defaultConfidence);
  
  // Counts state
  const [counts, setCounts] = useState({});
  
  // Species list
  const [species, setSpecies] = useState([]);
  
  // Frame data for label correction
  const [lastFrameData, setLastFrameData] = useState(null);
  
  // Capture interval ref
  const captureIntervalRef = useRef(null);

  // Fetch species list on mount
  useEffect(() => {
    const fetchSpecies = async () => {
      try {
        const response = await fetch(`${config.apiUrl}/api/species`);
        const data = await response.json();
        if (data.status === 'success') {
          setSpecies(data.species);
        }
      } catch (error) {
        console.error('Failed to fetch species:', error);
      }
    };
    
    fetchSpecies();
  }, []);

  // Update counts when new prediction arrives
  useEffect(() => {
    if (lastPrediction && lastPrediction.confidence >= confidenceThreshold) {
      const species = lastPrediction.species;
      setCounts(prev => ({
        ...prev,
        [species]: (prev[species] || 0) + 1
      }));
    }
  }, [lastPrediction, confidenceThreshold]);

  // Frame capture loop
  const captureAndSendFrame = useCallback(async () => {
    if (!isConnected || !cameraActive) {
      return;
    }

    try {
      const frameBlob = await captureFrame(quality);
      
      if (frameBlob) {
        setLastFrameData(frameBlob);
        const arrayBuffer = await frameBlob.arrayBuffer();
        sendFrame(arrayBuffer);
      }
    } catch (error) {
      console.error('Failed to capture/send frame:', error);
    }
  }, [isConnected, cameraActive, captureFrame, quality, sendFrame]);

  // Start/stop capture
  const handleToggleCapture = () => {
    if (isCapturing) {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
        captureIntervalRef.current = null;
      }
      disconnect();
      setIsCapturing(false);
    } else {
      if (!cameraActive) {
        alert('Please start the camera first');
        return;
      }
      
      connect();
      setIsCapturing(true);
      
      const interval = 1000 / fps;
      captureIntervalRef.current = setInterval(captureAndSendFrame, interval);
    }
  };

  // Update capture interval when FPS changes
  useEffect(() => {
    if (isCapturing && captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      const interval = 1000 / fps;
      captureIntervalRef.current = setInterval(captureAndSendFrame, interval);
    }
  }, [fps, isCapturing, captureAndSendFrame]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
      }
    };
  }, []);

  const handleResetCounts = () => {
    setCounts({});
  };

  const handleLabelSubmit = (correctedLabel) => {
    console.log('Label corrected to:', correctedLabel);
  };

  const handleSnapshot = async () => {
    if (!lastFrameData || !lastPrediction) {
      return;
    }

    try {
      const formData = new FormData();
      formData.append('file', lastFrameData, 'snapshot.jpg');
      formData.append('frame_id', `snapshot_${Date.now()}`);
      formData.append('prediction', JSON.stringify(lastPrediction));

      const response = await fetch(`${config.apiUrl}/api/snapshot`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        console.log('Snapshot saved');
      }
    } catch (error) {
      console.error('Failed to save snapshot:', error);
    }
  };

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Animated background particles */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-20 w-72 h-72 bg-cyan-500/10 rounded-full blur-3xl particle"></div>
        <div className="absolute top-40 right-32 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl particle" style={{animationDelay: '5s'}}></div>
        <div className="absolute bottom-20 left-1/3 w-80 h-80 bg-purple-500/10 rounded-full blur-3xl particle" style={{animationDelay: '10s'}}></div>
      </div>

      {/* Header with glassmorphism */}
      <header className="relative bg-white/5 backdrop-blur-xl border-b border-white/10 shadow-2xl">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-2xl blur-xl opacity-50"></div>
                <div className="relative bg-gradient-to-br from-cyan-500 to-blue-600 p-3 rounded-2xl shadow-lg">
                  <Microscope className="w-10 h-10 text-white" />
                </div>
              </div>
              <div>
                <h1 className="text-3xl font-bold gradient-text flex items-center gap-2">
                  Marine Organism Identification
                  <Sparkles className="w-6 h-6 text-cyan-400 animate-pulse" />
                </h1>
                <p className="text-sm text-cyan-300/80 flex items-center gap-2 mt-1">
                  <Waves className="w-4 h-4" />
                  SIH 2025 - PS25043: Intelligent Microscopy System
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <ConnectionStatus 
                isConnected={isConnected} 
                error={wsError} 
                reconnectAttempt={reconnectAttempt}
              />
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8 relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Camera and Controls */}
          <div className="lg:col-span-2 space-y-8">
            {/* Camera Feed with enhanced styling */}
            <div className="float-animation">
              <CameraCapture 
                onFrameCapture={captureAndSendFrame}
                isCapturing={isCapturing}
              >
                <PredictionOverlay 
                  prediction={lastPrediction}
                  confidenceThreshold={confidenceThreshold}
                />
              </CameraCapture>
            </div>

            {/* Capture Controls with glassmorphism */}
            <div className="card glow-pulse">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-xl font-bold text-white mb-2 flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></div>
                    Capture Control
                  </h3>
                  <p className="text-cyan-300/70 text-sm">
                    {isCapturing ? '🔴 Streaming frames to AI backend' : '⚪ Ready to start capture'}
                  </p>
                </div>
                
                <div className="flex gap-3">
                  <button
                    onClick={handleToggleCapture}
                    className={isCapturing ? 'btn-danger' : 'btn-primary'}
                    disabled={!cameraActive && !isCapturing}
                  >
                    {isCapturing ? (
                      <>
                        <Square className="w-5 h-5" />
                        Stop Capture
                      </>
                    ) : (
                      <>
                        <Play className="w-5 h-5" />
                        Start Capture
                      </>
                    )}
                  </button>
                  
                  <button
                    onClick={handleSnapshot}
                    className="btn-secondary"
                    disabled={!lastFrameData || !lastPrediction}
                    title="Save snapshot"
                  >
                    <CameraIcon className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>

            {/* Settings */}
            <ControlPanel
              fps={fps}
              setFps={setFps}
              confidenceThreshold={confidenceThreshold}
              setConfidenceThreshold={setConfidenceThreshold}
              quality={quality}
              setQuality={setQuality}
            />
          </div>

          {/* Right Column - Stats and Management */}
          <div className="space-y-8">
            {/* Organism Counts */}
            <div className="float-animation" style={{animationDelay: '1s'}}>
              <CountDisplay 
                counts={counts}
                onReset={handleResetCounts}
              />
            </div>

            {/* Label Correction */}
            <div className="float-animation" style={{animationDelay: '2s'}}>
              <LabelCorrection
                prediction={lastPrediction}
                species={species}
                onSubmit={handleLabelSubmit}
                frameData={lastFrameData}
              />
            </div>

            {/* Model Status */}
            <div className="float-animation" style={{animationDelay: '3s'}}>
              <ModelStatus />
            </div>
          </div>
        </div>
      </main>

      {/* Footer with glassmorphism */}
      <footer className="relative bg-white/5 backdrop-blur-xl border-t border-white/10 mt-16">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between text-sm text-cyan-300/70">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              Built for Smart India Hackathon 2025
            </div>
            <div className="flex items-center gap-2">
              Problem Statement: PS25043
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
