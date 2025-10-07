/**
 * Camera capture component with live video feed
 */
import React, { useEffect, useState, useRef } from 'react';
import { Camera, CameraOff, RefreshCw } from 'lucide-react';
import { useCamera } from '../hooks/useCamera';
import config from '../config';

export const CameraCapture = ({ onFrameCapture, isCapturing, onCameraStatusChange, children }) => {
  const {
    videoRef,
    isActive,
    error,
    devices,
    selectedDeviceId,
    startCamera,
    stopCamera,
    switchDevice,
    getDevices,
    captureFrame,
  } = useCamera();
  
  const [showDeviceSelect, setShowDeviceSelect] = useState(false);
  const captureIntervalRef = useRef(null);

  useEffect(() => {
    getDevices();
  }, [getDevices]);

  // Notify parent of camera status changes
  useEffect(() => {
    console.log('📷 CameraCapture: isActive changed to:', isActive);
    if (onCameraStatusChange) {
      console.log('📷 CameraCapture: calling onCameraStatusChange with:', isActive);
      onCameraStatusChange(isActive);
    } else {
      console.log('📷 CameraCapture: onCameraStatusChange callback not provided');
    }
  }, [isActive, onCameraStatusChange]);

  // Handle frame capture when isCapturing changes
  useEffect(() => {
    if (isCapturing && isActive && onFrameCapture) {
      console.log('🎬 Starting frame capture in CameraCapture...');
      const interval = 1000 / config.defaultFps; // Use default FPS
      console.log(`⏰ Frame capture interval: ${interval}ms (FPS: ${config.defaultFps})`);
      
      captureIntervalRef.current = setInterval(async () => {
        try {
          console.log('📸 Capturing frame...');
          const frameBlob = await captureFrame(config.defaultQuality);
          
          if (frameBlob && onFrameCapture) {
            console.log(`✅ Frame captured (${frameBlob.size} bytes), sending to callback...`);
            onFrameCapture(frameBlob);
          } else {
            console.log('❌ Frame capture failed - no blob returned');
          }
        } catch (error) {
          console.error('❌ Frame capture error:', error);
        }
      }, interval);
      
      return () => {
        if (captureIntervalRef.current) {
          console.log('🛑 Stopping frame capture in CameraCapture...');
          clearInterval(captureIntervalRef.current);
          captureIntervalRef.current = null;
        }
      };
    } else if (!isCapturing && captureIntervalRef.current) {
      console.log('🛑 Stopping frame capture in CameraCapture (isCapturing changed)...');
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
  }, [isCapturing, isActive, onFrameCapture, captureFrame]);

  const handleToggleCamera = async () => {
    if (isActive) {
      stopCamera();
    } else {
      await startCamera(selectedDeviceId);
    }
  };

  const handleDeviceChange = (e) => {
    const deviceId = e.target.value;
    switchDevice(deviceId);
  };

  return (
    <div className="card relative">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Camera className="w-5 h-5" />
          Camera Feed
        </h2>
        
        <div className="flex items-center gap-2">
          {/* Device selector */}
          {devices.length > 1 && (
            <div className="relative">
              <button
                onClick={() => setShowDeviceSelect(!showDeviceSelect)}
                className="btn-secondary text-sm py-1"
                title="Select camera"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
              
              {showDeviceSelect && (
                <div className="absolute right-0 mt-2 w-64 bg-white border border-gray-200 rounded-lg shadow-lg z-10">
                  <div className="p-2">
                    <label className="block text-sm font-medium mb-2">
                      Select Camera
                    </label>
                    <select
                      value={selectedDeviceId || ''}
                      onChange={handleDeviceChange}
                      className="input-field text-sm"
                    >
                      {devices.map((device) => (
                        <option key={device.deviceId} value={device.deviceId}>
                          {device.label || `Camera ${device.deviceId.slice(0, 8)}`}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              )}
            </div>
          )}
          
          {/* Start/Stop button */}
          <button
            onClick={handleToggleCamera}
            className={isActive ? 'btn-danger' : 'btn-primary'}
          >
            {isActive ? (
              <>
                <CameraOff className="w-4 h-4 mr-2" />
                Stop
              </>
            ) : (
              <>
                <Camera className="w-4 h-4 mr-2" />
                Start
              </>
            )}
          </button>
        </div>
      </div>

      {/* Video container */}
      <div className="relative bg-black rounded-lg overflow-hidden" style={{ aspectRatio: '4/3' }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-contain"
        />
        
        {/* Overlay children (predictions, bounding boxes, etc.) */}
        {isActive && children}
        
        {/* Status indicators */}
        {!isActive && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-gray-400">
              <CameraOff className="w-16 h-16 mx-auto mb-2" />
              <p>Camera is off</p>
              <p className="text-sm">Click Start to begin</p>
            </div>
          </div>
        )}
        
        {isCapturing && isActive && (
          <div className="absolute top-4 right-4 flex items-center gap-2 bg-red-600 text-white px-3 py-1 rounded-full text-sm recording-indicator">
            <div className="w-2 h-2 bg-white rounded-full"></div>
            Recording
          </div>
        )}
      </div>

      {/* Error message */}
      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Info message */}
      {!isActive && !error && (
        <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg text-blue-700 text-sm">
          <strong>Tip:</strong> Allow camera access when prompted. Ensure your microscope camera is connected.
        </div>
      )}
    </div>
  );
};

export default CameraCapture;
