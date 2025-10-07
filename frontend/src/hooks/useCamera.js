/**
 * Custom hook for camera access and frame capture
 */
import { useState, useRef, useCallback, useEffect } from 'react';
import config from '../config';

export const useCamera = () => {
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState(null);
  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState(null);
  
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const canvasRef = useRef(null);

  // Get available camera devices
  const getDevices = useCallback(async () => {
    try {
      const deviceList = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = deviceList.filter(device => device.kind === 'videoinput');
      setDevices(videoDevices);
      
      if (videoDevices.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoDevices[0].deviceId);
      }
      
      return videoDevices;
    } catch (err) {
      console.error('Failed to enumerate devices:', err);
      setError('Failed to get camera devices');
      return [];
    }
  }, [selectedDeviceId]);

  // Start camera stream
  const startCamera = useCallback(async (deviceId = null) => {
    try {
      setError(null);
      
      const constraints = {
        ...config.cameraConstraints,
        video: {
          ...config.cameraConstraints.video,
          ...(deviceId && { deviceId: { exact: deviceId } })
        }
      };
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      
      streamRef.current = stream;
      setIsActive(true);
      
      // Get devices after permission granted
      await getDevices();
      
      return true;
    } catch (err) {
      console.error('Failed to start camera:', err);
      setError(err.message || 'Failed to access camera');
      setIsActive(false);
      return false;
    }
  }, [getDevices]);

  // Stop camera stream
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    
    setIsActive(false);
  }, []);

  // Capture frame as JPEG blob
  const captureFrame = useCallback((quality = config.defaultQuality) => {
    if (!videoRef.current || !isActive) {
      console.log('❌ Capture frame: video ref or not active', { videoRef: !!videoRef.current, isActive });
      return null;
    }

    try {
      const video = videoRef.current;
      console.log('📹 Video element:', {
        videoWidth: video.videoWidth,
        videoHeight: video.videoHeight,
        readyState: video.readyState,
        paused: video.paused,
        ended: video.ended
      });

      // Create canvas if not exists
      if (!canvasRef.current) {
        canvasRef.current = document.createElement('canvas');
        console.log('🖼️ Created new canvas element');
      }

      const canvas = canvasRef.current;

      // Set canvas size to match video
      const width = video.videoWidth || config.canvasWidth;
      const height = video.videoHeight || config.canvasHeight;
      canvas.width = width;
      canvas.height = height;

      console.log('📐 Canvas size:', { width, height });

      // Draw video frame to canvas
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, width, height);

      // Convert to blob
      return new Promise((resolve) => {
        canvas.toBlob(
          (blob) => {
            if (blob) {
              console.log('✅ Canvas toBlob success:', { size: blob.size, type: blob.type });
              resolve(blob);
            } else {
              console.error('❌ Canvas toBlob returned null');
              resolve(null);
            }
          },
          'image/jpeg',
          quality / 100
        );
      });

    } catch (err) {
      console.error('❌ Failed to capture frame:', err);
      return null;
    }
  }, [isActive]);

  // Switch camera device
  const switchDevice = useCallback(async (deviceId) => {
    stopCamera();
    setSelectedDeviceId(deviceId);
    await startCamera(deviceId);
  }, [startCamera, stopCamera]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return {
    videoRef,
    isActive,
    error,
    devices,
    selectedDeviceId,
    startCamera,
    stopCamera,
    captureFrame,
    switchDevice,
    getDevices,
  };
};

export default useCamera;
