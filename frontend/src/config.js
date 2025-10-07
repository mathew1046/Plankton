/**
 * Application configuration
 */

export const config = {
  // Backend URLs
  wsUrl: import.meta.env.VITE_WS_URL || 'ws://localhost:8000',
  apiUrl: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  apiKey: import.meta.env.VITE_API_KEY || 'dev-key-12345',
  
  // Camera settings
  defaultFps: parseInt(import.meta.env.VITE_DEFAULT_FPS) || 6,
  defaultQuality: parseInt(import.meta.env.VITE_DEFAULT_QUALITY) || 85,
  defaultConfidence: parseFloat(import.meta.env.VITE_DEFAULT_CONFIDENCE) || 0.1,  // Temporarily lowered for testing
  
  // Camera constraints
  cameraConstraints: {
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: 'user',
      frameRate: { ideal: 30 }
    },
    audio: false
  },
  
  // Canvas settings
  canvasWidth: 640,
  canvasHeight: 480,
  
  // WebSocket settings
  reconnectDelay: 3000,
  maxReconnectAttempts: 5,
  
  // UI settings
  maxLogEntries: 100,
  toastDuration: 3000,
};

export default config;
