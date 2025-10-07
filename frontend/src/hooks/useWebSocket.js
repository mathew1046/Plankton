/**
 * Custom hook for WebSocket connection management
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import config from '../config';

export const useWebSocket = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastPrediction, setLastPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const shouldReconnectRef = useRef(true);

  const connect = useCallback(() => {
    try {
      const wsUrl = `${config.wsUrl}/ws/predict?token=${config.apiKey}`;
      console.log('Connecting to WebSocket:', wsUrl);
      
      const ws = new WebSocket(wsUrl);
      
      ws.binaryType = 'arraybuffer';
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setError(null);
        setReconnectAttempt(0);
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'prediction') {
            console.log('🧠 WebSocket prediction received:', data);
            setLastPrediction(data);
          } else if (data.type === 'error') {
            console.error('Server error:', data.message);
            setError(data.message);
          } else if (data.type === 'model_update') {
            console.log('Model updated:', data.model_version);
            // Could trigger a notification here
          }
        } catch (err) {
          console.error('Failed to parse message:', err);
        }
      };
      
      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError('WebSocket connection error');
      };
      
      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        setIsConnected(false);
        wsRef.current = null;
        
        // Attempt reconnection if not manually closed
        if (shouldReconnectRef.current && reconnectAttempt < config.maxReconnectAttempts) {
          console.log(`Reconnecting in ${config.reconnectDelay}ms... (attempt ${reconnectAttempt + 1})`);
          reconnectTimeoutRef.current = setTimeout(() => {
            setReconnectAttempt(prev => prev + 1);
            connect();
          }, config.reconnectDelay);
        } else if (reconnectAttempt >= config.maxReconnectAttempts) {
          setError('Max reconnection attempts reached');
        }
      };
      
      wsRef.current = ws;
      
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      setError(err.message);
    }
  }, [reconnectAttempt]);

  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false;
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setIsConnected(false);
  }, []);

  const sendFrame = useCallback((frameData) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(frameData);
        return true;
      } catch (err) {
        console.error('Failed to send frame:', err);
        setError('Failed to send frame');
        return false;
      }
    }
    return false;
  }, []);

  const isWebSocketConnected = useCallback(() => {
    return wsRef.current && wsRef.current.readyState === WebSocket.OPEN;
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      shouldReconnectRef.current = false;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return {
    isConnected,
    lastPrediction,
    error,
    connect,
    disconnect,
    sendFrame,
    reconnectAttempt,
    isWebSocketConnected,
  };
};

export default useWebSocket;
