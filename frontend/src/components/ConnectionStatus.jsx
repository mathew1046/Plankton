/**
 * WebSocket connection status indicator
 */
import React from 'react';
import { Wifi, WifiOff, AlertCircle } from 'lucide-react';

export const ConnectionStatus = ({ isConnected, error, reconnectAttempt }) => {
  if (isConnected) {
    return (
      <div className="flex items-center gap-2 text-green-600 text-sm">
        <Wifi className="w-4 h-4" />
        <span>Connected</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center gap-2 text-red-600 text-sm">
        <AlertCircle className="w-4 h-4" />
        <span>Error: {error}</span>
      </div>
    );
  }

  if (reconnectAttempt > 0) {
    return (
      <div className="flex items-center gap-2 text-yellow-600 text-sm">
        <WifiOff className="w-4 h-4 animate-pulse" />
        <span>Reconnecting... (attempt {reconnectAttempt})</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 text-gray-400 text-sm">
      <WifiOff className="w-4 h-4" />
      <span>Disconnected</span>
    </div>
  );
};

export default ConnectionStatus;
