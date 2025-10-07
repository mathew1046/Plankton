/**
 * Control panel for adjusting settings
 */
import React, { useState, useEffect } from 'react';
import { Settings, Thermometer } from 'lucide-react';
import config from '../config';

const ControlPanel = ({
  fps,
  setFps,
  confidenceThreshold,
  setConfidenceThreshold,
  quality,
  setQuality
}) => {
  const [temperature, setTemperature] = useState(1.3132);
  const [isLoading, setIsLoading] = useState(false);

  // Fetch current temperature on mount
  useEffect(() => {
    fetchTemperature();
  }, []);

  const fetchTemperature = async () => {
    try {
      const response = await fetch(`${config.apiUrl}/api/model/temperature`);
      if (response.ok) {
        const data = await response.json();
        setTemperature(data.temperature);
      }
    } catch (error) {
      console.error('Failed to fetch temperature:', error);
    }
  };

  const updateTemperature = async (newTemp) => {
    setIsLoading(true);
    try {
      const response = await fetch(`${config.apiUrl}/api/model/temperature`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ temperature: newTemp }),
      });

      if (response.ok) {
        setTemperature(newTemp);
        console.log('Temperature updated to:', newTemp);
      } else {
        console.error('Failed to update temperature');
      }
    } catch (error) {
      console.error('Failed to update temperature:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTemperatureChange = (e) => {
    const newTemp = parseFloat(e.target.value);
    setTemperature(newTemp);
    updateTemperature(newTemp);
  };

  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-4">
        <Settings className="w-5 h-5 text-cyan-400" />
        <h3 className="text-lg font-semibold">Settings</h3>
      </div>

      <div className="space-y-4">
        {/* FPS Control */}
        <div>
          <label className="block text-sm font-medium mb-2">
            Frame Rate: {fps} FPS
          </label>
          <input
            type="range"
            min="1"
            max="30"
            value={fps}
            onChange={(e) => setFps(parseInt(e.target.value))}
            className="w-full"
          />
        </div>

        {/* Quality Control */}
        <div>
          <label className="block text-sm font-medium mb-2">
            JPEG Quality: {quality}%
          </label>
          <input
            type="range"
            min="50"
            max="100"
            step="5"
            value={quality}
            onChange={(e) => setQuality(Number(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Low</span>
            <span>High</span>
          </div>
        </div>

        {/* Confidence Threshold */}
        <div>
          <label className="block text-sm font-medium mb-2">
            Confidence Threshold: {confidenceThreshold}
          </label>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.1"
            value={confidenceThreshold}
            onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>

        {/* Temperature Control */}
        <div>
          <label className="block text-sm font-medium mb-2 flex items-center gap-2">
            <Thermometer className="w-4 h-4" />
            Temperature: {temperature.toFixed(3)}
          </label>
          <input
            type="range"
            min="0.5"
            max="2.0"
            step="0.001"
            value={temperature}
            onChange={handleTemperatureChange}
            disabled={isLoading}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-400 mt-1">
            <span>0.5 (Conservative)</span>
            <span>2.0 (Aggressive)</span>
          </div>
          {isLoading && (
            <p className="text-xs text-cyan-400 mt-1">Updating temperature...</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;
