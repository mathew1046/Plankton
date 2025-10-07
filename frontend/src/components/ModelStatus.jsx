/**
 * Model status and management component
 */
import React, { useState, useEffect } from 'react';
import { Brain, RefreshCw, RotateCcw, Activity, Database } from 'lucide-react';
import config from '../config';

export const ModelStatus = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [bufferStats, setBufferStats] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchModelStatus = async () => {
    try {
      const response = await fetch(`${config.apiUrl}/api/model/status`);
      const data = await response.json();
      if (data.status === 'success') {
        setModelInfo(data.data);
      }
    } catch (error) {
      console.error('Failed to fetch model status:', error);
    }
  };

  const fetchTrainingStatus = async () => {
    try {
      const response = await fetch(`${config.apiUrl}/api/training/status`);
      const data = await response.json();
      if (data.status === 'success') {
        setTrainingStatus(data.data);
      }
    } catch (error) {
      console.error('Failed to fetch training status:', error);
    }
  };

  const fetchBufferStats = async () => {
    try {
      const response = await fetch(`${config.apiUrl}/api/training/buffer`);
      const data = await response.json();
      if (data.status === 'success') {
        setBufferStats(data.data);
      }
    } catch (error) {
      console.error('Failed to fetch buffer stats:', error);
    }
  };

  useEffect(() => {
    fetchModelStatus();
    fetchTrainingStatus();
    fetchBufferStats();
    
    const interval = setInterval(() => {
      fetchModelStatus();
      fetchTrainingStatus();
      fetchBufferStats();
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);

  const handleReload = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${config.apiUrl}/api/model/reload`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (response.ok) {
        await fetchModelStatus();
      }
    } catch (error) {
      console.error('Failed to reload model:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRollback = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${config.apiUrl}/api/model/rollback`, {
        method: 'POST',
      });
      if (response.ok) {
        await fetchModelStatus();
      }
    } catch (error) {
      console.error('Failed to rollback model:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleStartTraining = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${config.apiUrl}/api/training/start`, {
        method: 'POST',
      });
      if (response.ok) {
        await fetchTrainingStatus();
      }
    } catch (error) {
      console.error('Failed to start training:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Brain className="w-5 h-5" />
        Model Status
      </h3>

      {modelInfo && (
        <div className="space-y-3 mb-4">
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">Version:</span>
            <span className="font-medium">{modelInfo.model_version}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">Device:</span>
            <span className="font-medium">{modelInfo.device}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">Inferences:</span>
            <span className="font-medium">{modelInfo.inference_count}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">Avg Time:</span>
            <span className="font-medium">{modelInfo.avg_inference_time_ms}ms</span>
          </div>
        </div>
      )}

      {/* Training Buffer */}
      {bufferStats && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4">
          <div className="flex items-center gap-2 mb-2">
            <Database className="w-4 h-4 text-blue-600" />
            <span className="text-sm font-medium text-blue-900">Training Buffer</span>
          </div>
          <div className="text-sm text-blue-700">
            {bufferStats.size} / {bufferStats.max_size} samples
            <div className="w-full bg-blue-200 rounded-full h-2 mt-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all"
                style={{ width: `${bufferStats.utilization * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      )}

      {/* Training Status */}
      {trainingStatus && trainingStatus.status !== 'idle' && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-3 mb-4">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-green-600" />
            <span className="text-sm font-medium text-green-900">
              Training: {trainingStatus.status}
            </span>
          </div>
          {trainingStatus.progress !== undefined && (
            <div className="text-sm text-green-700">
              Progress: {(trainingStatus.progress * 100).toFixed(0)}%
              <div className="w-full bg-green-200 rounded-full h-2 mt-2">
                <div
                  className="bg-green-600 h-2 rounded-full transition-all"
                  style={{ width: `${trainingStatus.progress * 100}%` }}
                ></div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="space-y-2">
        <button
          onClick={handleReload}
          className="btn-secondary w-full text-sm"
          disabled={loading}
        >
          <RefreshCw className="w-4 h-4 mr-2" />
          Reload Model
        </button>
        
        <button
          onClick={handleRollback}
          className="btn-secondary w-full text-sm"
          disabled={loading}
        >
          <RotateCcw className="w-4 h-4 mr-2" />
          Rollback
        </button>
        
        <button
          onClick={handleStartTraining}
          className="btn-primary w-full text-sm"
          disabled={loading || !bufferStats || bufferStats.size < 10}
        >
          <Activity className="w-4 h-4 mr-2" />
          Start Training
        </button>
      </div>
    </div>
  );
};

export default ModelStatus;
