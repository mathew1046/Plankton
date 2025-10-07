/**
 * Overlay component for displaying predictions and bounding boxes
 */
import React from 'react';
import { AlertCircle } from 'lucide-react';

export const PredictionOverlay = ({ prediction, confidenceThreshold }) => {
  if (!prediction) return null;

  const { species, confidence, bbox } = prediction;
  
  // Filter by confidence threshold
  if (confidence < confidenceThreshold) return null;

  // Determine confidence color
  const getConfidenceColor = (conf) => {
    if (conf >= 0.8) return 'text-green-400 border-green-400';
    if (conf >= 0.6) return 'text-yellow-400 border-yellow-400';
    return 'text-orange-400 border-orange-400';
  };

  const confidenceColor = getConfidenceColor(confidence);

  return (
    <div className="absolute inset-0 pointer-events-none">
      {/* Bounding box */}
      {bbox && bbox.length === 4 && (
        <div
          className={`absolute border-2 ${confidenceColor} transition-all duration-300`}
          style={{
            left: `${bbox[0]}px`,
            top: `${bbox[1]}px`,
            width: `${bbox[2] - bbox[0]}px`,
            height: `${bbox[3] - bbox[1]}px`,
          }}
        >
          {/* Corner markers */}
          <div className={`absolute -top-1 -left-1 w-3 h-3 border-t-2 border-l-2 ${confidenceColor}`}></div>
          <div className={`absolute -top-1 -right-1 w-3 h-3 border-t-2 border-r-2 ${confidenceColor}`}></div>
          <div className={`absolute -bottom-1 -left-1 w-3 h-3 border-b-2 border-l-2 ${confidenceColor}`}></div>
          <div className={`absolute -bottom-1 -right-1 w-3 h-3 border-b-2 border-r-2 ${confidenceColor}`}></div>
        </div>
      )}

      {/* Prediction label */}
      <div className="absolute top-4 left-4 prediction-enter">
        <div className={`bg-black bg-opacity-75 backdrop-blur-sm rounded-lg px-4 py-2 border-2 ${confidenceColor}`}>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${confidence >= 0.8 ? 'bg-green-400' : confidence >= 0.6 ? 'bg-yellow-400' : 'bg-orange-400'}`}></div>
            <div>
              <div className="text-white font-semibold text-lg">{species}</div>
              <div className={`text-sm ${confidenceColor}`}>
                {(confidence * 100).toFixed(1)}% confidence
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Low confidence warning */}
      {confidence < 0.6 && (
        <div className="absolute bottom-4 left-4">
          <div className="bg-orange-500 bg-opacity-90 text-white px-3 py-2 rounded-lg flex items-center gap-2 text-sm">
            <AlertCircle className="w-4 h-4" />
            Low confidence - verify result
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionOverlay;
