/**
 * Component for correcting prediction labels
 */
import React, { useState } from 'react';
import { Edit3, Check, X } from 'lucide-react';
import config from '../config';

export const LabelCorrection = ({ 
  prediction, 
  species, 
  onSubmit, 
  frameData 
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [selectedSpecies, setSelectedSpecies] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleStartEdit = () => {
    setIsEditing(true);
    setSelectedSpecies(prediction?.species || '');
  };

  const handleCancel = () => {
    setIsEditing(false);
    setSelectedSpecies('');
  };

  const handleSubmit = async () => {
    if (!selectedSpecies || !prediction || !frameData) {
      return;
    }

    setIsSubmitting(true);
    
    try {
      const formData = new FormData();
      formData.append('file', frameData, 'frame.jpg');
      formData.append('frame_id', `frame_${Date.now()}`);
      formData.append('original_prediction', prediction.species);
      formData.append('corrected_label', selectedSpecies);
      formData.append('confidence', prediction.confidence);

      const response = await fetch(`${config.apiUrl}/api/training/submit`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        onSubmit?.(selectedSpecies);
        setIsEditing(false);
      } else {
        console.error('Failed to submit correction');
      }
    } catch (error) {
      console.error('Error submitting correction:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!prediction) return null;

  return (
    <div className="card">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Edit3 className="w-5 h-5" />
        Label Correction
      </h3>

      {!isEditing ? (
        <div>
          <p className="text-sm text-gray-600 mb-3">
            Current prediction: <strong>{prediction.species}</strong> ({(prediction.confidence * 100).toFixed(1)}%)
          </p>
          <button
            onClick={handleStartEdit}
            className="btn-secondary w-full"
            disabled={!frameData}
          >
            <Edit3 className="w-4 h-4 mr-2" />
            Correct Label
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium mb-2">
              Select Correct Species
            </label>
            <select
              value={selectedSpecies}
              onChange={(e) => setSelectedSpecies(e.target.value)}
              className="input-field"
              disabled={isSubmitting}
            >
              <option value="">-- Select Species --</option>
              {species.map((sp) => (
                <option key={sp} value={sp}>
                  {sp}
                </option>
              ))}
            </select>
          </div>

          <div className="flex gap-2">
            <button
              onClick={handleSubmit}
              className="btn-primary flex-1"
              disabled={!selectedSpecies || isSubmitting}
            >
              <Check className="w-4 h-4 mr-2" />
              {isSubmitting ? 'Submitting...' : 'Submit'}
            </button>
            <button
              onClick={handleCancel}
              className="btn-secondary"
              disabled={isSubmitting}
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default LabelCorrection;
