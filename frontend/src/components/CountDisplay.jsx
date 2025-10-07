/**
 * Display organism counts by species
 */
import React from 'react';
import { Hash, RotateCcw } from 'lucide-react';

export const CountDisplay = ({ counts, onReset }) => {
  const totalCount = Object.values(counts).reduce((sum, count) => sum + count, 0);
  
  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Hash className="w-5 h-5" />
          Organism Count
        </h3>
        <button
          onClick={onReset}
          className="btn-secondary text-sm py-1"
          title="Reset counts"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>

      {/* Total count */}
      <div className="bg-primary-50 border border-primary-200 rounded-lg p-3 mb-4">
        <div className="text-sm text-primary-600 font-medium">Total Detected</div>
        <div className="text-3xl font-bold text-primary-700">{totalCount}</div>
      </div>

      {/* Species breakdown */}
      <div className="space-y-2 max-h-64 overflow-y-auto">
        {Object.entries(counts).length === 0 ? (
          <div className="text-center text-gray-400 py-4">
            No organisms detected yet
          </div>
        ) : (
          Object.entries(counts)
            .sort(([, a], [, b]) => b - a)
            .map(([species, count]) => (
              <div
                key={species}
                className="flex items-center justify-between p-2 bg-gray-50 rounded hover:bg-gray-100 transition-colors"
              >
                <span className="font-medium text-gray-700">{species}</span>
                <span className="badge badge-info">{count}</span>
              </div>
            ))
        )}
      </div>
    </div>
  );
};

export default CountDisplay;
