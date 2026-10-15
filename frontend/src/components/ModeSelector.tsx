'use client';

import React from 'react';
import { Upload, Camera, Sparkles } from 'lucide-react';

interface ModeSelectorProps {
  mode: 'upload' | 'camera';
  onSelectMode: (mode: 'upload' | 'camera') => void;
}

export const ModeSelector: React.FC<ModeSelectorProps> = ({ mode, onSelectMode }) => {
  return (
    <div className="flex items-center justify-center my-6">
      <div className="bg-surface-card p-1.5 rounded-2xl border border-surface-border flex items-center space-x-2 shadow-inner">
        <button
          onClick={() => onSelectMode('upload')}
          className={`flex items-center space-x-2 px-6 py-2.5 rounded-xl font-medium text-sm transition-all duration-200 ${
            mode === 'upload'
              ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/25'
              : 'text-slate-400 hover:text-slate-200 hover:bg-surface-cardHover'
          }`}
        >
          <Upload className="h-4 w-4" />
          <span>Upload Video</span>
        </button>

        <button
          onClick={() => onSelectMode('camera')}
          className={`flex items-center space-x-2 px-6 py-2.5 rounded-xl font-medium text-sm transition-all duration-200 ${
            mode === 'camera'
              ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/25'
              : 'text-slate-400 hover:text-slate-200 hover:bg-surface-cardHover'
          }`}
        >
          <Camera className="h-4 w-4" />
          <span>Camera Recording</span>
          <span className="text-[10px] uppercase tracking-wider font-semibold px-1.5 py-0.5 rounded bg-cyan-900/60 text-cyan-300 border border-cyan-700/40">
            Milestone 2 Ready
          </span>
        </button>
      </div>
    </div>
  );
};
