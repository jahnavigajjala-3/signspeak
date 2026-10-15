'use client';

import React, { useState, useRef } from 'react';
import { UploadCloud, FileVideo, Sparkles, CheckCircle2 } from 'lucide-react';

interface VideoUploaderProps {
  onVideoSelected: (file: File) => void;
  onSampleSelected?: () => void;
  isProcessing?: boolean;
}

export const VideoUploader: React.FC<VideoUploaderProps> = ({
  onVideoSelected,
  onSampleSelected,
  isProcessing = false,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('video/')) {
        onVideoSelected(file);
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onVideoSelected(e.target.files[0]);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto my-4">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-3xl p-10 text-center cursor-pointer transition-all duration-300 ${
          isDragOver
            ? 'border-cyan-400 bg-cyan-500/10 scale-[1.01]'
            : 'border-surface-border bg-surface-card hover:border-slate-600 hover:bg-surface-cardHover'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/mp4,video/webm,video/quicktime,video/avi"
          className="hidden"
          onChange={handleFileChange}
          disabled={isProcessing}
        />

        <div className="h-16 w-16 mx-auto mb-4 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
          <UploadCloud className="h-8 w-8" />
        </div>

        <h3 className="text-lg font-semibold text-white mb-1">
          Upload a sign-language video
        </h3>
        <p className="text-sm text-slate-400 mb-4 max-w-md mx-auto">
          Drag and drop your video file here, or click to browse. Supports MP4, WebM, MOV.
        </p>

        <div className="flex flex-wrap items-center justify-center gap-2 text-xs text-slate-500">
          <span className="px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700">16 Frames Sampled</span>
          <span className="px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700">224×224 Normalization</span>
          <span className="px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700">CNN Features</span>
        </div>
      </div>

      {/* Quick Sample Selector */}
      {onSampleSelected && (
        <div className="mt-4 flex items-center justify-center">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onSampleSelected();
            }}
            className="flex items-center space-x-2 text-xs font-medium px-4 py-2 rounded-xl bg-surface-card border border-surface-border text-cyan-400 hover:text-cyan-300 hover:border-cyan-500/40 transition-all shadow-sm"
          >
            <Sparkles className="h-3.5 w-3.5" />
            <span>Load How2Sign Benchmark Sample Video</span>
          </button>
        </div>
      )}
    </div>
  );
};
