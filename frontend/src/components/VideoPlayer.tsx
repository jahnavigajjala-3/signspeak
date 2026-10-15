'use client';

import React, { useRef, useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Volume2, VolumeX, Sparkles, RefreshCw, Film } from 'lucide-react';

interface VideoPlayerProps {
  videoUrl: string;
  fileName?: string;
  fileSize?: string;
  onTranslate: () => void;
  onReplaceVideo: () => void;
  isProcessing?: boolean;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({
  videoUrl,
  fileName = 'Uploaded Sign Video',
  fileSize,
  onTranslate,
  onReplaceVideo,
  isProcessing = false,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleTimeUpdate = () => setCurrentTime(video.currentTime);
    const handleLoadedMetadata = () => setDuration(video.duration);
    const handleEnded = () => setIsPlaying(false);

    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    video.addEventListener('ended', handleEnded);

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('ended', handleEnded);
    };
  }, [videoUrl]);

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
    } else {
      videoRef.current.play();
      setIsPlaying(true);
    }
  };

  const toggleMute = () => {
    if (!videoRef.current) return;
    videoRef.current.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!videoRef.current) return;
    const seekTime = parseFloat(e.target.value);
    videoRef.current.currentTime = seekTime;
    setCurrentTime(seekTime);
  };

  const handleReplay = () => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = 0;
    videoRef.current.play();
    setIsPlaying(true);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="w-full max-w-2xl mx-auto my-4 bg-surface-card border border-surface-border rounded-3xl overflow-hidden shadow-2xl">
      {/* Top Bar / Metadata */}
      <div className="px-5 py-3 border-b border-surface-border flex items-center justify-between bg-surface-card/60">
        <div className="flex items-center space-x-2 overflow-hidden">
          <Film className="h-4 w-4 text-cyan-400 shrink-0" />
          <span className="text-xs font-medium text-slate-200 truncate">{fileName}</span>
          {fileSize && (
            <span className="text-[10px] text-slate-500 font-mono px-1.5 py-0.5 rounded bg-slate-800">
              {fileSize}
            </span>
          )}
        </div>
        <button
          onClick={onReplaceVideo}
          disabled={isProcessing}
          className="flex items-center space-x-1.5 text-xs text-slate-400 hover:text-cyan-400 transition-colors"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          <span>Change Video</span>
        </button>
      </div>

      {/* Video Container */}
      <div className="relative aspect-video bg-black flex items-center justify-center group">
        <video
          ref={videoRef}
          src={videoUrl}
          playsInline
          muted={isMuted}
          className="w-full h-full object-contain"
        />

        {/* Play Overlay */}
        {!isPlaying && (
          <button
            onClick={togglePlay}
            className="absolute inset-0 m-auto h-16 w-16 rounded-full bg-cyan-500/90 hover:bg-cyan-400 text-slate-950 flex items-center justify-center shadow-xl transition-all transform hover:scale-105"
          >
            <Play className="h-7 w-7 fill-slate-950 ml-1" />
          </button>
        )}
      </div>

      {/* Video Scrubber & Controls */}
      <div className="p-4 space-y-3 bg-surface-card">
        {/* Scrubber */}
        <div className="flex items-center space-x-3">
          <span className="text-xs font-mono text-slate-400 w-10 text-right">
            {formatTime(currentTime)}
          </span>
          <input
            type="range"
            min={0}
            max={duration || 100}
            step={0.1}
            value={currentTime}
            onChange={handleSeek}
            className="flex-1 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
          />
          <span className="text-xs font-mono text-slate-400 w-10">
            {formatTime(duration)}
          </span>
        </div>

        {/* Action Controls */}
        <div className="flex items-center justify-between pt-1">
          <div className="flex items-center space-x-2">
            <button
              onClick={togglePlay}
              aria-label={isPlaying ? 'Pause' : 'Play'}
              className="flex items-center space-x-1.5 px-3 py-2 rounded-xl hover:bg-surface-border text-slate-300 hover:text-white transition-colors text-xs"
            >
              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              <span>{isPlaying ? 'Pause' : 'Play'}</span>
            </button>
            <button
              onClick={handleReplay}
              aria-label="Replay from start"
              className="flex items-center space-x-1.5 px-3 py-2 rounded-xl hover:bg-surface-border text-slate-300 hover:text-white transition-colors text-xs"
            >
              <RotateCcw className="h-4 w-4" />
              <span>Replay</span>
            </button>
            <button
              onClick={toggleMute}
              aria-label={isMuted ? 'Unmute' : 'Mute'}
              className="flex items-center space-x-1.5 px-3 py-2 rounded-xl hover:bg-surface-border text-slate-300 hover:text-white transition-colors text-xs"
            >
              {isMuted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
              <span>{isMuted ? 'Unmute' : 'Mute'}</span>
            </button>
          </div>

          {/* Primary Translate Button */}
          <button
            onClick={onTranslate}
            disabled={isProcessing}
            className={`flex items-center space-x-2 px-6 py-2.5 rounded-xl font-semibold text-sm transition-all duration-200 ${
              isProcessing
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                : 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold shadow-lg shadow-cyan-500/30 hover:scale-[1.02]'
            }`}
          >
            <Sparkles className="h-4 w-4 fill-slate-950" />
            <span>{isProcessing ? 'Processing Model...' : 'Translate Sign Video'}</span>
          </button>
        </div>
      </div>
    </div>
  );
};
