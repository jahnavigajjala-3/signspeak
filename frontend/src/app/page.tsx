'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { AlertTriangle, X } from 'lucide-react';
import { Header } from '@/components/Header';
import { ModeSelector } from '@/components/ModeSelector';
import { VideoUploader } from '@/components/VideoUploader';
import { VideoPlayer } from '@/components/VideoPlayer';
import { ProcessingState } from '@/components/ProcessingState';
import { TranslationResult } from '@/components/TranslationResult';
import { SpeechPlayer } from '@/components/SpeechPlayer';
import { CameraWorkspace } from '@/components/CameraWorkspace';
import { translateVideo, checkHealth, ApiError, type TranslationResponse } from '@/lib/api';

// ─── Types ────────────────────────────────────────────────────────────────────

type WorkspaceMode = 'upload' | 'camera';
type TranslateStatus = 'idle' | 'processing' | 'success' | 'error';

// ─── Main Page ─────────────────────────────────────────────────────────────────

export default function HomePage() {
  const [mode, setMode] = useState<WorkspaceMode>('upload');
  const [activeTab, setActiveTab] = useState<'translate' | 'camera'>('translate');

  // Video state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoObjectUrl, setVideoObjectUrl] = useState<string | null>(null);

  // Translation state
  const [status, setStatus] = useState<TranslateStatus>('idle');
  const [result, setResult] = useState<TranslationResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Speech state
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speakTrigger, setSpeakTrigger] = useState(0);

  // System device info
  const [deviceStatus, setDeviceStatus] = useState('Checking...');

  // ── Fetch health / device info on mount
  useEffect(() => {
    checkHealth()
      .then((d) => {
        if (d.status === 'healthy') {
          setDeviceStatus(d.device);
        } else {
          setDeviceStatus('Checkpoint unavailable');
        }
      })
      .catch(() => setDeviceStatus('Backend unreachable'));
  }, []);

  // ── Sync header tab with mode
  useEffect(() => {
    setActiveTab(mode === 'upload' ? 'translate' : 'camera');
  }, [mode]);

  // ── Revoke videoObjectUrl on unmount to prevent memory leaks
  useEffect(() => {
    return () => {
      if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
    };
  }, [videoObjectUrl]);

  const handleTabChange = (tab: 'translate' | 'camera') => {
    setActiveTab(tab);
    setMode(tab === 'translate' ? 'upload' : 'camera');
  };

  // ── Video file selection (upload or camera)
  const handleVideoSelected = useCallback(
    (file: File) => {
      // Revoke previous object URL
      if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
      setVideoObjectUrl(URL.createObjectURL(file));
      setSelectedFile(file);
      // Reset translation state
      setStatus('idle');
      setResult(null);
      setErrorMsg(null);
    },
    [videoObjectUrl]
  );

  const handleReplaceVideo = useCallback(() => {
    if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
    setVideoObjectUrl(null);
    setSelectedFile(null);
    setStatus('idle');
    setResult(null);
    setErrorMsg(null);
  }, [videoObjectUrl]);

  // ── Core translate function — used by both upload and camera modes
  const runTranslation = useCallback(async (file: File) => {
    setStatus('processing');
    setResult(null);
    setErrorMsg(null);
    try {
      const data = await translateVideo(file);
      setResult(data);
      setStatus('success');
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : (err instanceof Error ? err.message : 'Unknown error occurred');
      setErrorMsg(msg);
      setStatus('error');
    }
  }, []);

  const handleTranslate = useCallback(async () => {
    if (!selectedFile) return;
    await runTranslation(selectedFile);
  }, [selectedFile, runTranslation]);

  // ── Camera: receives recorded file and immediately translates
  const handleCameraTranslate = useCallback(
    async (file: File) => {
      // Show it in the upload workspace for result display
      handleVideoSelected(file);
      setMode('upload');
      setActiveTab('translate');
      await runTranslation(file);
    },
    [handleVideoSelected, runTranslation]
  );

  const handleSpeak = () => {
    setSpeakTrigger((n) => n + 1);
  };

  // ── Format file size
  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="flex flex-col min-h-screen">
      <Header
        deviceStatus={deviceStatus}
        activeTab={activeTab}
        setActiveTab={handleTabChange}
      />

      <main className="flex-1 max-w-6xl mx-auto w-full px-4 sm:px-6 py-6">
        {/* Tagline */}
        <div className="text-center mb-2">
          <p className="text-xs font-mono uppercase tracking-widest text-slate-500">
            Custom CNN → LSTM → Additive Attention Decoder
          </p>
        </div>

        {/* Mode Selector */}
        <ModeSelector mode={mode} onSelectMode={setMode} />

        {/* ── Upload Workspace ── */}
        {mode === 'upload' && (
          <div>
            {/* Error Banner */}
            {status === 'error' && errorMsg && (
              <div className="max-w-2xl mx-auto my-4 p-4 rounded-2xl bg-rose-500/10 border border-rose-500/20 flex items-start justify-between space-x-3">
                <div className="flex items-start space-x-3">
                  <AlertTriangle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-semibold text-rose-300">Inference Failed</p>
                    <p className="text-xs text-rose-400/80 mt-0.5">{errorMsg}</p>
                  </div>
                </div>
                <button
                  onClick={() => setErrorMsg(null)}
                  className="text-slate-500 hover:text-white transition-colors shrink-0"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            )}

            {/* Step 1: Upload or Preview */}
            {!selectedFile ? (
              <VideoUploader
                onVideoSelected={handleVideoSelected}
                isProcessing={status === 'processing'}
              />
            ) : (
              <VideoPlayer
                videoUrl={videoObjectUrl!}
                fileName={selectedFile.name}
                fileSize={formatFileSize(selectedFile.size)}
                onTranslate={handleTranslate}
                onReplaceVideo={handleReplaceVideo}
                isProcessing={status === 'processing'}
              />
            )}

            {/* Step 2: Processing State */}
            {status === 'processing' && <ProcessingState />}

            {/* Step 3: Translation Result */}
            {status === 'success' && result && (
              <>
                <TranslationResult
                  translation={result.translation}
                  processingTimeMs={result.processing_time_ms}
                  videoInfo={result.video_info}
                  checkpointUsed={result.checkpoint_used}
                  onSpeak={handleSpeak}
                  isSpeaking={isSpeaking}
                />
                {result.translation && result.translation.trim().length > 0 && (
                  <SpeechPlayer
                    key={speakTrigger}
                    text={result.translation}
                    onStateChange={setIsSpeaking}
                  />
                )}
              </>
            )}
          </div>
        )}

        {/* ── Camera Workspace ── */}
        {mode === 'camera' && (
          <CameraWorkspace onTranslate={handleCameraTranslate} />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-surface-border py-4 mt-auto">
        <p className="text-center text-xs font-mono text-slate-600">
          SignSpeak · CNN Visual Encoder → LSTM Temporal Encoder → Additive Attention Decoder · Milestone 1
        </p>
      </footer>
    </div>
  );
}
