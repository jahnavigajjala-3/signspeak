'use client';

import React, { useRef, useState, useCallback, useEffect } from 'react';
import {
  Camera,
  CameraOff,
  Circle,
  Square,
  RotateCcw,
  Sparkles,
  AlertTriangle,
  Film,
  RefreshCw,
} from 'lucide-react';

type CameraState =
  | 'idle'
  | 'permission-pending'
  | 'permission-denied'
  | 'preview'
  | 'recording'
  | 'recorded'
  | 'processing';

interface CameraWorkspaceProps {
  onTranslate: (videoFile: File) => Promise<void>;
}

export const CameraWorkspace: React.FC<CameraWorkspaceProps> = ({ onTranslate }) => {
  const [state, setState] = useState<CameraState>('idle');
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [recordedBlobUrl, setRecordedBlobUrl] = useState<string | null>(null);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const recordedVideoRef = useRef<HTMLVideoElement>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordingChunksRef = useRef<Blob[]>([]);
  const durationTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const recordedBlobUrlRef = useRef<string | null>(null);

  const startCamera = useCallback(async () => {
    setState('permission-pending');
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setState('preview');
    } catch (err) {
      const e = err as DOMException;
      if (e.name === 'NotAllowedError' || e.name === 'PermissionDeniedError') {
        setState('permission-denied');
        setError('Camera permission denied. Please allow access in your browser settings.');
      } else {
        setState('idle');
        setError(`Unable to access camera: ${e.message}`);
      }
    }
  }, []);

  const startRecording = useCallback(() => {
    if (!mediaStreamRef.current) return;

    recordingChunksRef.current = [];
    const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
      ? 'video/webm;codecs=vp9'
      : MediaRecorder.isTypeSupported('video/webm')
      ? 'video/webm'
      : 'video/mp4';

    const recorder = new MediaRecorder(mediaStreamRef.current, { mimeType });
    mediaRecorderRef.current = recorder;

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        recordingChunksRef.current.push(e.data);
      }
    };

    recorder.onstop = () => {
      const blob = new Blob(recordingChunksRef.current, { type: mimeType });
      const url = URL.createObjectURL(blob);
      recordedBlobUrlRef.current = url;
      setRecordedBlob(blob);
      setRecordedBlobUrl(url);
      setState('recorded');
    };

    recorder.start(100);
    setState('recording');
    setRecordingDuration(0);

    durationTimerRef.current = setInterval(() => {
      setRecordingDuration((d) => d + 1);
    }, 1000);
  }, []);

  const stopRecording = useCallback(() => {
    if (durationTimerRef.current) {
      clearInterval(durationTimerRef.current);
    }
    mediaRecorderRef.current?.stop();
  }, []);

  const retake = useCallback(() => {
    if (recordedBlobUrl) {
      URL.revokeObjectURL(recordedBlobUrl);
      recordedBlobUrlRef.current = null;
      setRecordedBlobUrl(null);
    }
    setRecordedBlob(null);
    setRecordingDuration(0);
    setState('preview');
  }, [recordedBlobUrl]);

  const stopCamera = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (recordedBlobUrl) URL.revokeObjectURL(recordedBlobUrl);
    recordedBlobUrlRef.current = null;
    setRecordedBlobUrl(null);
    setRecordedBlob(null);
    setRecordingDuration(0);
    setState('idle');
  }, [recordedBlobUrl]);

  const handleTranslateRecording = useCallback(async () => {
    if (!recordedBlob) return;
    const ext = recordedBlob.type.includes('mp4') ? 'mp4' : 'webm';
    const file = new File([recordedBlob], `camera_recording.${ext}`, { type: recordedBlob.type });
    setState('processing');
    try {
      await onTranslate(file);
    } finally {
      setState('recorded');
    }
  }, [recordedBlob, onTranslate]);

  useEffect(() => {
    return () => {
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      }
      if (durationTimerRef.current) clearInterval(durationTimerRef.current);
      if (recordedBlobUrlRef.current) URL.revokeObjectURL(recordedBlobUrlRef.current);
    };
  }, []);

  const formatDuration = (secs: number) => {
    const m = Math.floor(secs / 60).toString().padStart(2, '0');
    const s = (secs % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  return (
    <div className="w-full max-w-2xl mx-auto my-4">
      {/* Status bar */}
      <div className="flex items-center justify-between mb-3 px-1">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-bold uppercase tracking-widest text-slate-400">
            Camera Recording
          </span>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-300 border border-amber-500/20 font-semibold">
            Milestone 2 Ready
          </span>
        </div>
        {(state === 'preview' || state === 'recording' || state === 'recorded') && (
          <button
            onClick={stopCamera}
            className="flex items-center space-x-1 text-xs text-slate-500 hover:text-rose-400 transition-colors"
          >
            <CameraOff className="h-3.5 w-3.5" />
            <span>Stop Camera</span>
          </button>
        )}
      </div>

      {/* Main Camera Area */}
      <div className="bg-surface-card border border-surface-border rounded-3xl overflow-hidden shadow-2xl">
        {/* Camera viewport */}
        <div className="relative aspect-video bg-slate-950 flex items-center justify-center">
          {/* Live camera preview */}
          <video
            ref={videoRef}
            className={`w-full h-full object-cover ${
              state === 'preview' || state === 'recording' ? 'block' : 'hidden'
            }`}
            muted
            playsInline
          />

          {/* Recorded clip preview */}
          {(state === 'recorded' || state === 'processing') && recordedBlobUrl && (
            <video
              ref={recordedVideoRef}
              src={recordedBlobUrl}
              className="w-full h-full object-cover"
              controls
              playsInline
            />
          )}

          {/* Idle state */}
          {state === 'idle' && (
            <div className="flex flex-col items-center space-y-4 text-center p-8">
              <div className="h-20 w-20 rounded-2xl bg-slate-800/60 border border-slate-700 flex items-center justify-center">
                <Camera className="h-10 w-10 text-slate-500" />
              </div>
              <div>
                <h3 className="text-base font-semibold text-white mb-1">Record Sign Language</h3>
                <p className="text-sm text-slate-400 max-w-xs">
                  Record yourself signing and process it through the SignSpeak ML pipeline.
                </p>
              </div>
              <button
                onClick={startCamera}
                className="flex items-center space-x-2 px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 text-slate-950 font-bold text-sm shadow-lg shadow-cyan-500/25 hover:scale-[1.02] transition-transform"
              >
                <Camera className="h-4 w-4" />
                <span>Enable Camera</span>
              </button>
            </div>
          )}

          {/* Permission pending */}
          {state === 'permission-pending' && (
            <div className="flex flex-col items-center space-y-3 text-center p-8">
              <div className="h-12 w-12 rounded-full border-2 border-cyan-400/40 border-t-cyan-400 animate-spin" />
              <p className="text-sm text-slate-300">Requesting camera permission...</p>
            </div>
          )}

          {/* Permission denied */}
          {state === 'permission-denied' && (
            <div className="flex flex-col items-center space-y-3 text-center p-8">
              <div className="h-16 w-16 rounded-2xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center">
                <AlertTriangle className="h-8 w-8 text-rose-400" />
              </div>
              <div>
                <p className="text-base font-semibold text-white mb-1">Camera Access Denied</p>
                <p className="text-sm text-slate-400 max-w-xs">
                  Allow camera access in your browser settings, then try again.
                </p>
              </div>
              <button
                onClick={startCamera}
                className="flex items-center space-x-2 px-4 py-2 rounded-xl border border-slate-600 text-slate-300 hover:text-white hover:border-slate-400 text-sm transition-all"
              >
                <RefreshCw className="h-4 w-4" />
                <span>Try Again</span>
              </button>
            </div>
          )}

          {/* Recording indicator overlay */}
          {state === 'recording' && (
            <div className="absolute top-4 left-4 flex items-center space-x-2 bg-rose-600/90 backdrop-blur-sm px-3 py-1.5 rounded-full shadow-lg">
              <div className="h-2.5 w-2.5 rounded-full bg-white animate-pulse" />
              <span className="text-xs font-bold text-white font-mono">
                REC {formatDuration(recordingDuration)}
              </span>
            </div>
          )}

          {/* Recorded clip chip */}
          {state === 'recorded' && (
            <div className="absolute top-4 left-4 flex items-center space-x-2 bg-emerald-600/90 backdrop-blur-sm px-3 py-1.5 rounded-full shadow-lg">
              <Film className="h-3.5 w-3.5 text-white" />
              <span className="text-xs font-bold text-white">
                Clip Ready · {formatDuration(recordingDuration)}
              </span>
            </div>
          )}
        </div>

        {/* Camera Controls Bar */}
        <div className="p-5 bg-surface-card border-t border-surface-border">
          {(state === 'preview') && (
            <div className="flex items-center justify-center">
              <button
                onClick={startRecording}
                className="flex items-center space-x-2 px-8 py-3 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-bold text-sm shadow-lg shadow-rose-600/30 hover:scale-[1.02] transition-all"
              >
                <Circle className="h-4 w-4 fill-white" />
                <span>Start Recording</span>
              </button>
            </div>
          )}

          {state === 'recording' && (
            <div className="flex items-center justify-center space-x-4">
              <span className="text-sm font-mono text-rose-300 font-bold">
                {formatDuration(recordingDuration)}
              </span>
              <button
                onClick={stopRecording}
                className="flex items-center space-x-2 px-8 py-3 rounded-xl bg-rose-600/20 border border-rose-500/40 text-rose-300 hover:bg-rose-600/30 font-bold text-sm transition-all"
              >
                <Square className="h-4 w-4 fill-current" />
                <span>Stop Recording</span>
              </button>
            </div>
          )}

          {(state === 'recorded' || state === 'processing') && (
            <div className="flex flex-wrap items-center justify-between gap-3">
              <button
                onClick={retake}
                disabled={state === 'processing'}
                className="flex items-center space-x-2 px-5 py-2.5 rounded-xl border border-surface-border text-slate-300 hover:text-white hover:border-slate-500 text-sm transition-all bg-surface-cardHover"
              >
                <RotateCcw className="h-4 w-4" />
                <span>Retake</span>
              </button>

              <button
                onClick={handleTranslateRecording}
                disabled={state === 'processing'}
                className={`flex items-center space-x-2 px-6 py-2.5 rounded-xl font-bold text-sm transition-all ${
                  state === 'processing'
                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                    : 'bg-gradient-to-r from-cyan-500 to-blue-600 text-slate-950 shadow-lg shadow-cyan-500/25 hover:scale-[1.02]'
                }`}
              >
                <Sparkles className="h-4 w-4" />
                <span>
                  {state === 'processing' ? 'Running Model...' : 'Translate Recording'}
                </span>
              </button>
            </div>
          )}

          {(state === 'idle' || state === 'permission-denied') && (
            <p className="text-xs text-slate-500 text-center font-mono">
              Live camera inference is enabled in Milestone 2 — camera recording sends through the same ML pipeline as video upload
            </p>
          )}
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="mt-3 p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 flex items-start space-x-2">
          <AlertTriangle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
          <p className="text-xs text-rose-300">{error}</p>
        </div>
      )}
    </div>
  );
};
