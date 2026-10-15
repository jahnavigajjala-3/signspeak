'use client';

import React, { useState } from 'react';
import { Copy, Check, Volume2, ChevronDown, ChevronUp, Clock, Cpu } from 'lucide-react';

interface TranslationResultProps {
  translation: string;
  processingTimeMs: number;
  videoInfo: {
    duration_sec: number;
    fps: number;
    width: number;
    height: number;
    frame_count: number;
  };
  checkpointUsed: string;
  onSpeak: () => void;
  isSpeaking: boolean;
}

export const TranslationResult: React.FC<TranslationResultProps> = ({
  translation,
  processingTimeMs,
  videoInfo,
  checkpointUsed,
  onSpeak,
  isSpeaking,
}) => {
  const [copied, setCopied] = useState(false);
  const [showMeta, setShowMeta] = useState(false);

  const handleCopy = async () => {
    if (!translation) return;
    try {
      await navigator.clipboard.writeText(translation);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard write failed silently
    }
  };

  const hasTranslation = translation && translation.trim().length > 0;

  return (
    <div className="w-full max-w-2xl mx-auto my-4 bg-surface-card border border-surface-border rounded-3xl overflow-hidden shadow-2xl">
      {/* Result Header */}
      <div className="px-6 py-4 border-b border-surface-border bg-gradient-to-r from-cyan-500/5 to-blue-600/5 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-xs font-bold uppercase tracking-widest text-emerald-400">
            Translation
          </span>
          <span className="text-xs text-slate-500 font-mono ml-2">
            SignSpeak PyTorch Model Output
          </span>
        </div>
        <div className="flex items-center space-x-1 text-xs text-slate-500">
          <Clock className="h-3.5 w-3.5" />
          <span>{(processingTimeMs / 1000).toFixed(2)}s</span>
        </div>
      </div>

      {/* Translation Text */}
      <div className="px-6 py-8">
        {hasTranslation ? (
          <p className="text-2xl sm:text-3xl font-bold text-white leading-relaxed tracking-tight">
            &ldquo;{translation}&rdquo;
          </p>
        ) : (
          <div className="text-center py-4">
            <p className="text-lg text-slate-500 italic">No translation generated.</p>
            <p className="text-sm text-slate-600 mt-2">
              The model did not produce a token sequence for this video. Try a different clip.
            </p>
          </div>
        )}
      </div>

      {/* Action Row */}
      {hasTranslation && (
        <div className="px-6 pb-6 flex flex-wrap items-center gap-3">
          {/* Speak Button */}
          <button
            onClick={onSpeak}
            disabled={isSpeaking}
            className={`flex items-center space-x-2 px-5 py-2.5 rounded-xl font-semibold text-sm transition-all duration-200 ${
              isSpeaking
                ? 'bg-cyan-600/40 text-cyan-200 cursor-default'
                : 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold shadow-md shadow-cyan-500/25 hover:scale-[1.02]'
            }`}
          >
            <Volume2 className="h-4 w-4" />
            <span>{isSpeaking ? 'Speaking...' : 'Speak Translation'}</span>
          </button>

          {/* Copy Button */}
          <button
            onClick={handleCopy}
            className="flex items-center space-x-2 px-4 py-2.5 rounded-xl text-sm text-slate-300 hover:text-white border border-surface-border hover:border-slate-500 transition-all bg-surface-cardHover"
          >
            {copied ? (
              <>
                <Check className="h-4 w-4 text-emerald-400" />
                <span className="text-emerald-400">Copied!</span>
              </>
            ) : (
              <>
                <Copy className="h-4 w-4" />
                <span>Copy Text</span>
              </>
            )}
          </button>
        </div>
      )}

      {/* Inference Metadata (collapsible) */}
      <div className="border-t border-surface-border">
        <button
          onClick={() => setShowMeta(!showMeta)}
          className="w-full flex items-center justify-between px-6 py-3 text-xs text-slate-500 hover:text-slate-300 transition-colors"
        >
          <div className="flex items-center space-x-2">
            <Cpu className="h-3.5 w-3.5" />
            <span className="font-mono">Model &amp; Video Metadata</span>
          </div>
          {showMeta ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
        </button>

        {showMeta && (
          <div className="px-6 pb-5 grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs font-mono">
            {[
              { label: 'Checkpoint', value: checkpointUsed.split('/').pop() || checkpointUsed },
              { label: 'Inference Time', value: `${processingTimeMs.toFixed(0)} ms` },
              { label: 'Video Duration', value: `${videoInfo.duration_sec}s` },
              { label: 'FPS', value: videoInfo.fps },
              { label: 'Resolution', value: `${videoInfo.width}×${videoInfo.height}` },
              { label: 'Frame Count', value: videoInfo.frame_count },
            ].map(({ label, value }) => (
              <div key={label} className="bg-slate-900/60 rounded-lg p-2.5 border border-slate-800">
                <p className="text-slate-500 mb-1">{label}</p>
                <p className="text-slate-200 truncate">{String(value)}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
