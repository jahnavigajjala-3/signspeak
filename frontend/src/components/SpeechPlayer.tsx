'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Play, Square, RotateCcw } from 'lucide-react';

interface SpeechPlayerProps {
  text: string;
  onStateChange?: (speaking: boolean) => void;
}

const BAR_COUNT = 32;

export const SpeechPlayer: React.FC<SpeechPlayerProps> = ({ text, onStateChange }) => {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [hasSpoken, setHasSpoken] = useState(false);
  const [barHeights, setBarHeights] = useState<number[]>(Array(BAR_COUNT).fill(0.25));
  const animFrameRef = useRef<number | null>(null);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  // Animate waveform bars while speaking
  const animateWaveform = useCallback(() => {
    if (typeof window === 'undefined') return;
    setBarHeights(
      Array.from({ length: BAR_COUNT }, (_, i) => {
        const base = 0.25 + 0.5 * Math.sin((Date.now() / 250) + i * 0.6);
        const noise = 0.15 * Math.random();
        return Math.max(0.1, Math.min(1.0, base + noise));
      })
    );
    animFrameRef.current = requestAnimationFrame(animateWaveform);
  }, []);

  const stopAnimation = useCallback(() => {
    if (animFrameRef.current !== null) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    setBarHeights(Array(BAR_COUNT).fill(0.25));
  }, []);

  const speak = useCallback(() => {
    if (typeof window === 'undefined' || !window.speechSynthesis) return;

    // Cancel any in-progress speech
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    utterance.onstart = () => {
      setIsSpeaking(true);
      setHasSpoken(true);
      onStateChange?.(true);
      animateWaveform();
    };
    utterance.onend = () => {
      setIsSpeaking(false);
      onStateChange?.(false);
      stopAnimation();
    };
    utterance.onerror = () => {
      setIsSpeaking(false);
      onStateChange?.(false);
      stopAnimation();
    };

    utteranceRef.current = utterance;
    window.speechSynthesis.speak(utterance);
  }, [text, onStateChange, animateWaveform, stopAnimation]);

  const stop = useCallback(() => {
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    setIsSpeaking(false);
    onStateChange?.(false);
    stopAnimation();
  }, [onStateChange, stopAnimation]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (typeof window !== 'undefined' && window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
      stopAnimation();
    };
  }, [stopAnimation]);

  return (
    <div className="w-full max-w-2xl mx-auto my-4 bg-surface-card border border-surface-border rounded-3xl p-5 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-bold uppercase tracking-widest text-slate-400">
            Speech Output
          </span>
          <span
            className={`text-[10px] px-2 py-0.5 rounded-full font-semibold border transition-all ${
              isSpeaking
                ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30'
                : hasSpoken
                ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                : 'bg-slate-700/40 text-slate-500 border-slate-700'
            }`}
          >
            {isSpeaking ? 'Speaking' : hasSpoken ? 'Finished' : 'Idle'}
          </span>
        </div>

        {/* Controls */}
        <div className="flex items-center space-x-2">
          {isSpeaking ? (
            <button
              onClick={stop}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-rose-500/20 text-rose-300 border border-rose-500/30 hover:bg-rose-500/30 transition-all"
            >
              <Square className="h-3 w-3 fill-current" />
              <span>Stop</span>
            </button>
          ) : (
            <button
              onClick={speak}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 hover:bg-cyan-500/30 transition-all"
            >
              <Play className="h-3 w-3 fill-current" />
              <span>{hasSpoken ? 'Replay' : 'Play'}</span>
            </button>
          )}
          {hasSpoken && !isSpeaking && (
            <button
              onClick={speak}
              className="flex items-center space-x-1.5 p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-surface-border transition-all"
              title="Replay"
              aria-label="Replay speech"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              <span className="text-xs">Replay</span>
            </button>
          )}
        </div>
      </div>

      {/* Animated Waveform */}
      <div className="flex items-center justify-center space-x-0.5 h-14" aria-hidden="true">
        {barHeights.map((h, i) => (
          <div
            key={i}
            className={`w-1 rounded-full transition-all duration-75 ${
              isSpeaking ? 'bg-cyan-400' : 'bg-slate-700'
            }`}
            style={{
              height: `${Math.round(h * 100)}%`,
              opacity: isSpeaking ? 0.6 + h * 0.4 : 0.4,
            }}
          />
        ))}
      </div>

      <p className="text-[11px] text-slate-600 text-center mt-3 font-mono">
        {isSpeaking
          ? 'Browser Web Speech API · Speaking translation aloud'
          : 'Web Speech API · Click Play to hear the translation'}
      </p>
    </div>
  );
};
