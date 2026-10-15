'use client';

import React, { useEffect, useState } from 'react';

const STAGES = [
  {
    id: 1,
    label: 'Sampling Video Frames',
    sublabel: '16 frames extracted at uniform intervals',
    icon: '▶',
  },
  {
    id: 2,
    label: 'CNN Spatial Encoding',
    sublabel: 'Custom Conv2d → MaxPool → AdaptiveAvgPool → 512d projection',
    icon: '⬡',
  },
  {
    id: 3,
    label: 'LSTM Temporal Tracking',
    sublabel: 'Bidirectional sequence modeling across frame features',
    icon: '⟳',
  },
  {
    id: 4,
    label: 'Additive Attention Decoding',
    sublabel: 'Bahdanau attention + autoregressive greedy decoding',
    icon: '◈',
  },
];

export const ProcessingState: React.FC = () => {
  const [activeStage, setActiveStage] = useState(0);

  // Purely visual stage cycling — does NOT represent actual backend progress
  useEffect(() => {
    const interval = setInterval(() => {
      setActiveStage((prev) => (prev < STAGES.length - 1 ? prev + 1 : prev));
    }, 1400);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="w-full max-w-2xl mx-auto my-6 bg-surface-card border border-surface-border rounded-3xl p-6 shadow-2xl">
      <div className="flex items-center space-x-3 mb-6">
        <div className="h-8 w-8 rounded-full bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center">
          <div className="h-3 w-3 rounded-full bg-cyan-400 animate-ping" />
        </div>
        <div>
          <h3 className="text-sm font-bold text-white">Running Inference Pipeline</h3>
          <p className="text-xs text-slate-400">PyTorch SignSpeak model processing your video</p>
        </div>
      </div>

      {/* Pipeline Stages — conceptual representation */}
      <div className="relative space-y-3">
        {STAGES.map((stage, idx) => {
          const isDone = idx < activeStage;
          const isActive = idx === activeStage;

          return (
            <div
              key={stage.id}
              className={`flex items-start space-x-4 p-3 rounded-xl border transition-all duration-500 ${
                isActive
                  ? 'border-cyan-500/40 bg-cyan-500/5'
                  : isDone
                  ? 'border-emerald-500/20 bg-emerald-500/5 opacity-60'
                  : 'border-surface-border opacity-30'
              }`}
            >
              {/* Stage indicator */}
              <div
                className={`h-8 w-8 rounded-lg flex items-center justify-center text-sm font-mono shrink-0 border ${
                  isActive
                    ? 'bg-cyan-500/20 border-cyan-400/60 text-cyan-300 animate-pulse'
                    : isDone
                    ? 'bg-emerald-500/20 border-emerald-400/40 text-emerald-300'
                    : 'bg-slate-800/60 border-slate-700 text-slate-500'
                }`}
              >
                {isDone ? '✓' : stage.icon}
              </div>

              <div className="min-w-0">
                <p
                  className={`text-sm font-semibold ${
                    isActive ? 'text-white' : isDone ? 'text-slate-300' : 'text-slate-600'
                  }`}
                >
                  {stage.label}
                </p>
                <p className="text-xs text-slate-500 mt-0.5 font-mono">{stage.sublabel}</p>
              </div>

              {isActive && (
                <div className="ml-auto flex space-x-0.5 self-center shrink-0">
                  {[0, 1, 2, 3].map((i) => (
                    <div
                      key={i}
                      className="w-0.5 bg-cyan-400 rounded-full"
                      style={{
                        height: '16px',
                        animation: `wave 1.2s ease-in-out infinite`,
                        animationDelay: `${i * 0.15}s`,
                        transform: 'scaleY(0.3)',
                      }}
                    />
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <p className="text-[11px] text-slate-600 mt-5 text-center font-mono">
        Stage indicators are conceptual — the model runs in a single synchronous inference pass
      </p>
    </div>
  );
};
