'use client';

import React from 'react';
import { Activity, Sparkles, Cpu } from 'lucide-react';

interface HeaderProps {
  deviceStatus?: string;
  activeTab: 'translate' | 'camera';
  setActiveTab: (tab: 'translate' | 'camera') => void;
}

export const Header: React.FC<HeaderProps> = ({ deviceStatus = 'Backend Active', activeTab, setActiveTab }) => {
  return (
    <header className="border-b border-surface-border bg-surface-base/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center space-x-3">
          <div className="h-9 w-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center text-white shadow-lg shadow-cyan-500/20">
            <Sparkles className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-lg tracking-tight text-white">SignSpeak</span>
              <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                PyTorch ML
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">Continuous Sign-Language Understanding</p>
          </div>
        </div>

        {/* Center Nav */}
        <nav className="flex items-center space-x-1 bg-surface-card p-1 rounded-xl border border-surface-border">
          <button
            onClick={() => setActiveTab('translate')}
            aria-label="Translate workspace"
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
              activeTab === 'translate'
                ? 'bg-cyan-500 text-slate-950 font-semibold shadow-md'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Translate
          </button>
          <button
            onClick={() => setActiveTab('camera')}
            aria-label="Camera recording workspace"
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all flex items-center space-x-1.5 ${
              activeTab === 'camera'
                ? 'bg-cyan-500 text-slate-950 font-semibold shadow-md'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <span>Camera</span>
            <span className="text-[9px] px-1.5 py-0.5 rounded bg-cyan-950/60 text-cyan-300 font-mono">NEXT</span>
          </button>
        </nav>

        {/* Backend / Device Status */}
        <div className="hidden md:flex items-center space-x-2 bg-surface-card px-3 py-1.5 rounded-lg border border-surface-border text-xs text-slate-300">
          <Cpu className="h-3.5 w-3.5 text-cyan-400" />
          <span className="font-mono text-[11px] text-slate-300">{deviceStatus}</span>
          <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
        </div>
      </div>
    </header>
  );
};
