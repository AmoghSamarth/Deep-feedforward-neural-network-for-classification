"use client";

import React from "react";
import { Brain, Sparkles, Github } from "lucide-react";

export type TabKey =
  | "home"
  | "dataset"
  | "preprocessing"
  | "architecture"
  | "training"
  | "evaluation"
  | "predict";

interface NavbarProps {
  activeTab: TabKey;
  onSelectTab: (tab: TabKey) => void;
}

const TABS: { key: TabKey; label: string; icon: string }[] = [
  { key: "home", label: "Home", icon: "🏠" },
  { key: "dataset", label: "Dataset", icon: "📊" },
  { key: "preprocessing", label: "Preprocessing", icon: "⚙️" },
  { key: "architecture", label: "Architecture", icon: "🧠" },
  { key: "training", label: "Training", icon: "🚀" },
  { key: "evaluation", label: "Evaluation", icon: "📈" },
  { key: "predict", label: "Predict", icon: "🔮" },
];

export function Navbar({ activeTab, onSelectTab }: NavbarProps) {
  return (
    <header className="mb-8 space-y-5">
      {/* Top Header Row */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-slate-200/80 pb-5">
        <div className="flex items-center gap-3.5">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-blue-600 text-white shadow-md shadow-blue-500/25">
            <Brain className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-slate-900 sm:text-2xl">
              Deep Feedforward Neural Network
            </h1>
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Pattern Recognition &bull; TAE 1: Project Based Learning &bull; Phase I
            </p>
          </div>
        </div>

        {/* Student & USN Tag */}
        <div className="flex items-center gap-2.5">
          <div className="inline-flex items-center gap-1.5 rounded-full border border-blue-200 bg-blue-50/80 px-3 py-1.5 text-xs font-semibold text-blue-700">
            <Sparkles className="h-3.5 w-3.5 text-blue-600" />
            <span>Amogh Samarth &bull; CM23034</span>
          </div>
          <a
            href="https://github.com/AmoghSamarth/Deep-feedforward-neural-network-for-classification"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-600 shadow-sm transition hover:bg-slate-50 hover:text-slate-900"
          >
            <Github className="h-3.5 w-3.5" />
            <span>GitHub</span>
          </a>
        </div>
      </div>

      {/* Modern Navigation Tabs */}
      <nav aria-label="Dashboard Navigation">
        <div className="flex flex-wrap items-center justify-center gap-1.5 rounded-2xl border border-slate-200/80 bg-white p-1.5 shadow-sm sm:justify-start">
          {TABS.map((tab) => {
            const isActive = activeTab === tab.key;
            return (
              <button
                key={tab.key}
                onClick={() => onSelectTab(tab.key)}
                className={`flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold transition-all duration-150 ${
                  isActive
                    ? "border border-blue-200 bg-blue-50 text-blue-600 shadow-sm shadow-blue-500/10"
                    : "border border-transparent text-slate-600 hover:bg-slate-100/80 hover:text-slate-900"
                }`}
              >
                <span>{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </nav>
    </header>
  );
}
