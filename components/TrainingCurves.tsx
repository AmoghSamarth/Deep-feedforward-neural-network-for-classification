"use client";

import React, { useState } from "react";

interface TrainingCurvesProps {
  history: {
    epoch: number[];
    loss: number[];
    val_loss: number[];
    accuracy: number[];
    val_accuracy: number[];
  };
  earlyStoppingRestoredEpoch?: number;
}

export function TrainingCurves({ history, earlyStoppingRestoredEpoch = 68 }: TrainingCurvesProps) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const totalEpochs = history.epoch.length;
  const width = 500;
  const height = 220;
  const padding = { top: 25, right: 25, bottom: 35, left: 45 };

  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  // Coordinate mappers
  const getX = (epoch: number) => padding.left + ((epoch - 1) / (totalEpochs - 1)) * plotW;

  // Accuracy: 0 to 105%
  const getAccY = (val: number) => padding.top + plotH - (val / 105) * plotH;

  // Loss: 0 to max loss (e.g., round up to 1.2)
  const maxLoss = Math.max(...history.loss, ...history.val_loss, 0.5);
  const getLossY = (val: number) => padding.top + plotH - (val / (maxLoss * 1.1)) * plotH;

  // Build SVG path strings
  const buildPath = (values: number[], getYFn: (v: number) => number) => {
    return values
      .map((val, idx) => {
        const x = getX(history.epoch[idx]);
        const y = getYFn(val);
        return `${idx === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");
  };

  const trainAccPath = buildPath(history.accuracy, getAccY);
  const valAccPath = buildPath(history.val_accuracy, getAccY);
  const trainLossPath = buildPath(history.loss, getLossY);
  const valLossPath = buildPath(history.val_loss, getLossY);

  const activeEpoch = hoverIndex !== null ? history.epoch[hoverIndex] : totalEpochs;
  const activeTrainAcc = hoverIndex !== null ? history.accuracy[hoverIndex] : history.accuracy[totalEpochs - 1];
  const activeValAcc = hoverIndex !== null ? history.val_accuracy[hoverIndex] : history.val_accuracy[totalEpochs - 1];
  const activeTrainLoss = hoverIndex !== null ? history.loss[hoverIndex] : history.loss[totalEpochs - 1];
  const activeValLoss = hoverIndex !== null ? history.val_loss[hoverIndex] : history.val_loss[totalEpochs - 1];

  return (
    <div className="space-y-6">
      {/* Current Hover Info Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-200 bg-slate-50/80 px-4 py-3 text-xs">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-slate-700">Inspecting Epoch:</span>
          <span className="rounded bg-blue-100 px-2 py-0.5 font-mono font-bold text-blue-800">
            {activeEpoch} / {totalEpochs}
          </span>
          {activeEpoch === earlyStoppingRestoredEpoch && (
            <span className="rounded bg-emerald-100 px-2 py-0.5 font-semibold text-emerald-800">
              Optimal Weights Restored
            </span>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-4 font-medium">
          <span className="text-blue-600">
            Train Acc: <b>{activeTrainAcc.toFixed(1)}%</b>
          </span>
          <span className="text-slate-600">
            Val Acc: <b>{activeValAcc.toFixed(1)}%</b>
          </span>
          <span className="text-blue-600">
            Train Loss: <b>{activeTrainLoss.toFixed(4)}</b>
          </span>
          <span className="text-slate-600">
            Val Loss: <b>{activeValLoss.toFixed(4)}</b>
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Accuracy Chart */}
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">
              Accuracy Progression (%)
            </h4>
            <div className="flex items-center gap-3 text-[11px] font-semibold">
              <span className="flex items-center gap-1 text-blue-600">
                <span className="inline-block h-2 w-3 rounded-full bg-blue-600"></span>
                Train
              </span>
              <span className="flex items-center gap-1 text-slate-500">
                <span className="inline-block h-0.5 w-3 border-t-2 border-dashed border-slate-500"></span>
                Validation
              </span>
            </div>
          </div>

          <div className="mt-3 overflow-x-auto">
            <svg
              viewBox={`0 0 ${width} ${height}`}
              className="h-auto w-full cursor-crosshair select-none"
              onMouseMove={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const mouseX = ((e.clientX - rect.left) / rect.width) * width;
                if (mouseX >= padding.left && mouseX <= width - padding.right) {
                  const ratio = (mouseX - padding.left) / plotW;
                  const idx = Math.round(ratio * (totalEpochs - 1));
                  setHoverIndex(Math.max(0, Math.min(totalEpochs - 1, idx)));
                }
              }}
              onMouseLeave={() => setHoverIndex(null)}
            >
              {/* Horizontal grid lines */}
              {[0, 25, 50, 75, 100].map((tick) => {
                const y = getAccY(tick);
                return (
                  <g key={`grid-acc-${tick}`}>
                    <line
                      x1={padding.left}
                      y1={y}
                      x2={width - padding.right}
                      y2={y}
                      stroke="#f1f5f9"
                      strokeWidth={1}
                    />
                    <text
                      x={padding.left - 8}
                      y={y + 3.5}
                      textAnchor="end"
                      className="fill-slate-400 text-[10px] font-mono"
                    >
                      {tick}%
                    </text>
                  </g>
                );
              })}

              {/* Early stopping marker line */}
              {earlyStoppingRestoredEpoch && (
                <line
                  x1={getX(earlyStoppingRestoredEpoch)}
                  y1={padding.top}
                  x2={getX(earlyStoppingRestoredEpoch)}
                  y2={height - padding.bottom}
                  stroke="#10b981"
                  strokeWidth={1.2}
                  strokeDasharray="3 3"
                />
              )}

              {/* Data Lines */}
              <path d={valAccPath} fill="none" stroke="#64748b" strokeWidth={2} strokeDasharray="4 3" />
              <path d={trainAccPath} fill="none" stroke="#2563eb" strokeWidth={2.2} />

              {/* Hover vertical line */}
              {hoverIndex !== null && (
                <line
                  x1={getX(activeEpoch)}
                  y1={padding.top}
                  x2={getX(activeEpoch)}
                  y2={height - padding.bottom}
                  stroke="#3b82f6"
                  strokeWidth={1}
                  strokeDasharray="2 2"
                />
              )}

              {/* X Axis labels */}
              <text x={padding.left} y={height - 12} className="fill-slate-400 text-[10px]">
                Epoch 1
              </text>
              <text x={width / 2} y={height - 12} textAnchor="middle" className="fill-slate-500 text-[11px] font-medium">
                Training Epochs
              </text>
              <text x={width - padding.right} y={height - 12} textAnchor="end" className="fill-slate-400 text-[10px]">
                Epoch {totalEpochs}
              </text>
            </svg>
          </div>
        </div>

        {/* Loss Chart */}
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">
              Loss Progression (Categorical Cross-Entropy)
            </h4>
            <div className="flex items-center gap-3 text-[11px] font-semibold">
              <span className="flex items-center gap-1 text-blue-600">
                <span className="inline-block h-2 w-3 rounded-full bg-blue-600"></span>
                Train
              </span>
              <span className="flex items-center gap-1 text-slate-500">
                <span className="inline-block h-0.5 w-3 border-t-2 border-dashed border-slate-500"></span>
                Validation
              </span>
            </div>
          </div>

          <div className="mt-3 overflow-x-auto">
            <svg
              viewBox={`0 0 ${width} ${height}`}
              className="h-auto w-full cursor-crosshair select-none"
              onMouseMove={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const mouseX = ((e.clientX - rect.left) / rect.width) * width;
                if (mouseX >= padding.left && mouseX <= width - padding.right) {
                  const ratio = (mouseX - padding.left) / plotW;
                  const idx = Math.round(ratio * (totalEpochs - 1));
                  setHoverIndex(Math.max(0, Math.min(totalEpochs - 1, idx)));
                }
              }}
              onMouseLeave={() => setHoverIndex(null)}
            >
              {/* Horizontal grid lines */}
              {[0, 0.2, 0.5, 0.8, 1.0].map((tick) => {
                const y = getLossY(tick);
                return (
                  <g key={`grid-loss-${tick}`}>
                    <line
                      x1={padding.left}
                      y1={y}
                      x2={width - padding.right}
                      y2={y}
                      stroke="#f1f5f9"
                      strokeWidth={1}
                    />
                    <text
                      x={padding.left - 8}
                      y={y + 3.5}
                      textAnchor="end"
                      className="fill-slate-400 text-[10px] font-mono"
                    >
                      {tick}
                    </text>
                  </g>
                );
              })}

              {/* Early stopping marker line */}
              {earlyStoppingRestoredEpoch && (
                <line
                  x1={getX(earlyStoppingRestoredEpoch)}
                  y1={padding.top}
                  x2={getX(earlyStoppingRestoredEpoch)}
                  y2={height - padding.bottom}
                  stroke="#10b981"
                  strokeWidth={1.2}
                  strokeDasharray="3 3"
                />
              )}

              {/* Data Lines */}
              <path d={valLossPath} fill="none" stroke="#64748b" strokeWidth={2} strokeDasharray="4 3" />
              <path d={trainLossPath} fill="none" stroke="#2563eb" strokeWidth={2.2} />

              {/* Hover vertical line */}
              {hoverIndex !== null && (
                <line
                  x1={getX(activeEpoch)}
                  y1={padding.top}
                  x2={getX(activeEpoch)}
                  y2={height - padding.bottom}
                  stroke="#3b82f6"
                  strokeWidth={1}
                  strokeDasharray="2 2"
                />
              )}

              {/* X Axis labels */}
              <text x={padding.left} y={height - 12} className="fill-slate-400 text-[10px]">
                Epoch 1
              </text>
              <text x={width / 2} y={height - 12} textAnchor="middle" className="fill-slate-500 text-[11px] font-medium">
                Training Epochs
              </text>
              <text x={width - padding.right} y={height - 12} textAnchor="end" className="fill-slate-400 text-[10px]">
                Epoch {totalEpochs}
              </text>
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}
