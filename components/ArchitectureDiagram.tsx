"use client";

import React, { useState } from "react";

export function ArchitectureDiagram() {
  const [hoveredLayer, setHoveredLayer] = useState<number | null>(null);

  // Layer configuration: [Count to render, label, subtitle, params, act]
  const layers = [
    { name: "INPUT", sub: "4 Features", count: 4, act: "None", weights: 0, biases: 0, total: 0 },
    { name: "HIDDEN 1", sub: "16 Neurons", count: 12, act: "ReLU", weights: 64, biases: 16, total: 80 },
    { name: "HIDDEN 2", sub: "16 Neurons", count: 12, act: "ReLU", weights: 256, biases: 16, total: 272 },
    { name: "HIDDEN 3", sub: "8 Neurons", count: 8, act: "ReLU", weights: 128, biases: 8, total: 136 },
    { name: "OUTPUT", sub: "3 Classes", count: 3, act: "Softmax", weights: 24, biases: 3, total: 27 },
  ];

  const svgWidth = 800;
  const svgHeight = 260;
  const xCoords = [80, 240, 400, 560, 720];

  // Compute node Y positions
  const nodeCoords = layers.map((layer, lIdx) => {
    const x = xCoords[lIdx];
    const n = layer.count;
    const spacing = 190 / (n + 1);
    const ys = [];
    for (let i = 0; i < n; i++) {
      ys.push(40 + (i + 1) * spacing);
    }
    return { x, ys, layerIdx: lIdx };
  });

  return (
    <div className="space-y-6">
      {/* SVG Topology Diagram */}
      <div className="overflow-x-auto rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="mx-auto h-auto w-full max-w-[800px] select-none"
        >
          {/* Background subtle grid pattern or gradient */}
          <defs>
            <linearGradient id="lineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#93c5fd" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.4" />
            </linearGradient>
          </defs>

          {/* Connective Weight Lines */}
          {nodeCoords.slice(0, -1).map((curr, idx) => {
            const next = nodeCoords[idx + 1];
            const isLayerActive = hoveredLayer === idx || hoveredLayer === idx + 1;
            return (
              <g key={`layer-conn-${idx}`}>
                {curr.ys.map((y1, i) =>
                  next.ys.map((y2, j) => (
                    <line
                      key={`line-${idx}-${i}-${j}`}
                      x1={curr.x}
                      y1={y1}
                      x2={next.x}
                      y2={y2}
                      stroke={isLayerActive ? "#2563eb" : "#cbd5e1"}
                      strokeOpacity={isLayerActive ? 0.35 : 0.15}
                      strokeWidth={isLayerActive ? 1.2 : 0.7}
                    />
                  ))
                )}
              </g>
            );
          })}

          {/* Layer Headers & Nodes */}
          {nodeCoords.map((layerData, idx) => {
            const info = layers[idx];
            const isHovered = hoveredLayer === idx;
            const isOutput = idx === 4;
            const isInput = idx === 0;

            const fillColor = isOutput
              ? "#1d4ed8"
              : isInput
              ? "#3b82f6"
              : isHovered
              ? "#2563eb"
              : "#60a5fa";

            return (
              <g
                key={`nodes-layer-${idx}`}
                onMouseEnter={() => setHoveredLayer(idx)}
                onMouseLeave={() => setHoveredLayer(null)}
                className="cursor-pointer"
              >
                {/* Header Title */}
                <text
                  x={layerData.x}
                  y={22}
                  textAnchor="middle"
                  className="fill-slate-900 text-[11px] font-bold uppercase tracking-wider"
                >
                  {info.name}
                </text>
                {/* Header Subtitle */}
                <text
                  x={layerData.x}
                  y={34}
                  textAnchor="middle"
                  className="fill-slate-500 text-[9.5px] font-medium"
                >
                  {info.sub} ({info.act})
                </text>

                {/* Nodes */}
                {layerData.ys.map((y, nodeIdx) => (
                  <circle
                    key={`circle-${idx}-${nodeIdx}`}
                    cx={layerData.x}
                    cy={y}
                    r={isHovered ? 7.5 : 6}
                    fill={fillColor}
                    stroke="#ffffff"
                    strokeWidth={1.8}
                    className="transition-all duration-150"
                  />
                ))}
              </g>
            );
          })}
        </svg>

        <div className="mt-2 text-center text-xs font-semibold text-slate-500">
          Input Layer (4) &rarr; Hidden Layers (16 &rarr; 16 &rarr; 8) &rarr; Output Layer (3) &bull; Hover columns to highlight connections
        </div>
      </div>

      {/* Parameter Count Breakdown Table */}
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-700">
          Trainable Parameter Count Breakdown
        </h3>
        <p className="mt-0.5 text-xs text-slate-500">
          Rigorous layer-by-layer accounting matching the Keras model architecture.
        </p>

        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-700">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-[11px] font-bold uppercase tracking-wider text-slate-600">
                <th className="py-2.5 px-3">Layer</th>
                <th className="py-2.5 px-3">Type & Activation</th>
                <th className="py-2.5 px-3">Input Dim</th>
                <th className="py-2.5 px-3">Output Dim</th>
                <th className="py-2.5 px-3">Weight Matrix (W)</th>
                <th className="py-2.5 px-3">Bias Vector (b)</th>
                <th className="py-2.5 px-3 text-right">Total Parameters</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              <tr className="hover:bg-slate-50/60">
                <td className="py-2.5 px-3 font-semibold text-slate-900">Layer 1</td>
                <td className="py-2.5 px-3">Dense + ReLU</td>
                <td className="py-2.5 px-3">4</td>
                <td className="py-2.5 px-3">16</td>
                <td className="py-2.5 px-3 font-mono">4 &times; 16 = 64</td>
                <td className="py-2.5 px-3 font-mono">16</td>
                <td className="py-2.5 px-3 font-mono font-bold text-slate-900 text-right">80</td>
              </tr>
              <tr className="hover:bg-slate-50/60">
                <td className="py-2.5 px-3 font-semibold text-slate-900">Layer 2</td>
                <td className="py-2.5 px-3">Dense + ReLU</td>
                <td className="py-2.5 px-3">16</td>
                <td className="py-2.5 px-3">16</td>
                <td className="py-2.5 px-3 font-mono">16 &times; 16 = 256</td>
                <td className="py-2.5 px-3 font-mono">16</td>
                <td className="py-2.5 px-3 font-mono font-bold text-slate-900 text-right">272</td>
              </tr>
              <tr className="hover:bg-slate-50/60">
                <td className="py-2.5 px-3 font-semibold text-slate-900">Layer 3</td>
                <td className="py-2.5 px-3">Dense + ReLU</td>
                <td className="py-2.5 px-3">16</td>
                <td className="py-2.5 px-3">8</td>
                <td className="py-2.5 px-3 font-mono">16 &times; 8 = 128</td>
                <td className="py-2.5 px-3 font-mono">8</td>
                <td className="py-2.5 px-3 font-mono font-bold text-slate-900 text-right">136</td>
              </tr>
              <tr className="hover:bg-slate-50/60">
                <td className="py-2.5 px-3 font-semibold text-slate-900">Layer 4</td>
                <td className="py-2.5 px-3">Dense + Softmax</td>
                <td className="py-2.5 px-3">8</td>
                <td className="py-2.5 px-3">3</td>
                <td className="py-2.5 px-3 font-mono">8 &times; 3 = 24</td>
                <td className="py-2.5 px-3 font-mono">3</td>
                <td className="py-2.5 px-3 font-mono font-bold text-slate-900 text-right">27</td>
              </tr>
              <tr className="bg-blue-50/60 font-bold text-blue-900">
                <td className="py-3 px-3 uppercase tracking-wider" colSpan={4}>
                  Cumulative Network Parameters
                </td>
                <td className="py-3 px-3 font-mono">472 Weights</td>
                <td className="py-3 px-3 font-mono">43 Biases</td>
                <td className="py-3 px-3 font-mono text-base text-blue-700 text-right">
                  515 Parameters
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
