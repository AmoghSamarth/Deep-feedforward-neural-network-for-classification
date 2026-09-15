"use client";

import React, { useState, useMemo } from "react";
import { FnnModelArtifact, predictIrisSample } from "@/lib/inference";
import { Sparkles, CheckCircle2, Sliders } from "lucide-react";

interface PredictorProps {
  model: FnnModelArtifact;
}

export function Predictor({ model }: PredictorProps) {
  // Default values matching sample Versicolor
  const [sepalLength, setSepalLength] = useState<number>(6.0);
  const [sepalWidth, setSepalWidth] = useState<number>(2.9);
  const [petalLength, setPetalLength] = useState<number>(4.5);
  const [petalWidth, setPetalWidth] = useState<number>(1.3);

  // Quick Iris presets from original application
  const presets = [
    { label: "Sample: Setosa", values: [5.0, 3.5, 1.4, 0.2] as const, color: "text-blue-600" },
    { label: "Sample: Versicolor", values: [6.0, 2.9, 4.5, 1.3] as const, color: "text-sky-600" },
    { label: "Sample: Virginica", values: [6.7, 3.1, 5.6, 2.4] as const, color: "text-indigo-600" },
  ];

  const applyPreset = (values: readonly [number, number, number, number]) => {
    setSepalLength(values[0]);
    setSepalWidth(values[1]);
    setPetalLength(values[2]);
    setPetalWidth(values[3]);
  };

  // Real-time FNN forward pass inference
  const prediction = useMemo(() => {
    return predictIrisSample([sepalLength, sepalWidth, petalLength, petalWidth], model);
  }, [sepalLength, sepalWidth, petalLength, petalWidth, model]);

  const speciesShort = prediction.predictedClass.replace("Iris ", "");

  return (
    <div className="space-y-6">
      {/* Preset Quick Buttons */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-500">
          <Sliders className="h-4 w-4 text-blue-600" />
          <span>Select an Iris Sample Preset or Adjust Sliders:</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {presets.map((preset) => (
            <button
              key={preset.label}
              onClick={() => applyPreset(preset.values)}
              className="rounded-xl border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 shadow-sm transition hover:border-blue-300 hover:bg-blue-50/50 hover:text-blue-700"
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Input Sliders Column (7 cols) */}
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm lg:col-span-7">
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-700">
            Iris Flower Morphological Measurements
          </h3>
          <p className="text-xs text-slate-500">
            Adjust the sliders below. Real-time inference executes on every change.
          </p>

          <div className="mt-6 space-y-5">
            {/* Sepal Length */}
            <div>
              <div className="flex justify-between text-xs font-semibold">
                <label htmlFor="sl-slider" className="text-slate-700">
                  Sepal Length (cm)
                </label>
                <span className="font-mono text-blue-600">{sepalLength.toFixed(1)} cm</span>
              </div>
              <input
                id="sl-slider"
                type="range"
                min="4.0"
                max="8.0"
                step="0.1"
                value={sepalLength}
                onChange={(e) => setSepalLength(parseFloat(e.target.value))}
                className="mt-2"
              />
              <div className="mt-0.5 flex justify-between text-[10px] text-slate-400">
                <span>4.0 cm</span>
                <span>8.0 cm</span>
              </div>
            </div>

            {/* Sepal Width */}
            <div>
              <div className="flex justify-between text-xs font-semibold">
                <label htmlFor="sw-slider" className="text-slate-700">
                  Sepal Width (cm)
                </label>
                <span className="font-mono text-blue-600">{sepalWidth.toFixed(1)} cm</span>
              </div>
              <input
                id="sw-slider"
                type="range"
                min="2.0"
                max="4.5"
                step="0.1"
                value={sepalWidth}
                onChange={(e) => setSepalWidth(parseFloat(e.target.value))}
                className="mt-2"
              />
              <div className="mt-0.5 flex justify-between text-[10px] text-slate-400">
                <span>2.0 cm</span>
                <span>4.5 cm</span>
              </div>
            </div>

            {/* Petal Length */}
            <div>
              <div className="flex justify-between text-xs font-semibold">
                <label htmlFor="pl-slider" className="text-slate-700">
                  Petal Length (cm)
                </label>
                <span className="font-mono text-blue-600">{petalLength.toFixed(1)} cm</span>
              </div>
              <input
                id="pl-slider"
                type="range"
                min="1.0"
                max="7.0"
                step="0.1"
                value={petalLength}
                onChange={(e) => setPetalLength(parseFloat(e.target.value))}
                className="mt-2"
              />
              <div className="mt-0.5 flex justify-between text-[10px] text-slate-400">
                <span>1.0 cm</span>
                <span>7.0 cm</span>
              </div>
            </div>

            {/* Petal Width */}
            <div>
              <div className="flex justify-between text-xs font-semibold">
                <label htmlFor="pw-slider" className="text-slate-700">
                  Petal Width (cm)
                </label>
                <span className="font-mono text-blue-600">{petalWidth.toFixed(1)} cm</span>
              </div>
              <input
                id="pw-slider"
                type="range"
                min="0.1"
                max="2.6"
                step="0.1"
                value={petalWidth}
                onChange={(e) => setPetalWidth(parseFloat(e.target.value))}
                className="mt-2"
              />
              <div className="mt-0.5 flex justify-between text-[10px] text-slate-400">
                <span>0.1 cm</span>
                <span>2.6 cm</span>
              </div>
            </div>
          </div>
        </div>

        {/* Prediction Results Column (5 cols) */}
        <div className="flex flex-col justify-between space-y-5 rounded-2xl border border-blue-200 bg-gradient-to-br from-blue-50/70 via-white to-indigo-50/40 p-6 shadow-sm lg:col-span-5">
          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-blue-700">
                FNN Prediction Result
              </span>
              <span className="inline-flex items-center gap-1 rounded-full bg-blue-100 px-2.5 py-0.5 text-xs font-bold text-blue-800">
                <Sparkles className="h-3 w-3" />
                Live Inference
              </span>
            </div>

            {/* Prominent Predicted Class */}
            <div className="mt-5 text-center">
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Identified Species
              </div>
              <div className="mt-1 text-3xl font-extrabold tracking-tight text-blue-900">
                {prediction.predictedClass}
              </div>
              <div className="mt-2 inline-flex items-center gap-1.5 rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 shadow-sm border border-slate-200">
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
                <span>Confidence: {(prediction.confidence * 100).toFixed(2)}%</span>
              </div>
            </div>

            {/* Softmax Probability Distribution */}
            <div className="mt-6 space-y-3">
              <div className="text-xs font-bold uppercase tracking-wider text-slate-600">
                Softmax Class Probabilities
              </div>

              {model.classNames.map((cName) => {
                const prob = prediction.probabilities[cName] ?? 0;
                const percent = (prob * 100).toFixed(1);
                const isWinner = cName === prediction.predictedClass;

                return (
                  <div key={cName} className="space-y-1">
                    <div className="flex justify-between text-xs font-semibold">
                      <span className={isWinner ? "font-bold text-blue-700" : "text-slate-600"}>
                        {cName.replace("Iris ", "")}
                      </span>
                      <span className={isWinner ? "font-bold text-blue-700 font-mono" : "text-slate-500 font-mono"}>
                        {percent}%
                      </span>
                    </div>
                    <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
                      <div
                        className={`h-full rounded-full transition-all duration-300 ${
                          isWinner ? "bg-blue-600" : "bg-slate-300"
                        }`}
                        style={{ width: `${Math.max(2, parseFloat(percent))}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Mathematical Verification Details */}
          <div className="rounded-xl border border-slate-200 bg-white/90 p-3 text-[11px] text-slate-600">
            <div className="font-semibold text-slate-700">Technical Vector Pipeline:</div>
            <div className="mt-1 flex justify-between font-mono text-[10px]">
              <span>Raw [x]:</span>
              <span>[{sepalLength.toFixed(1)}, {sepalWidth.toFixed(1)}, {petalLength.toFixed(1)}, {petalWidth.toFixed(1)}]</span>
            </div>
            <div className="mt-0.5 flex justify-between font-mono text-[10px]">
              <span>Scaled [z]:</span>
              <span>[{prediction.scaledInput.map((v) => v.toFixed(2)).join(", ")}]</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
