import React from "react";

interface ConfusionMatrixProps {
  matrix: number[][]; // 3x3
  classNames: string[];
}

export function ConfusionMatrix({ matrix, classNames }: ConfusionMatrixProps) {
  const shortNames = classNames.map((c) => c.replace("Iris ", ""));

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-100 pb-3">
        <div>
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-700">
            Confusion Matrix
          </h3>
          <p className="text-xs text-slate-500">
            Holdout Test Set &bull; 23 unseen samples
          </p>
        </div>
        <div className="rounded-full bg-blue-50 px-2.5 py-1 text-xs font-semibold text-blue-700">
          22 / 23 Correct (95.7%)
        </div>
      </div>

      {/* Heatmap Grid */}
      <div className="mt-5 flex flex-col items-center">
        <div className="mb-2 text-center text-xs font-semibold uppercase tracking-wider text-slate-500">
          Predicted Class &rarr;
        </div>

        <div className="flex items-center">
          {/* Y-Axis Label */}
          <div className="mr-3 -rotate-90 select-none text-xs font-semibold uppercase tracking-wider text-slate-500">
            Actual Class
          </div>

          <div>
            {/* Column Headers */}
            <div className="grid grid-cols-3 gap-2 pb-2 text-center text-xs font-bold text-slate-600">
              {shortNames.map((name) => (
                <div key={`col-${name}`} className="w-20 sm:w-24">
                  {name}
                </div>
              ))}
            </div>

            {/* Matrix Rows */}
            <div className="space-y-2">
              {matrix.map((row, rIdx) => (
                <div key={`row-${rIdx}`} className="flex items-center gap-2">
                  <div className="w-16 text-right text-xs font-bold text-slate-600">
                    {shortNames[rIdx]}
                  </div>
                  {row.map((val, cIdx) => {
                    const isDiagonal = rIdx === cIdx;
                    const isError = !isDiagonal && val > 0;

                    let bgClass = "bg-slate-50 border-slate-200 text-slate-400";
                    if (isDiagonal && val > 0) {
                      bgClass = "bg-blue-600 border-blue-600 text-white font-bold shadow-sm shadow-blue-500/20";
                    } else if (isError) {
                      bgClass = "bg-amber-100 border-amber-300 text-amber-900 font-bold";
                    }

                    return (
                      <div
                        key={`cell-${rIdx}-${cIdx}`}
                        className={`flex h-14 w-20 sm:w-24 flex-col items-center justify-center rounded-xl border text-center transition-all ${bgClass}`}
                      >
                        <span className="text-lg font-bold">{val}</span>
                        <span className="text-[10px] opacity-80">
                          {isDiagonal ? "True Pos" : val > 0 ? "Misclassified" : ""}
                        </span>
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap items-center justify-center gap-4 text-xs text-slate-500">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 rounded bg-blue-600"></span>
            <span>Diagonal: Correct Predictions</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 rounded bg-amber-100 border border-amber-300"></span>
            <span>Off-Diagonal: 1 Versicolor predicted as Virginica</span>
          </span>
        </div>
      </div>
    </div>
  );
}
