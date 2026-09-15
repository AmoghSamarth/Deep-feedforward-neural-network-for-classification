import React from "react";

interface DatasetRecord {
  id: number;
  sepalLength: number;
  sepalWidth: number;
  petalLength: number;
  petalWidth: number;
  species: string;
}

interface FeatureDistributionsProps {
  dataset: DatasetRecord[];
}

export function FeatureDistributions({ dataset }: FeatureDistributionsProps) {
  const speciesList = ["Setosa", "Versicolor", "Virginica"];

  const features = [
    { key: "sepalLength" as const, label: "Sepal Length", unit: "cm", minLimit: 4.0, maxLimit: 8.0 },
    { key: "sepalWidth" as const, label: "Sepal Width", unit: "cm", minLimit: 2.0, maxLimit: 4.5 },
    { key: "petalLength" as const, label: "Petal Length", unit: "cm", minLimit: 1.0, maxLimit: 7.0 },
    { key: "petalWidth" as const, label: "Petal Width", unit: "cm", minLimit: 0.1, maxLimit: 2.6 },
  ];

  // Calculate min, mean, max for each species and feature
  const stats = features.map((feat) => {
    const speciesStats = speciesList.map((spec) => {
      const vals = dataset
        .filter((d) => d.species === spec)
        .map((d) => d[feat.key])
        .sort((a, b) => a - b);

      const min = vals[0] ?? 0;
      const max = vals[vals.length - 1] ?? 0;
      const avg = vals.reduce((a, b) => a + b, 0) / (vals.length || 1);
      const median = vals[Math.floor(vals.length / 2)] ?? 0;

      return { species: spec, min, max, avg, median };
    });

    return { ...feat, speciesStats };
  });

  const speciesColors: Record<string, { bg: string; border: string; text: string; bar: string }> = {
    Setosa: { bg: "bg-blue-50", border: "border-blue-200", text: "text-blue-700", bar: "bg-blue-600" },
    Versicolor: { bg: "bg-sky-50", border: "border-sky-200", text: "text-sky-700", bar: "bg-sky-500" },
    Virginica: { bg: "bg-indigo-50", border: "border-indigo-200", text: "text-indigo-700", bar: "bg-indigo-600" },
  };

  return (
    <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
      {stats.map((item) => (
        <div key={item.key} className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="border-b border-slate-100 pb-2.5">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">
              {item.label}
            </h4>
            <p className="text-[11px] text-slate-400">Values in {item.unit}</p>
          </div>

          <div className="mt-4 space-y-3.5">
            {item.speciesStats.map((s) => {
              const range = item.maxLimit - item.minLimit;
              const leftPercent = Math.max(0, ((s.min - item.minLimit) / range) * 100);
              const widthPercent = Math.max(8, ((s.max - s.min) / range) * 100);
              const medianPercent = Math.max(0, ((s.median - item.minLimit) / range) * 100);
              const color = speciesColors[s.species];

              return (
                <div key={s.species} className="text-xs">
                  <div className="flex items-center justify-between pb-1 text-[11px] font-semibold">
                    <span className={color.text}>{s.species}</span>
                    <span className="font-mono text-slate-600">
                      &mu; {s.avg.toFixed(1)} {item.unit}
                    </span>
                  </div>

                  {/* Range visualizer track */}
                  <div className="relative h-4 rounded-md bg-slate-100 p-0.5">
                    {/* Range span */}
                    <div
                      className={`absolute top-1 bottom-1 rounded-sm opacity-60 ${color.bar}`}
                      style={{ left: `${leftPercent}%`, width: `${widthPercent}%` }}
                    />
                    {/* Median pin */}
                    <div
                      className="absolute top-0 bottom-0 w-1 rounded-full bg-slate-900 shadow-sm"
                      style={{ left: `${medianPercent}%` }}
                    />
                  </div>

                  <div className="mt-1 flex justify-between text-[10px] text-slate-400">
                    <span>Min: {s.min.toFixed(1)}</span>
                    <span>Max: {s.max.toFixed(1)}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
