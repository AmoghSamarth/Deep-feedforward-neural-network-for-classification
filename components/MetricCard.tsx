import React from "react";

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  highlight?: boolean;
}

export function MetricCard({ title, value, subtitle, highlight }: MetricCardProps) {
  return (
    <div
      className={`rounded-xl border p-5 text-center transition-all duration-200 ${
        highlight
          ? "border-blue-200 bg-blue-50/50 shadow-sm shadow-blue-500/10"
          : "border-slate-200 bg-white hover:border-slate-300 hover:shadow-sm"
      }`}
    >
      <div className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
        {title}
      </div>
      <div className={`mt-1.5 text-2xl font-bold tracking-tight ${highlight ? "text-blue-600" : "text-slate-900"}`}>
        {value}
      </div>
      {subtitle && (
        <div className="mt-1 text-xs text-slate-500">
          {subtitle}
        </div>
      )}
    </div>
  );
}
