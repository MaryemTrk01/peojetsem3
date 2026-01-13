import React from "react";

type Props = {
  title: string;
  value: string;
  subtitle?: string;
  tone?: "good" | "warn" | "bad" | "neutral";
};

const toneClass: Record<NonNullable<Props["tone"]>, string> = {
  good: "border-emerald-300/50 bg-gradient-to-br from-emerald-100/60 to-emerald-200/60 text-emerald-900 shadow-lg shadow-emerald-200/30",
  warn: "border-amber-300/50 bg-gradient-to-br from-amber-100/60 to-amber-200/60 text-amber-900 shadow-lg shadow-amber-200/30",
  bad: "border-rose-300/50 bg-gradient-to-br from-rose-100/60 to-rose-200/60 text-rose-900 shadow-lg shadow-rose-200/30",
  neutral:
    "border-sky-300/50 bg-gradient-to-br from-sky-100/60 to-sky-200/60 text-sky-900 shadow-lg shadow-sky-200/30",
};

export default function StatCard({
  title,
  value,
  subtitle,
  tone = "neutral",
}: Props) {
  return (
    <div
      className={`rounded-2xl border p-6 backdrop-blur-sm transform transition-all duration-300 hover:scale-105 hover:-translate-y-2 ${toneClass[tone]}`}
    >
      <div className="text-xs font-semibold uppercase tracking-widest opacity-75">
        {title}
      </div>
      <div className="mt-4 text-4xl font-bold drop-shadow-lg">{value}</div>
      {subtitle ? (
        <div className="mt-2 text-sm opacity-60 font-medium">{subtitle}</div>
      ) : null}

      {/* 3D Border Effect */}
      <div className="absolute inset-0 rounded-2xl border-t border-white/20 pointer-events-none"></div>
      <div className="absolute inset-0 rounded-2xl border-b border-black/40 pointer-events-none"></div>
    </div>
  );
}
