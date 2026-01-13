import React from 'react';

type Props = {
  title: string;
  value: string;
  subtitle?: string;
  tone?: 'good' | 'warn' | 'bad' | 'neutral';
};

const toneClass: Record<NonNullable<Props['tone']>, string> = {
  good: 'border-emerald-200 bg-emerald-50 text-emerald-900',
  warn: 'border-amber-200 bg-amber-50 text-amber-900',
  bad: 'border-rose-200 bg-rose-50 text-rose-900',
  neutral: 'border-slate-200 bg-white text-slate-900',
};

export default function StatCard({ title, value, subtitle, tone='neutral' }: Props) {
  return (
    <div className={`rounded-2xl border p-4 shadow-sm ${toneClass[tone]}`}>
      <div className="text-sm font-medium opacity-80">{title}</div>
      <div className="mt-2 text-3xl font-semibold">{value}</div>
      {subtitle ? <div className="mt-1 text-xs opacity-70">{subtitle}</div> : null}
    </div>
  );
}
