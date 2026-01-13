import React, { useEffect, useMemo, useState } from "react";
import StatCard from "./components/StatCard";
import HealthChart from "./components/HealthChart";
import { getJSON } from "./api";

type VibPoint = { index: number; health: number };
type AudioPoint = { index: number; health: number; file: string };
type FusionPoint = {
  index: number;
  health_vib: number;
  health_audio: number;
  temp_c: number;
  health_global: number;
  rul_days: number;
};

function toneFromHealth(h: number) {
  if (h >= 80) return "good";
  if (h >= 50) return "warn";
  return "bad";
}

export default function App() {
  const [vib, setVib] = useState<VibPoint[]>([]);
  const [aud, setAud] = useState<AudioPoint[]>([]);
  const [fus, setFus] = useState<FusionPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const latest = useMemo(() => {
    const lastV = vib.length ? vib[vib.length - 1].health : null;
    const lastA = aud.length ? aud[aud.length - 1].health : null;
    const lastG = fus.length ? fus[fus.length - 1].health_global : null;
    const lastR = fus.length ? fus[fus.length - 1].rul_days : null;
    const temp = fus.length ? fus[fus.length - 1].temp_c : null;
    return { lastV, lastA, lastG, lastR, temp };
  }, [vib, aud, fus]);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      console.log("[App] Loading data from backend...");
      const [v, a, f] = await Promise.all([
        getJSON<VibPoint[]>("/api/vibration/timeseries"),
        getJSON<AudioPoint[]>("/api/audio/timeseries?limit=30"),
        getJSON<FusionPoint[]>("/api/fusion/timeseries"),
      ]);
      console.log("[App] Data loaded successfully:", {
        vib: v.length,
        audio: a.length,
        fusion: f.length,
      });
      setVib(v);
      setAud(a);
      setFus(f);
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      console.error("[App] Error loading data:", errMsg);
      setError(`Failed to load data: ${errMsg}`);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4">
          <div>
            <div className="text-lg font-semibold text-slate-900">
              PiCube Dashboard
            </div>
            <div className="text-xs text-slate-600">
              Vibration + Audio + Fusion/RUL (demo from training data)
            </div>
          </div>
          <button
            onClick={load}
            className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-900 shadow-sm hover:bg-slate-50"
          >
            Refresh
          </button>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6">
        {error && (
          <div className="mb-4 rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            {error}
          </div>
        )}
        {loading ? (
          <div className="rounded-2xl border border-slate-200 bg-white p-6 text-sm text-slate-700 shadow-sm">
            Loading…
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
              <StatCard
                title="Vibration Health"
                value={
                  latest.lastV === null ? "—" : `${latest.lastV.toFixed(1)}%`
                }
                subtitle="From vib_health_timeseries.csv"
                tone={
                  latest.lastV === null
                    ? "neutral"
                    : (toneFromHealth(latest.lastV) as any)
                }
              />
              <StatCard
                title="Audio Health"
                value={
                  latest.lastA === null ? "—" : `${latest.lastA.toFixed(1)}%`
                }
                subtitle="Worst segment of last WAV"
                tone={
                  latest.lastA === null
                    ? "neutral"
                    : (toneFromHealth(latest.lastA) as any)
                }
              />
              <StatCard
                title="Global Health"
                value={
                  latest.lastG === null ? "—" : `${latest.lastG.toFixed(1)}%`
                }
                subtitle={
                  latest.temp === null
                    ? ""
                    : `Temp: ${latest.temp.toFixed(1)}°C`
                }
                tone={
                  latest.lastG === null
                    ? "neutral"
                    : (toneFromHealth(latest.lastG) as any)
                }
              />
              <StatCard
                title="RUL"
                value={
                  latest.lastR === null
                    ? "—"
                    : `${latest.lastR.toFixed(1)} days`
                }
                subtitle="Fusion model (configurable)"
                tone="neutral"
              />
            </div>

            <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
              <HealthChart
                data={vib}
                yKey="health"
                label="Vibration Health (%)"
              />
              <HealthChart data={aud} yKey="health" label="Audio Health (%)" />
            </div>

            <div className="mt-6">
              <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                <div className="text-sm font-medium text-slate-900">
                  Fusion & RUL
                </div>
                <div className="mt-3 h-64">
                  <HealthChart
                    data={fus as any}
                    yKey="health_global"
                    label="Global Health (%)"
                  />
                </div>
                <div className="mt-3 text-xs text-slate-600">
                  Note: For demo, temperature is simulated. Replace
                  `/api/fusion/timeseries` to use real sensor values on
                  Raspberry Pi.
                </div>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
