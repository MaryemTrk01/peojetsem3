import React, { useEffect, useRef, useState } from "react";
import StatCard from "./components/StatCard";
import HealthChart from "./components/HealthChart";
import { getJSON } from "./api";

type StreamPoint = {
  t: number;
  health_vib: number;
  health_audio: number;
  health_global: number;
  rul_days: number;
  temp_c: number;
};

function toneFromHealth(h: number) {
  if (h >= 80) return "good";
  if (h >= 50) return "warn";
  return "bad";
}

export default function App() {
  const [data, setData] = useState<StreamPoint[]>([]);
  const [latest, setLatest] = useState<StreamPoint | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(true);
  const refreshRate = 10000; // 10 seconds in ms
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const dataRef = useRef<StreamPoint[]>([]);

  // Fetch latest point and add to rolling window
  const fetchLatest = async () => {
    try {
      const point = await getJSON<StreamPoint>("/api/stream/summary");

      // Add to rolling window (keep last 300 points)
      const newData = [...dataRef.current, point];
      if (newData.length > 300) {
        newData.shift();
      }
      dataRef.current = newData;
      setData(newData);
      setLatest(point);

      if (loading) setLoading(false);
      if (error) setError(null);
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      console.error("[App] Error fetching data:", errMsg);
      setError(`Connection failed: ${errMsg}`);
    }
  };

  // Start polling when component mounts
  useEffect(() => {
    console.log("[App] Starting real-time data stream...");
    setLoading(true);

    // Fetch immediately
    fetchLatest();

    // Then poll every refreshRate ms
    if (isRunning) {
      intervalRef.current = setInterval(fetchLatest, refreshRate);
    }

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [refreshRate, isRunning]);

  const handleTogglePlayback = async () => {
    try {
      const newRunning = !isRunning;
      await getJSON(
        `/api/stream/controls?action=${newRunning ? "start" : "stop"}`
      );
      setIsRunning(newRunning);
    } catch (err) {
      console.error("Error toggling playback:", err);
    }
  };

  const handleReset = async () => {
    try {
      await getJSON("/api/stream/controls?action=reset");
      dataRef.current = [];
      setData([]);
      setLatest(null);
      fetchLatest();
    } catch (err) {
      console.error("Error resetting:", err);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
      {/* Header */}
      <header className="border-b border-slate-700 bg-slate-900/90 backdrop-blur">
        <div className="mx-auto max-w-7xl px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-2xl font-bold text-white">
                PiCube Dashboard
              </div>
              <div className="mt-1 text-sm text-slate-400">
                Virtual Simulation Mode | Real-time Health & RUL Monitoring
              </div>
            </div>
            <div className="flex items-center gap-4">
              <button
                onClick={handleTogglePlayback}
                className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
                  isRunning
                    ? "bg-red-600 hover:bg-red-700 text-white"
                    : "bg-green-600 hover:bg-green-700 text-white"
                }`}
              >
                {isRunning ? "⏸ Stop" : "▶ Play"}
              </button>
              <button
                onClick={handleReset}
                className="rounded-lg bg-slate-700 hover:bg-slate-600 px-4 py-2 text-sm font-medium text-white transition"
              >
                🔄 Reset
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8">
        {/* Error Alert */}
        {error && (
          <div className="mb-6 rounded-lg border border-red-500/50 bg-red-500/10 p-4 text-sm text-red-300">
            ⚠️ {error}
          </div>
        )}

        {/* Loading State */}
        {loading && !latest ? (
          <div className="flex h-96 items-center justify-center">
            <div className="text-center">
              <div className="mb-4 animate-spin text-4xl">⏳</div>
              <div className="text-lg text-slate-300">
                Loading simulation data...
              </div>
            </div>
          </div>
        ) : (
          <>
            {/* Stat Cards */}
            <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <StatCard
                title="Vibration Health"
                value={latest ? `${latest.health_vib.toFixed(1)}%` : "—"}
                subtitle="Current sensor reading"
                tone={
                  latest
                    ? (toneFromHealth(latest.health_vib) as any)
                    : "neutral"
                }
              />
              <StatCard
                title="Audio Health"
                value={latest ? `${latest.health_audio.toFixed(1)}%` : "—"}
                subtitle="Sound analysis"
                tone={
                  latest
                    ? (toneFromHealth(latest.health_audio) as any)
                    : "neutral"
                }
              />
              <StatCard
                title="Global Health"
                value={latest ? `${latest.health_global.toFixed(1)}%` : "—"}
                subtitle={latest ? `Temp: ${latest.temp_c.toFixed(1)}°C` : "—"}
                tone={
                  latest
                    ? (toneFromHealth(latest.health_global) as any)
                    : "neutral"
                }
              />
              <StatCard
                title="RUL (Days)"
                value={latest ? `${latest.rul_days.toFixed(1)}` : "—"}
                subtitle="Remaining useful life"
                tone="neutral"
              />
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <div className="rounded-lg border border-slate-700 bg-slate-800/50 p-6 backdrop-blur">
                <div className="mb-4 text-sm font-semibold text-slate-200">
                  Vibration Health Over Time
                </div>
                <div className="h-64">
                  {data.length > 0 ? (
                    <HealthChart
                      data={data}
                      dataKey="health_vib"
                      label="Vibration"
                      color="#3b82f6"
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center text-slate-500">
                      Waiting for data...
                    </div>
                  )}
                </div>
              </div>

              <div className="rounded-lg border border-slate-700 bg-slate-800/50 p-6 backdrop-blur">
                <div className="mb-4 text-sm font-semibold text-slate-200">
                  Audio Health Over Time
                </div>
                <div className="h-64">
                  {data.length > 0 ? (
                    <HealthChart
                      data={data}
                      dataKey="health_audio"
                      label="Audio"
                      color="#8b5cf6"
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center text-slate-500">
                      Waiting for data...
                    </div>
                  )}
                </div>
              </div>

              <div className="rounded-lg border border-slate-700 bg-slate-800/50 p-6 backdrop-blur lg:col-span-2">
                <div className="mb-4 text-sm font-semibold text-slate-200">
                  Global Health & RUL Prediction
                </div>
                <div className="h-64">
                  {data.length > 0 ? (
                    <HealthChart
                      data={data}
                      dataKey="health_global"
                      label="Global Health"
                      color="#10b981"
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center text-slate-500">
                      Waiting for data...
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Data Stats */}
            <div className="mt-8 rounded-lg border border-slate-700 bg-slate-800/50 p-6 backdrop-blur">
              <div className="text-sm text-slate-400">
                <div>
                  Points collected:{" "}
                  <span className="font-semibold text-slate-200">
                    {data.length}
                  </span>
                </div>
                <div className="mt-2 text-xs">
                  Simulation data loaded from: training/demo CSV & WAV files (or
                  synthetic if missing)
                </div>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
