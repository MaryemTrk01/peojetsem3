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
  if (h >= 50) return "good";
  if (h >= 30) return "warn";
  return "bad";
}

export default function App() {
  const [data, setData] = useState<StreamPoint[]>([]);
  const [latest, setLatest] = useState<StreamPoint | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(true);
  const refreshRate = 20000; // 20 seconds in ms
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
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800">
      {/* Header */}
      <header className="border-b border-slate-700/50 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="px-8 py-5">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-3">
                <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-red-500">
                  <span className="text-lg font-bold text-white">⚙️</span>
                </div>
                <div>
                  <div className="text-2xl font-bold text-white">
                    Predictive Maintenance
                  </div>
                  <div className="text-xs text-slate-400">
                    Real-time Health & RUL Monitoring
                  </div>
                </div>
              </div>
            </div>

            {/* Professional Control Group */}
            <div className="flex items-center gap-6 pl-8">
              {/* Status LED Indicators */}
              <div className="flex items-center gap-3">
                <div className="text-xs font-semibold text-slate-300 uppercase tracking-wide">
                  Status:
                </div>
                <div className="flex items-center gap-2">
                  {/* Green LED - Good */}
                  <div
                    className={`w-3 h-3 rounded-full transition-all duration-300 ${
                      latest && toneFromHealth(latest.health_global) === "good"
                        ? "bg-green-500 shadow-lg shadow-green-500/60 animate-pulse"
                        : "bg-slate-600"
                    }`}
                  ></div>

                  {/* Orange LED - Warning */}
                  <div
                    className={`w-3 h-3 rounded-full transition-all duration-300 ${
                      latest && toneFromHealth(latest.health_global) === "warn"
                        ? "bg-orange-500 shadow-lg shadow-orange-500/60 animate-pulse"
                        : "bg-slate-600"
                    }`}
                  ></div>

                  {/* Red LED - Bad */}
                  <div
                    className={`w-3 h-3 rounded-full transition-all duration-300 ${
                      latest && toneFromHealth(latest.health_global) === "bad"
                        ? "bg-red-500 shadow-lg shadow-red-500/60 animate-pulse"
                        : "bg-slate-600"
                    }`}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="px-8 py-8">
        {/* Error Alert */}
        {error && (
          <div className="mb-6 rounded-lg border border-red-500/40 bg-red-500/5 p-4 text-sm text-red-300 backdrop-blur">
            <div className="flex items-center gap-2">
              <span className="text-lg">⚠️</span>
              <span>{error}</span>
            </div>
          </div>
        )}

        {/* Loading State */}
        {loading && !latest ? (
          <div className="flex h-96 items-center justify-center">
            <div className="text-center">
              <div className="mb-4 animate-spin text-5xl">⏳</div>
              <div className="text-lg text-slate-300 font-medium">
                Loading simulation data...
              </div>
            </div>
          </div>
        ) : (
          <>
            {/* Stat Cards - Full Width Landscape Layout */}
            <div className="mb-10 grid grid-cols-2 gap-5 lg:grid-cols-5">
              <div className="group relative">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-blue-400 rounded-lg opacity-0 group-hover:opacity-10 blur transition-opacity duration-300"></div>
                <StatCard
                  title="Vibration Health"
                  value={latest ? `${latest.health_vib.toFixed(1)}%` : "—"}
                  subtitle="Current sensor"
                  tone={
                    latest
                      ? (toneFromHealth(latest.health_vib) as any)
                      : "neutral"
                  }
                />
              </div>

              <div className="group relative">
                <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-purple-400 rounded-lg opacity-0 group-hover:opacity-10 blur transition-opacity duration-300"></div>
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
              </div>

              <div className="group relative">
                <div className="absolute inset-0 bg-gradient-to-r from-emerald-600 to-emerald-400 rounded-lg opacity-0 group-hover:opacity-10 blur transition-opacity duration-300"></div>
                <StatCard
                  title="Global Health"
                  value={latest ? `${latest.health_global.toFixed(1)}%` : "—"}
                  subtitle="System status"
                  tone={
                    latest
                      ? (toneFromHealth(latest.health_global) as any)
                      : "neutral"
                  }
                />
              </div>

              <div className="group relative">
                <div className="absolute inset-0 bg-gradient-to-r from-amber-600 to-amber-400 rounded-lg opacity-0 group-hover:opacity-10 blur transition-opacity duration-300"></div>
                <StatCard
                  title="RUL (Days)"
                  value={latest ? `${latest.rul_days.toFixed(1)}` : "—"}
                  subtitle="Remaining useful life"
                  tone="neutral"
                />
              </div>

              <div className="group relative">
                <div className="absolute inset-0 bg-gradient-to-r from-red-600 to-orange-400 rounded-lg opacity-0 group-hover:opacity-10 blur transition-opacity duration-300"></div>
                <StatCard
                  title="Temperature"
                  value={latest ? `${latest.temp_c.toFixed(1)}°C` : "—"}
                  subtitle="System temperature"
                  tone="neutral"
                />
              </div>
            </div>

            {/* Charts Layout - 2 Top, 1 Bottom Full Width */}
            <div className="grid grid-cols-1 gap-6 mb-8">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Vibration Chart */}
                <div className="rounded-xl border border-slate-700/40 bg-gradient-to-br from-slate-800/60 to-slate-800/40 p-6 backdrop-blur-sm hover:border-slate-600/60 transition-all duration-300 shadow-xl">
                  <div className="mb-5 flex items-center gap-3">
                    <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-blue-500/20 border border-blue-500/40">
                      <span className="text-sm">📊</span>
                    </div>
                    <div className="font-semibold text-slate-100">
                      Vibration Health
                    </div>
                  </div>
                  <div className="h-72">
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

                {/* Audio Chart */}
                <div className="rounded-xl border border-slate-700/40 bg-gradient-to-br from-slate-800/60 to-slate-800/40 p-6 backdrop-blur-sm hover:border-slate-600/60 transition-all duration-300 shadow-xl">
                  <div className="mb-5 flex items-center gap-3">
                    <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-purple-500/20 border border-purple-500/40">
                      <span className="text-sm">🔊</span>
                    </div>
                    <div className="font-semibold text-slate-100">
                      Audio Health
                    </div>
                  </div>
                  <div className="h-72">
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
              </div>

              {/* Global Health Chart - Full Width */}
              <div className="rounded-xl border border-slate-700/40 bg-gradient-to-br from-slate-800/60 to-slate-800/40 p-6 backdrop-blur-sm hover:border-slate-600/60 transition-all duration-300 shadow-xl">
                <div className="mb-5 flex items-center gap-3">
                  <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-emerald-500/20 border border-emerald-500/40">
                    <span className="text-sm">💚</span>
                  </div>
                  <div className="font-semibold text-slate-100">
                    Global Health & RUL Prediction
                  </div>
                </div>
                <div className="h-72">
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

            {/* Data Stats Footer */}
            <div className="rounded-xl border border-slate-700/40 bg-gradient-to-r from-slate-800/40 to-slate-800/20 p-6 backdrop-blur-sm">
              <div className="flex items-center justify-between">
                <div className="space-y-2">
                  <div className="text-xs uppercase tracking-widest text-slate-400 font-semibold">
                    Monitoring Status
                  </div>
                  <div className="flex items-center gap-4">
                    <div>
                      <span className="text-slate-400 text-sm">
                        Data Points:{" "}
                      </span>
                      <span className="font-bold text-cyan-400 text-lg">
                        {data.length}
                      </span>
                    </div>
                    <div className="h-6 w-px bg-slate-700/50"></div>
                    <div>
                      <span className="text-slate-400 text-sm">Status: </span>
                      <span
                        className={`font-bold text-sm ${
                          isRunning ? "text-green-400" : "text-orange-400"
                        }`}
                      >
                        {isRunning ? "● Streaming" : "● Paused"}
                      </span>
                    </div>
                  </div>
                </div>
                <div className="text-right text-xs text-slate-500">
                  Powered by TensorFlow • Real-time Simulation
                </div>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
