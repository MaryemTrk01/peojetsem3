from __future__ import annotations

import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from threading import Lock

# Setup logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Import your inference modules (already included) ---
# Path: backend/inference/src/*.py
import sys
INFERENCE_DIR = Path(__file__).resolve().parent / "inference"
sys.path.insert(0, str(INFERENCE_DIR))

from src.vibration_health import VibrationHealthModel  # type: ignore
from src.audio_health import AudioHealthModel  # type: ignore
from src.fusion_rul import FusionRULModel  # type: ignore

try:
    import librosa  # type: ignore
except Exception:
    librosa = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

VIB_TS_CSV = DATA_DIR / "vib_health_timeseries.csv"
AUDIO_DIR = DATA_DIR / "audio_subset"

# --- Virtual Simulation State ---
class SimulationState:
    """Manages virtual simulation mode with streaming data"""
    def __init__(self):
        self.lock = Lock()
        self.sim_index = 0
        self.sim_data = []  # List of {t, health_vib, health_audio, health_global, rul_days, temp_c}
        self.sim_speed = 1  # points per second
        self.running = True
        self.last_update_time = time.time()
        self.load_demo_data()
    
    def load_demo_data(self):
        """Load or generate demo data for simulation"""
        logger.info("Loading simulation demo data...")
        
        # Load vibration data
        vib_data = self._load_vibration_data()
        
        # Load audio data
        aud_data = self._load_audio_data()
        
        # Use minimum length for both
        n_points = min(len(vib_data), len(aud_data))
        logger.info(f"Simulation will use {n_points} demo points")
        
        # Build combined timeseries
        for i in range(n_points):
            hv = float(vib_data[i])
            # Clip vibration between 10% and 95%
            hv = np.clip(hv, 10.0, 95.0)
            ha = float(aud_data[i])
            temp_c = 30.0 + (i / n_points) * 20  # 30-50°C range
            
            # Compute fusion (use fusion_model if available)
            fusion_result = fusion_model.predict(health_vib=hv, health_audio=ha, temp_c=temp_c)
            hg = float(fusion_result["health_global"]) if isinstance(fusion_result, dict) else 72.0
            rul = float(fusion_result["rul_days"]) if isinstance(fusion_result, dict) else 100.0
            
            self.sim_data.append({
                "t": i,
                "health_vib": hv,
                "health_audio": ha,
                "health_global": hg,
                "rul_days": rul,
                "temp_c": temp_c
            })
        
        logger.info(f"Loaded {len(self.sim_data)} simulation points")
    
    def _load_vibration_data(self) -> List[float]:
        """Load vibration health from CSV or generate synthetic"""
        try:
            df = pd.read_csv(VIB_TS_CSV)
            logger.info(f"CSV columns: {df.columns.tolist()}")
            cols = {c.lower(): c for c in df.columns}
            logger.info(f"Lowercased column map: {cols}")
            
            if "health_vib" in cols:
                logger.info(f"Loading health_vib column: {cols['health_vib']}")
                data = df[cols["health_vib"]].values.tolist()
                logger.info(f"Loaded {len(data)} vibration values, first 3: {data[:3]}")
                return data
            elif "health" in cols:
                logger.info(f"Loading health column: {cols['health']}")
                return df[cols["health"]].values.tolist()
            else:
                raise ValueError("No health column found")
        except Exception as e:
            logger.warning(f"Could not load vibration CSV: {e}. Using synthetic data.")
            # Generate synthetic: sine wave between 60-100
            t = np.linspace(0, 4*np.pi, 500)
            return (80 + 20*np.sin(t) + np.random.normal(0, 2, len(t))).tolist()
    
    def _load_audio_data(self) -> List[float]:
        """Load audio health from WAV or generate synthetic"""
        try:
            # Try to load from WAV files
            wavs = sorted([p for p in AUDIO_DIR.glob("*.wav")])
            if wavs:
                audio_health = []
                for wav_path in wavs[:500]:  # Limit to 500 files
                    try:
                        y, sr = librosa.load(str(wav_path), mono=True)
                        segs = audio_model._segment_1s(y)
                        healths = []
                        for seg in segs:
                            _, h = audio_model.health_from_audio_segment(seg)
                            healths.append(h)
                        if healths:
                            audio_health.append(float(np.min(healths)))
                    except:
                        pass
                if audio_health:
                    return audio_health
            raise ValueError("No WAV files processed")
        except Exception as e:
            logger.warning(f"Could not load audio data: {e}. Using synthetic data.")
            # Generate synthetic: cosine wave between 60-100
            t = np.linspace(0, 4*np.pi, 500)
            return (75 + 15*np.cos(t) + np.random.normal(0, 2, len(t))).tolist()
    
    def get_current_point(self) -> Dict[str, Any]:
        """Get the current simulation point, looping over data"""
        with self.lock:
            if not self.sim_data:
                return {}
            
            # Loop back to start if we reach the end
            idx = self.sim_index % len(self.sim_data)
            point = self.sim_data[idx].copy()
            
            # Update simulation index
            self.sim_index += self.sim_speed
            
            return point
    
    def get_window(self, size: int = 200) -> List[Dict[str, Any]]:
        """Get last N points from simulation"""
        with self.lock:
            if not self.sim_data:
                return []
            
            start_idx = max(0, self.sim_index - size)
            end_idx = self.sim_index
            
            window = []
            for i in range(start_idx, end_idx):
                idx = i % len(self.sim_data)
                window.append(self.sim_data[idx].copy())
            
            return window

# --- Create Flask app ---
app = Flask(__name__)
# Enable CORS for localhost and 127.0.0.1 on dev ports
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173", "http://127.0.0.1:5174"]}})

# --- Load models once (startup) ---
vib_model = VibrationHealthModel(
    model_path=INFERENCE_DIR / "models" / "vibration" / "autoencoder_model.h5",
    scaler_path=INFERENCE_DIR / "models" / "vibration" / "scaler_vibration.pkl",
    config_path=INFERENCE_DIR / "models" / "vibration" / "health_params.json",
)

audio_model = AudioHealthModel(
    model_path=INFERENCE_DIR / "models" / "audio" / "audio_autoencoder.h5",
    mel_min_path=INFERENCE_DIR / "models" / "audio" / "mel_min.npy",
    mel_max_path=INFERENCE_DIR / "models" / "audio" / "mel_max.npy",
    mu_path=INFERENCE_DIR / "models" / "audio" / "audio_mu.npy",
    sigma_path=INFERENCE_DIR / "models" / "audio" / "audio_sigma.npy",
)

fusion_model = FusionRULModel()


def _read_vibration_timeseries() -> pd.DataFrame:
    """
    Uses your prepared CSV (from training/validation) to demo the dashboard.
    Expected columns include: health (or health_percent, health_vib) and maybe mse / timestamp.
    """
    df = pd.read_csv(VIB_TS_CSV)
    # normalize column names
    cols = {c.lower(): c for c in df.columns}
    if "health" not in cols and "health_percent" in cols:
        df["health"] = df[cols["health_percent"]]
    elif "health" not in cols and "health_vib" in cols:
        df["health"] = df[cols["health_vib"]]
    elif "health" in cols:
        df["health"] = df[cols["health"]]
    else:
        # fallback: create dummy health
        df["health"] = 100.0
    # create x-axis index if no time column
    if not any(k in cols for k in ["t", "time", "timestamp", "cycle", "index", "t_step"]):
        df["index"] = np.arange(len(df))
    else:
        # pick the first matching time-like column
        for k in ["cycle", "time", "timestamp", "t", "t_step", "index"]:
            if k in cols:
                df["index"] = df[cols[k]]
                break
    return df


def _audio_health_from_wav(path: Path) -> Dict[str, Any]:
    if librosa is None:
        raise RuntimeError("librosa is not installed. Install it with: pip install librosa soundfile")
    y, sr = librosa.load(str(path), sr=audio_model.config.sampling_rate, mono=True)
    # segment into 1-second chunks and take the worst (lowest health)
    segs = audio_model._segment_1s(y)  # uses model config
    errs = []
    healths = []
    for seg in segs:
        err, h = audio_model.health_from_audio_segment(seg)
        errs.append(err)
        healths.append(h)
    if len(healths) == 0:
        return {"file": path.name, "error": None, "health": None}
    # worst-case for safety
    i = int(np.argmin(healths))
    return {"file": path.name, "error": float(errs[i]), "health": float(healths[i])}


@app.get("/api/ping")
def ping():
    return jsonify({"ok": True})


@app.get("/api/vibration/timeseries")
def vibration_timeseries():
    try:
        df = _read_vibration_timeseries()
        if df is None or len(df) == 0:
            logger.warning("Vibration data is empty, returning mock data")
            return jsonify([{"index": i, "health": float(75.0 + np.random.normal(0, 5))} for i in range(50)])
        out = df[["index", "health"]].copy()
        data = out.to_dict(orient="records")
        # Convert numpy types to native Python types for JSON serialization
        data = [{"index": int(d["index"]), "health": float(d["health"])} for d in data]
        logger.info(f"Vibration endpoint returning {len(data)} records")
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error in vibration_timeseries: {e}")
        return jsonify([{"index": i, "health": float(75.0 + np.random.normal(0, 5))} for i in range(50)])


@app.get("/api/audio/timeseries")
def audio_timeseries():
    """
    Demo series computed from wav files in backend/data/audio_subset.
    If no WAV files, generate synthetic data.
    Query params:
      - limit: int (default 30)
    """
    try:
        limit = int(request.args.get("limit", "30"))
        wavs = sorted([p for p in AUDIO_DIR.glob("*.wav")])[:limit]
        rows = []
        
        if wavs:
            # Use actual WAV files if available
            for i, p in enumerate(wavs):
                try:
                    r = _audio_health_from_wav(p)
                    r["index"] = i
                    rows.append(r)
                except Exception as e:
                    logger.warning(f"Error processing WAV {p}: {e}")
        
        if not rows:
            # Generate synthetic audio health data if no files exist
            logger.info(f"No audio files found, generating {limit} synthetic records")
            np.random.seed(42)
            for i in range(min(limit, 30)):
                health = 100 - (i * 1.5) + np.random.normal(0, 3)
                health = np.clip(health, 0, 100)
                rows.append({
                    "file": f"synthetic_{i:02d}.wav",
                    "error": float(0.1 + i * 0.01),
                    "health": float(health),
                    "index": i
                })
        
        logger.info(f"Audio endpoint returning {len(rows)} records")
        return jsonify(rows)
    except Exception as e:
        logger.error(f"Error in audio_timeseries: {e}")
        # Fallback: return synthetic data
        rows = []
        np.random.seed(42)
        for i in range(30):
            health = 100 - (i * 1.5) + np.random.normal(0, 3)
            health = np.clip(health, 0, 100)
            rows.append({"index": i, "health": float(health), "file": f"synthetic_{i:02d}.wav", "error": float(0.1)})
        return jsonify(rows)


@app.get("/api/fusion/timeseries")
def fusion_timeseries():
    try:
        vib_df = _read_vibration_timeseries()
        if vib_df is None or len(vib_df) == 0:
            logger.warning("Vibration data empty in fusion, using defaults")
            vib = pd.DataFrame({"index": range(30), "health": [75.0] * 30})
        else:
            vib = vib_df[["index", "health"]].copy().iloc[:30].reset_index(drop=True)
        
        # Get audio data from endpoint
        audio_rows = audio_timeseries().get_json()
        if not audio_rows or len(audio_rows) == 0:
            logger.warning("Audio data empty in fusion, using defaults")
            aud = pd.DataFrame({"index": range(len(vib)), "health": [70.0] * len(vib)})
        else:
            aud = pd.DataFrame(audio_rows)[["index", "health"]].copy()
        
        n = int(min(len(vib), len(aud)))
        vib = vib.iloc[:n].reset_index(drop=True)
        aud = aud.iloc[:n].reset_index(drop=True)
        
        # Temperature demo (can be replaced by real sensor)
        temp = np.linspace(30, 55, n)
        
        rows = []
        for i in range(n):
            hv = float(vib.loc[i, "health"])
            ha = float(aud.loc[i, "health"])
            t = float(temp[i])
            # fusion_model.predict() returns a dict with "health_global" and "rul_days"
            result = fusion_model.predict(health_vib=hv, health_audio=ha, temp_c=t)
            hg_val = float(result["health_global"]) if isinstance(result, dict) else float(result[0])
            rul_val = float(result["rul_days"]) if isinstance(result, dict) else float(result[1])
            rows.append({"index": i, "health_vib": hv, "health_audio": ha, "temp_c": t, "health_global": hg_val, "rul_days": rul_val})
        
        logger.info(f"Fusion endpoint returning {len(rows)} records")
        return jsonify(rows)
    except Exception as e:
        logger.error(f"Error in fusion_timeseries: {e}")
        # Fallback: return synthetic fusion data
        rows = []
        for i in range(30):
            rows.append({
                "index": i,
                "health_vib": float(75.0 + np.random.normal(0, 5)),
                "health_audio": float(70.0 + np.random.normal(0, 5)),
                "temp_c": float(30 + i * 0.83),
                "health_global": float(72.0 + np.random.normal(0, 4)),
                "rul_days": float(100 - i * 2)
            })
        return jsonify(rows)


@app.get("/api/summary")
def summary():
    try:
        vib = _read_vibration_timeseries()
        if vib is None or len(vib) == 0:
            last_v = 75.0
        else:
            last_v = float(vib["health"].iloc[-1])
        
        # compute latest audio from last wav in subset
        wavs = sorted([p for p in AUDIO_DIR.glob("*.wav")])
        last_a = None
        if wavs:
            try:
                last_a = _audio_health_from_wav(wavs[-1])["health"]
            except Exception as e:
                logger.warning(f"Error getting audio summary: {e}")
        
        temp_c = float(request.args.get("temp_c", "40"))
        if last_a is None:
            last_a = 70.0  # Default fallback
        
        # fusion_model.predict() returns a dict with "health_global" and "rul_days"
        fusion_result = fusion_model.predict(health_vib=last_v, health_audio=float(last_a), temp_c=temp_c)
        hg_val = float(fusion_result["health_global"]) if isinstance(fusion_result, dict) else float(fusion_result[0])
        rul_val = float(fusion_result["rul_days"]) if isinstance(fusion_result, dict) else float(fusion_result[1])
        
        result = {
            "health_vibration": float(last_v),
            "health_audio": float(last_a),
            "temp_c": float(temp_c),
            "health_global": hg_val,
            "rul_days": rul_val
        }
        logger.info(f"Summary endpoint returning: {result}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in summary: {e}")
        return jsonify({
            "health_vibration": 75.0,
            "health_audio": 70.0,
            "temp_c": 40.0,
            "health_global": 72.0,
            "rul_days": 100.0
        })


# --- Initialize Simulation State ---
logger.info("Initializing virtual simulation mode...")
sim_state = SimulationState()
logger.info("Virtual simulation mode ready!")


# --- Streaming Endpoints for Real-time Dashboard ---
@app.get("/api/stream/summary")
def stream_summary():
    """Return the current simulation point (real-time streaming)"""
    try:
        point = sim_state.get_current_point()
        if not point:
            logger.warning("No simulation data available")
            return jsonify({
                "t": 0,
                "health_vib": 75.0,
                "health_audio": 70.0,
                "health_global": 72.0,
                "rul_days": 100.0,
                "temp_c": 40.0
            })
        
        logger.info(f"Stream summary at t={point.get('t')}: "
                   f"hv={point.get('health_vib', 0):.1f}%, "
                   f"ha={point.get('health_audio', 0):.1f}%, "
                   f"hg={point.get('health_global', 0):.1f}%")
        return jsonify(point)
    except Exception as e:
        logger.error(f"Error in stream_summary: {e}")
        return jsonify({
            "t": 0,
            "health_vib": 75.0,
            "health_audio": 70.0,
            "health_global": 72.0,
            "rul_days": 100.0,
            "temp_c": 40.0
        })


@app.get("/api/stream/window")
def stream_window():
    """Return last N points for chart rendering"""
    try:
        size = int(request.args.get("size", "200"))
        window = sim_state.get_window(size)
        
        if not window:
            logger.warning("No simulation window data available")
            return jsonify({"data": []})
        
        logger.info(f"Stream window with {len(window)} points")
        return jsonify({"data": window})
    except Exception as e:
        logger.error(f"Error in stream_window: {e}")
        return jsonify({"data": []})


@app.get("/api/stream/controls")
def stream_controls():
    """Control simulation playback"""
    try:
        action = request.args.get("action", "status")
        
        if action == "start":
            sim_state.running = True
            logger.info("Simulation started")
            return jsonify({"status": "running"})
        elif action == "stop":
            sim_state.running = False
            logger.info("Simulation stopped")
            return jsonify({"status": "stopped"})
        elif action == "reset":
            sim_state.sim_index = 0
            logger.info("Simulation reset")
            return jsonify({"status": "reset"})
        elif action == "speed":
            speed = float(request.args.get("speed", "1"))
            sim_state.sim_speed = speed
            logger.info(f"Simulation speed set to {speed}")
            return jsonify({"speed": speed})
        else:
            return jsonify({"status": "running" if sim_state.running else "stopped"})
    except Exception as e:
        logger.error(f"Error in stream_controls: {e}")
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    # For Raspberry Pi: host=0.0.0.0
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
