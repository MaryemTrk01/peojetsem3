from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, request, jsonify

from src.vibration_health import VibrationHealthModel
from src.audio_health import AudioHealthModel
from src.fusion_rul import FusionConfig, global_health, rul_from_health_linear

app = Flask(__name__)

BASE_DIR = Path(os.environ.get("PICUBE_BASE", str(Path(__file__).resolve().parent.parent)))
MODELS_DIR = BASE_DIR / "models"

# Lazy-loaded singletons (Pi-friendly)
_vib = None
_aud = None
_fcfg = FusionConfig()


def get_vib() -> VibrationHealthModel:
    global _vib
    if _vib is None:
        vib_dir = MODELS_DIR / "vibration"
        model_path = vib_dir / ("autoencoder_model.tflite" if os.environ.get("PICUBE_VIB_TFLITE") == "1" else "autoencoder_model.h5")
        use_tflite = model_path.suffix.lower() == ".tflite"
        _vib = VibrationHealthModel(
            model_path=str(model_path),
            scaler_path=str(vib_dir / "scaler_vibration.pkl"),
            health_params_path=str(vib_dir / "health_params.json"),
            use_tflite=use_tflite,
        )
    return _vib


def get_aud() -> AudioHealthModel:
    global _aud
    if _aud is None:
        aud_dir = MODELS_DIR / "audio"
        model_path = aud_dir / ("audio_autoencoder.tflite" if os.environ.get("PICUBE_AUD_TFLITE") == "1" else "audio_autoencoder.h5")
        use_tflite = model_path.suffix.lower() == ".tflite"
        _aud = AudioHealthModel(
            model_path=str(model_path),
            stats_dir=str(aud_dir),
            use_tflite=use_tflite,
        )
    return _aud


@app.get("/api/health/vibration")
def api_vibration_health():
    """
    Body JSON:
      {"accel": [[x,y,z], ...]}  length >= 2048
    """
    payload = request.get_json(force=True)
    accel = payload.get("accel", [])
    import numpy as np

    arr = np.asarray(accel, dtype=np.float32)
    vib = get_vib()
    mse, health = vib.health_from_samples(arr)
    if mse.size == 0:
        return jsonify({"error": "need at least 2048 samples"}), 400
    return jsonify({"mse": float(mse[-1]), "health": float(health[-1])})


@app.get("/api/health/audio")
def api_audio_health():
    """
    Body JSON:
      {"samples": [...], "sr": 16000}
    """
    payload = request.get_json(force=True)
    samples = payload.get("samples", [])
    import numpy as np

    segment = np.asarray(samples, dtype=np.float32)
    aud = get_aud()
    err, health = aud.health_from_audio_segment(segment)
    return jsonify({"error": float(err), "health": float(health)})


@app.get("/api/fusion")
def api_fusion():
    """
    Body JSON:
      {"health_vib": 90.2, "health_audio": 88.1, "temp_c": 75.0}
    """
    payload = request.get_json(force=True)
    hv = float(payload.get("health_vib", 0.0))
    ha = float(payload.get("health_audio", 0.0))
    t = float(payload.get("temp_c", 0.0))
    H = global_health(hv, ha, t, _fcfg)
    rul = rul_from_health_linear(H, _fcfg)
    return jsonify({"health_global": float(H), "rul_days": float(rul)})


@app.get("/api/ping")
def ping():
    return jsonify({"ok": True})


if __name__ == "__main__":
    # On Pi: python -m src.flask_app
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
