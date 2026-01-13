from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.vibration_health import VibrationHealthModel
from src.audio_health import AudioHealthModel
from src.fusion_rul import FusionConfig, global_health, rul_from_health_linear


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vib_csv", help="CSV with AccX,AccY,AccZ")
    ap.add_argument("--audio_npy", help="NPY with audio samples (mono)")
    ap.add_argument("--temp_c", type=float, default=75.0)
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()

    base = Path(args.base)
    vib_dir = base / "models" / "vibration"
    aud_dir = base / "models" / "audio"

    vib = VibrationHealthModel(
        model_path=str(vib_dir / "autoencoder_model.h5"),
        scaler_path=str(vib_dir / "scaler_vibration.pkl"),
        health_params_path=str(vib_dir / "health_params.json"),
        use_tflite=False,
    )
    aud = AudioHealthModel(
        model_path=str(aud_dir / "audio_autoencoder.h5"),
        stats_dir=str(aud_dir),
        use_tflite=False,
    )

    hv = None
    if args.vib_csv:
        df = pd.read_csv(args.vib_csv)
        accel = df[["AccX", "AccY", "AccZ"]].values.astype(np.float32)
        mse, health = vib.health_from_samples(accel)
        hv = float(health[-1])
        print("Vibration:", {"mse": float(mse[-1]), "health": hv})

    ha = None
    if args.audio_npy:
        audio = np.load(args.audio_npy).astype(np.float32)
        err, health = aud.health_from_audio_segment(audio)
        ha = float(health)
        print("Audio:", {"error": float(err), "health": ha})

    if hv is not None and ha is not None:
        cfg = FusionConfig()
        H = global_health(hv, ha, float(args.temp_c), cfg)
        rul = rul_from_health_linear(H, cfg)
        print("Fusion:", {"health_global": H, "rul_days": rul})


if __name__ == "__main__":
    main()
