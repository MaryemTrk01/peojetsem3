from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Iterable, Dict, Any, Union

import numpy as np
import joblib

try:
    from tensorflow.keras.models import load_model  # type: ignore
except Exception:
    load_model = None

from .utils_tflite import TFLiteModel


def create_windows(signal: np.ndarray, window_size: int = 2048, hop_size: int = 1024) -> np.ndarray:
    """Convert (T, 3) accel samples to (N, window_size, 3) windows."""
    if signal.ndim != 2 or signal.shape[1] != 3:
        raise ValueError(f"Expected signal shape (T,3), got {signal.shape}")
    if signal.shape[0] < window_size:
        return np.empty((0, window_size, 3), dtype=np.float32)

    windows = []
    for start in range(0, signal.shape[0] - window_size + 1, hop_size):
        end = start + window_size
        windows.append(signal[start:end, :])
    return np.asarray(windows, dtype=np.float32)


@dataclass
class VibrationHealthConfig:
    window_size: int = 2048
    hop_size: int = 1024


class VibrationHealthModel:
    """
    Wrapper inference for vibration autoencoder.

    Required files:
      - model: autoencoder_model.h5 (or .tflite if use_tflite=True)
      - scaler: scaler_vibration.pkl
      - health params JSON: health_params.json containing mse_min_ref and mse_ref

    Compatibility:
      - app.py may pass config_path=...  (alias of health_params_path)
      - older code may pass health_params_path=...
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        scaler_path: Union[str, Path],
        health_params_path: Optional[Union[str, Path]] = None,
        *,
        config_path: Optional[Union[str, Path]] = None,  # alias
        use_tflite: bool = False,
        config: Optional[VibrationHealthConfig] = None,
    ):
        self.config = config or VibrationHealthConfig()

        model_path = Path(model_path)
        scaler_path = Path(scaler_path)

        # config_path is just an alias for health_params_path
        if health_params_path is None and config_path is not None:
            health_params_path = config_path

        if health_params_path is None:
            raise ValueError(
                "Missing health params JSON path. Provide health_params_path=... or config_path=... "
                "(expected a JSON with keys: mse_min_ref, mse_ref)."
            )
        health_params_path = Path(health_params_path)

        # Validate files early
        if not model_path.exists():
            raise FileNotFoundError(f"Vibration model not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Vibration scaler not found: {scaler_path}")
        if not health_params_path.exists():
            raise FileNotFoundError(f"Vibration health params JSON not found: {health_params_path}")

        # Load scaler + params
        self.scaler = joblib.load(scaler_path)

        params = json.loads(health_params_path.read_text(encoding="utf-8"))
        if "mse_min_ref" not in params or "mse_ref" not in params:
            raise ValueError(
                f"Invalid health params JSON at {health_params_path}. Expected keys: mse_min_ref and mse_ref."
            )
        self.mse_min_ref = float(params["mse_min_ref"])
        self.mse_ref = float(params["mse_ref"])

        self.use_tflite = bool(use_tflite)
        self._tflite: Optional[TFLiteModel] = None
        self._keras = None

        if self.use_tflite:
            self._tflite = TFLiteModel.from_file(str(model_path))
        else:
            if load_model is None:
                raise RuntimeError(
                    "TensorFlow/Keras is not installed. Either install tensorflow, "
                    "or convert to TFLite and set use_tflite=True."
                )
            # ✅ FIX: Keras3 compatibility: do not deserialize training/compile config
            self._keras = load_model(str(model_path), compile=False)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        if self._tflite is not None:
            return self._tflite.predict(x)
        assert self._keras is not None
        return self._keras.predict(x, verbose=0)

    def health_from_samples(self, accel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        accel: (T,3) array.
        Returns per-window (mse, health_percent).
        """
        cfg = self.config
        X = create_windows(accel, cfg.window_size, cfg.hop_size)
        if X.shape[0] == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        # scale exactly like training: scaler applied on flattened (N*window,3)
        X_scaled = self.scaler.transform(X.reshape(-1, 3)).reshape(
            X.shape[0], cfg.window_size, 3
        ).astype(np.float32)

        X_pred = self._predict(X_scaled)
        mse = np.mean((X_scaled - X_pred) ** 2, axis=(1, 2)).astype(np.float32)

        # Linear mapping using stored references
        denom = max(1e-12, (self.mse_ref - self.mse_min_ref))
        health = 100.0 * (1.0 - np.clip((mse - self.mse_min_ref) / denom, 0.0, 1.0))
        return mse.astype(np.float32), health.astype(np.float32)


class CircularBuffer3:
    """Simple ring buffer for (window_size, 3)."""

    def __init__(self, window_size: int):
        self.window_size = int(window_size)
        self._buf = np.zeros((self.window_size, 3), dtype=np.float32)
        self._i = 0
        self.full = False

    def append(self, sample_xyz: np.ndarray) -> None:
        self._buf[self._i, :] = sample_xyz
        self._i = (self._i + 1) % self.window_size
        if self._i == 0:
            self.full = True

    def get(self) -> np.ndarray:
        if not self.full:
            raise RuntimeError("Buffer not full yet")
        # return in chronological order
        return np.concatenate([self._buf[self._i :], self._buf[: self._i]], axis=0)


def stream_vibration_health(
    model: VibrationHealthModel,
    sample_iter: Iterable[np.ndarray],
    compute_every: int = 1024,
) -> Iterable[Dict[str, Any]]:
    """
    Stream inference helper:
      - Fill a 2048-sample ring buffer
      - Compute health every `compute_every` new samples (default 1024 => 50% overlap)
    """
    cfg = model.config
    ring = CircularBuffer3(cfg.window_size)
    counter = 0

    for sample in sample_iter:
        ring.append(np.asarray(sample, dtype=np.float32))
        counter += 1
        if ring.full and counter % int(compute_every) == 0:
            window = ring.get()
            mse, health = model.health_from_samples(window)
            yield {"mse": float(mse[-1]), "health": float(health[-1])}
