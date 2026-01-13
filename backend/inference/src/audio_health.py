from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

try:
    from tensorflow.keras.models import load_model  # type: ignore
except Exception:
    load_model = None

from .utils_tflite import TFLiteModel


# -----------------------------
# Mel spectrogram (no librosa)
# -----------------------------
def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: Optional[float] = None) -> np.ndarray:
    """
    Build mel filterbank matrix (n_mels, 1 + n_fft//2)
    Pure numpy implementation (portable to Raspberry Pi).
    """
    if fmax is None:
        fmax = sr / 2.0

    # FFT freq bins
    n_freqs = 1 + n_fft // 2
    freqs = np.linspace(0, sr / 2.0, n_freqs)

    # Mel points
    mels = np.linspace(hz_to_mel(np.array([fmin]))[0], hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz = mel_to_hz(mels)

    # Convert hz to fft bin numbers
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left:
            center += 1
        if right == center:
            right += 1
        if right <= left:
            continue

        # Rising slope
        for j in range(left, center):
            fb[i, j] = (j - left) / max(1, (center - left))
        # Falling slope
        for j in range(center, right):
            fb[i, j] = (right - j) / max(1, (right - center))

    return fb


def stft_mag(y: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """
    Compute magnitude STFT: (n_frames, 1+n_fft//2)
    """
    y = np.asarray(y, dtype=np.float32).flatten()
    if y.size < n_fft:
        # pad
        pad = n_fft - y.size
        y = np.pad(y, (0, pad), mode="constant")

    window = np.hanning(n_fft).astype(np.float32)

    n_frames = 1 + (y.size - n_fft) // hop_length if y.size >= n_fft else 1
    frames = []
    for i in range(n_frames):
        start = i * hop_length
        frame = y[start : start + n_fft]
        if frame.size < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.size), mode="constant")
        frames.append(frame * window)

    X = np.fft.rfft(np.stack(frames, axis=0), n=n_fft, axis=1)
    mag = np.abs(X).astype(np.float32)
    return mag


def audio_to_mel(y: np.ndarray, sr: int = 16000, n_fft: int = 1024, hop_length: int = 512, n_mels: int = 64) -> np.ndarray:
    """
    Returns mel spectrogram in shape (n_mels, n_frames)
    """
    mag = stft_mag(y, n_fft=n_fft, hop_length=hop_length)  # (frames, freqs)
    fb = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)  # (mels, freqs)
    mel = fb @ mag.T  # (mels, frames)
    mel = np.log10(np.maximum(mel, 1e-10)).astype(np.float32)
    return mel


# -----------------------------
# Model wrapper
# -----------------------------
@dataclass
class AudioHealthConfig:
    sr: int = 16000
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 64


class AudioHealthModel:
    """
    Audio autoencoder inference.

    Expected artifacts:
      - audio_autoencoder.h5 (or .tflite if use_tflite=True)
      - mel_min.npy, mel_max.npy (for normalization)
      - audio_mu.npy, audio_sigma.npy (for health mapping)

    Compatibility with app.py:
      app.py may call:
        AudioHealthModel(model_path=..., mel_min_path=..., mel_max_path=..., mu_path=..., sigma_path=...)
      Some older code may use:
        mel_min=..., mel_max=..., audio_mu=..., audio_sigma=...
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        *,
        mel_min_path: Optional[Union[str, Path]] = None,
        mel_max_path: Optional[Union[str, Path]] = None,
        mu_path: Optional[Union[str, Path]] = None,
        sigma_path: Optional[Union[str, Path]] = None,
        # aliases (optional)
        mel_min: Optional[Union[str, Path]] = None,
        mel_max: Optional[Union[str, Path]] = None,
        audio_mu: Optional[Union[str, Path]] = None,
        audio_sigma: Optional[Union[str, Path]] = None,
        use_tflite: bool = False,
        config: Optional[AudioHealthConfig] = None,
    ):
        self.config = config or AudioHealthConfig()

        model_path = Path(model_path)

        # Resolve aliases
        mel_min_path = mel_min_path or mel_min
        mel_max_path = mel_max_path or mel_max
        mu_path = mu_path or audio_mu
        sigma_path = sigma_path or audio_sigma

        if mel_min_path is None or mel_max_path is None or mu_path is None or sigma_path is None:
            raise ValueError(
                "Missing required paths. Provide mel_min_path, mel_max_path, mu_path, sigma_path "
                "(or aliases mel_min, mel_max, audio_mu, audio_sigma)."
            )

        mel_min_path = Path(mel_min_path)
        mel_max_path = Path(mel_max_path)
        mu_path = Path(mu_path)
        sigma_path = Path(sigma_path)

        # Validate files
        for p, name in [
            (model_path, "model_path"),
            (mel_min_path, "mel_min_path"),
            (mel_max_path, "mel_max_path"),
            (mu_path, "mu_path"),
            (sigma_path, "sigma_path"),
        ]:
            if not p.exists():
                raise FileNotFoundError(f"Audio artifact not found ({name}): {p}")

        # Load normalization + mapping stats
        self.mel_min = np.load(mel_min_path).astype(np.float32)
        self.mel_max = np.load(mel_max_path).astype(np.float32)

        self.mu = float(np.load(mu_path).reshape(-1)[0])
        self.sigma = float(np.load(sigma_path).reshape(-1)[0])

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
            # ✅ Keras 3 fix
            self._keras = load_model(str(model_path), compile=False)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        if self._tflite is not None:
            return self._tflite.predict(x)
        assert self._keras is not None
        return self._keras.predict(x, verbose=0)

    def _normalize_mel(self, mel: np.ndarray) -> np.ndarray:
        """
        Normalize mel using stored min/max.
        Works if mel_min/mel_max are scalars or per-bin arrays.
        """
        mmin = self.mel_min
        mmax = self.mel_max
        denom = np.maximum(mmax - mmin, 1e-12)
        x = (mel - mmin) / denom
        return np.clip(x, 0.0, 1.0).astype(np.float32)

    def health_from_audio(self, y: np.ndarray) -> Tuple[float, float]:
        """
        y: raw audio samples (float32) at 16kHz (or will be treated as such).
        Returns (reconstruction_error, health_percent)
        """
        cfg = self.config
        mel = audio_to_mel(y, sr=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length, n_mels=cfg.n_mels)
        mel_n = self._normalize_mel(mel)

        # Model expects (batch, n_mels, n_frames, 1)
        x = mel_n[np.newaxis, :, :, np.newaxis].astype(np.float32)
        x_pred = self._predict(x)

        err = float(np.mean((x - x_pred) ** 2))

        # Health mapping from your guide:
        # health = 100 * (1 - clip((err - mu)/(3*sigma), 0, 1))
        denom = max(1e-12, 3.0 * self.sigma)
        z = (err - self.mu) / denom
        health = 100.0 * (1.0 - float(np.clip(z, 0.0, 1.0)))

        return err, float(health)
