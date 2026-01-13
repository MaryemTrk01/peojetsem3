# PiCube – Raspberry Pi inference pack

This folder contains **inference-only** code for:
- Vibration health (Autoencoder)
- Audio health (CNN Autoencoder)
- Fusion + RUL (demo rule)

## Folder structure
- models/vibration/
  - autoencoder_model.h5 (or autoencoder_model.tflite)
  - scaler_vibration.pkl
  - health_params.json
- models/audio/
  - audio_autoencoder.h5 (or audio_autoencoder.tflite)
  - mel_min.npy / mel_max.npy
  - audio_mu.npy / audio_sigma.npy
  - health_params_audio.json (optional)

## Install (Raspberry Pi)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# if you converted models to tflite:
pip install tflite-runtime
```

## Run API (Flask)
```bash
python -m src.flask_app
# API: http://<pi-ip>:5000/api/ping
```

## Convert Keras -> TFLite (run on PC)
```bash
python -m src.convert_to_tflite --keras models/vibration/autoencoder_model.h5 --out models/vibration/autoencoder_model.tflite --fp16
python -m src.convert_to_tflite --keras models/audio/audio_autoencoder.h5 --out models/audio/audio_autoencoder.tflite --fp16
```
