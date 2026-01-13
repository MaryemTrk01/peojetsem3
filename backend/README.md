# Backend (Flask) - PiCube Dashboard Demo

## Run locally
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

API:
- GET /api/ping
- GET /api/vibration/timeseries
- GET /api/audio/timeseries?limit=30
- GET /api/fusion/timeseries
- GET /api/summary

## Raspberry Pi 4 notes
- Prefer `tflite-runtime` + convert models to `.tflite` for best performance.
- If TensorFlow install is hard on Pi, keep inference in TFLite mode.
