from __future__ import annotations

import argparse
from pathlib import Path


def convert_keras_to_tflite(keras_path: Path, out_path: Path, fp16: bool = False) -> None:
    import tensorflow as tf  # type: ignore

    model = tf.keras.models.load_model(str(keras_path))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if fp16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keras", required=True, help="Path to .h5/.keras model")
    ap.add_argument("--out", required=True, help="Output .tflite path")
    ap.add_argument("--fp16", action="store_true", help="Use float16 weights (smaller, faster on many devices)")
    args = ap.parse_args()

    convert_keras_to_tflite(Path(args.keras), Path(args.out), fp16=bool(args.fp16))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
