from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def load_tflite_interpreter(model_path: str):
    """
    Load a TFLite interpreter.

    On Raspberry Pi it's common to install `tflite-runtime` (lighter than full TensorFlow).
    This helper will prefer `tflite_runtime` and fall back to `tensorflow.lite` if available.
    """
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
    except Exception:
        from tensorflow.lite import Interpreter  # type: ignore

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


@dataclass
class TFLiteModel:
    interpreter: object
    input_index: int
    output_index: int
    input_shape: Tuple[int, ...]

    @classmethod
    def from_file(cls, model_path: str) -> "TFLiteModel":
        itp = load_tflite_interpreter(model_path)
        input_details = itp.get_input_details()
        output_details = itp.get_output_details()
        if not input_details or not output_details:
            raise RuntimeError("Invalid TFLite model: missing input/output details")

        inp = input_details[0]
        out = output_details[0]
        return cls(
            interpreter=itp,
            input_index=int(inp["index"]),
            output_index=int(out["index"]),
            input_shape=tuple(int(x) for x in inp["shape"]),
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        self.interpreter.set_tensor(self.input_index, x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)
