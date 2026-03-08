from pathlib import Path
import onnx
import numpy as np

from onnxruntime.quantization import QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import (
    get_qnn_qdq_config,
    qnn_preprocess_model
)


class DummyCalibrationReader:
    """
    Basic calibration reader.
    QNN requires a data reader even for simple cases.
    This version feeds random tensors matching model inputs.
    """

    def __init__(self, model_path: str, num_samples: int = 10):

        self.model = onnx.load(model_path)
        self.num_samples = num_samples
        self.count = 0

        self.inputs = {}

        for inp in self.model.graph.input:
            shape = [
                d.dim_value if d.dim_value > 0 else 1
                for d in inp.type.tensor_type.shape.dim
            ]
            self.inputs[inp.name] = shape


    def get_next(self):

        if self.count >= self.num_samples:
            return None

        self.count += 1

        sample = {}

        for name, shape in self.inputs.items():
            sample[name] = np.random.rand(*shape).astype(np.float32)

        return sample


def quantize_onnx_model(model_path: Path):

    model_path = Path(model_path)

    output_path = model_path.with_suffix(".qdq.onnx")
    preproc_path = model_path.with_suffix(".preproc.onnx")

    print(f"Quantizing {model_path.name}")

    reader = DummyCalibrationReader(str(model_path))

    # Preprocess model
    model_changed = qnn_preprocess_model(str(model_path), str(preproc_path))

    model_to_quantize = preproc_path if model_changed else model_path

    # QNN config
    qnn_config = get_qnn_qdq_config(
        str(model_to_quantize),
        reader,
        activation_type=QuantType.QUInt16,
        weight_type=QuantType.QUInt8,
    )

    # Quantize
    quantize(
        str(model_to_quantize),
        str(output_path),
        qnn_config,
    )

    print(f"Quantized model saved to {output_path}")

    return output_path