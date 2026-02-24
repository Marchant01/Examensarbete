import onnxruntime as ort
from typing import Dict, Any, List
from pathlib import Path

# Text-to-Text models
class BaseRunner:
    def __init__(self, model_path, input, providers=None):
        self.model_path = model_path
        self.providers = self._default_providers()
        self.session = self._createSession()
    
    def _default_providers(self):
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _createSession(self):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_profiling=True

        return ort.InferenceSession(
            self.model_path,
            sess_options=so,
            providers=self.providers # will need to be changed to be picked by the user, cpu for npu!
        )
        
    def run(self, inputs):
        model = _createSession()
        return self.session.run(None, inputs)

class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, BaseRunner] = {}

    def load(self, model_id: str, model_path: str, providers=None):
        if model_id not in self._models:
            self._models[model_id] = BaseRunner(model_path, providers)
        return self._models[model_id]

    def get(self, model_id: str):
        return self._models.get(model_id)

    def list_loaded(self):
        return list(self._models.keys())

    def clear_loaded_models(self):
        return self._models.clear()

model_registry = ModelRegistry()

def run_flow(flow_steps: List[Dict], initial_input: Dict):
    """
    flow_steps example:
    [
        {"model_id": "modelA"},
        {"model_id": "modelB"}
    ]
    """

    data = initial_input

    for step in flow_steps:
        model_id = step["model_id"]
        runner = model_registry.get(model_id)

        if not runner:
            raise ValueError(f"Model '{model_id}' not loaded")

        data = runner.run(data)

    return data

