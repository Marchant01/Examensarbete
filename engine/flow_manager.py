import asyncio
import onnxruntime as ort
from typing import Any, Dict, List, Optional
from transformers import pipeline as hf_pipeline
from optimum.onnxruntime import (
    ORTModelForCausalLM,
    ORTModelForSeq2SeqLM,
    ORTModelForFeatureExtraction,
    ORTModelForImageClassification,
    ORTModelForCustomTasks,
)
from model_loader import TASK_CONFIG

print(ort.__version__)

def _try_import_diffusion():
    try:
        from optimum.onnxruntime.modeling_diffusion import (
            ORTStableDiffusionPipeline,
            ORTStableDiffusionImg2ImgPipeline,
        )
        return ORTStableDiffusionPipeline, ORTStableDiffusionImg2ImgPipeline
    except ImportError:
        return None, None

# Text-to-Text models
class ModelRunner:
    """
    Wraps an optimum ORT model together with its HF pipeline so that
    .run(input) accepts plain Python values and returns plain Python values
    suitable for chaining to the next step.
    """
    def __init__(self, repo_id: str, task: str, provider: str = "QNNExecutionProvider"):
        self.repo_id = repo_id
        self.task = task
        self.provider = provider
        self._pipe = self._load(repo_id, task, provider)
    
    def _load(self, repo_id: str, task: str, provider: str):
        if task in ("text-to-image", "image-to-image"):
            OrtDiff, OrtImg2Img = _try_import_diffusion()
            cls = OrtImg2Img if task == "image-to-image" else OrtDiff
            if cls is None:
                raise ImportError("Install optimum[diffusers] for diffusion model support.")
            # Diffusion pipelines are self-contained (tokenizer + scheduler + unet etc.)
            return cls.from_pretrained(repo_id, providers=[provider], subfolder="onnx")

        cfg = TASK_CONFIG.get(task)
        if cfg is None:
            raise ValueError(f"Unsupported task '{task}'.")

        ort_class = cfg["ort_class"]
        processor_fn = cfg["processor_fn"]
        pipeline_task = cfg["pipeline_task"]

        # Session Options
        so = ort.SessionOptions()
        so.enable_profiling = True
        # , provider_options=[{"backend_path": "QnnHtp.dll"}]

        model = ort_class.from_pretrained(repo_id, session_options=so, providers=[provider], provider_options=[{"backend_path": "QnnHtp.dll"}], subfolder="onnx")
        print(model.model.get_providers())
        processor = processor_fn(repo_id)

        # Build a standard HF pipeline backed by the ORT model.
        # This handles tokenization, batching, and decoding automatically.
        return hf_pipeline(pipeline_task, model=model, tokenizer=processor)

    def run(self, inputs: Any) -> Any:
        output = self._pipe(inputs)

        # ---- TEXT GENERATION ----
        if self.task in ("text-generation", "text-to-text"):
            if isinstance(output, list) and len(output) > 0:
                first = output[0]
                if "generated_text" in first:
                    return first["generated_text"]
                if "translation_text" in first:
                    return first["translation_text"]
                if "summary_text" in first:
                    return first["summary_text"]

        # ---- TEXT CLASSIFICATION ----
        if self.task == "text-classification":
            if isinstance(output, list) and len(output) > 0:
                return output[0]["label"]

        # ---- FEATURE EXTRACTION ----
        if self.task == "feature-extraction":
            return output[0]

        # ---- IMAGE CLASSIFICATION ----
        if self.task == "image-classification":
            if isinstance(output, list) and len(output) > 0:
                return output[0]["label"]

        # ---- DIFFUSION ----
        if self.task in ("text-to-image", "image-to-image"):
            # Diffusion pipelines return object with .images
            if hasattr(output, "images"):
                return output.images[0]

        return output

class ModelRegistry:
    """Keeps loaded ModelRunner instances in memory to avoid reloading."""

    def __init__(self):
        self._runners: Dict[str, ModelRunner] = {}

    def load(self, model_id: str, task: str, provider: str = "QNNExecutionProvider") -> ModelRunner:
        if model_id not in self._runners:
            self._runners[model_id] = ModelRunner(model_id, task, provider)
        return self._runners[model_id]

    def get(self, model_id: str) -> Optional[ModelRunner]:
        return self._runners.get(model_id)

    def list_loaded(self) -> List[str]:
        return list(self._runners.keys())

    def clear_loaded_models(self):
        self._runners.clear()

model_registry = ModelRegistry()

def run_flow(flow_steps: List[Dict], initial_input: Any) -> Any:
    """
    Execute a chain of models, passing each model's output as the next model's input.

    flow_steps example:
        [
            {"model_id": "Xenova/t5-small", "task": "text-to-text"},
            {"model_id": "Xenova/vit-base-patch16-224", "task": "image-classification"},
        ]

    The caller is responsible for ensuring the output type of step N is
    compatible with the input type expected by step N+1.
    """
    data = initial_input

    for step in flow_steps:
        model_id = step["model_id"]
        runner = model_registry.get(model_id)

        if runner is None:
            raise ValueError(
                f"Model '{model_id}' is not loaded. "
                f"Call model_registry.load('{model_id}', task='...') first."
            )

        data = runner.run(data)

    return data
