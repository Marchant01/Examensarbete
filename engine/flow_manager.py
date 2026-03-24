import math
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import onnxruntime as ort
from diffusers import DDIMScheduler
from huggingface_hub import hf_hub_download
from qai_hub_models.models._shared.stable_diffusion.app import StableDiffusionApp
from qai_hub_models.utils.onnx.torch_wrapper import OnnxModelTorchWrapper
from transformers import CLIPTokenizer, pipeline as hf_pipeline

from model_loader import (
    STABLE_DIFFUSION_DEFAULT_GUIDANCE_SCALE,
    QUALCOMM_REFERENCE_ONNXRUNTIME_QNN_VERSION,
    STABLE_DIFFUSION_DEFAULT_SEED,
    STABLE_DIFFUSION_DEFAULT_STEPS,
    STABLE_DIFFUSION_SCHEDULER_CONFIG_NAME,
    STABLE_DIFFUSION_SCHEDULER_SUBFOLDER,
    TASK_CONFIG,
    resolve_model_source,
)

print(ort.__version__)

QNN_PROVIDER = "QNNExecutionProvider"
QNN_PROVIDER_OPTIONS = {
    "backend_path": "QnnHtp.dll",
    "htp_performance_mode": "high_performance",
    "enable_htp_fp16_precision": "1",
}

def _session_options() -> ort.SessionOptions:
    session_options = ort.SessionOptions()
    session_options.enable_profiling = True
    session_options.add_session_config_entry("ep.context_embed_mode", "1")
    session_options.add_session_config_entry("ep.context_enable", "1")
    return session_options


@dataclass(frozen=True)
class TextToImageInputs:
    prompt: str
    num_steps: int
    guidance_scale: float
    seed: int


def _read_scheduler_config(config_path: Path) -> dict:
    try:
        with config_path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to read Stable Diffusion scheduler config at '{config_path}'."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Stable Diffusion scheduler config at '{config_path}' is not valid JSON."
        ) from exc


def _load_scheduler_config(model_spec: dict) -> tuple[dict, str]:
    scheduler_config_path = model_spec.get("scheduler_config_path")
    if isinstance(scheduler_config_path, Path) and scheduler_config_path.is_file():
        return _read_scheduler_config(scheduler_config_path), str(scheduler_config_path)

    try:
        config_path = Path(
            hf_hub_download(
                repo_id=model_spec["hf_repo"],
                filename=f"{STABLE_DIFFUSION_SCHEDULER_SUBFOLDER}/{STABLE_DIFFUSION_SCHEDULER_CONFIG_NAME}",
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to fetch Stable Diffusion scheduler config from Hugging Face repo "
            f"'{model_spec['hf_repo']}'."
        ) from exc

    return _read_scheduler_config(config_path), str(config_path)


def _make_stable_diffusion_scheduler(model_spec: dict):
    config, config_source = _load_scheduler_config(model_spec)
    config = dict(config)
    config.pop("_class_name", None)

    try:
        scheduler = DDIMScheduler.from_config(config)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to instantiate Stable Diffusion scheduler 'DDIMScheduler' from config '{config_source}'."
        ) from exc

    print("Stable Diffusion scheduler:", "DDIMScheduler", f"(config_source={config_source})")
    return scheduler


def _coerce_int_field(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Text-to-image field '{field_name}' must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"Text-to-image field '{field_name}' must be an integer.")
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Text-to-image field '{field_name}' must be an integer.") from exc


def _coerce_non_negative_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Text-to-image field '{field_name}' must be a number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Text-to-image field '{field_name}' must be a number.") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(
            f"Text-to-image field '{field_name}' must be a finite number greater than or equal to 0."
        )
    return parsed


def _coerce_text_to_image_inputs(inputs: Any) -> TextToImageInputs:
    if not isinstance(inputs, dict):
        raise TypeError(
            "Text-to-image input must be an object with 'prompt', 'num_steps', "
            "'guidance_scale', and 'seed' fields."
        )

    prompt = inputs.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Text-to-image input requires a non-empty prompt.")

    num_steps = _coerce_int_field(
        inputs.get("num_steps", STABLE_DIFFUSION_DEFAULT_STEPS),
        "num_steps",
    )
    if num_steps <= 0:
        raise ValueError("Text-to-image field 'num_steps' must be greater than 0.")

    guidance_scale = _coerce_non_negative_float(
        inputs.get("guidance_scale", STABLE_DIFFUSION_DEFAULT_GUIDANCE_SCALE),
        "guidance_scale",
    )
    seed = _coerce_int_field(inputs.get("seed", STABLE_DIFFUSION_DEFAULT_SEED), "seed")

    return TextToImageInputs(
        prompt=prompt.strip(),
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )


def _read_exported_onnxruntime_version(model_spec: dict) -> str | None:
    metadata_path = model_spec.get("metadata_path")
    if not metadata_path:
        return None

    try:
        text = metadata_path.read_text(encoding="utf-8")
    except OSError:
        return None

    match = re.search(r"^\s*onnx_runtime:\s*([^\s]+)\s*$", text, re.MULTILINE)
    if match:
        return match.group(1)
    return None


def _version_series(version: str | None) -> str | None:
    if not version:
        return None
    parts = version.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else version


def _validate_stable_diffusion_runtime(model_spec: dict) -> None:
    installed_ort = ort.__version__
    exported_ort = _read_exported_onnxruntime_version(model_spec)
    installed_series = _version_series(installed_ort)
    exported_series = _version_series(exported_ort)
    reference_series = _version_series(QUALCOMM_REFERENCE_ONNXRUNTIME_QNN_VERSION)

    if exported_series and installed_series == exported_series:
        return

    message = (
        "Stable Diffusion Qualcomm runtime mismatch. "
        f"Installed onnxruntime is {installed_ort}."
    )

    if exported_ort:
        message += f" This model bundle metadata reports onnx_runtime {exported_ort}."

    if reference_series:
        message += (
            " The Qualcomm reference StableDiffusion venv is using "
            f"onnxruntime-qnn {QUALCOMM_REFERENCE_ONNXRUNTIME_QNN_VERSION}."
        )

    message += (
        " Align this engine to the same 1.24.x onnxruntime-qnn family first. "
        "If the error remains, regenerate the model bundle with the same runtime "
        "family as the engine or replace it with a matching Qualcomm export."
    )

    raise RuntimeError(message)


class StableDiffusionRunner:
    def __init__(self, model_spec: dict):
        self.model_spec = model_spec
        _validate_stable_diffusion_runtime(model_spec)
        self._app = StableDiffusionApp(
            OnnxModelTorchWrapper.OnNPU(str(model_spec["text_encoder"])),
            OnnxModelTorchWrapper.OnNPU(str(model_spec["vae_decoder"])),
            OnnxModelTorchWrapper.OnNPU(str(model_spec["unet"])),
            CLIPTokenizer.from_pretrained(model_spec["hf_repo"], subfolder="tokenizer"),
            _make_stable_diffusion_scheduler(model_spec),
            channel_last_latent=True,
        )

    def run(self, inputs: Any) -> Any:
        text_to_image_inputs = _coerce_text_to_image_inputs(inputs)
        return self._app.generate_image(
            prompt=text_to_image_inputs.prompt,
            num_steps=text_to_image_inputs.num_steps,
            seed=text_to_image_inputs.seed,
            guidance_scale=text_to_image_inputs.guidance_scale,
        )


class ModelRunner:
    """
    Wrap an ORT-backed pipeline so .run(input) accepts plain Python values
    and returns plain Python values suitable for chaining.
    """

    def __init__(self, repo_id: str, task: str):
        self.repo_id = repo_id
        self.task = task
        self._pipe = self._load(repo_id, task)

    def _load(self, repo_id: str, task: str):
        if task == "text-to-image":
            model_spec = resolve_model_source(repo_id, task)
            return StableDiffusionRunner(model_spec)

        if task == "image-to-image":
            raise NotImplementedError("image-to-image is not wired to the Qualcomm demo setup.")

        cfg = TASK_CONFIG.get(task)
        if cfg is None:
            raise ValueError(f"Unsupported task '{task}'.")

        ort_class = cfg["ort_class"]
        if ort_class is None:
            raise ImportError(f"Missing runtime support for task '{task}'.")
        processor_fn = cfg["processor_fn"]
        pipeline_task = cfg["pipeline_task"]

        model = ort_class.from_pretrained(
            repo_id,
            session_options=_session_options(),
            providers=[QNN_PROVIDER],
            provider_options=[QNN_PROVIDER_OPTIONS],
        )

        processor = processor_fn(repo_id)
        if task == "image-classification":
            return hf_pipeline(pipeline_task, model=model, feature_extractor=processor)

        return hf_pipeline(pipeline_task, model=model, tokenizer=processor)

    def run(self, inputs: Any) -> Any:
        if self.task == "text-to-image":
            return self._pipe.run(inputs)

        output = self._pipe(inputs)

        if self.task in ("text-generation", "text-to-text"):
            if isinstance(output, list) and output:
                first = output[0]
                if "generated_text" in first:
                    return first["generated_text"]
                if "translation_text" in first:
                    return first["translation_text"]
                if "summary_text" in first:
                    return first["summary_text"]

        if self.task == "text-classification":
            if isinstance(output, list) and output:
                return output[0]["label"]

        if self.task == "feature-extraction":
            return output[0]

        if self.task == "image-classification":
            if isinstance(output, list) and output:
                return output[0]["label"]

        if self.task == "image-to-image" and hasattr(output, "images"):
            return output.images[0]

        return output


class ModelRegistry:
    """Keeps loaded ModelRunner instances in memory to avoid reloading."""

    def __init__(self):
        self._runners: Dict[str, ModelRunner] = {}

    def load(self, model_id: str, task: str) -> ModelRunner:
        if model_id not in self._runners:
            self._runners[model_id] = ModelRunner(model_id, task)
        return self._runners[model_id]

    def get(self, model_id: str) -> Optional[ModelRunner]:
        return self._runners.get(model_id)

    def list_loaded(self) -> List[str]:
        return list(self._runners.keys())

    def clear_loaded_models(self):
        self._runners.clear()


model_registry = ModelRegistry()


def run_flow(flow_steps: List[Dict], initial_input: Any) -> Any:
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
