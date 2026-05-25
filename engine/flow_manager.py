import base64
import json
import math
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import onnxruntime as ort
import torch
from diffusers import DDIMScheduler
from huggingface_hub import hf_hub_download
from PIL import Image
from qai_hub_models.models._shared.stable_diffusion.app import (
    OUT_H,
    OUT_W,
    StableDiffusionApp,
)
from qai_hub_models.utils.onnx.torch_wrapper import OnnxModelTorchWrapper
from transformers import CLIPTokenizer

from model_loader import (
    CONTROLNET_CANNY_DEFAULT_HIGH_THRESHOLD,
    CONTROLNET_CANNY_DEFAULT_LOW_THRESHOLD,
    QUALCOMM_REFERENCE_ONNXRUNTIME_QNN_VERSION,
    STABLE_DIFFUSION_DEFAULT_GUIDANCE_SCALE,
    STABLE_DIFFUSION_DEFAULT_SEED,
    STABLE_DIFFUSION_DEFAULT_STEPS,
    STABLE_DIFFUSION_SCHEDULER_CONFIG_NAME,
    STABLE_DIFFUSION_SCHEDULER_SUBFOLDER,
    normalize_model_id,
    resolve_model_source,
)


@dataclass(frozen=True)
class GenerationInputs:
    prompt: str
    num_steps: int
    guidance_scale: float
    seed: int


@dataclass(frozen=True)
class ControlNetCannyInputs(GenerationInputs):
    image: Image.Image
    canny_low_threshold: int
    canny_high_threshold: int


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
        raise ValueError(f"Field '{field_name}' must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"Field '{field_name}' must be an integer.")
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field '{field_name}' must be an integer.") from exc


def _coerce_float_field(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Field '{field_name}' must be a number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field '{field_name}' must be a number.") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(
            f"Field '{field_name}' must be a finite number greater than or equal to 0."
        )
    return parsed


def _coerce_generation_inputs(inputs: Any) -> GenerationInputs:
    if not isinstance(inputs, dict):
        raise TypeError(
            "Model input must be an object with 'prompt', 'num_steps', "
            "'guidance_scale', and 'seed' fields."
        )

    prompt = inputs.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Model input requires a non-empty prompt.")

    num_steps = _coerce_int_field(
        inputs.get("num_steps", STABLE_DIFFUSION_DEFAULT_STEPS),
        "num_steps",
    )
    if num_steps <= 0:
        raise ValueError("Field 'num_steps' must be greater than 0.")

    guidance_scale = _coerce_float_field(
        inputs.get("guidance_scale", STABLE_DIFFUSION_DEFAULT_GUIDANCE_SCALE),
        "guidance_scale",
    )
    seed = _coerce_int_field(inputs.get("seed", STABLE_DIFFUSION_DEFAULT_SEED), "seed")
    if seed < 0:
        raise ValueError("Field 'seed' must be greater than or equal to 0.")

    return GenerationInputs(
        prompt=prompt.strip(),
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )


def _decode_image_data_url(image_data_url: Any) -> Image.Image:
    if not isinstance(image_data_url, str) or not image_data_url.strip():
        raise ValueError("ControlNet-Canny requires a reference image data URL.")

    raw = image_data_url.strip()
    if "," in raw and raw.startswith("data:image"):
        raw = raw.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw, validate=True)
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("ControlNet-Canny reference image could not be decoded.") from exc


def _coerce_threshold(value: Any, field_name: str, default: int) -> int:
    parsed = _coerce_int_field(value if value is not None else default, field_name)
    if parsed < 0 or parsed > 255:
        raise ValueError(f"Field '{field_name}' must be between 0 and 255.")
    return parsed


def _coerce_controlnet_canny_inputs(inputs: Any) -> ControlNetCannyInputs:
    generation_inputs = _coerce_generation_inputs(inputs)
    assert isinstance(inputs, dict)

    low = _coerce_threshold(
        inputs.get("canny_low_threshold"),
        "canny_low_threshold",
        CONTROLNET_CANNY_DEFAULT_LOW_THRESHOLD,
    )
    high = _coerce_threshold(
        inputs.get("canny_high_threshold"),
        "canny_high_threshold",
        CONTROLNET_CANNY_DEFAULT_HIGH_THRESHOLD,
    )
    if low > high:
        raise ValueError(
            "Field 'canny_low_threshold' must be less than or equal to 'canny_high_threshold'."
        )

    return ControlNetCannyInputs(
        prompt=generation_inputs.prompt,
        num_steps=generation_inputs.num_steps,
        guidance_scale=generation_inputs.guidance_scale,
        seed=generation_inputs.seed,
        image=_decode_image_data_url(inputs.get("image_data_url")),
        canny_low_threshold=low,
        canny_high_threshold=high,
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


def _validate_qualcomm_runtime(model_spec: dict) -> None:
    installed_ort = ort.__version__
    exported_ort = _read_exported_onnxruntime_version(model_spec)
    installed_series = _version_series(installed_ort)
    exported_series = _version_series(exported_ort)
    reference_series = _version_series(QUALCOMM_REFERENCE_ONNXRUNTIME_QNN_VERSION)

    if exported_series and installed_series == exported_series:
        return

    message = (
        "Qualcomm model runtime mismatch. "
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


def _nchw_to_nhwc(tensor: torch.Tensor) -> torch.Tensor:
    return torch.permute(torch.as_tensor(tensor), (0, 2, 3, 1)).contiguous()


def _nhwc_to_nchw(tensor: torch.Tensor) -> torch.Tensor:
    return torch.permute(torch.as_tensor(tensor), (0, 3, 1, 2)).contiguous()


class ControlNetOnNPU:
    def __init__(self, model_path: Path):
        self._model = OnnxModelTorchWrapper.OnNPU(str(model_path))

    def __call__(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        text_emb: torch.Tensor,
        image_cond: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        outputs = self._model(
            _nchw_to_nhwc(latent),
            timestep,
            text_emb,
            _nchw_to_nhwc(image_cond),
        )
        if not isinstance(outputs, tuple):
            raise TypeError("ControlNet-Canny model returned an unexpected single output.")
        return tuple(_nhwc_to_nchw(output) for output in outputs)


class ControlUnetOnNPU:
    def __init__(self, model_path: Path):
        self._model = OnnxModelTorchWrapper.OnNPU(str(model_path))

    def __call__(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        text_emb: torch.Tensor,
        *controlnet_outputs: torch.Tensor,
    ) -> torch.Tensor:
        output = self._model(
            _nchw_to_nhwc(latent),
            timestep,
            text_emb,
            *[_nchw_to_nhwc(output) for output in controlnet_outputs],
        )
        if not isinstance(output, torch.Tensor):
            raise TypeError("ControlNet-Canny UNet returned an unexpected output tuple.")
        return _nhwc_to_nchw(output)


class VaeDecoderOnNPU:
    def __init__(self, model_path: Path):
        self._model = OnnxModelTorchWrapper.OnNPU(str(model_path))

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        output = self._model(_nchw_to_nhwc(latent))
        if not isinstance(output, torch.Tensor):
            raise TypeError("VAE decoder returned an unexpected output tuple.")
        return output


class StableDiffusionRunner:
    def __init__(self, model_spec: dict):
        self.model_spec = model_spec
        _validate_qualcomm_runtime(model_spec)
        self._app = StableDiffusionApp(
            OnnxModelTorchWrapper.OnNPU(str(model_spec["text_encoder"])),
            OnnxModelTorchWrapper.OnNPU(str(model_spec["vae_decoder"])),
            OnnxModelTorchWrapper.OnNPU(str(model_spec["unet"])),
            CLIPTokenizer.from_pretrained(model_spec["hf_repo"], subfolder="tokenizer"),
            _make_stable_diffusion_scheduler(model_spec),
            channel_last_latent=True,
        )

    def run(self, inputs: Any) -> Any:
        model_inputs = _coerce_generation_inputs(inputs)
        return self._app.generate_image(
            prompt=model_inputs.prompt,
            num_steps=model_inputs.num_steps,
            seed=model_inputs.seed,
            guidance_scale=model_inputs.guidance_scale,
        )


class ControlNetCannyRunner:
    def __init__(self, model_spec: dict):
        from qai_hub_models.models.controlnet_canny import Model as ControlNetCannyModel

        self.model_spec = model_spec
        _validate_qualcomm_runtime(model_spec)
        self._app = StableDiffusionApp(
            text_encoder=OnnxModelTorchWrapper.OnNPU(str(model_spec["text_encoder"])),
            vae_decoder=VaeDecoderOnNPU(model_spec["vae_decoder"]),
            unet=ControlUnetOnNPU(model_spec["unet"]),
            tokenizer=ControlNetCannyModel.make_tokenizer(),
            scheduler=ControlNetCannyModel.make_scheduler("DEFAULT"),
            channel_last_latent=False,
            controlnet=ControlNetOnNPU(model_spec["controlnet"]),
        )

    def run(self, inputs: Any) -> Any:
        from qai_hub_models.models._shared.stable_diffusion.utils import make_canny

        model_inputs = _coerce_controlnet_canny_inputs(inputs)
        cond_image = make_canny(
            model_inputs.image,
            OUT_H,
            OUT_W,
            model_inputs.canny_low_threshold,
            model_inputs.canny_high_threshold,
        )
        return self._app.generate_image(
            prompt=model_inputs.prompt,
            num_steps=model_inputs.num_steps,
            seed=model_inputs.seed,
            guidance_scale=model_inputs.guidance_scale,
            cond_image=cond_image,
        )


class ModelRunner:
    def __init__(self, model_id: str):
        self.model_id = normalize_model_id(model_id)
        self._runner = self._load(self.model_id)

    def _load(self, model_id: str):
        model_spec = resolve_model_source(model_id)
        runner = model_spec["runner"]
        if runner == "stable-diffusion":
            return StableDiffusionRunner(model_spec)
        if runner == "controlnet-canny":
            return ControlNetCannyRunner(model_spec)
        raise ValueError(f"Unsupported model runner '{runner}'.")

    def run(self, inputs: Any) -> Any:
        return self._runner.run(inputs)


class ModelRegistry:
    """Keeps loaded ModelRunner instances in memory to avoid reloading."""

    def __init__(self):
        self._runners: dict[str, ModelRunner] = {}

    def load(self, model_id: str, _task: str | None = None) -> ModelRunner:
        normalized = normalize_model_id(model_id)
        if normalized not in self._runners:
            self._runners[normalized] = ModelRunner(normalized)
        return self._runners[normalized]

    def get(self, model_id: str) -> Optional[ModelRunner]:
        return self._runners.get(normalize_model_id(model_id))

    def list_loaded(self) -> list[str]:
        return list(self._runners.keys())

    def clear_loaded_models(self):
        self._runners.clear()


model_registry = ModelRegistry()


def run_model(model_id: str, inputs: Any) -> Any:
    runner = model_registry.get(model_id)
    if runner is None:
        raise ValueError(
            f"Model '{model_id}' is not loaded. Call load_model first."
        )
    return runner.run(inputs)
