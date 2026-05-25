import os
from pathlib import Path, PureWindowsPath

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = ROOT_DIR / "models"

STABLE_DIFFUSION_MODEL_ID = "stable-diffusion"
STABLE_DIFFUSION_HF_REPO = "sd2-community/stable-diffusion-2-1-base"
STABLE_DIFFUSION_DEFAULT_STEPS = 20
STABLE_DIFFUSION_DEFAULT_SEED = 41
STABLE_DIFFUSION_DEFAULT_GUIDANCE_SCALE = 7.5
STABLE_DIFFUSION_BUILD_DIR = "stable_diffusion_v2_1_w8a16"

CONTROLNET_CANNY_MODEL_ID = "controlnet-canny"
CONTROLNET_CANNY_BUILD_DIR = "controlnet_canny"
CONTROLNET_CANNY_DEFAULT_LOW_THRESHOLD = 100
CONTROLNET_CANNY_DEFAULT_HIGH_THRESHOLD = 200

QUALCOMM_TARGET = "qualcomm-snapdragon-x-elite"
STABLE_DIFFUSION_SCHEDULER_SUBFOLDER = "scheduler"
STABLE_DIFFUSION_SCHEDULER_CONFIG_NAME = "scheduler_config.json"
QUALCOMM_REFERENCE_ONNXRUNTIME_QNN_VERSION = "1.24.4"

SUPPORTED_MODELS = {
    STABLE_DIFFUSION_MODEL_ID: {
        "id": STABLE_DIFFUSION_MODEL_ID,
        "label": "Stable Diffusion",
        "task": "text-to-image",
        "runner": "stable-diffusion",
    },
    CONTROLNET_CANNY_MODEL_ID: {
        "id": CONTROLNET_CANNY_MODEL_ID,
        "label": "ControlNet-Canny",
        "task": "controlnet-canny",
        "runner": "controlnet-canny",
    },
}

MODEL_ALIASES = {
    "": STABLE_DIFFUSION_MODEL_ID,
    STABLE_DIFFUSION_MODEL_ID: STABLE_DIFFUSION_MODEL_ID,
    STABLE_DIFFUSION_HF_REPO: STABLE_DIFFUSION_MODEL_ID,
    STABLE_DIFFUSION_BUILD_DIR: STABLE_DIFFUSION_MODEL_ID,
    CONTROLNET_CANNY_MODEL_ID: CONTROLNET_CANNY_MODEL_ID,
    CONTROLNET_CANNY_BUILD_DIR: CONTROLNET_CANNY_MODEL_ID,
    "ControlNet-Canny": CONTROLNET_CANNY_MODEL_ID,
    "qualcomm/ControlNet-Canny": CONTROLNET_CANNY_MODEL_ID,
}


def get_models_dir(models_dir: str | Path | None = None) -> Path:
    if models_dir is None:
        models_dir = os.environ.get("MODELS_DIR", str(DEFAULT_MODELS_DIR))
    return Path(models_dir).expanduser()


def _to_windows_path(path: Path) -> str:
    resolved = path.resolve()
    parts = resolved.parts

    if len(parts) >= 4 and parts[0] == "/" and parts[1] == "mnt" and len(parts[2]) == 1:
        drive = f"{parts[2].upper()}:"
        return str(PureWindowsPath(drive, *parts[3:]))

    return str(PureWindowsPath(*parts))


def normalize_model_id(model_id: str | None) -> str:
    key = model_id or ""
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    raise ValueError(
        f"Unsupported model '{model_id}'. Supported models: {list(SUPPORTED_MODELS)}"
    )


def _stable_diffusion_candidates(models_dir: Path) -> list[Path]:
    base = models_dir / "text-to-image"
    return [
        base / STABLE_DIFFUSION_MODEL_ID / "precompiled" / QUALCOMM_TARGET,
        base / STABLE_DIFFUSION_MODEL_ID,
        base / STABLE_DIFFUSION_BUILD_DIR / "precompiled" / QUALCOMM_TARGET,
        base / STABLE_DIFFUSION_BUILD_DIR,
    ]


def _controlnet_canny_candidates(models_dir: Path) -> list[Path]:
    base = models_dir / "text-to-image"
    return [
        base / CONTROLNET_CANNY_BUILD_DIR / "precompiled" / QUALCOMM_TARGET,
        base / CONTROLNET_CANNY_BUILD_DIR,
        base / CONTROLNET_CANNY_MODEL_ID / "precompiled" / QUALCOMM_TARGET,
        base / CONTROLNET_CANNY_MODEL_ID,
    ]


def _missing(paths: dict[str, Path]) -> list[str]:
    return [name for name, path in paths.items() if not path.is_file()]


def resolve_stable_diffusion_assets(models_dir: str | Path | None = None) -> dict:
    models_dir = get_models_dir(models_dir)
    checked_paths: list[str] = []

    for candidate in _stable_diffusion_candidates(models_dir):
        checked_paths.append(_to_windows_path(candidate))

        component_paths = {
            "text_encoder": candidate / "text_encoder" / "model.onnx",
            "unet": candidate / "unet" / "model.onnx",
            "vae_decoder": candidate / "vae_decoder" / "model.onnx",
        }
        missing = _missing(component_paths)
        if not missing:
            metadata_path = candidate / "metadata.yaml"
            scheduler_config = (
                candidate
                / STABLE_DIFFUSION_SCHEDULER_SUBFOLDER
                / STABLE_DIFFUSION_SCHEDULER_CONFIG_NAME
            )
            return {
                "id": STABLE_DIFFUSION_MODEL_ID,
                "runner": "stable-diffusion",
                "task": "text-to-image",
                "hf_repo": STABLE_DIFFUSION_HF_REPO,
                "model_root": candidate,
                "model_root_windows": _to_windows_path(candidate),
                "metadata_path": metadata_path,
                "metadata_path_windows": _to_windows_path(metadata_path),
                "scheduler_config_path": scheduler_config if scheduler_config.is_file() else None,
                **component_paths,
            }

    raise FileNotFoundError(
        "Stable Diffusion assets were not found. Expected Qualcomm export under one of: "
        + ", ".join(checked_paths)
    )


def resolve_controlnet_canny_assets(models_dir: str | Path | None = None) -> dict:
    models_dir = get_models_dir(models_dir)
    checked_paths: list[str] = []

    for candidate in _controlnet_canny_candidates(models_dir):
        checked_paths.append(_to_windows_path(candidate))

        component_paths = {
            "text_encoder": candidate / "text_encoder" / "text_encoder.onnx",
            "controlnet": candidate / "controlnet" / "controlnet.onnx",
            "unet": candidate / "unet" / "unet.onnx",
            "vae_decoder": candidate / "vae_decoder" / "vae.onnx",
        }
        missing = _missing(component_paths)
        if not missing:
            metadata_path = candidate / "metadata.yaml"
            return {
                "id": CONTROLNET_CANNY_MODEL_ID,
                "runner": "controlnet-canny",
                "task": "controlnet-canny",
                "model_root": candidate,
                "model_root_windows": _to_windows_path(candidate),
                "metadata_path": metadata_path,
                "metadata_path_windows": _to_windows_path(metadata_path),
                **component_paths,
            }

    raise FileNotFoundError(
        "ControlNet-Canny assets were not found. Expected Qualcomm export under one of: "
        + ", ".join(checked_paths)
    )


def resolve_model_source(
    model_id: str | None,
    _task: str | None = None,
    models_dir: str | Path | None = None,
) -> dict:
    normalized = normalize_model_id(model_id)
    if normalized == STABLE_DIFFUSION_MODEL_ID:
        return resolve_stable_diffusion_assets(models_dir)
    if normalized == CONTROLNET_CANNY_MODEL_ID:
        return resolve_controlnet_canny_assets(models_dir)
    raise ValueError(f"Unsupported model '{model_id}'.")


def _supported_model_status(model_id: str, models_dir: str | Path | None = None) -> dict:
    info = dict(SUPPORTED_MODELS[model_id])
    try:
        spec = resolve_model_source(model_id, models_dir=models_dir)
        info.update({
            "installed": True,
            "path": spec["model_root_windows"],
            "missing_components": [],
        })
    except FileNotFoundError as exc:
        info.update({
            "installed": False,
            "path": None,
            "missing_components": [str(exc)],
        })
    return info


def list_supported_models(models_dir: str | Path | None = None) -> dict:
    return {
        "models": [
            _supported_model_status(STABLE_DIFFUSION_MODEL_ID, models_dir),
            _supported_model_status(CONTROLNET_CANNY_MODEL_ID, models_dir),
        ]
    }
