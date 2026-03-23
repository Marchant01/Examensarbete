import asyncio
import os
from pathlib import Path, PureWindowsPath
from huggingface_hub import list_models

try:
    from optimum.onnxruntime import (
        ORTModelForCausalLM,
        ORTModelForFeatureExtraction,
        ORTModelForImageClassification,
        ORTModelForSeq2SeqLM,
    )
except ImportError:
    ORTModelForCausalLM = None
    ORTModelForFeatureExtraction = None
    ORTModelForImageClassification = None
    ORTModelForSeq2SeqLM = None

try:
    from transformers import AutoFeatureExtractor, AutoTokenizer
except ImportError:
    AutoFeatureExtractor = None
    AutoTokenizer = None

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = ROOT_DIR / "models"

STABLE_DIFFUSION_MODEL_ID = "stable-diffusion"
STABLE_DIFFUSION_HF_REPO = "sd2-community/stable-diffusion-2-1-base"
STABLE_DIFFUSION_DEFAULT_STEPS = 20
STABLE_DIFFUSION_DEFAULT_SEED = 41
STABLE_DIFFUSION_BUILD_DIR = "stable_diffusion_v2_1_w8a16"
STABLE_DIFFUSION_TARGET = "qualcomm-snapdragon-x-elite"
QUALCOMM_REFERENCE_ONNXRUNTIME_QNN_VERSION = "1.24.4"


def _load_tokenizer(repo_id: str):
    if AutoTokenizer is None:
        raise ImportError("Install `transformers` to use text tasks.")
    return AutoTokenizer.from_pretrained(repo_id)


def _load_feature_extractor(repo_id: str):
    if AutoFeatureExtractor is None:
        raise ImportError("Install `transformers` to use image classification.")
    return AutoFeatureExtractor.from_pretrained(repo_id)


TASK_CONFIG = {
    "text-generation": {
        "ort_class": ORTModelForCausalLM,
        "pipeline_task": "text-generation",
        "processor_fn": _load_tokenizer,
    },
    "text-to-text": {
        "ort_class": ORTModelForSeq2SeqLM,
        "pipeline_task": "text2text-generation",
        "processor_fn": _load_tokenizer,
    },
    "feature-extraction": {
        "ort_class": ORTModelForFeatureExtraction,
        "pipeline_task": "feature-extraction",
        "processor_fn": _load_tokenizer,
    },
    "image-classification": {
        "ort_class": ORTModelForImageClassification,
        "pipeline_task": "image-classification",
        "processor_fn": _load_feature_extractor,
    },
    "text-to-image": {
        "ort_class": None,
        "pipeline_task": "text-to-image",
        "processor_fn": None,
    },
    "image-to-image": {
        "ort_class": None,
        "pipeline_task": "image-to-image",
        "processor_fn": None,
    },
}

MODEL_FILTERS = list(TASK_CONFIG.keys())

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


def _stable_diffusion_candidates(models_dir: Path) -> list[Path]:
    base = models_dir / "text-to-image"
    return [
        base / STABLE_DIFFUSION_MODEL_ID / "precompiled" / STABLE_DIFFUSION_TARGET,
        base / STABLE_DIFFUSION_MODEL_ID,
        base / STABLE_DIFFUSION_BUILD_DIR / "precompiled" / STABLE_DIFFUSION_TARGET,
        base / STABLE_DIFFUSION_BUILD_DIR,
    ]


def resolve_stable_diffusion_assets(models_dir: str | Path | None = None) -> dict:
    models_dir = get_models_dir(models_dir)
    checked_paths: list[str] = []

    for candidate in _stable_diffusion_candidates(models_dir):
        checked_paths.append(_to_windows_path(candidate))

        text_encoder = candidate / "text_encoder" / "model.onnx"
        unet = candidate / "unet" / "model.onnx"
        vae_decoder = candidate / "vae_decoder" / "model.onnx"

        if text_encoder.is_file() and unet.is_file() and vae_decoder.is_file():
            metadata_path = candidate / "metadata.yaml"
            return {
                "repo_id": STABLE_DIFFUSION_MODEL_ID,
                "task": "text-to-image",
                "hf_repo": STABLE_DIFFUSION_HF_REPO,
                "model_root": candidate,
                "model_root_windows": _to_windows_path(candidate),
                "metadata_path": metadata_path,
                "metadata_path_windows": _to_windows_path(metadata_path),
                "text_encoder": text_encoder,
                "text_encoder_windows": _to_windows_path(text_encoder),
                "unet": unet,
                "unet_windows": _to_windows_path(unet),
                "vae_decoder": vae_decoder,
                "vae_decoder_windows": _to_windows_path(vae_decoder),
            }

    raise FileNotFoundError(
        "Stable Diffusion assets were not found. Expected Qualcomm export under one of: "
        + ", ".join(checked_paths)
    )


def resolve_model_source(model_id: str, task: str, models_dir: str | Path | None = None) -> dict:
    if task == "text-to-image":
        allowed_ids = {
            "",
            STABLE_DIFFUSION_MODEL_ID,
            STABLE_DIFFUSION_HF_REPO,
            STABLE_DIFFUSION_BUILD_DIR,
        }
        if model_id not in allowed_ids:
            raise ValueError(
                f"Unsupported text-to-image model '{model_id}'. "
                f"Use '{STABLE_DIFFUSION_MODEL_ID}'."
            )
        return resolve_stable_diffusion_assets(models_dir)

    return {
        "repo_id": model_id,
        "task": task,
    }


async def list_available_models(task: str, limit: int = 20):
    if task not in TASK_CONFIG:
        raise ValueError(f"Unknown task '{task}'. Valid tasks: {MODEL_FILTERS}")

    if task == "text-to-image":
        try:
            spec = resolve_stable_diffusion_assets()
        except FileNotFoundError:
            return []

        return [
            {
                "id": STABLE_DIFFUSION_MODEL_ID,
                "last_modified": None,
                "downloads": None,
                "path": spec["model_root_windows"],
            }
        ]

    models = await asyncio.to_thread(
        list_models,
        limit=limit,
        filter=(task, "onnx"),
    )
    return [
        {
            "id": m.id,
            "last_modified": m.last_modified,
            "downloads": m.downloads,
        }
        for m in models
    ]


async def install_model(repo_id: str, task: str, model_dir: str):
    if task not in TASK_CONFIG:
        raise ValueError(f"Unknown task '{task}'. Valid tasks: {MODEL_FILTERS}")

    if task == "text-to-image":
        spec = resolve_stable_diffusion_assets(model_dir)
        return {
            "repo_id": STABLE_DIFFUSION_MODEL_ID,
            "task": task,
            "model_name": STABLE_DIFFUSION_MODEL_ID,
            "cache_dir": spec["model_root_windows"],
        }

    local_dir = Path(model_dir) / task / repo_id
    local_dir.mkdir(parents=True, exist_ok=True)

    ort_class = TASK_CONFIG[task]["ort_class"]
    if ort_class is None:
        raise NotImplementedError(f"Installing '{task}' is not wired in this engine yet.")
    await asyncio.to_thread(
        ort_class.from_pretrained,
        repo_id,
        export=False,
        cache_dir=str(local_dir),
    )

    return {
        "repo_id": repo_id,
        "task": task,
        "model_name": repo_id.split("/")[-1],
        "cache_dir": _to_windows_path(local_dir),
    }


def list_installed_models(models_dir: str) -> dict:
    out = {}
    model_root = get_models_dir(models_dir)

    for task in MODEL_FILTERS:
        if task == "text-to-image":
            try:
                resolve_stable_diffusion_assets(model_root)
                out[task] = [STABLE_DIFFUSION_MODEL_ID]
            except FileNotFoundError:
                out[task] = []
            continue

        task_dir = model_root / task
        if not task_dir.is_dir():
            out[task] = []
            continue

        out[task] = [
            f"{author.name}/{model.name}"
            for author in task_dir.iterdir()
            if author.is_dir()
            for model in author.iterdir()
            if model.is_dir()
        ]

    return out


def list_model_filters():
    return MODEL_FILTERS
