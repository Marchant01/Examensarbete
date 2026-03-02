import asyncio
from pathlib import Path
from huggingface_hub import list_models, snapshot_download
from transformers import AutoTokenizer, AutoProcessor, AutoFeatureExtractor, pipeline
from optimum.onnxruntime import (
    ORTModelForCausalLM,
    ORTModelForSeq2SeqLM,
    ORTModelForFeatureExtraction,
    ORTModelForImageClassification,
    ORTModelForCustomTasks,
)

def _try_import_diffusion():
    try:
        from optimum.onnxruntime.modeling_diffusion import ORTStableDiffusionPipeline, ORTStableDiffusionImg2ImgPipeline
        return ORTStableDiffusionPipeline, ORTStableDiffusionImg2ImgPipeline
    except ImportError:
        return None, None

# MODEL_FILTERS = ["text-to-image", "text-generation", "image-to-image"]

TASK_CONFIG = {
    "text-generation": {
        "ort_class": ORTModelForCausalLM,
        "pipeline_task": "text-generation",
        "processor_fn": lambda repo_id: AutoTokenizer.from_pretrained(repo_id),
    },
    "text-to-text": {
        "ort_class": ORTModelForSeq2SeqLM,
        "pipeline_task": "text2text-generation",
        "processor_fn": lambda repo_id: AutoTokenizer.from_pretrained(repo_id),
    },
    "feature-extraction": {
        "ort_class": ORTModelForFeatureExtraction,
        "pipeline_task": "feature-extraction",
        "processor_fn": lambda repo_id: AutoTokenizer.from_pretrained(repo_id),
    },
    "image-classification": {
        "ort_class": ORTModelForImageClassification,
        "pipeline_task": "image-classification",
        "processor_fn": lambda repo_id: AutoFeatureExtractor.from_pretrained(repo_id),
    },
    # Diffusion tasks are handled separately via ORTStableDiffusion* pipelines.
    "text-to-image": {"ort_class": None, "pipeline_task": "text-to-image", "processor_fn": None},
    "image-to-image": {"ort_class": None, "pipeline_task": "image-to-image", "processor_fn": None},
}

MODEL_FILTERS = list(TASK_CONFIG.keys())

async def list_available_models(task: str, limit: int = 20):
    if task not in TASK_CONFIG:
        raise ValueError(f"Unknown task '{task}'. Valid tasks: {MODEL_FILTERS}")

    models = await asyncio.to_thread(
        list_models,
        limit=limit, 
        filter=(task, "onnx"),
    )
    return [
        {
            "id": m.id, 
            "last_modified": m.last_modified, 
            "downloads": m.downloads
        } 
        for m in models
    ]

async def install_model(repo_id: str, task: str, model_dir: str):
    """
    Downloads models using Optimum/HF-hub to the models repository.
    """
    if task not in TASK_CONFIG:
        raise ValueError(f"Unknown task '{task}'. Valid tasks: {MODEL_FILTERS}")
        
    local_dir = Path(model_dir) / task / repo_id
    local_dir.mkdir(parents=True, exist_ok=True)

    if task in ("text-to-image", "image-to-image"):
        OrtDiff, OrtImg2Img = _try_import_diffusion()
        cls = OrtImg2Img if task == "image-to-image" else OrtDiff
        if cls is None:
            raise ImportError(
                "Diffusion support requires `optimum[diffusers]`. "
                "Install with: pip install optimum[diffusers]"
            )
        await asyncio.to_thread(
            cls.from_pretrained,
            repo_id,
            export=False,           # model is already ONNX on HF hub
            cache_dir=str(local_dir),
        )
    else:
        ort_class = TASK_CONFIG[task]["ort_class"]
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
        "cache_dir": str(local_dir),
    }

def list_installed_models(models_dir: str) -> dict:
    """Walk the models directory and list installed repo_ids per task."""
    out = {}
    model_root = Path(models_dir)
    for task in MODEL_FILTERS:
        task_dir = model_root / task
        if not task_dir.is_dir():
            out[task] = []
            continue
        out[task] = [
            f"{author.name}/{model.name}"
            for author in task_dir.iterdir() if author.is_dir()
            for model in author.iterdir() if model.is_dir()
        ]
    return out

# def find_onnx_model_path(repo_id: str) -> str:
#     """
#     Find directory matching repo_id anywhere under models_root
#     """
#     for path in repo_id.rglob(repo_id):
#         if path.is_dir() and path.name == repo_id:
#             return str(path)

#     raise FileNotFoundError(f"Model directory '{repo_id}' not found")

def list_model_filters():
    return MODEL_FILTERS