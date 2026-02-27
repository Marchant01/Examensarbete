import asyncio
from pathlib import Path
from huggingface_hub import list_models, snapshot_download

MODEL_FILTERS = ["text-to-image", "text-generation", "image-to-image"]

async def list_available_models(task: str, limit: int = 20):
    if task not in MODEL_FILTERS:
        raise ValueError(f"Uknown task")

    models = await asyncio.to_thread(
        list_models,
        limit=limit, 
        filter=(task, "onnx")
    )

    return [{
        "id": m.id, 
        "last_modified": m.last_modified, 
        "downloads": m.downloads} for m in models]

async def install_model(repo_id: str, task: str, model_dir: str):
    if task not in MODEL_FILTERS:
        raise ValueError(f"Unknown task")
    local_dir = Path(model_dir) / task / repo_id
    local_dir.mkdir(parents=True, exist_ok=True)

    await asyncio.to_thread(
        snapshot_download,
        repo_id=repo_id, 
        local_dir=str(local_dir)
    )

    model_name = repo_id.split("/")[-1]

    return {
        "repo_id": repo_id,
        "task": task,
        "model_name": model_name,
        "installed_dir": str(local_dir)
    }

def list_installed_models(model_dir: str) -> dict:
    out = {}
    model_root = Path(model_dir)
    for task in MODEL_FILTERS:
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

def find_onnx_model_path(repo_id: str) -> str:
    """
    Find directory matching repo_id anywhere under models_root
    """
    for path in repo_id.rglob(repo_id):
        if path.is_dir() and path.name == repo_id:
            return str(path)

    raise FileNotFoundError(f"Model directory '{repo_id}' not found")

def list_model_filters():
    return MODEL_FILTERS

async def onnx_converter():
    pass
