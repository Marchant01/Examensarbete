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
        
def list_installed_models(model_dir: str):
    out = {}
    model_root = Path(model_dir)
    for task in MODEL_FILTERS:
        task_dir = model_root / task
        if not task_dir.is_dir():
            out[task] = []
            continue
        out[task] = [
            entry.name for entry in task_dir.iterdir()
            if entry.is_dir()
        ]
    return out

def find_onnx_model_path(model_root: str):
    """
    Recursively find first .onnx file in a model directory
    """
    model_root = Path(model_root)

    for path in model_root.rglob("*.onnx"):
        return str(path)

    raise FileNotFoundError("No ONNX file found in model directory")

def list_model_filters():
    return MODEL_FILTERS
