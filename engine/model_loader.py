import os
import asyncio
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

async def install_model(repo_id: str, task: str):
    if task not in MODEL_FILTERS:
        raise ValueError(f"Unknown task")
    local_dir = f"models/{task}/{repo_id}"

    await asyncio.to_thread(
        snapshot_download,
        repo_id=repo_id, 
        local_dir=local_dir
    )

    model_name = repo_id.split("/")[-1]

    return {
        "repo_id": repo_id,
        "task": task,
        "model_name": model_name,
        "installed_dir": local_dir
    }
        
def list_installed_models(model_dir: str):
    out = {}
    for task in MODEL_FILTERS:
        task_dir = os.path.join(model_dir, task)
        if not os.path.isdir(task_dir):
            out[task] = []
            continue
        out[task] = [
            name for name in os.listdir(task_dir)
            if os.path.isdir(os.path.join(task_dir, name))
        ]
    return out

def list_model_filters():
    return MODEL_FILTERS