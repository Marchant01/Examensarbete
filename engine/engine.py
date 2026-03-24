import asyncio
import base64
import json
import os
import sys
import traceback
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from qai_hub_models.utils.display import to_uint8

from flow_manager import model_registry, run_flow
from model_loader import (
    list_available_models,
    list_installed_models,
    install_model,
    list_model_filters,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = ROOT_DIR / "models"


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _prepare_flow_input(flow_steps, raw_input):
    if not flow_steps:
        return raw_input

    first_task = flow_steps[0].get("task")
    if first_task != "text-to-image":
        return raw_input

    if isinstance(raw_input, dict):
        return raw_input

    if raw_input is None:
        raise ValueError(
            "Text-to-image flow requires an input object with at least a non-empty 'prompt' field."
        )

    raise TypeError(
        "Text-to-image flow input must be an object with 'prompt', 'num_steps', "
        "'guidance_scale', and 'seed' fields."
    )


def _image_to_data_url(image_output) -> str:
    if isinstance(image_output, Image.Image):
        pil_image = image_output
    else:
        array = to_uint8(np.asarray(image_output))
        if array.ndim == 4:
            array = array[0]
        pil_image = Image.fromarray(array)

    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _serialize_result(result):
    if isinstance(result, Image.Image):
        return _image_to_data_url(result)

    try:
        array = np.asarray(result)
    except Exception:
        return result

    if array.ndim >= 3:
        return _image_to_data_url(result)

    return result


async def main():
    models_dir = Path(os.environ.get("MODELS_DIR", str(DEFAULT_MODELS_DIR))).expanduser()
    models_dir.mkdir(parents=True, exist_ok=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        req = {}
        try:
            req = json.loads(line)
            req_id = req.get("id", "unknown")
            cmd = req.get("cmd")
            args = req.get("args", {})

            if cmd == "list_available_models":
                task = args["task"]
                limit = int(args.get("limit", 20))
                try:
                    models = await list_available_models(task, limit)
                    send({
                        "id": req_id,
                        "type": "done",
                        "data": {"models": models},
                    })
                except Exception as e:
                    send({
                        "id": req_id,
                        "type": "error",
                        "data": {
                            "message": str(e),
                            "trace": traceback.format_exc(),
                        },
                    })
                continue

            if cmd == "list_installed_models":
                data = list_installed_models(str(models_dir))
                send({"id": req_id, "type": "done", "data": data})
                continue

            if cmd == "install_model":
                task = args["task"]
                repo_id = args["repo_id"]
                try:
                    model = await install_model(repo_id, task, str(models_dir))
                    send({
                        "id": req_id,
                        "type": "done",
                        "data": {"model": model},
                    })
                except Exception as e:
                    send({
                        "id": req_id,
                        "type": "error",
                        "data": {
                            "message": str(e),
                            "trace": traceback.format_exc(),
                        },
                    })
                continue

            if cmd == "list_model_filters":
                filters = list_model_filters()
                send({"id": req_id, "type": "done", "data": {"filters": filters}})
                continue

            if cmd == "load_model":
                repo_id = args["repo_id"]
                task = args["task"]
                try:
                    await asyncio.to_thread(model_registry.load, repo_id, task)
                    send({"id": req_id, "type": "done", "data": {"loaded": repo_id}})
                except Exception as e:
                    send({
                        "id": req_id,
                        "type": "error",
                        "data": {
                            "message": str(e),
                            "trace": traceback.format_exc(),
                        },
                    })
                continue

            if cmd == "clear_loaded_models":
                try:
                    model_registry.clear_loaded_models()
                    send({"id": req_id, "type": "done", "data": "cleared"})
                except Exception as e:
                    send({
                        "id": req_id,
                        "type": "error",
                        "data": {
                            "message": str(e),
                            "trace": traceback.format_exc(),
                        },
                    })
                continue

            if cmd == "run_flow":
                try:
                    flow = args["flow"]
                    initial_input = _prepare_flow_input(flow, args.get("input"))
                    result = await asyncio.to_thread(run_flow, flow, initial_input)
                    send({
                        "id": req_id,
                        "type": "done",
                        "data": {"outputs": _serialize_result(result)},
                    })
                except Exception as e:
                    send({
                        "id": req_id,
                        "type": "error",
                        "data": {
                            "message": str(e),
                            "trace": traceback.format_exc(),
                        },
                    })
                continue

            send({"id": req_id, "type": "error", "data": {"message": f"Unknown cmd: {cmd}"}})

        except Exception as e:
            send({
                "id": req.get("id") if isinstance(req, dict) else "unknown",
                "type": "error",
                "data": {"message": str(e), "trace": traceback.format_exc()},
            })


if __name__ == "__main__":
    asyncio.run(main())
