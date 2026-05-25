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

from flow_manager import model_registry, run_model
from model_loader import list_supported_models

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = ROOT_DIR / "models"


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


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

            if cmd == "list_supported_models":
                try:
                    data = list_supported_models(str(models_dir))
                    send({"id": req_id, "type": "done", "data": data})
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

            if cmd == "load_model":
                model_id = args.get("model_id") or args.get("repo_id")
                task = args.get("task")
                try:
                    runner = await asyncio.to_thread(model_registry.load, model_id, task)
                    send({"id": req_id, "type": "done", "data": {"loaded": runner.model_id}})
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

            if cmd == "run_model":
                try:
                    model_id = args["model_id"]
                    result = await asyncio.to_thread(run_model, model_id, args.get("input"))
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

            send({"id": req_id, "type": "error", "data": {"message": f"Unknown cmd: {cmd}"}})

        except Exception as e:
            send({
                "id": req.get("id") if isinstance(req, dict) else "unknown",
                "type": "error",
                "data": {"message": str(e), "trace": traceback.format_exc()},
            })


if __name__ == "__main__":
    asyncio.run(main())
