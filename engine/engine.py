import json
import sys
import os
import traceback
import asyncio
from pathlib import Path

from model_loader import list_available_models, list_installed_models, install_model, list_model_filters

# from flow_manager import model_registry, run_flow

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = ROOT_DIR / "models"

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

async def main():
    models_dir = Path(os.environ.get("MODELS_DIR", str(DEFAULT_MODELS_DIR))).expanduser()
    models_dir.mkdir(parents=True, exist_ok=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

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
                    send({"id": req_id, 
                    "type": "done", 
                    "data": {"models": models}})
                except Exception as e:
                    send({"id": req_id, 
                    "type": "error", 
                    "data": {"message": str(e), 
                    "trace": traceback.format_exc()}})
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
                    send({"id": req_id, "type": "done", "data": {"model": model}})
                except Exception as e:
                    send({
                        "id": req_id,
                        "type": "error",
                        "data": {"message": str(e), "trace": traceback.format_exc()}
                    })
                continue

            if cmd == "list_model_filters":
                filters = list_model_filters()
                send({"id": req_id, "type": "done", "data": {"filters": filters}})
                continue

            # if cmd == "load_model":
            #     task = args["task"]
            #     repo_id = args["repo_id"]

            #     model_path_root = models_dir / task / repo_id
            #     onnx_path = find_onnx_model_path(str(model_path_root))

            #     model_registry.load(repo_id, onnx_path)

            #     send({"id": req_id, "type": "done", "data": {"loaded": repo_id}})
            #     continue

            # if cmd == "run_flow":
            #     flow = args["flow"]
            #     initial_input = args["input"]

            #     result = run_flow(flow, initial_input)

            #     send({"id": req_id, "type": "done", "data": {"outputs": result}})
            #     continue

            send({"id": req_id, "type": "error", "data": {"message": f"Unknown cmd: {cmd}"}})
        
        except Exception as e:
            send({
                "id": req.get("id") if isinstance(req, dict) else "unknown",
                "type": "error",
                "data": {"message": str(e), "trace": traceback.format_exc()}
            })

if __name__ == "__main__":
    asyncio.run(main())
