import json
import sys
import os
import traceback
import asyncio

from model_loader import list_available_models, install_model, list_installed_models, list_model_filters

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

async def main():
    models_dir = os.environ.get("MODELS_DIR", "models")

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
                    continue
                except Exception as e:
                    send({"id": req_id, 
                    "type": "error", 
                    "data": {"message": str(e), 
                    "trace": traceback.format_exc()}})

            if cmd == "list_installed_models":
                data = list_installed_models(models_dir)
                send({"id": req_id, "type": "done", "data": data})
                continue
            
            if cmd == "install_model":
                task = args["task"]
                repo_id = args["repo_id"]
                try:
                    model = await install_model(repo_id, task)
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

            send({"id": req_id, "type": "error", "data": {"message": f"Unknown cmd: {cmd}"}})
        
        except Exception as e:
            send({
                "id": req.get("id") if isinstance(req, dict) else "unknown",
                "type": "error",
                "data": {"message": str(e), "trace": traceback.format_exc()}
            })

if __name__ == "__main__":
    asyncio.run(main())