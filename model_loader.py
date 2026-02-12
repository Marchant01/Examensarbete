from huggingface_hub import list_models, snapshot_download

# Test version, loads available HF models based on filters and installs selected model.
def select_model():
    print("Select a filter")
    
    model_filter = ["text-to-image", "text-generation", "image-to-image"]
    print(*model_filter, sep=", ")

    selected = input()
    print("listing models based on fitler")

    models = list_models(limit=20, library="onnx", filter=selected)

    for model in models:
        print(model)

    print("Select a model to isntall with repo_id")
    repo_id = input()
    snapshot_download(repo_id=repo_id, local_dir=f"models/{repo_id}")

