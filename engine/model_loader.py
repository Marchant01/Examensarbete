import os
from huggingface_hub import list_models, snapshot_download

def install_new_models():
    print("Select a filter")
    
    model_filter = ["text-to-image", "text-generation", "image-to-image"]
    print(*model_filter, sep=", ")

    selected = input()
    print("listing models based on filter")

    models = list_models(limit=20, library="onnx", filter=selected)

    for model in models:
        print(model.id)

    print("Select a model to isntall with repo_id")
    repo_id = input()
    snapshot_download(repo_id=repo_id, 
    local_dir=f"models/{selected}/{repo_id}",
    )
    
    return f"Installed: {repo_id}"

def select_model():
    installed_models = [f for f in os.listdir("models/") if os.path.isdir(f)]
    print(len(installed_models))
    
    if len(installed_models) < 1:
        install_new_models()

    print(installed_models, sep="\n")

    selected_model = input()

    return selected_model
        
        


