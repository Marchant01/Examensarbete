from huggingface_hub import list_models

def select_model():
    print("Select a filter")
    
    model_filter = ["text-to-image", "text-generation", "image-to-image"]
    for filters in model_filter:
        print(filter)

    selected = input()
    print("listing models based on fitler")

    models = list_models(limit=20, library="onnx", filter=selected)

    for model in models:
        print(model)

    
