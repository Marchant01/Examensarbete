import onnxruntime as ort
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
from onnxruntime import SessionOptions, GraphOptimizationLevel
from model_loader import select_model

def main():
    print("Hello from examensarbete!")
    print("ORT version:", ort.__version__)
    print("Providers:", ort.get_available_providers())

    so = ort.SessionOptions()
    so.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_LAYOUT

    model = select_model()

    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    model = ORTModelForCausalLM.from_pretrained(
        model,
        subfolder="onnx",
        file_name="model_fp16.onnx",
        providers=["CPUExecutionProvider"],
        session_options=so,
    )

    prompt = "Skriv tre idéer för ett examensarbete om lokal AI."
    inputs = tokenizer(prompt, return_tensors="pt")

    out = model.generate(**inputs, max_new_tokens=80)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
