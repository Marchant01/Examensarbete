# Source Material #
https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
https://www.qualcomm.com/developer/blog/2025/05/deploy-ai-models-on-snapdragon-x-elite-with-qualcomm-ai-hub
https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization
https://docs.qualcomm.com/doc/80-63442-10/topic/QNN_general_overview.html#developers-on-linux
https://onnx.ai/onnx/intro/concepts.html
https://onnx.ai/onnx/operators/index.html#l-onnx-operators

## Link to excalidraw for flowchart ## 
https://excalidraw.com/#json=W3ZIn1Fl4boE2uhtUXnke,1ZR0tDtwZ6Hl40B-DIW4GA

## Structure ##
Projektets “kärnmoduler” (Python engine)

Tänk så här:

Model Registry

metadata: modelltyp, källa, version, licensnotis, kompat-status

lokala paths till bundles

Model Builder

download weights

export to ONNX

validate (snabb testkörning)

(valfritt) quantize profile (INT8/INT4)

packa som en “bundle” (manifest + onnx + tokenizer + config)

Runtime Runner

ORT session med QNN EP

standardiserade input/output adapters

mätning: latency, ev minne, loggar

Flow Engine

DAG av noder

noder = “LLM prompt”, “text encoder”, “denoise step”, “vae decode”, “save image”

kör med event-streaming till UI

Project Store

project.yaml/json + flows/*.json

outputs, cache, models


## Workflow ##

### Picking a model ###

Where do you get them? Could be installed from huggingface using huggingface cli.
You can use Huggingface_hub in order to auth and search models. This will probably be the best approach

log in on hf > search models > install them

It is possible to add search filters, could add some mandatory filters to make sure that the user doesn't pull an unusable model.

### Selecting model ###

With decorators, we can make sure that the models follow the correct behaviour for the onnx runtime 
