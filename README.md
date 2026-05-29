# Examensarbete

Det här projektet är en lokal, flödesbaserad AI-applikation för att köra
generativa modeller direkt på en Windows-dator. Målet är att kunna bygga enkla
AI-flöden i ett desktopgränssnitt och köra dem via en Python-baserad motor med
ONNX Runtime och Qualcomm/QNN-stöd.

![Uploading clipboard_2026-05-28_17-00.png…]()

Projektet består av två huvuddelar:

- `engine/` innehåller Python-motorn som laddar modeller, hanterar flöden och
  kör inferens.
- `app/app/` innehåller desktopapplikationen byggd med Tauri, Svelte och Bun.

Modellerna ligger utanför Git-repot eftersom de är stora och ofta behöver
genereras eller exporteras för rätt hårdvara. Användaren behöver därför själv
hämta och lägga till modeller från Qualcomm AI Hub.

## Förutsättningar

Installationen är tänkt att göras på Windows.

Du behöver installera:

- Git
- Python 3.13.9
- uv
- Bun
- Rust och de vanliga Tauri-förutsättningarna för Windows

Kontrollera att kommandona fungerar i PowerShell:

```powershell
git --version
python --version
uv --version
bun --version
cargo --version
```

## Installation

Klona projektet:

```powershell
git clone <repo-url>
cd examensarbete
```

Installera Python-motorn och alla Python-paket:

```powershell
cd engine
uv sync
uv pip install -r requirements-qualcomm-stablediffusion.txt
```

`uv sync` skapar och hanterar den lokala virtuella miljön i `engine/.venv`.
Paketlistan i `requirements-qualcomm-stablediffusion.txt` innehåller de
versioner som behövs för Qualcomm/Stable Diffusion-miljön och ska installeras
efter `uv sync`.

Installera desktopappen:

```powershell
cd ..\app\app
bun install
```

Starta applikationen i utvecklingsläge:

```powershell
bun run tauri dev
```

När Tauri-appen startar försöker den köra Python-motorn från
`engine/.venv/Scripts/python.exe`.

## Modeller

Modeller ingår inte i Git-repot och ska inte checkas in. Katalogen `models/` är
ignorerad av Git eftersom modellfilerna kan vara mycket stora.

Du behöver själv hämta eller exportera modeller från Qualcomm AI Hub och lägga
dem i `models/` i repo-roten. Appen kommer att visa modeller som saknade tills
rätt filer finns på plats.

Förväntad struktur på hög nivå:

```text
models/
  text-to-image/
    stable-diffusion/
    controlnet_canny/
```

För Stable Diffusion letar motorn efter exporterade ONNX/QNN-komponenter som
bland annat innehåller:

- `text_encoder/model.onnx`
- `unet/model.onnx`
- `vae_decoder/model.onnx`

För ControlNet-Canny letar motorn efter motsvarande QAI Hub-export med:

- `text_encoder/text_encoder.onnx`
- `controlnet/controlnet.onnx`
- `unet/unet.onnx`
- `vae_decoder/vae.onnx`

Om modellen ligger i en `precompiled/qualcomm-snapdragon-x-elite/`-katalog kan
motorn också hitta den där. Det viktiga är att QAI Hub-exporten behåller sin
förväntade katalogstruktur.

## Vanliga problem

Om applikationen startar men modellerna visas som saknade betyder det normalt
att `models/` saknas, att modellen ligger i fel underkatalog eller att exporten
inte innehåller alla ONNX-filer som motorn förväntar sig.

Om Python-motorn inte startar, kontrollera först att `engine/.venv` finns och
att paketen har installerats:

```powershell
cd engine
uv sync
uv pip install -r requirements-qualcomm-stablediffusion.txt
```

Om ONNX Runtime eller QNN ger versionsfel behöver den installerade
`onnxruntime-qnn`-versionen matcha modellens export. Den här prototypen är
anpassad för `onnxruntime-qnn==1.24.4` enligt paketlistan.

## Utvecklingsstatus

Projektet är en tidig prototyp. Fokus ligger på lokal körning av AI-flöden,
modellhantering och experiment med Qualcomm AI Hub, ONNX Runtime och QNN/NPU.
