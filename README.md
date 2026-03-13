# Examensarbete

This repository contains a prototype for a local, flow-based AI application.

The project is focused on exploring how creative AI workflows can run locally on a laptop, with support for ONNX Runtime and hardware-accelerated execution as part of the longer-term goal.

## Current Scope

The application currently supports:

- browsing available ONNX models by task
- installing selected models locally
- loading installed models into the runtime
- building and running simple model flows in a desktop UI

The repository is split into two main parts:

- `engine/` for the Python-based runtime and model handling
- `app/app/` for the Tauri + Svelte desktop interface

## Development

```bash
cd engine
uv sync
```

```bash
cd app/app
bun install
bun run tauri dev
```

## Notes

- The project is under active development and the scope may change.
- The current version is still an early prototype.
- Work related to NPU/QNN support, optimization, and quantization is still in progress.
