<script lang="ts">
    import { onDestroy, onMount } from "svelte";
    import { get } from "svelte/store";
    import {
        CONTROLNET_CANNY_MODEL_ID,
        STABLE_DIFFUSION_MODEL_ID,
        clearLoadedModels,
        engineEvents,
        getSupportedModels,
        loadModel,
        runModel,
        type ControlNetCannyInput,
        type SupportedModel,
        type TextToImageInput,
    } from "$lib/engine";

    const DEFAULT_NUM_STEPS = 20;
    const DEFAULT_GUIDANCE_SCALE = 7.5;
    const DEFAULT_SEED = 41;
    const DEFAULT_CANNY_LOW_THRESHOLD = 100;
    const DEFAULT_CANNY_HIGH_THRESHOLD = 200;
    const MAX_SEED = 2147483647;
    const REQUEST_TIMEOUT_MS = 5 * 60 * 1000;

    type EngineEvent = {
        id?: string;
        type?: "done" | "error" | string;
        data?: any;
    };

    type FlowOutput = { outputs: any } | null;

    const FALLBACK_MODELS: SupportedModel[] = [
        {
            id: STABLE_DIFFUSION_MODEL_ID,
            label: "Stable Diffusion",
            task: "text-to-image",
            runner: "stable-diffusion",
            installed: true,
            path: null,
            missing_components: [],
        },
        {
            id: CONTROLNET_CANNY_MODEL_ID,
            label: "ControlNet-Canny",
            task: "controlnet-canny",
            runner: "controlnet-canny",
            installed: true,
            path: null,
            missing_components: [],
        },
    ];

    let supportedModels: SupportedModel[] = FALLBACK_MODELS;
    let selectedModelId = STABLE_DIFFUSION_MODEL_ID;
    let selectedModel = FALLBACK_MODELS[0];
    let selectedModelLoaded = false;
    let selectedModelInstalled = true;
    let isControlNet = false;
    let loadedModelIds: string[] = [];

    let promptInput = "";
    let referenceImageDataUrl = "";
    let referenceImageName = "";
    let referenceImageSize = 0;
    let referenceImageType = "";
    let referenceImageInput: HTMLInputElement | null = null;
    let numSteps = DEFAULT_NUM_STEPS;
    let guidanceScale = DEFAULT_GUIDANCE_SCALE;
    let seed = DEFAULT_SEED;
    let cannyLowThreshold = DEFAULT_CANNY_LOW_THRESHOLD;
    let cannyHighThreshold = DEFAULT_CANNY_HIGH_THRESHOLD;

    let isLoadingModel = false;
    let isRunning = false;
    let isRefreshingModels = false;
    let flowOutput: FlowOutput = null;
    let lastError: any = null;
    let generationStartedAt = 0;
    let generationElapsedMs = 0;
    let generationTimer: ReturnType<typeof setInterval> | null = null;

    function waitForEngineEvent(id: string): Promise<EngineEvent> {
        const existingEvent = get(engineEvents).find((event) => event?.id === id);
        if (existingEvent) return Promise.resolve(existingEvent);

        return new Promise((resolve, reject) => {
            let unsubscribe = () => {};
            const timeout = setTimeout(() => {
                unsubscribe();
                reject(new Error("Timed out waiting for the engine response."));
            }, REQUEST_TIMEOUT_MS);

            unsubscribe = engineEvents.subscribe((events) => {
                const event = events.find((item) => item?.id === id);
                if (!event) return;

                clearTimeout(timeout);
                unsubscribe();
                resolve(event);
            });
        });
    }

    function getSelectedModel(): SupportedModel {
        return (
            supportedModels.find((model) => model.id === selectedModelId) ||
            FALLBACK_MODELS[0]
        );
    }

    function syncSelectedModel() {
        selectedModel = getSelectedModel();
        selectedModelLoaded = modelIsLoaded(selectedModelId);
        selectedModelInstalled = selectedModel.installed !== false;
        isControlNet = selectedModelId === CONTROLNET_CANNY_MODEL_ID;
    }

    function modelIsLoaded(modelId: string): boolean {
        for (const loadedId of loadedModelIds) {
            if (loadedId === modelId) return true;
        }
        return false;
    }

    function normalizeError(error: unknown) {
        return { message: error instanceof Error ? error.message : String(error) };
    }

    function updateGenerationElapsed() {
        if (generationStartedAt === 0) return;
        generationElapsedMs = performance.now() - generationStartedAt;
    }

    function startGenerationTimer() {
        stopGenerationTimer();
        generationStartedAt = performance.now();
        generationElapsedMs = 0;
        generationTimer = setInterval(updateGenerationElapsed, 100);
    }

    function stopGenerationTimer() {
        if (generationTimer) {
            clearInterval(generationTimer);
            generationTimer = null;
        }
        updateGenerationElapsed();
    }

    function resetGenerationTimer() {
        stopGenerationTimer();
        generationStartedAt = 0;
        generationElapsedMs = 0;
    }

    function formatElapsedTime(milliseconds: number): string {
        const totalSeconds = Math.max(0, milliseconds / 1000);

        if (totalSeconds < 60) {
            return `${totalSeconds.toFixed(1)}s`;
        }

        const minutes = Math.floor(totalSeconds / 60);
        const seconds = Math.floor(totalSeconds % 60);
        return `${minutes}m ${seconds.toString().padStart(2, "0")}s`;
    }

    function getValidationError(): string | null {
        if (!selectedModelInstalled) {
            return `${selectedModel.label} is missing local model files.`;
        }
        if (!selectedModelLoaded) {
            return "Load the selected model before running it.";
        }
        if (promptInput.trim().length === 0) {
            return "Enter a prompt before running the model.";
        }
        if (Number(cannyLowThreshold) > Number(cannyHighThreshold)) {
            return "Canny low threshold must be less than or equal to the high threshold.";
        }
        if (isControlNet && referenceImageDataUrl.length === 0) {
            return "Choose a reference image before running ControlNet-Canny.";
        }
        return null;
    }

    async function refreshSupportedModels() {
        try {
            isRefreshingModels = true;
            const requestId = await getSupportedModels();
            const event = await waitForEngineEvent(requestId);

            if (event.type === "done" && Array.isArray(event.data?.models)) {
                supportedModels = event.data.models;
                syncSelectedModel();
                lastError = null;
                return;
            }

            if (event.type === "error") {
                lastError = event.data;
            }
        } catch (error) {
            lastError = normalizeError(error);
        } finally {
            isRefreshingModels = false;
        }
    }

    async function loadSelectedModel() {
        if (isLoadingModel || !selectedModelInstalled || selectedModelLoaded) {
            return;
        }

        try {
            lastError = null;
            isLoadingModel = true;
            const requestId = await loadModel(selectedModelId);
            const event = await waitForEngineEvent(requestId);

            if (event.type === "done" && typeof event.data?.loaded === "string") {
                if (!modelIsLoaded(event.data.loaded)) {
                    loadedModelIds = [...loadedModelIds, event.data.loaded];
                }
                syncSelectedModel();
                return;
            }

            if (event.type === "error") {
                lastError = event.data;
            }
        } catch (error) {
            lastError = normalizeError(error);
        } finally {
            isLoadingModel = false;
        }
    }

    async function clearLoaded() {
        try {
            lastError = null;
            const requestId = await clearLoadedModels();
            const event = await waitForEngineEvent(requestId);

            if (event.type === "done") {
                loadedModelIds = [];
                syncSelectedModel();
                flowOutput = null;
                resetGenerationTimer();
                return;
            }

            if (event.type === "error") {
                lastError = event.data;
            }
        } catch (error) {
            lastError = normalizeError(error);
        }
    }

    function buildModelInput(): TextToImageInput | ControlNetCannyInput {
        const baseInput: TextToImageInput = {
            prompt: promptInput.trim(),
            num_steps: Number(numSteps),
            guidance_scale: Number(guidanceScale),
            seed: Number(seed),
        };

        if (!isControlNet) return baseInput;

        return {
            ...baseInput,
            image_data_url: referenceImageDataUrl,
            canny_low_threshold: Number(cannyLowThreshold),
            canny_high_threshold: Number(cannyHighThreshold),
        };
    }

    async function executeModel() {
        resetGenerationTimer();
        const validationError = getValidationError();
        if (validationError) {
            lastError = { message: validationError };
            return;
        }

        try {
            flowOutput = null;
            lastError = null;
            isRunning = true;
            startGenerationTimer();
            const requestId = await runModel(selectedModelId, buildModelInput());
            const event = await waitForEngineEvent(requestId);

            if (event.type === "done") {
                flowOutput = event.data;
                return;
            }

            if (event.type === "error") {
                lastError = event.data;
            }
        } catch (error) {
            lastError = normalizeError(error);
        } finally {
            stopGenerationTimer();
            isRunning = false;
        }
    }

    function randomizeSeed() {
        seed = Math.floor(Math.random() * (MAX_SEED + 1));
    }

    function handleModelChange(event: Event) {
        selectedModelId = (event.target as HTMLSelectElement).value;
        syncSelectedModel();
    }

    function resetReferenceImage(clearInput = true) {
        referenceImageDataUrl = "";
        referenceImageName = "";
        referenceImageSize = 0;
        referenceImageType = "";

        if (clearInput && referenceImageInput) {
            referenceImageInput.value = "";
        }
    }

    function isSupportedImageFile(file: File): boolean {
        if (file.type.startsWith("image/")) return true;

        return /\.(apng|avif|bmp|gif|jpe?g|png|webp)$/i.test(file.name);
    }

    function setReferenceImageFromFile(file: File) {
        if (!isSupportedImageFile(file)) {
            resetReferenceImage();
            lastError = { message: "Choose an image file for ControlNet-Canny." };
            return;
        }

        const reader = new FileReader();
        reader.onload = () => {
            if (typeof reader.result !== "string") {
                resetReferenceImage();
                lastError = { message: "Failed to read the reference image." };
                return;
            }

            referenceImageDataUrl = reader.result;
            referenceImageName = file.name;
            referenceImageSize = file.size;
            referenceImageType = file.type || "image";
            lastError = null;
        };
        reader.onerror = () => {
            resetReferenceImage();
            lastError = { message: "Failed to read the reference image." };
        };
        reader.readAsDataURL(file);
    }

    function handleReferenceImageChange(event: Event) {
        const input = event.target as HTMLInputElement;
        const file = input.files?.[0];

        if (!file) {
            resetReferenceImage(false);
            return;
        }

        setReferenceImageFromFile(file);
    }

    function formatFileSize(bytes: number): string {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    }

    function formatOutput(data: any): string {
        if (data === null || data === undefined) return "";
        if (typeof data === "string") return data;
        return JSON.stringify(data, null, 2);
    }

    function isImageOutput(data: FlowOutput): boolean {
        const output = data?.outputs;
        return typeof output === "string" && output.startsWith("data:image");
    }

    onMount(() => {
        void refreshSupportedModels();
    });

    onDestroy(() => {
        resetGenerationTimer();
    });
</script>

<div class="diagram-container">
    <div class="tool-bar">
        <select value={selectedModelId} on:change={handleModelChange}>
            {#each supportedModels as model}
                <option value={model.id}>{model.label}</option>
            {/each}
        </select>

        <button
            on:click={loadSelectedModel}
            disabled={!selectedModelInstalled || selectedModelLoaded || isLoadingModel}
        >
            {#if isLoadingModel}
                Loading...
            {:else if selectedModelLoaded}
                Loaded
            {:else}
                Load model
            {/if}
        </button>

        <button on:click={clearLoaded} disabled={loadedModelIds.length === 0}>
            Clear loaded
        </button>

        <button
            class="run-btn"
            on:click={executeModel}
            disabled={isRunning || !selectedModelLoaded || !selectedModelInstalled}
        >
            {isRunning ? "Running..." : "Run"}
        </button>
    </div>

    {#if isRefreshingModels}
        <p>Loading supported models...</p>
    {/if}

    {#if !selectedModelInstalled}
        <div class="output-error">
            {selectedModel.label} is missing local model files.
        </div>
    {/if}

    <div class="diagram">
        <div class="flow-node" class:loaded={selectedModelLoaded}>
            <span class="node-task">{selectedModel.task}</span>
            <span class="node-id">{selectedModel.label}</span>
            <span class="node-status">{selectedModelLoaded ? "loaded" : "not loaded"}</span>
        </div>
    </div>

    <div class="io-panel">
        <div class="io-input">
            <label for="prompt-input">Prompt</label>
            <textarea
                id="prompt-input"
                bind:value={promptInput}
                rows="4"
                placeholder="Describe the image you want to generate..."
            />

            {#if isControlNet}
                <div class="image-import-panel">
                    <div class="parameter-label-row">
                        <label for="reference-image-input">Reference image</label>
                        {#if referenceImageDataUrl}
                            <button
                                type="button"
                                class="image-remove-button"
                                on:click={() => resetReferenceImage()}
                            >
                                Remove
                            </button>
                        {/if}
                    </div>

                    <input
                        bind:this={referenceImageInput}
                        id="reference-image-input"
                        type="file"
                        accept="image/*"
                        on:change={handleReferenceImageChange}
                    />

                    {#if referenceImageDataUrl}
                        <div class="image-preview">
                            <img src={referenceImageDataUrl} alt="Reference preview" />
                            <div class="image-meta">
                                <span>{referenceImageName}</span>
                                <span>{referenceImageType} - {formatFileSize(referenceImageSize)}</span>
                            </div>
                        </div>
                    {/if}
                </div>
            {/if}

            <div class="parameter-grid">
                <div class="parameter-field">
                    <div class="parameter-label-row">
                        <label for="steps-input">Number of steps</label>
                        <span class="parameter-value">{numSteps}</span>
                    </div>
                    <input id="steps-input" type="range" bind:value={numSteps} min="1" max="50" step="1" />
                </div>

                <div class="parameter-field">
                    <div class="parameter-label-row">
                        <label for="guidance-input">Guidance scale</label>
                        <span class="parameter-value">{Number(guidanceScale).toFixed(1)}</span>
                    </div>
                    <input id="guidance-input" type="range" bind:value={guidanceScale} min="0" max="20" step="0.5" />
                </div>

                <div class="parameter-field parameter-field-wide">
                    <div class="parameter-label-row">
                        <label for="seed-input">Seed</label>
                        <span class="parameter-value">{seed}</span>
                    </div>
                    <div class="seed-row">
                        <input id="seed-input" type="number" bind:value={seed} min="0" max={MAX_SEED} step="1" />
                        <button type="button" class="seed-randomize" on:click={randomizeSeed}>
                            Randomize
                        </button>
                    </div>
                </div>

                {#if isControlNet}
                    <div class="parameter-field">
                        <div class="parameter-label-row">
                            <label for="canny-low-input">Canny low threshold</label>
                            <span class="parameter-value">{cannyLowThreshold}</span>
                        </div>
                        <input id="canny-low-input" type="range" bind:value={cannyLowThreshold} min="0" max="255" step="1" />
                    </div>

                    <div class="parameter-field">
                        <div class="parameter-label-row">
                            <label for="canny-high-input">Canny high threshold</label>
                            <span class="parameter-value">{cannyHighThreshold}</span>
                        </div>
                        <input id="canny-high-input" type="range" bind:value={cannyHighThreshold} min="0" max="255" step="1" />
                    </div>
                {/if}
            </div>
        </div>

        <div class="io-output" class:has-content={!!flowOutput || !!lastError}>
            <label class="io-label" for="output">
                Output
                {#if flowOutput}
                    <span class="output-badge">done</span>
                {:else if lastError}
                    <span class="output-badge error">error</span>
                {:else if isRunning}
                    <span class="output-badge running">running</span>
                {/if}
                {#if generationElapsedMs > 0 || isRunning}
                    <span class="output-timer">{formatElapsedTime(generationElapsedMs)}</span>
                {/if}
            </label>

            <div class="output-body">
                {#if isRunning}
                    <div class="output-running">
                        <span class="spinner-large">Processing...</span>
                    </div>
                {:else if lastError}
                    <pre class="output-error">{lastError.message ?? formatOutput(lastError)}</pre>
                    {#if lastError.trace}
                        <details class="trace-details">
                            <summary>Traceback</summary>
                            <pre class="output-trace">{lastError.trace}</pre>
                        </details>
                    {/if}
                {:else if flowOutput}
                    {#if isImageOutput(flowOutput)}
                        <img class="output-image" src={flowOutput.outputs} alt="Generated output" />
                    {:else}
                        <pre class="output-text">{formatOutput(flowOutput.outputs)}</pre>
                    {/if}
                {:else}
                    <p class="output-placeholder">Output will appear here after the model runs.</p>
                {/if}
            </div>
        </div>
    </div>
</div>
