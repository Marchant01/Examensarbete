<script lang="ts">
    import { onMount } from "svelte";
    import {
        engineEvents,
        loadModel,
        clearLoadedModels,
        getInstalledModels,
        runFlow,
        type TextToImageInput,
    } from "$lib/engine";

    const TASKS: string[] = ["text-to-image", "text-generation", "image-to-image"];
    const DEFAULT_NUM_STEPS = 20;
    const DEFAULT_GUIDANCE_SCALE = 7.5;
    const DEFAULT_SEED = 41;
    const MAX_SEED = 2147483647;

    let selectedModel = "";
    let selectedTask  = "";

    type InstalledModel = { repo_id: string; task: string };

    $: installedModels = (() => {
        const event = [...$engineEvents]
            .reverse()
            .find(
                (ev) =>
                    ev?.type === "done" &&
                    ev?.data &&
                    TASKS.some((t) => t in ev.data)
            );
        if (!event) return [] as InstalledModel[];

        return Object.entries(event.data as Record<string, string[]>).flatMap(
            ([task, ids]) => ids.map((repo_id) => ({ repo_id, task }))
        );
    })();

    type FlowNode = { repo_id: string; task: string; loaded: boolean };
    let flowNodes: FlowNode[] = [];

    $: {
        const loadedEvent = [...$engineEvents]
            .reverse()
            .find((ev) => ev?.type === "done" && ev?.data?.loaded);

        if (loadedEvent) {
            const loadedId = loadedEvent.data.loaded;
            flowNodes = flowNodes.map((n) =>
                n.repo_id === loadedId ? { ...n, loaded: true } : n
            );
        };
    };

    $: {
        const cleared = [...$engineEvents]
            .reverse()
            .find((ev) => ev?.type === "done" && ev?.data === "cleared");
        if (cleared) {
            flowNodes = flowNodes.map((n) => ({ ...n, loaded: false }));
        };
    };

    // Derive flow output from the most recent run_flow response
    type FlowOutput = { outputs: any } | null;
    $: flowOutput = (() => {
        const ev = [...$engineEvents]
            .reverse()
            .find((ev) => ev?.type === "done" && ev?.data?.outputs !== undefined);
        return ev ? ev.data : null as FlowOutput;
    })();

    // Derive any engine errors
    $: lastError = [...$engineEvents]
        .reverse()
        .find((ev) => ev?.type === "error")?.data ?? null;

    // Whether a flow is currently running (sent but no response yet)
    let isRunning = false;
    $: if (flowOutput || lastError) isRunning = false;

    function addModel() {
        if (!selectedModel || !selectedTask) return;
        if (flowNodes.some((n) => n.repo_id === selectedModel)) return;

        loadModel(selectedModel, selectedTask);
        flowNodes = [...flowNodes, { repo_id: selectedModel, task: selectedTask, loaded: false }];
        selectedModel = "";
        selectedTask  = "";
    };

    function removeNode(repo_id: string) {
        flowNodes = flowNodes.filter((n) => n.repo_id !== repo_id);
    };

    function handleClear() {
        clearLoadedModels();
        flowNodes = flowNodes.map((n) => ({ ...n, loaded: false }));
    };

    let flowInput = "";
    let promptInput = "";
    let numSteps = DEFAULT_NUM_STEPS;
    let guidanceScale = DEFAULT_GUIDANCE_SCALE;
    let seed = DEFAULT_SEED;

    $: firstInputTask = flowNodes[0]?.task ?? selectedTask;
    $: isTextToImageFlow = firstInputTask === "text-to-image";
    $: canRunFlow =
        flowNodes.length > 0 &&
        flowNodes.every((n) => n.loaded) &&
        (!isTextToImageFlow || promptInput.trim().length > 0);

    function randomizeSeed() {
        seed = Math.floor(Math.random() * (MAX_SEED + 1));
    };

    function buildFlowInput(): string | TextToImageInput {
        if (!isTextToImageFlow) {
            return flowInput;
        }

        return {
            prompt: promptInput.trim(),
            num_steps: numSteps,
            guidance_scale: guidanceScale,
            seed,
        };
    };

    function executeFlow() {
        if (!canRunFlow) return;
        isRunning = true;
        const flow = flowNodes.map((n) => ({ model_id: n.repo_id, task: n.task }));
        runFlow(flow, buildFlowInput());
    };

    // Format output for display
    function formatOutput(data: any): string {
        if (data === null || data === undefined) return "";
        if (typeof data === "string") return data;
        return JSON.stringify(data, null, 2);
    };

    // Detect if output is an image (base64 or URL)
    function isImageOutput(data: any): boolean {
        if (!data?.outputs) return false;
        const out = data.outputs;
        if (typeof out === "string") {
            return out.startsWith("data:image") || out.startsWith("http");
        }
        // HF pipeline text-to-image returns [{ "generated_image": ... }] or similar
        if (Array.isArray(out) && out[0]?.url) return true;
        return false;
    };

    onMount(() => {
        getInstalledModels();
    });
</script>

<div class="diagram-container">

    <div class="tool-bar">
        <select bind:value={selectedTask}>
            <option value="" disabled>Select task</option>
            {#each TASKS as task}
                <option value={task}>{task}</option>
            {/each}
        </select>

        <select bind:value={selectedModel}>
            <option value="" disabled>Select model</option>
            {#each installedModels.filter(m => !selectedTask || m.task === selectedTask) as m}
                <option value={m.repo_id}>{m.repo_id}</option>
            {/each}
        </select>

        <button on:click={addModel} disabled={!selectedModel || !selectedTask}>
            Add to flow
        </button>

        <button on:click={handleClear} disabled={flowNodes.length === 0}>
            Clear loaded
        </button>

        <button
            class="run-btn"
            on:click={executeFlow}
            disabled={!canRunFlow || isRunning}
        >
            {#if isRunning}
                <span class="spinner">Running…</span>
            {:else}
                ▶ Run flow
            {/if}
        </button>
    </div>

    <div class="diagram">
        {#if flowNodes.length === 0}
            <p class="empty-hint">Add a model above to build your flow.</p>
        {:else}
            {#each flowNodes as node, i}
                {#if i > 0}
                    <div class="connector">→</div>
                {/if}

                <div class="flow-node" class:loaded={node.loaded}>
                    <span class="node-task">{node.task}</span>
                    <span class="node-id">{node.repo_id}</span>
                    <span class="node-status">{node.loaded ? "✓ loaded" : "loading…"}</span>
                    <button class="node-remove" on:click={() => removeNode(node.repo_id)}>✕</button>
                </div>
            {/each}
        {/if}
    </div>

    <div class="io-panel">
        <div class="io-input">
            {#if isTextToImageFlow}
                <label for="prompt-input">Prompt</label>
                <textarea
                    id="prompt-input"
                    bind:value={promptInput}
                    rows="4"
                    placeholder="Describe the image you want to generate…"
                />

                <div class="parameter-grid">
                    <div class="parameter-field">
                        <div class="parameter-label-row">
                            <label for="steps-input">Number of steps</label>
                            <span class="parameter-value">{numSteps}</span>
                        </div>
                        <input
                            id="steps-input"
                            type="range"
                            bind:value={numSteps}
                            min="1"
                            max="50"
                            step="1"
                        />
                    </div>

                    <div class="parameter-field">
                        <div class="parameter-label-row">
                            <label for="guidance-input">Guidance scale</label>
                            <span class="parameter-value">{guidanceScale.toFixed(1)}</span>
                        </div>
                        <input
                            id="guidance-input"
                            type="range"
                            bind:value={guidanceScale}
                            min="0"
                            max="20"
                            step="0.5"
                        />
                    </div>

                    <div class="parameter-field parameter-field-wide">
                        <div class="parameter-label-row">
                            <label for="seed-input">Seed</label>
                            <span class="parameter-value">{seed}</span>
                        </div>
                        <div class="seed-row">
                            <input
                                id="seed-input"
                                type="number"
                                bind:value={seed}
                                min="0"
                                max={MAX_SEED}
                                step="1"
                            />
                            <button type="button" class="seed-randomize" on:click={randomizeSeed}>
                                Randomize
                            </button>
                        </div>
                    </div>
                </div>
            {:else}
                <label for="flow-input">Input</label>
                <input
                    id="flow-input"
                    bind:value={flowInput}
                    placeholder="Enter prompt or data…"
                />
            {/if}
        </div>

        <div class="io-output" class:has-content={!!flowOutput || !!lastError}>
            <label class="io-label" for="id">
                Output
                {#if flowOutput}
                    <span class="output-badge">done</span>
                {:else if lastError}
                    <span class="output-badge error">error</span>
                {:else if isRunning}
                    <span class="output-badge running">running</span>
                {/if}
            </label>

            <div class="output-body">
                {#if isRunning}
                    <div class="output-running">
                        <span class="spinner-large">Processing flow…</span>
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
                        <!-- Image output (text-to-image pipelines) -->
                        {#if typeof flowOutput.outputs === "string"}
                            <img class="output-image" src={flowOutput.outputs} alt="Generated output" />
                        {:else if Array.isArray(flowOutput.outputs)}
                            {#each flowOutput.outputs as item}
                                {#if item?.url}
                                    <img class="output-image" src={item.url} alt="Generated output" />
                                {/if}
                            {/each}
                        {/if}
                    {:else}
                        <!-- Text / JSON output -->
                        <pre class="output-text">{formatOutput(flowOutput.outputs)}</pre>
                    {/if}

                {:else}
                    <p class="output-placeholder">Output will appear here after the flow runs.</p>
                {/if}
            </div>
        </div>
    </div>

</div>
