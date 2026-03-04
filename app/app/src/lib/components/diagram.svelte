<script lang="ts">
    import { onMount } from "svelte";
    import { engineEvents, loadModel, clearLoadedModels, getInstalledModels, runFlow } from "$lib/engine";

    const TASKS: string[] = ["text-to-image", "text-generation", "image-to-image"];

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

    let flowInput: string = "";

    function executeFlow() {
        if (flowNodes.length === 0 || !flowNodes.every((n) => n.loaded)) return;
        isRunning = true;
        const flow = flowNodes.map((n) => ({ model_id: n.repo_id, task: n.task }));
        runFlow(flow, flowInput);
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
            disabled={flowNodes.length === 0 || isRunning || !flowNodes.every((n) => n.loaded)}
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
            <label for="id">Input</label>
            <input bind:value={flowInput} placeholder="Enter prompt or data…" />
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
