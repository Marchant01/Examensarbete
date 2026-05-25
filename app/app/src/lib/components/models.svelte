<script lang="ts">
    import { onMount } from "svelte";
    import { engineEvents, getSupportedModels, type SupportedModel } from "$lib/engine";

    let requestId = "";
    let lastError: any = null;

    $: event = requestId
        ? $engineEvents.find((ev) => ev?.id === requestId)
        : null;
    $: models = (
        event?.type === "done" && Array.isArray(event?.data?.models)
            ? event.data.models
            : []
    ) as SupportedModel[];
    $: if (event?.type === "error") {
        lastError = event.data;
    }

    onMount(async () => {
        try {
            requestId = await getSupportedModels();
        } catch (error) {
            lastError = { message: error instanceof Error ? error.message : String(error) };
        }
    });
</script>

<div class="models-container">
    <div class="installed-model-selection">
        <h2>Supported models</h2>

        {#if lastError}
            <pre class="output-error">{lastError.message ?? JSON.stringify(lastError, null, 2)}</pre>
        {:else if models.length === 0}
            <p>Loading supported models...</p>
        {:else}
            {#each models as model}
                <div class="model-card">
                    <h2>{model.label}</h2>
                    <p>Task: {model.task}</p>
                    <p>Status: {model.installed ? "installed" : "missing local files"}</p>
                    {#if model.path}
                        <p>{model.path}</p>
                    {/if}
                </div>
            {/each}
        {/if}
    </div>
</div>
