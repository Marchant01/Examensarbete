<script lang="ts">
    import { engineEvents, getModels, installModel } from "$lib/engine";

    $: models = $engineEvents
        .filter((ev) => ev?.type === "done" && ev?.data.models)
        .flatMap((ev) => ev.data.models);

    const options: string[] = ["text-to-image", "text-generation", "image-to-image"];
    let selected = options[0];

    let limit: number = 5;

    function normalizeLimit(value: number): number {
        if (!Number.isFinite(value)) return 5;
        return Math.max(5, Math.min(40, Math.trunc(value)));
    }

    function fetchByFilter() {
        limit = normalizeLimit(limit);
        engineEvents.set([]);
        getModels(selected, limit);
    }

    function installByTask(repoID: string) {
        installModel(repoID, selected);
    }
</script>

<div class="models">
    <div class="model-selection">
        <h3>Filters</h3>
        <select bind:value={selected}>
            {#each options as o}
            <option value={o}>{o}</option>
            {/each}
        </select>
        <input type="number" name="limit" bind:value={limit} min="5" max="40" step="5" />
    </div>
    <button on:click={fetchByFilter}>Get models</button>
    {#each models as model}
    <div class="model-card">
        <h2>{model.id}</h2>
        <!-- <p>{model.last_modified}</p> -->
        <p>Downloads: {model.downloads}</p>
        <button on:click={() => installByTask(model.id)}>Download</button>
    </div>
    {/each}
</div>
