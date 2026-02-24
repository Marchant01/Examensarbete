<script lang="ts">
    import { onMount } from "svelte";
    import { engineEvents, getModels, installModel, getInstalledModels } from "$lib/engine";

    const options: string[] = ["text-to-image", "text-generation", "image-to-image"];
    let selectedAvailable = options[0];
    let selectedInstalled = options[0];

    let limit: number = 5;
    let installedModels: Record<string, string[]> = {};

    $: models =
        [...$engineEvents]
            .reverse()
            .find((ev) => ev?.type === "done" && ev?.data?.models)?.data?.models ?? [];

    $: installedModels =
        [...$engineEvents]
            .reverse()
            .find(
                (ev) =>
                    ev?.type === "done" &&
                    ev?.data &&
                    options.every((task) => task in ev.data)
            )?.data ?? {};

    $: installedForSelected = installedModels[selectedInstalled] ?? [];

    onMount(() => {
        getInstalledModels(selectedInstalled);
    });

    function normalizeLimit(value: number): number {
        if (!Number.isFinite(value)) return 5;
        return Math.max(5, Math.min(40, Math.trunc(value)));
    };

    function fetchByFilter() {
        limit = normalizeLimit(limit);
        getModels(selectedAvailable, limit);
    }

    async function installByTask(repoID: string) {
        await installModel(repoID, selectedAvailable);
        await getInstalledModels(selectedInstalled);
    }

</script>

<div class="models">
    <div class="installed-model-selection">
        <h2>Installed models</h2>
        <h3>Filter</h3>
        <select bind:value={selectedInstalled}>
            {#each options as o}
            <option value={o}>{o}</option>
            {/each}
        </select>

        {#if installedForSelected.length === 0}
        <p>No installed models for {selectedInstalled}.</p>
        {:else}
        {#each installedForSelected as modelName}
        <div class="model-card">
            <h2>{modelName}</h2>
            <p>Task: {selectedInstalled}</p>
        </div>
        {/each}
        {/if}
    </div>
    <div class="install-model-selection">
        <h2>Install new models</h2>
        <h3>Filters</h3>
        <select bind:value={selectedAvailable}>
            {#each options as o}
            <option value={o}>{o}</option>
            {/each}
        </select>
        <input type="number" name="limit" bind:value={limit} min="5" max="40" step="5" />
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
</div>
