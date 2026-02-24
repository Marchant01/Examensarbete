<script lang="ts">
    import { onMount } from "svelte";
    import { engineEvents, loadModel, clearLoadedModels, getInstalledModels } from "$lib/engine";

    const options: string[] = ["text-to-image", "text-generation", "image-to-image"];
    let models: string[] = [];
    let installedModels: Record<string, string[]> = {};
    let selectedInstalled = options[0];


    // Function for adding selected model to the diagram
    function addModelBox() {

    }

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

    onMount(() => {
        getInstalledModels(selectedInstalled);
    });


</script>

<div class="diagram-container">
    <div class="tool-bar">
            <!-- select a install model -->
        <select bind:value={selectedInstalled}>
            {#each options as o}
            <option value={o}>{o}</option>
            {/each}
        </select>
        <button>Add model</button>
    </div>
    <div class="diagram">
    <!-- Element representing a model  -->
     <div class="diagram-models">

     </div>
     <button on:click={clearLoadedModels}>Clear Models</button>
</div>
</div>