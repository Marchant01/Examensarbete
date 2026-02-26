<script lang="ts">
    import { onMount } from "svelte";
    import { engineEvents, loadModel, clearLoadedModels, getInstalledModels, runFlow } from "$lib/engine";

    const options: string[] = ["text-to-image", "text-generation", "image-to-image"];

    let selectedModel: string = "";
    let installedModels: Record<string, string[]> = {};
    let flowList: Flow[] = [];
    let loadedModels: string[] = [];
    
    type Flow = {
        flow_steps: string[];
    };

    let initial_input: any;
    
    $: installedModels =
        [...$engineEvents]
            .reverse()
            .find(
                (ev) =>
                    ev?.type === "done" &&
                    ev?.data &&
                    options.every((task) => task in ev.data)
            )?.data ?? {};

    // Flatten all installed models from every task into one list
    $: models = Object.values(installedModels).flat();

    onMount(() => {
        getInstalledModels();
    });

    // Models are loaded here but currently no flow is structured here
    async function addModel() {
        if(selectedModel) {
            loadModel(selectedModel)
            loadedModels.push(selectedModel)
            flowList.push({
                flow_steps: [selectedModel]
            });
        };
    };

    // Adds the model and the steps to a list for tracking the 
    // async function addFlow() {
    //     flowList.push({
    //         flow_steps: [selectedModel]
    //     });
    // };

    async function executeFlow() {
        runFlow(flowList, initial_input)
    }
    // Probably need a function that will handle creating an object that 
    // will contain the input and modelnames and so on for running the flow
</script>

<div class="diagram-container">
    <div class="tool-bar">
        <select bind:value={selectedModel}>
            <option value="" disabled>Select model</option>
            {#each models as model}
            <option value={model}>{model}</option>
            {/each}
        </select>
        <button on:click={addModel}>Add model</button>
        <button on:click={clearLoadedModels}>Clear Models</button>
        <button on:click={executeFlow}>Run Flow</button>
    </div>
    <div class="diagram">
        <div class="diagram-models"></div>
        {#if loadedModels.length == 0}
        <p>Select a model to load</p>
        {:else}
        {#each loadedModels as loadedModel}
            <h1>{loadedModel}</h1>
        {/each}
        {/if}
    </div>
</div>