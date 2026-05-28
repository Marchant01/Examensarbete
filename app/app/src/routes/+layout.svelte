<script lang="ts">
    import "../app.css";
    import { onMount, onDestroy } from "svelte";
    import { startEngine, stopEngine } from "$lib/engine";
    
    onMount(() => {
      startEngine();
    });
    
    onDestroy(() => {
      stopEngine();
    });
</script>
<main>
    <slot />
</main>
