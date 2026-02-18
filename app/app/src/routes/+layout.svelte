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

<nav>
    <a href="/models">Browse models</a>
    <a href="/">Installed models</a>
</nav>

<main>
    <slot />
</main>
