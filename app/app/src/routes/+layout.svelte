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

<div class="main-nav-container">
  <nav class="main-nav">
    <a href="/models">Browse models</a>
    <a href="/flows">Flow</a>
  </nav>
</div>

<main>
    <slot />
</main>
