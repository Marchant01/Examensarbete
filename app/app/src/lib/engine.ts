import { writable } from 'svelte/store';
import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';

export const engineEvents = writable<any[]>([]);
export const engineRaw = writable<string[]>([]);
export const engineErr = writable<string[]>([]);

export type TextToImageInput = {
    prompt: string;
    num_steps: number;
    guidance_scale: number;
    seed: number;
};

let startPromise: Promise<void> | null = null;
let listenersPromise: Promise<void> | null = null;
let unlistenFns: UnlistenFn[] = [];

async function ensureListeners() {
    if (listenersPromise) return listenersPromise;

    listenersPromise = (async () => {
        const stdoutUnlisten = await listen<string>("engine-stdout", (event) => {
            const line = event.payload;

            try {
                const ev = JSON.parse(line);
                engineEvents.update((xs) => [...xs, ev]);
                console.log("ENGINE EVENT:", ev);
            } catch {
                console.log("ENGINE RAW", line);
                engineRaw.update((xs) => [...xs, line]);
            }
        });

        const stderrUnlisten = await listen<string>("engine-stderr", (event) => {
            const line = event.payload;
            engineErr.update((xs) => [...xs, line]);
            console.log("ENGINE STDERR:", line);
        });

        unlistenFns = [stdoutUnlisten, stderrUnlisten];
    })();

    return listenersPromise;
}

// Starts the python engine and reads from the IO stream
export async function startEngine() {
    if (startPromise) return startPromise;

    startPromise = (async () => {
        try {
            await ensureListeners();
            await invoke('start_engine');
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            engineErr.update((xs) => [...xs, message]);
            console.log("ENGINE START ERROR:", message);
            throw error;
        } finally {
            startPromise = null;
        }
    })();

    return startPromise;
};

export async function stopEngine() {
    await invoke('stop_engine');

    for (const unlisten of unlistenFns) {
        unlisten();
    }
    unlistenFns = [];
    listenersPromise = null;
};

export async function sendJson(obj: unknown) {
    await startEngine();
    await invoke('send_engine_json', { payload: obj });
};

export async function getModels(task: string, limit: number = 10) {
    await sendJson({
        id: crypto.randomUUID(),
        cmd: "list_available_models",
        args: {"task": task, "limit": limit}
    });
};

export async function getInstalledModels() {
    await sendJson({
        id: crypto.randomUUID(),
        cmd: "list_installed_models",
        args: {}
    });
};

// Could be used for future features where more 
// model categories are added and make it dynamic in the frontend
export async function getFilters() {
    await sendJson({
        id: crypto.randomUUID(),
        cmd: "list_model_filters",
        args: {}
    });
};

export async function installModel(repoID: string, task: string) {
    await sendJson({
        id: crypto.randomUUID(),
        cmd: "install_model",
        args: {"repo_id": repoID, "task": task}
    });
};

export async function loadModel(repoID: string, task: string) {
    await sendJson({
        id: crypto.randomUUID(),
        cmd: "load_model",
        args: {"repo_id": repoID, "task": task}
    });
};

export async function clearLoadedModels() {
    await sendJson({
        id: crypto.randomUUID(),
        cmd: "clear_loaded_models",
        args: {}
    });
};

export async function runFlow(flow: any[], input: any) {
    await sendJson({
        id: crypto.randomUUID(),
        cmd: "run_flow",
        args: {"flow": flow, "input": input}
    });
};
