import { writable } from 'svelte/store';
import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';

export const engineEvents = writable<any[]>([]);
export const engineRaw = writable<string[]>([]);
export const engineErr = writable<string[]>([]);

export const STABLE_DIFFUSION_MODEL_ID = "stable-diffusion";
export const CONTROLNET_CANNY_MODEL_ID = "controlnet-canny";

export type SupportedModel = {
    id: string;
    label: string;
    task: string;
    runner: string;
    installed: boolean;
    path: string | null;
    missing_components: string[];
};

export type TextToImageInput = {
    prompt: string;
    num_steps: number;
    guidance_scale: number;
    seed: number;
};

export type ControlNetCannyInput = TextToImageInput & {
    image_data_url: string;
    canny_low_threshold: number;
    canny_high_threshold: number;
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

// Starts the python engine and reads from the IO stream.
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

async function sendJson(obj: unknown) {
    await startEngine();
    await invoke('send_engine_json', { payload: obj });
};

async function sendCommand(cmd: string, args: Record<string, unknown> = {}) {
    const id = crypto.randomUUID();
    await sendJson({ id, cmd, args });
    return id;
};

export async function getSupportedModels() {
    return sendCommand("list_supported_models");
};

export async function loadModel(modelID: string) {
    return sendCommand("load_model", { model_id: modelID });
};

export async function clearLoadedModels() {
    return sendCommand("clear_loaded_models");
};

export async function runModel(
    modelID: string,
    input: TextToImageInput | ControlNetCannyInput,
) {
    return sendCommand("run_model", { model_id: modelID, input });
};
