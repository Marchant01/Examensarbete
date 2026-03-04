import { writable } from 'svelte/store';
import { Command } from '@tauri-apps/plugin-shell'

export const engineEvents = writable<any[]>([]);
export const engineRaw = writable<string[]>([]);
export const engineErr = writable<string[]>([]);

let child:any;

// Starts the python engine and reads from the IO stream
export async function startEngine() {
    const cmd = Command.create('bash', [
        'engine/run.sh'
    ],
    {
        cwd: '../../..',
        env: {
            MODELS_DIR: 'models'
        }
    }
);

    cmd.stdout.on("data", (line: string) => {
        try {
            const ev = JSON.parse(line);
            engineEvents.update((xs) => [...xs, ev]);
            console.log("ENGINE EVENT: " + ev);
        } catch {
            console.log("ENGINE RAW" + line);
            engineRaw.update((xs) => [...xs, line]);
        }
    });

    cmd.stderr.on("data", (line: string) => {
        engineErr.update((xs) => [...xs, line]);
        console.log("ENGINE STDERR: " + line);
    });

    child = await cmd.spawn();
};

export async function stopEngine() {
    if (child) {
        await child.kill();
    }
};

export async function sendJson(obj: unknown) {
    if (!child) throw new Error("Engine not started");
    await child.write(JSON.stringify(obj) + "\n");
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