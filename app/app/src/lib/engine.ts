import { Command } from '@tauri-apps/plugin-shell'

const cmd = Command.create('Python', [
    '-u',
    '../../engine/engine.py'
]);

cmd.stdout.on("data", (line: string) => {
    try {
        const ev = JSON.parse(line);
        console.log("ENGINE EVENT: " + ev)
    } catch {
        console.log("ENGINE RAW" + line)
    }
});

cmd.stderr.on("data", (line: string) => {
    console.log("ENGINE STDERR: " + line)
});

const child = await cmd.spawn();

