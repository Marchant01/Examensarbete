use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, State};

struct EngineProcess {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
}

struct EngineState {
    process: Mutex<EngineProcess>,
}

impl Default for EngineState {
    fn default() -> Self {
        Self {
            process: Mutex::new(EngineProcess {
                child: None,
                stdin: None,
            }),
        }
    }
}

fn repo_root() -> Result<PathBuf, String> {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .canonicalize()
        .map_err(|e| format!("Failed to resolve repo root: {e}"))
}

fn engine_python(root: &PathBuf) -> PathBuf {
    let windows_venv = root.join("engine").join(".venv").join("Scripts").join("python.exe");
    if windows_venv.exists() {
        return windows_venv;
    }

    let unix_venv = root.join("engine").join(".venv").join("bin").join("python");
    if unix_venv.exists() {
        return unix_venv;
    }

    PathBuf::from("python")
}

fn spawn_output_reader(app: AppHandle, event_name: &'static str, reader: impl std::io::Read + Send + 'static) {
    std::thread::spawn(move || {
        let buffered = BufReader::new(reader);

        for line in buffered.lines() {
            match line {
                Ok(line) => {
                    let _ = app.emit(event_name, line);
                }
                Err(err) => {
                    let _ = app.emit("engine-stderr", format!("Failed to read engine output: {err}"));
                    break;
                }
            }
        }
    });
}

fn take_engine_pipes(child: &mut Child) -> Result<(ChildStdin, ChildStdout, ChildStderr), String> {
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "Engine stdin was not available".to_string())?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Engine stdout was not available".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "Engine stderr was not available".to_string())?;

    Ok((stdin, stdout, stderr))
}

#[tauri::command]
fn start_engine(app: AppHandle, state: State<'_, EngineState>) -> Result<(), String> {
    let root = repo_root()?;
    let python = engine_python(&root);
    let engine_script = root.join("engine").join("engine.py");
    let models_dir = root.join("models");

    let mut process = state
        .process
        .lock()
        .map_err(|_| "Engine state lock was poisoned".to_string())?;

    if let Some(child) = process.child.as_mut() {
        match child.try_wait() {
            Ok(None) => return Ok(()),
            Ok(Some(_)) => {
                process.child = None;
                process.stdin = None;
            }
            Err(err) => return Err(format!("Failed to inspect engine process: {err}")),
        }
    }

    let mut child = Command::new(&python)
        .arg(&engine_script)
        .current_dir(&root)
        .env("MODELS_DIR", &models_dir)
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            format!(
                "Failed to start engine with python '{}' and script '{}': {e}",
                python.display(),
                engine_script.display()
            )
        })?;

    let (stdin, stdout, stderr) = take_engine_pipes(&mut child)?;

    spawn_output_reader(app.clone(), "engine-stdout", stdout);
    spawn_output_reader(app, "engine-stderr", stderr);

    process.stdin = Some(stdin);
    process.child = Some(child);

    Ok(())
}

#[tauri::command]
fn stop_engine(state: State<'_, EngineState>) -> Result<(), String> {
    let mut process = state
        .process
        .lock()
        .map_err(|_| "Engine state lock was poisoned".to_string())?;

    process.stdin = None;

    if let Some(mut child) = process.child.take() {
        child.kill().map_err(|e| format!("Failed to stop engine: {e}"))?;
        let _ = child.wait();
    }

    Ok(())
}

#[tauri::command]
fn send_engine_json(state: State<'_, EngineState>, payload: Value) -> Result<(), String> {
    let line = serde_json::to_string(&payload)
        .map_err(|e| format!("Failed to serialize engine payload: {e}"))?;

    let mut process = state
        .process
        .lock()
        .map_err(|_| "Engine state lock was poisoned".to_string())?;

    let stdin = process
        .stdin
        .as_mut()
        .ok_or_else(|| "Engine not started".to_string())?;

    stdin
        .write_all(line.as_bytes())
        .map_err(|e| format!("Failed to write engine payload: {e}"))?;
    stdin
        .write_all(b"\n")
        .map_err(|e| format!("Failed to write engine newline: {e}"))?;
    stdin
        .flush()
        .map_err(|e| format!("Failed to flush engine payload: {e}"))?;

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(EngineState::default())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            start_engine,
            stop_engine,
            send_engine_json
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
