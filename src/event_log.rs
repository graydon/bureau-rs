//! Append-only JSONL event log.
//!
//! Subscribes to the broadcast `UiEvent` channel and writes every event
//! to disk as one JSON object per line. The log is forensic — it lets
//! operators reconstruct what happened during a run after the fact, and
//! survives engine crashes. It is NOT a checkpoint: there is no replay
//! and no "resume from log" feature. Restarts pick up project state
//! from `.bureau/graph.json` + git on `main`; the log is read-only.

use crate::state::StateHandle;
use anyhow::{Context, Result};
use chrono::Utc;
use std::io::Write;
use std::path::PathBuf;
use tokio::task::JoinHandle;

pub fn spawn(state: &StateHandle, log_path: PathBuf) -> Result<JoinHandle<()>> {
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("opening {}", log_path.display()))?;
    let header = serde_json::json!({"session_start": Utc::now().to_rfc3339()});
    if let Err(e) = writeln!(file, "{}", serde_json::to_string(&header).unwrap_or_default()) {
        tracing::warn!("event log header write failed: {e}");
    }
    let mut rx = state.subscribe();
    let log_display = log_path.display().to_string();
    let handle = tokio::spawn(async move {
        // After the first write failure, log once and stop trying so we
        // don't spam the tracing layer on every event.
        let mut log_dead = false;
        loop {
            match rx.recv().await {
                Ok(ev) => {
                    if log_dead {
                        continue;
                    }
                    // Skip streaming chunks. They're for the live UI;
                    // a single multi-turn LLM call easily produces
                    // hundreds of them and they drown out the actual
                    // record-of-events in the forensic log. The
                    // canonical assistant text lands as a
                    // `TranscriptAppended` at end-of-turn — that's
                    // what we want to see when reconstructing what
                    // happened.
                    if matches!(ev, crate::state::UiEvent::AssistantChunk { .. }) {
                        continue;
                    }
                    let wrapper = serde_json::json!({
                        "at": Utc::now().to_rfc3339(),
                        "ev": ev,
                    });
                    if let Ok(line) = serde_json::to_string(&wrapper) {
                        if let Err(e) = writeln!(file, "{}", line) {
                            tracing::error!(
                                "event log write to {log_display} failed: {e}; subsequent events will be dropped"
                            );
                            log_dead = true;
                            continue;
                        }
                        if let Err(e) = file.flush() {
                            tracing::error!(
                                "event log flush to {log_display} failed: {e}; subsequent events will be dropped"
                            );
                            log_dead = true;
                        }
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("event log lagged, dropped {n} events");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });
    Ok(handle)
}
