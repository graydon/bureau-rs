use anyhow::{Context, Result};
use bureau_rs::{
    checkpoint, config::Config, engine::Engine, state::EngineState, state::StateHandle, web,
};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "bureau-rs", about = "Hierarchical decomposition agent for Rust projects")]
struct Cli {
    /// Configuration directory containing problem.md and config.toml.
    config_dir: PathBuf,
    /// Working directory where the generated project is built.
    work_dir: PathBuf,
    /// Web UI port.
    #[arg(long, default_value_t = 8765)]
    port: u16,
    /// Resume from a JSON checkpoint instead of starting fresh.
    #[arg(long)]
    resume: Option<PathBuf>,
    /// Don't start the web server.
    #[arg(long)]
    no_ui: bool,
    /// Exit immediately when the pipeline finishes (default: keep UI alive
    /// until Ctrl+C so you can browse the result).
    #[arg(long)]
    exit_when_done: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("bureau_rs=info,warn")),
        )
        .init();

    let cli = Cli::parse();
    let config = Arc::new(
        Config::load(&cli.config_dir)
            .with_context(|| format!("loading config from {}", cli.config_dir.display()))?,
    );
    std::fs::create_dir_all(&cli.work_dir).context("creating workdir")?;

    let initial_state = if let Some(p) = &cli.resume {
        checkpoint::load(p)?
    } else {
        EngineState::new(
            cli.work_dir.clone(),
            cli.config_dir.clone(),
            config.toml.project_name.clone(),
        )
    };
    let state = StateHandle::new(initial_state);

    // Append-only event log.
    let log_path = cli.work_dir.join(".bureau").join("log.jsonl");
    let logger_handle = match checkpoint::spawn_event_logger(&state, log_path.clone()) {
        Ok(h) => {
            tracing::info!("event log: {}", log_path.display());
            Some(h)
        }
        Err(e) => {
            tracing::warn!("could not start event log: {e}");
            None
        }
    };

    // Construct the engine first so we can share its graph mutex with the
    // web UI (the web's reset_node mutates the engine's authoritative
    // graph directly).
    let engine = Arc::new(Engine::new(config.clone(), state.clone())?);

    let ui_handle = if !cli.no_ui {
        let app = web::AppState {
            state: state.clone(),
            workdir: cli.work_dir.clone(),
            graph: engine.graph.clone(),
        };
        let port = cli.port;
        Some(tokio::spawn(async move {
            if let Err(e) = web::serve(app, port).await {
                tracing::error!("web server: {e:#}");
            }
        }))
    } else {
        None
    };

    let result = engine.clone().run().await;

    // Final checkpoint.
    let ckpt_dir = cli.work_dir.join(".bureau").join("checkpoints");
    let snap = state.snapshot();
    if let Ok(p) = checkpoint::save(&snap, &ckpt_dir) {
        tracing::info!("final checkpoint: {}", p.display());
    }
    let _ = checkpoint::save_latest(&snap, &ckpt_dir);

    match &result {
        Ok(_) => tracing::info!("pipeline complete"),
        Err(e) => tracing::error!("pipeline error: {e:#}"),
    }

    if !cli.exit_when_done && ui_handle.is_some() {
        tracing::info!(
            "pipeline complete — web UI still running at http://0.0.0.0:{}. Press Ctrl+C to exit.",
            cli.port
        );
        wait_for_shutdown().await;
    }

    if let Some(h) = ui_handle {
        h.abort();
    }
    if let Some(h) = logger_handle {
        h.abort();
    }
    result
}

async fn wait_for_shutdown() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        if let Ok(mut sigterm) = signal(SignalKind::terminate()) {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {},
                _ = sigterm.recv() => {},
            }
        } else {
            let _ = tokio::signal::ctrl_c().await;
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
