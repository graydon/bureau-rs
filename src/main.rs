use anyhow::{Context, Result};
use bureau_rs::{
    config::Config, engine::Engine, event_log, state::EngineState, state::StateHandle, web,
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
    ///
    /// Restarting in an existing workdir resumes from on-disk state:
    /// stages marked Done in `.bureau/graph.json` are skipped, in-flight
    /// worktrees from a prior crashed run are pruned automatically, and
    /// the cost/budget counter resets to zero.
    work_dir: PathBuf,
    /// Web UI port.
    #[arg(long, default_value_t = 8765)]
    port: u16,
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

    // EngineState is fresh on every start. The project's actual state
    // (graph + authored files) lives in `.bureau/graph.json` + git on
    // main, so restarting in an existing workdir naturally resumes from
    // where the last run left off. The transient cost / scheduler /
    // history fields reset to zero — `.bureau/log.jsonl` keeps the
    // forensic record of prior sessions for anyone who needs it.
    let state = StateHandle::new(EngineState::new(
        cli.work_dir.clone(),
        cli.config_dir.clone(),
        config.toml.project_name.clone(),
    ));

    // Append-only event log.
    let log_path = cli.work_dir.join(".bureau").join("log.jsonl");
    let logger_handle = match event_log::spawn(&state, log_path.clone()) {
        Ok(h) => {
            tracing::info!("event log: {}", log_path.display());
            Some(h)
        }
        Err(e) => {
            tracing::warn!("could not start event log: {e}");
            None
        }
    };

    // Fetch live OpenRouter prices once at startup. If the call fails
    // (offline, network blip, etc.) we fall through with an empty table
    // and `compute_total_cost` falls back to built-in approximations.
    let prices = bureau_rs::pricing::fetch(config.toml.provider.base_url.as_deref()).await;

    // Construct the engine first so we can share its graph mutex with the
    // web UI (the web's reset_node mutates the engine's authoritative
    // graph directly).
    let driver: Arc<dyn bureau_rs::engine::LlmDriver> =
        Arc::new(bureau_rs::engine::OpenRouterDriver::from_config(&config)?);
    let engine = Arc::new(Engine::with_driver_and_prices(
        config.clone(),
        state.clone(),
        driver,
        prices,
    )?);

    let ui_handle = if !cli.no_ui {
        let app = web::AppState {
            state: state.clone(),
            workdir: cli.work_dir.clone(),
            layout: config.layout(),
            worktrees: Some(engine.worktrees.clone()),
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

    match &result {
        Ok(_) => tracing::info!("pipeline complete"),
        Err(e) => tracing::error!("pipeline error: {e:#}"),
    }

    if !cli.exit_when_done && ui_handle.is_some() {
        let kind = if result.is_ok() { "complete" } else { "halted" };
        tracing::info!(
            "pipeline {kind} — web UI still running at http://0.0.0.0:{}. Press Ctrl+C to exit.",
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
