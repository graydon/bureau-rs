use anyhow::{Context, Result, anyhow};
use bureau_rs::{
    checkpoint, config::Config, phase::Phase, scheduler::Orchestrator, state::OrchestratorState,
    state::StateHandle, web, worktree::Workspace,
};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "bureau-rs", about = "Hierarchical multi-phase Rust code agent orchestrator")]
struct Cli {
    /// Configuration directory containing problem.md, phases.toml, prompts/
    config_dir: PathBuf,
    /// Working directory where the generated crate is built
    work_dir: PathBuf,
    /// Web UI port
    #[arg(long, default_value_t = 8765)]
    port: u16,
    /// Resume from a checkpoint JSON file
    #[arg(long)]
    resume: Option<PathBuf>,
    /// Start from a specific phase (skip earlier phases)
    #[arg(long)]
    phase: Option<String>,
    /// Decompose and show task graph but don't execute
    #[arg(long)]
    dry_run: bool,
    /// Don't start the web server
    #[arg(long)]
    no_ui: bool,
    /// Exit immediately after the pipeline finishes instead of keeping the
    /// web UI alive for browsing.
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
    let workspace = Workspace::init(&cli.work_dir)?;

    let initial_state = if let Some(p) = &cli.resume {
        checkpoint::load(p)?
    } else {
        OrchestratorState::new(cli.work_dir.clone(), cli.config_dir.clone())
    };
    let handle = StateHandle::new(initial_state);

    // Append-only log of every UI event (prompts, tool calls, results, costs,
    // file changes...) at <workdir>/.bureau/log.jsonl. Survives restarts.
    let log_path = cli.work_dir.join(".bureau").join("log.jsonl");
    let logger_handle = match checkpoint::spawn_event_logger(&handle, log_path.clone()) {
        Ok(h) => {
            tracing::info!("event log: {}", log_path.display());
            Some(h)
        }
        Err(e) => {
            tracing::warn!("could not start event log at {}: {e:#}", log_path.display());
            None
        }
    };

    let start_phase = match &cli.phase {
        Some(p) => Phase::parse(p).ok_or_else(|| anyhow!("unknown phase '{}'", p))?,
        None => Phase::Spec,
    };

    if cli.dry_run {
        println!("dry-run: would start at phase {start_phase}");
        return Ok(());
    }

    let orch = Arc::new(Orchestrator::new(
        config.clone(),
        handle.clone(),
        workspace.clone(),
    )?);

    let ui_handle = if !cli.no_ui {
        let app = web::AppState {
            state: handle.clone(),
            workdir: cli.work_dir.clone(),
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

    let result = orch.run(start_phase).await;

    // Final checkpoint regardless of success/failure.
    let ckpt_dir = cli.work_dir.join(".bureau").join("checkpoints");
    match checkpoint::save(&handle.snapshot(), &ckpt_dir) {
        Ok(p) => tracing::info!("final checkpoint saved: {}", p.display()),
        Err(e) => tracing::warn!("final checkpoint failed: {e:#}"),
    }
    let _ = checkpoint::save_latest(&handle.snapshot(), &ckpt_dir);

    match &result {
        Ok(_) => tracing::info!("pipeline finished"),
        Err(e) => tracing::error!("pipeline error: {e:#}"),
    }

    // Keep the UI alive so the user can browse the session state. Exit on
    // Ctrl+C (or SIGTERM on Unix). This is the default; --exit-when-done
    // restores the old "exit immediately" behaviour.
    if !cli.exit_when_done && ui_handle.is_some() && !cli.no_ui {
        tracing::info!(
            "pipeline complete — web UI still running at http://0.0.0.0:{}. Press Ctrl+C to exit.",
            cli.port
        );
        wait_for_shutdown().await;
        tracing::info!("shutdown requested, exiting");
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
        let mut sigterm = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
                return;
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = sigterm.recv() => {},
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
