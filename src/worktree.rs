//! Git worktree management. Each parallel task is given its own worktree in
//! which it does its work, and merges back into the main tree on completion.

use crate::merge;
use anyhow::{Context, Result, anyhow};
use git2::{Repository, Signature};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

pub struct Workspace {
    pub root: PathBuf,
    pub repo: Mutex<Repository>,
}

impl Workspace {
    /// Initialize (or reuse) the workspace as a git repo. The default branch
    /// `main` will be created if necessary. The orchestrator's main loop
    /// commits to this repo at phase boundaries and after each task merge.
    pub fn init(root: &Path) -> Result<Arc<Self>> {
        std::fs::create_dir_all(root).context("creating workdir")?;
        let repo = match Repository::open(root) {
            Ok(r) => r,
            Err(_) => Repository::init(root).context("git init")?,
        };
        // Ensure .gitignore covers our bookkeeping directories before we ever
        // run `git add -A`. We do this before the initial commit so the
        // .gitignore is part of history from commit #1.
        ensure_gitignore(root)?;
        // Ensure there's at least one commit so worktrees can branch from it.
        let needs_initial = repo.head().is_err();
        if needs_initial {
            let sig = Signature::now("bureau-rs", "bureau-rs@local")?;
            let mut idx = repo.index()?;
            idx.add_path(std::path::Path::new(".gitignore")).ok();
            idx.write()?;
            let tree_id = idx.write_tree()?;
            let tree = repo.find_tree(tree_id)?;
            repo.commit(
                Some("HEAD"),
                &sig,
                &sig,
                "initial: gitignore",
                &tree,
                &[],
            )?;
        }
        Ok(Arc::new(Self {
            root: root.to_path_buf(),
            repo: Mutex::new(repo),
        }))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Commit current state of the main worktree. Returns the new HEAD oid.
    pub async fn commit(&self, message: &str) -> Result<git2::Oid> {
        let repo = self.repo.lock().await;
        do_commit(&repo, message)
    }

    /// Commit only if the index after `add_all` differs from HEAD's tree.
    /// Returns Some(oid) if a commit was made, None if nothing changed.
    pub async fn commit_if_dirty(&self, message: &str) -> Result<Option<git2::Oid>> {
        let repo = self.repo.lock().await;
        let mut idx = repo.index()?;
        idx.add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None)?;
        idx.write()?;
        let tree_id = idx.write_tree()?;
        let parent_tree_id = repo
            .head()
            .ok()
            .and_then(|h| h.target())
            .and_then(|t| repo.find_commit(t).ok())
            .map(|c| c.tree_id());
        if Some(tree_id) == parent_tree_id {
            return Ok(None);
        }
        let sig = Signature::now("bureau-rs", "bureau-rs@local")?;
        let tree = repo.find_tree(tree_id)?;
        let parent_commit = repo
            .head()
            .ok()
            .and_then(|h| h.target())
            .and_then(|t| repo.find_commit(t).ok());
        let parents: Vec<&git2::Commit> = parent_commit.iter().collect();
        let oid = repo.commit(Some("HEAD"), &sig, &sig, message, &tree, &parents)?;
        Ok(Some(oid))
    }
}

fn do_commit(repo: &Repository, message: &str) -> Result<git2::Oid> {
    let sig = Signature::now("bureau-rs", "bureau-rs@local")?;
    let mut idx = repo.index()?;
    idx.add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None)?;
    idx.write()?;
    let tree_id = idx.write_tree()?;
    let tree = repo.find_tree(tree_id)?;
    let parent_commit = repo
        .head()
        .ok()
        .and_then(|h| h.target())
        .and_then(|t| repo.find_commit(t).ok());
    let parents: Vec<&git2::Commit> = parent_commit.iter().collect();
    let oid = repo.commit(Some("HEAD"), &sig, &sig, message, &tree, &parents)?;
    Ok(oid)
}

/// One scratch worktree for a parallel task.
pub struct Worktree {
    pub id: Uuid,
    pub path: PathBuf,
}

/// Worktree pool. We don't actually use `git worktree` (which requires linked
/// worktrees with branches); instead we copy-on-write the working set of
/// declared read/write files into a per-task scratch dir. This is simpler and
/// avoids libgit2 worktree complexity, while preserving the "isolated tree
/// per task" semantics the spec requires.
pub struct WorktreePool {
    pub workspace: Arc<Workspace>,
    pub scratch_root: PathBuf,
}

impl WorktreePool {
    pub fn new(workspace: Arc<Workspace>) -> Result<Self> {
        let scratch_root = workspace.root.join(".bureau").join("worktrees");
        std::fs::create_dir_all(&scratch_root)?;
        Ok(Self {
            workspace,
            scratch_root,
        })
    }

    /// Allocate a new worktree by copying the entire main workdir into a
    /// scratch directory. All paths under .git, target, and .bureau are
    /// excluded. The returned path is the scratch root that the agent should
    /// see as its workdir.
    pub fn allocate(&self) -> Result<Worktree> {
        let id = Uuid::new_v4();
        let path = self.scratch_root.join(id.to_string());
        std::fs::create_dir_all(&path)?;
        copy_workdir(&self.workspace.root, &path)?;
        Ok(Worktree { id, path })
    }

    /// Merge a worktree's writes back to the main workdir. For files that
    /// appear in both, we attempt content-aware merging (Cargo.toml, mod
    /// declaration files); other files are copied straight across (write-set
    /// disjointness should make this safe per scheduler invariants).
    /// Returns the list of files that were written or modified in the main
    /// workdir.
    pub async fn merge_back(
        &self,
        worktree: &Worktree,
        written_files: &std::collections::HashSet<PathBuf>,
    ) -> Result<Vec<PathBuf>> {
        let mut changed = Vec::new();
        for rel in written_files {
            let src = worktree.path.join(rel);
            let dst = self.workspace.root.join(rel);
            if !src.exists() {
                continue;
            }
            let new_content = std::fs::read_to_string(&src)
                .with_context(|| format!("reading scratch {}", src.display()))?;
            if dst.exists() {
                let existing = std::fs::read_to_string(&dst)
                    .with_context(|| format!("reading existing {}", dst.display()))?;
                if existing == new_content {
                    continue;
                }
                match merge::three_way_merge(rel, None, &existing, &new_content)? {
                    merge::MergeOutcome::Clean(merged) => {
                        std::fs::write(&dst, merged.as_bytes())?;
                    }
                    merge::MergeOutcome::Conflict { theirs, .. } => {
                        // For conflicts on files outside our merge drivers, prefer the
                        // newer (theirs) value — the scheduler's write-set guards should
                        // generally prevent this case.
                        std::fs::write(&dst, theirs.as_bytes())?;
                    }
                }
            } else {
                if let Some(parent) = dst.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&dst, new_content.as_bytes())?;
            }
            changed.push(rel.clone());
        }
        Ok(changed)
    }

    pub fn release(&self, worktree: Worktree) -> Result<()> {
        if worktree.path.exists() {
            std::fs::remove_dir_all(&worktree.path)?;
        }
        Ok(())
    }
}

/// Ensure `<root>/.gitignore` contains the entries we need to keep
/// orchestrator bookkeeping (logs, checkpoints, scratch worktrees) and Rust
/// build artifacts out of the generated crate's git history. Idempotent:
/// existing entries are preserved; only missing ones are appended.
fn ensure_gitignore(root: &Path) -> Result<()> {
    const REQUIRED: &[&str] = &[
        "/.bureau/",
        "/target/",
        "**/*.rs.bk",
        "Cargo.lock.bak",
    ];
    let path = root.join(".gitignore");
    let mut existing = if path.exists() {
        std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?
    } else {
        String::new()
    };
    let mut changed = false;
    let lines: std::collections::HashSet<&str> = existing
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    let mut to_append = Vec::new();
    for entry in REQUIRED {
        if !lines.contains(*entry) {
            to_append.push(*entry);
            changed = true;
        }
    }
    if changed {
        if !existing.is_empty() && !existing.ends_with('\n') {
            existing.push('\n');
        }
        if !existing.contains("# bureau-rs") {
            existing.push_str("# bureau-rs: orchestrator bookkeeping\n");
        }
        for e in to_append {
            existing.push_str(e);
            existing.push('\n');
        }
        std::fs::write(&path, existing.as_bytes())
            .with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

fn copy_workdir(from: &Path, to: &Path) -> Result<()> {
    for entry in walkdir::WalkDir::new(from).min_depth(1).into_iter() {
        let entry = entry?;
        let p = entry.path();
        let rel = p
            .strip_prefix(from)
            .map_err(|e| anyhow!("strip prefix: {}", e))?;
        // Filter on the path RELATIVE to the source root so a workspace
        // path that itself contains `.bureau` (e.g. tests, weird mounts)
        // doesn't inadvertently exclude everything.
        if rel.components().any(|c| {
            let s = c.as_os_str().to_string_lossy();
            s == ".git" || s == "target" || s == ".bureau"
        }) {
            continue;
        }
        let dst = to.join(rel);
        if entry.file_type().is_dir() {
            std::fs::create_dir_all(&dst)?;
        } else if entry.file_type().is_file() {
            if let Some(parent) = dst.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(p, &dst)?;
        }
    }
    Ok(())
}
