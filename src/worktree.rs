//! Per-task git worktrees. Each parallel task does its work on its own
//! branch in its own worktree, then merges back to main on success.
//!
//! Why: parallel tasks each render the workspace and run cargo. With a
//! single shared workdir, parallel cargos contend for `target/` and
//! parallel renders contend for the same files. With a worktree per task,
//! each task has its own checkout (linked to the same `.git`) and its
//! own `target/`, so they truly run in parallel.
//!
//! The framework renders the "spine" deterministically from the shared
//! graph, so two parallel tasks rendering at adjacent moments produce
//! mostly-identical output for nodes neither of them owns. The engine
//! re-renders each worktree from the canonical shared graph just before
//! merging, which makes git's three-way merge clean: any node already
//! landed on main has identical content on both sides.
//!
//! All git operations go through `git2` (libgit2 bindings) — no
//! subprocess calls. Repositories are opened per-operation rather than
//! kept long-lived to keep the locking story simple.

use anyhow::{Context, Result};
use git2::{
    BranchType, IndexAddOption, Repository, RepositoryInitOptions, Signature,
    WorktreeAddOptions, WorktreePruneOptions, build::CheckoutBuilder,
};
use parking_lot::Mutex as ParkingMutex;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct Worktree {
    pub task_id: Uuid,
    pub branch: String,
    pub path: PathBuf,
    /// libgit2 worktree NAME — registered in `$GIT_DIR/worktrees/<name>`.
    /// We use the task UUID as the name so it's unique and easy to
    /// look up later for pruning.
    pub name: String,
}

/// The single main-branch repo at the workdir root. There's exactly one;
/// per-task worktrees are linked to it.
pub struct Workspace {
    pub root: PathBuf,
}

impl Workspace {
    /// Initialize (or reuse) the workdir as a git repo on `main`. Creates
    /// `.gitignore` covering bureau-rs bookkeeping and an initial commit
    /// so worktrees can branch from a real ref.
    pub fn init(root: &Path) -> Result<Arc<Self>> {
        std::fs::create_dir_all(root)?;
        let already_inited = root.join(".git").exists();
        if !already_inited {
            // Initial branch = main, regardless of user's git config.
            let mut opts = RepositoryInitOptions::new();
            opts.initial_head("main");
            let repo = Repository::init_opts(root, &opts).context("Repository::init_opts")?;
            configure_repo(&repo)?;
            ensure_gitignore(root)?;
            // Stage .gitignore and make the seed commit.
            let mut idx = repo.index()?;
            idx.add_path(Path::new(".gitignore")).ok();
            idx.write()?;
            let tree_id = idx.write_tree()?;
            let tree = repo.find_tree(tree_id)?;
            let sig = bureau_signature();
            repo.commit(Some("HEAD"), &sig, &sig, "scaffold", &tree, &[])?;
        } else {
            // Existing repo: still ensure config + .gitignore are sane.
            let repo = Repository::open(root).context("open existing repo")?;
            configure_repo(&repo)?;
            ensure_gitignore(root)?;
        }
        Ok(Arc::new(Self {
            root: root.to_path_buf(),
        }))
    }

    /// Stage everything in the main workdir and commit if anything changed.
    /// Returns true if a commit was made.
    pub fn commit_main(&self, message: &str) -> Result<bool> {
        let repo = Repository::open(&self.root).context("open main repo")?;
        let mut idx = repo.index()?;
        idx.add_all(["*"].iter(), IndexAddOption::DEFAULT, None)?;
        idx.write()?;
        let tree_id = idx.write_tree()?;
        let head_tree_id = repo
            .head()
            .ok()
            .and_then(|h| h.peel_to_tree().ok())
            .map(|t| t.id());
        if Some(tree_id) == head_tree_id {
            return Ok(false);
        }
        let tree = repo.find_tree(tree_id)?;
        let sig = bureau_signature();
        let parent = repo.head().ok().and_then(|h| h.peel_to_commit().ok());
        let parents: Vec<&git2::Commit> = parent.iter().collect();
        repo.commit(Some("HEAD"), &sig, &sig, message, &tree, &parents)?;
        Ok(true)
    }
}

/// Manages worktree allocation, merge-back, and cleanup.
pub struct WorktreePool {
    pub workspace: Arc<Workspace>,
    pub scratch_root: PathBuf,
    /// Serializes operations that need exclusive access to the main
    /// repo's working tree — worktree-add, merge, branch ops. Held only
    /// for the duration of those calls.
    main_lock: tokio::sync::Mutex<()>,
    /// Live record of currently-allocated worktrees (UI introspection).
    active: ParkingMutex<Vec<Worktree>>,
}

impl WorktreePool {
    pub fn new(workspace: Arc<Workspace>) -> Result<Self> {
        let scratch_root = workspace.root.join(".bureau").join("worktrees");
        std::fs::create_dir_all(&scratch_root)?;
        // Prune any leftover worktree records from a prior crashed run.
        if let Ok(repo) = Repository::open(&workspace.root) {
            if let Ok(names) = repo.worktrees() {
                for name in names.iter().flatten() {
                    if let Ok(wt) = repo.find_worktree(name) {
                        let mut o = WorktreePruneOptions::new();
                        o.valid(true).working_tree(true).locked(true);
                        let _ = wt.prune(Some(&mut o));
                    }
                }
            }
        }
        Ok(Self {
            workspace,
            scratch_root,
            main_lock: tokio::sync::Mutex::new(()),
            active: ParkingMutex::new(Vec::new()),
        })
    }

    /// All currently-allocated worktrees. Used by the web UI to show
    /// in-progress work alongside what's on main.
    pub fn active_worktrees(&self) -> Vec<Worktree> {
        self.active.lock().clone()
    }

    /// Allocate a worktree branched from the current main HEAD. The
    /// branch is named `task/<task-id>`; the worktree lives at
    /// `<workdir>/.bureau/worktrees/<task-id>/`.
    pub async fn allocate(&self, task_id: Uuid) -> Result<Worktree> {
        let _guard = self.main_lock.lock().await;
        let id_str = task_id.to_string();
        let path = self.scratch_root.join(&id_str);
        // Clear any stale dir from a prior crashed run.
        if path.exists() {
            let _ = std::fs::remove_dir_all(&path);
        }
        let branch_name = format!("task/{id_str}");
        let repo = Repository::open(&self.workspace.root)?;
        // Drop any stale branch with this name (force).
        if let Ok(mut b) = repo.find_branch(&branch_name, BranchType::Local) {
            let _ = b.delete();
        }
        // Create the branch from current HEAD.
        let head_commit = repo.head()?.peel_to_commit()?;
        let branch = repo.branch(&branch_name, &head_commit, false)?;
        let branch_ref = branch.into_reference();
        // Create the worktree pointing at the branch.
        let mut opts = WorktreeAddOptions::new();
        opts.reference(Some(&branch_ref));
        let _wt = repo
            .worktree(&id_str, &path, Some(&opts))
            .context("Repository::worktree")?;
        // Inherit our config knobs (gpgsign etc.) into the worktree's
        // local config; libgit2 normally points worktrees at the parent
        // repo's config but commit signing depends on local fallback.
        if let Ok(wt_repo) = Repository::open(&path) {
            let _ = configure_repo(&wt_repo);
        }
        let wt = Worktree {
            task_id,
            branch: branch_name,
            path,
            name: id_str,
        };
        self.active.lock().push(wt.clone());
        Ok(wt)
    }

    /// Commit the worktree's state and merge it into main. The CALLER
    /// is responsible for re-rendering the worktree from the canonical
    /// shared graph state before invoking this — that's what makes the
    /// merge clean. With every task's pre-merge tree reflecting the same
    /// shared graph state, the three-way merge resolves cleanly: any
    /// node already landed on main has identical content on both sides;
    /// only the truly-new content from this task is the diff.
    ///
    /// We deliberately do NOT use a "favor task" or "favor main" merge
    /// strategy — if the merge has real conflicts, that's a render
    /// determinism bug and we want to surface it as a task failure.
    pub async fn merge_and_release(&self, wt: Worktree, message: &str) -> Result<()> {
        if let Err(e) = commit_in_worktree(&wt.path, message) {
            tracing::warn!("commit in worktree {}: {e:#}", wt.path.display());
        }
        let _guard = self.main_lock.lock().await;
        let result = self.do_merge(&wt, message);
        self.cleanup(&wt);
        result
    }

    fn do_merge(&self, wt: &Worktree, message: &str) -> Result<()> {
        let repo = Repository::open(&self.workspace.root)?;
        let main_tip = repo.head()?.peel_to_commit()?;
        let task_branch = repo.find_branch(&wt.branch, BranchType::Local)?;
        let task_tip = task_branch.into_reference().peel_to_commit()?;
        if main_tip.id() == task_tip.id() {
            // Branch and main point at the same commit — nothing to merge.
            return Ok(());
        }
        // In-memory three-way merge.
        let mut idx = repo.merge_commits(&main_tip, &task_tip, None)?;
        if idx.has_conflicts() {
            anyhow::bail!(
                "merge of branch {} into main has conflicts — refusing to land \
                 (likely a render-determinism bug; the task is failed)",
                wt.branch
            );
        }
        let tree_id = idx.write_tree_to(&repo)?;
        let tree = repo.find_tree(tree_id)?;
        let sig = bureau_signature();
        let merge_commit_id = repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            message,
            &tree,
            &[&main_tip, &task_tip],
        )?;
        // Update main's working tree to match the new HEAD.
        let merge_commit = repo.find_commit(merge_commit_id)?;
        let merge_tree = merge_commit.tree()?;
        let mut co = CheckoutBuilder::new();
        co.force();
        repo.checkout_tree(merge_tree.as_object(), Some(&mut co))?;
        Ok(())
    }

    /// Drop a worktree without merging — used on task failure.
    pub async fn abandon(&self, wt: Worktree) -> Result<()> {
        let _guard = self.main_lock.lock().await;
        self.cleanup(&wt);
        Ok(())
    }

    fn cleanup(&self, wt: &Worktree) {
        // Remove on-disk working dir first (force).
        let _ = std::fs::remove_dir_all(&wt.path);
        // Prune the worktree admin record + delete the branch.
        if let Ok(repo) = Repository::open(&self.workspace.root) {
            if let Ok(wt_handle) = repo.find_worktree(&wt.name) {
                let mut o = WorktreePruneOptions::new();
                o.valid(true).working_tree(true).locked(true);
                let _ = wt_handle.prune(Some(&mut o));
            }
            if let Ok(mut b) = repo.find_branch(&wt.branch, BranchType::Local) {
                let _ = b.delete();
            }
        }
        self.active.lock().retain(|w| w.task_id != wt.task_id);
    }
}

/// Stage everything in the worktree and commit on its branch. If the
/// tree matches HEAD's tree (nothing to commit), we silently no-op —
/// `do_merge` will then short-circuit because main and the branch tip
/// are equal.
fn commit_in_worktree(path: &Path, message: &str) -> Result<()> {
    let repo = Repository::open(path).context("open worktree as repo")?;
    let mut idx = repo.index()?;
    idx.add_all(["*"].iter(), IndexAddOption::DEFAULT, None)?;
    idx.write()?;
    let tree_id = idx.write_tree()?;
    let head_tree_id = repo
        .head()
        .ok()
        .and_then(|h| h.peel_to_tree().ok())
        .map(|t| t.id());
    if Some(tree_id) == head_tree_id {
        return Ok(());
    }
    let tree = repo.find_tree(tree_id)?;
    let sig = bureau_signature();
    let parent = repo.head().ok().and_then(|h| h.peel_to_commit().ok());
    let parents: Vec<&git2::Commit> = parent.iter().collect();
    repo.commit(Some("HEAD"), &sig, &sig, message, &tree, &parents)?;
    Ok(())
}

/// Build a `bureau-rs` signature. We use `Signature::now` (which carries
/// `'static` lifetime) regardless of the repo's signature() — `configure_repo`
/// already sets the repo's user.name/user.email to `bureau-rs`/`bureau-rs@local`,
/// so this is consistent.
fn bureau_signature() -> Signature<'static> {
    Signature::now("bureau-rs", "bureau-rs@local").expect("static signature")
}

fn configure_repo(repo: &Repository) -> Result<()> {
    let mut cfg = repo.config()?;
    if cfg.get_string("user.name").is_err() {
        cfg.set_str("user.name", "bureau-rs")?;
    }
    if cfg.get_string("user.email").is_err() {
        cfg.set_str("user.email", "bureau-rs@local")?;
    }
    // Disable GPG signing locally — bureau-rs commits are bookkeeping,
    // and if the user's global config has signing on but the env can't
    // sign (e.g. devcontainer without the key), commits would fail.
    cfg.set_bool("commit.gpgsign", false)?;
    cfg.set_bool("tag.gpgsign", false)?;
    Ok(())
}

fn ensure_gitignore(root: &Path) -> Result<()> {
    const REQUIRED: &[&str] = &["/.bureau/", "/target/", "**/target/"];
    let path = root.join(".gitignore");
    let mut existing = if path.exists() {
        std::fs::read_to_string(&path)?
    } else {
        String::new()
    };
    let lines: std::collections::HashSet<&str> = existing
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    let mut to_append = Vec::new();
    for entry in REQUIRED {
        if !lines.contains(*entry) {
            to_append.push(*entry);
        }
    }
    if !to_append.is_empty() {
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
        std::fs::write(&path, existing.as_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[tokio::test]
    async fn allocate_creates_branch_and_worktree() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = Workspace::init(tmp.path()).unwrap();
        let pool = WorktreePool::new(ws.clone()).unwrap();
        let id = Uuid::new_v4();
        let wt = pool.allocate(id).await.unwrap();
        assert!(wt.path.exists());
        assert!(wt.branch.starts_with("task/"));
        let repo = Repository::open(&ws.root).unwrap();
        assert!(repo.find_branch(&wt.branch, BranchType::Local).is_ok());
        assert_eq!(pool.active_worktrees().len(), 1);
    }

    #[tokio::test]
    async fn merge_and_release_lands_changes_on_main_and_cleans_up() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = Workspace::init(tmp.path()).unwrap();
        let pool = WorktreePool::new(ws.clone()).unwrap();
        let wt = pool.allocate(Uuid::new_v4()).await.unwrap();
        fs::write(wt.path.join("hello.txt"), "hi from task\n").unwrap();
        let wt_clone = wt.clone();
        pool.merge_and_release(wt, "spec: thing").await.unwrap();
        assert!(!wt_clone.path.exists());
        assert!(ws.root.join("hello.txt").exists());
        assert_eq!(pool.active_worktrees().len(), 0);
        let repo = Repository::open(&ws.root).unwrap();
        assert!(repo.find_branch(&wt_clone.branch, BranchType::Local).is_err());
    }

    #[tokio::test]
    async fn abandon_drops_worktree_without_merging() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = Workspace::init(tmp.path()).unwrap();
        let pool = WorktreePool::new(ws.clone()).unwrap();
        let wt = pool.allocate(Uuid::new_v4()).await.unwrap();
        fs::write(wt.path.join("noped.txt"), "should not land\n").unwrap();
        let wt_clone = wt.clone();
        pool.abandon(wt).await.unwrap();
        assert!(!wt_clone.path.exists());
        assert!(!ws.root.join("noped.txt").exists());
    }

    #[tokio::test]
    async fn two_parallel_worktrees_merge_back_independently() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = Workspace::init(tmp.path()).unwrap();
        let pool = Arc::new(WorktreePool::new(ws.clone()).unwrap());
        let wta = pool.allocate(Uuid::new_v4()).await.unwrap();
        let wtb = pool.allocate(Uuid::new_v4()).await.unwrap();
        fs::write(wta.path.join("a.txt"), "from a\n").unwrap();
        fs::write(wtb.path.join("b.txt"), "from b\n").unwrap();
        pool.merge_and_release(wta, "spec: a").await.unwrap();
        pool.merge_and_release(wtb, "spec: b").await.unwrap();
        assert!(ws.root.join("a.txt").exists());
        assert!(ws.root.join("b.txt").exists());
    }

    #[tokio::test]
    async fn conflicting_merge_is_rejected_not_silently_resolved() {
        // Both worktrees write the SAME file with different content.
        // Their merge should hit a real conflict — and we should reject,
        // not silently pick a side.
        let tmp = tempfile::tempdir().unwrap();
        let ws = Workspace::init(tmp.path()).unwrap();
        let pool = Arc::new(WorktreePool::new(ws.clone()).unwrap());
        let wta = pool.allocate(Uuid::new_v4()).await.unwrap();
        fs::write(wta.path.join("contested.txt"), "version A\n").unwrap();
        let wtb = pool.allocate(Uuid::new_v4()).await.unwrap();
        fs::write(wtb.path.join("contested.txt"), "version B\n").unwrap();
        pool.merge_and_release(wta, "spec: a").await.unwrap();
        let err = pool.merge_and_release(wtb, "spec: b").await.unwrap_err();
        assert!(format!("{err:#}").contains("conflicts"));
        // After failure, main is left at version A (the first merge),
        // and B's worktree is cleaned up.
        let on_main = fs::read_to_string(ws.root.join("contested.txt")).unwrap();
        assert_eq!(on_main, "version A\n");
    }
}
