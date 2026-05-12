//! Per-task git worktrees. Each parallel task does its work on its own
//! branch in its own worktree, then rebases + ff-merges back to main.
//!
//! Why: parallel tasks each render the workspace and run cargo. With a
//! single shared workdir, parallel cargos contend for `target/` and
//! parallel renders contend for the same files. With a worktree per task,
//! each task has its own checkout (linked to the same `.git`) and its
//! own `target/`, so they truly run in parallel.
//!
//! Landing model: rebase the task branch onto current main HEAD, re-gate
//! on the rebased state, then fast-forward main. The rebase is what
//! resolves the "two tasks rendered the workspace at different snapshots"
//! problem: instead of a three-way merge (which conflicts when both sides
//! happened to author the same `.bureau/nodes/x.json`), the second-lander
//! simply re-applies its diff atop the first-lander's tip and re-runs the
//! gate. If the rebase conflicts, the branch is abandoned and the task
//! retries from scratch.
//!
//! All git operations go through `git2` (libgit2 bindings) — no
//! subprocess calls. Repositories are opened per-operation rather than
//! kept long-lived to keep the locking story simple.

use anyhow::{Context, Result};
use git2::{
    BranchType, IndexAddOption, RebaseOptions, Repository, RepositoryInitOptions, Signature,
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
            let repo = Repository::open(root).context("open existing repo")?;
            configure_repo(&repo)?;
            ensure_gitignore(root)?;
        }
        Ok(Arc::new(Self {
            root: root.to_path_buf(),
        }))
    }

    /// Read main's current HEAD commit id.
    pub fn head_commit_id(&self) -> Result<git2::Oid> {
        let repo = Repository::open(&self.root).context("open main repo")?;
        Ok(repo.head()?.peel_to_commit()?.id())
    }

    /// Hard-reset main back to the given commit. Kept around as a
    /// reasonable primitive even though the new engine flow doesn't use
    /// it — operators may still need it for recovery.
    pub fn reset_main_hard(&self, target: git2::Oid) -> Result<()> {
        let repo = Repository::open(&self.root).context("open main repo")?;
        let commit = repo
            .find_commit(target)
            .with_context(|| format!("find target commit {target}"))?;
        let mut co = CheckoutBuilder::new();
        co.force();
        repo.reset(commit.as_object(), git2::ResetType::Hard, Some(&mut co))
            .context("reset main")?;
        Ok(())
    }

    /// Stage everything in the main workdir and commit if anything changed.
    /// Used at scaffold time only. Returns true if a commit was made.
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

    /// Rebase `branch_name` onto current `main` HEAD. On conflict the
    /// rebase is aborted (the branch is left at its original tip) and Err
    /// returns. On success the branch's tip is a linear descendant of
    /// main's HEAD AND the working tree at `worktree_path` (which is
    /// checked out on `branch_name`) reflects the rebased state.
    ///
    /// We open the worktree's repo handle so libgit2 can update its
    /// working tree as it replays each commit — that's exactly what we
    /// need before running the post-rebase gate.
    pub fn rebase_branch_onto_main(
        &self,
        worktree_path: &Path,
        branch_name: &str,
    ) -> Result<()> {
        let repo = Repository::open(worktree_path).context("open worktree repo")?;
        let main_tip = {
            // Peel the main branch by name; the worktree's HEAD is on
            // `branch_name`, so we have to ask by ref rather than by HEAD.
            let main_ref = repo
                .find_branch("main", BranchType::Local)
                .context("find main branch")?;
            main_ref.into_reference().peel_to_commit()?
        };
        let branch_ref_obj = repo
            .find_branch(branch_name, BranchType::Local)
            .with_context(|| format!("find branch {branch_name}"))?
            .into_reference();
        let branch_tip = branch_ref_obj.peel_to_commit()?;
        if branch_tip.id() == main_tip.id() {
            // Branch is already at main's tip — nothing to rebase.
            return Ok(());
        }
        // Common ancestor = where the branch was originally created.
        let base = repo
            .merge_base(branch_tip.id(), main_tip.id())
            .context("find merge base")?;
        if base == main_tip.id() {
            // Branch already descends from main — fast-forward case, nothing
            // to do.
            return Ok(());
        }
        // IMPORTANT: build the branch annotated commit from the REFERENCE,
        // not from the bare OID. Libgit2 uses the ref name (carried by the
        // annotated commit) to know which branch to update on finish; if
        // we pass `find_annotated_commit(oid)` instead, finish updates
        // HEAD but leaves the branch ref pointing at the old tip.
        let branch_ac = repo.reference_to_annotated_commit(&branch_ref_obj)?;
        let upstream_ac = repo.find_annotated_commit(base)?;
        let onto_ac = repo.find_annotated_commit(main_tip.id())?;
        let mut opts = RebaseOptions::new();
        let mut rebase = repo
            .rebase(Some(&branch_ac), Some(&upstream_ac), Some(&onto_ac), Some(&mut opts))
            .context("rebase init")?;
        let sig = bureau_signature();
        loop {
            match rebase.next() {
                None => break,
                Some(Err(e)) => {
                    let _ = rebase.abort();
                    return Err(anyhow::anyhow!("rebase step: {e}"));
                }
                Some(Ok(_op)) => {
                    // After each pick, libgit2 has staged the patch into
                    // the worktree's index. If there are conflicts, the
                    // commit will fail; we abort and bail.
                    if repo.index()?.has_conflicts() {
                        let _ = rebase.abort();
                        return Err(anyhow::anyhow!(
                            "rebase of {branch_name} onto main hit conflicts"
                        ));
                    }
                    if let Err(e) = rebase.commit(None, &sig, None) {
                        let _ = rebase.abort();
                        return Err(anyhow::anyhow!("rebase commit: {e}"));
                    }
                }
            }
        }
        rebase.finish(Some(&sig)).context("rebase finish")?;
        Ok(())
    }

    /// Move `main` to point at `branch_name`'s tip. Pre-condition: the
    /// branch must already be a descendant of main (i.e., rebased).
    /// Errors if it isn't.
    pub fn fast_forward_main(&self, branch_name: &str) -> Result<()> {
        let repo = Repository::open(&self.root).context("open main repo")?;
        let main_ref = repo.find_branch("main", BranchType::Local)?;
        let main_tip = main_ref.into_reference().peel_to_commit()?;
        let branch_ref = repo
            .find_branch(branch_name, BranchType::Local)
            .with_context(|| format!("find branch {branch_name}"))?;
        let branch_tip = branch_ref.into_reference().peel_to_commit()?;
        if branch_tip.id() == main_tip.id() {
            return Ok(());
        }
        // Confirm branch descends from main.
        let base = repo.merge_base(main_tip.id(), branch_tip.id())?;
        if base != main_tip.id() {
            anyhow::bail!(
                "fast_forward_main: branch {branch_name} is not a descendant of main \
                 (base {base} != main tip {})",
                main_tip.id()
            );
        }
        // Update the main branch ref. We modify the ref directly rather
        // than going through HEAD because main isn't HEAD here — the
        // main workdir's HEAD is `main`, but `repo.head()` only matters
        // for ref name; what we're updating is the `main` branch itself.
        let mut main_branch = repo.find_branch("main", BranchType::Local)?;
        main_branch
            .get_mut()
            .set_target(branch_tip.id(), "ff-merge from task branch")?;
        // Check out the new tree into the main working directory so the
        // files match HEAD.
        let new_commit = repo.find_commit(branch_tip.id())?;
        let new_tree = new_commit.tree()?;
        let mut co = CheckoutBuilder::new();
        co.force();
        repo.checkout_tree(new_tree.as_object(), Some(&mut co))?;
        // Move HEAD to the updated main branch ref.
        repo.set_head("refs/heads/main")?;
        Ok(())
    }
}

/// Manages worktree allocation and cleanup. Land-time serialization is
/// achieved via `main_lock()` — callers hold it across rebase + ff-merge.
pub struct WorktreePool {
    pub workspace: Arc<Workspace>,
    pub scratch_root: PathBuf,
    /// Serializes the rebase + ff-merge cycle so that only one branch is
    /// landing at a time. Callers acquire this BEFORE rebasing onto main
    /// so that no other land can move main's HEAD between the rebase and
    /// the fast-forward.
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

    /// The serialization lock on landing. Callers hold this across the
    /// rebase + re-gate + ff-merge sequence so main only advances under
    /// exclusive access.
    pub fn main_lock(&self) -> &tokio::sync::Mutex<()> {
        &self.main_lock
    }

    /// Allocate a worktree branched from the current main HEAD. The
    /// branch is named `task/<task-id>`; the worktree lives at
    /// `<workdir>/.bureau/worktrees/<task-id>/`.
    pub async fn allocate(&self, task_id: Uuid) -> Result<Worktree> {
        let _guard = self.main_lock.lock().await;
        let id_str = task_id.to_string();
        let path = self.scratch_root.join(&id_str);
        if path.exists() {
            let _ = std::fs::remove_dir_all(&path);
        }
        let branch_name = format!("task/{id_str}");
        let repo = Repository::open(&self.workspace.root)?;
        if let Ok(mut b) = repo.find_branch(&branch_name, BranchType::Local) {
            let _ = b.delete();
        }
        let head_commit = repo.head()?.peel_to_commit()?;
        let branch = repo.branch(&branch_name, &head_commit, false)?;
        let branch_ref = branch.into_reference();
        let mut opts = WorktreeAddOptions::new();
        opts.reference(Some(&branch_ref));
        let _wt = repo
            .worktree(&id_str, &path, Some(&opts))
            .context("Repository::worktree")?;
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

    /// Stage everything in the worktree and commit on its branch. No-op
    /// if the tree matches HEAD. Used by the engine before landing so
    /// the model's edits are captured as a commit on the branch.
    pub fn commit_in_worktree(&self, wt: &Worktree, message: &str) -> Result<()> {
        commit_in_worktree(&wt.path, message)
    }

    /// Drop a worktree without merging — used on task failure.
    pub async fn abandon(&self, wt: Worktree) -> Result<()> {
        let _guard = self.main_lock.lock().await;
        self.cleanup(&wt);
        Ok(())
    }

    fn cleanup(&self, wt: &Worktree) {
        let _ = std::fs::remove_dir_all(&wt.path);
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

/// Stage everything in the worktree and commit on its branch.
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
    cfg.set_bool("commit.gpgsign", false)?;
    cfg.set_bool("tag.gpgsign", false)?;
    Ok(())
}

fn ensure_gitignore(root: &Path) -> Result<()> {
    const REQUIRED: &[&str] = &["/target/", "**/target/", "/.bureau/worktrees/"];
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
    async fn rebase_onto_main_when_branch_unchanged_is_noop() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = Workspace::init(tmp.path()).unwrap();
        let pool = WorktreePool::new(ws.clone()).unwrap();
        let wt = pool.allocate(Uuid::new_v4()).await.unwrap();
        ws.rebase_branch_onto_main(&wt.path, &wt.branch).unwrap();
    }

    #[tokio::test]
    async fn rebase_then_ff_merge_lands_non_conflicting_changes() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = Workspace::init(tmp.path()).unwrap();
        let pool = WorktreePool::new(ws.clone()).unwrap();
        // Branch A writes file_a.
        let wta = pool.allocate(Uuid::new_v4()).await.unwrap();
        fs::write(wta.path.join("file_a.txt"), "A\n").unwrap();
        pool.commit_in_worktree(&wta, "add a").unwrap();
        // Land A: rebase (no-op since main hasn't moved), then ff-merge.
        ws.rebase_branch_onto_main(&wta.path, &wta.branch).unwrap();
        ws.fast_forward_main(&wta.branch).unwrap();
        assert!(ws.root.join("file_a.txt").exists());
        pool.abandon(wta).await.unwrap();
        // Branch B was allocated from OLD main; needs rebase to land.
        let wtb = pool.allocate(Uuid::new_v4()).await.unwrap();
        fs::write(wtb.path.join("file_b.txt"), "B\n").unwrap();
        pool.commit_in_worktree(&wtb, "add b").unwrap();
        // Allocate ran from current main (which now has file_a) so the
        // rebase is also a no-op in this ordering. To get a non-trivial
        // rebase, write to main directly after allocate.
        fs::write(ws.root.join("file_c.txt"), "C\n").unwrap();
        ws.commit_main("add c on main").unwrap();
        ws.rebase_branch_onto_main(&wtb.path, &wtb.branch).unwrap();
        ws.fast_forward_main(&wtb.branch).unwrap();
        assert!(ws.root.join("file_a.txt").exists());
        assert!(ws.root.join("file_b.txt").exists());
        assert!(ws.root.join("file_c.txt").exists());
        // Branch B's worktree now reflects the rebased state (file_c is there too).
        assert!(wtb.path.join("file_c.txt").exists());
        pool.abandon(wtb).await.unwrap();
    }

    #[tokio::test]
    async fn rebase_with_conflict_returns_err_and_aborts() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = Workspace::init(tmp.path()).unwrap();
        let pool = WorktreePool::new(ws.clone()).unwrap();
        // Branch A writes contested.txt with one content; commit.
        let wta = pool.allocate(Uuid::new_v4()).await.unwrap();
        fs::write(wta.path.join("contested.txt"), "from A\n").unwrap();
        pool.commit_in_worktree(&wta, "A version").unwrap();
        // Meanwhile main writes contested.txt with DIFFERENT content.
        fs::write(ws.root.join("contested.txt"), "from main\n").unwrap();
        ws.commit_main("main version").unwrap();
        // Now rebase A onto main — should conflict.
        let err = ws
            .rebase_branch_onto_main(&wta.path, &wta.branch)
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("conflict") || msg.contains("rebase"), "got: {msg}");
        pool.abandon(wta).await.unwrap();
    }

    #[tokio::test]
    async fn fast_forward_main_rejects_non_descendant_branch() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = Workspace::init(tmp.path()).unwrap();
        let pool = WorktreePool::new(ws.clone()).unwrap();
        let wt = pool.allocate(Uuid::new_v4()).await.unwrap();
        fs::write(wt.path.join("a.txt"), "a\n").unwrap();
        pool.commit_in_worktree(&wt, "add a").unwrap();
        // Move main forward independently so the branch is no longer a
        // descendant.
        fs::write(ws.root.join("b.txt"), "b\n").unwrap();
        ws.commit_main("add b on main").unwrap();
        let err = ws.fast_forward_main(&wt.branch).unwrap_err();
        assert!(format!("{err:#}").contains("not a descendant"));
        pool.abandon(wt).await.unwrap();
    }
}
