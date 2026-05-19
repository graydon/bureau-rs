//! The decomposition graph.
//!
//! Each `Node` represents one abstraction with a public interface and hidden
//! internals. Nodes are arranged in two overlaid graphs:
//!
//! - A **parent → child tree** that determines on-disk module nesting. Every
//!   node has at most one parent; the root has none.
//! - A **dependency DAG**, where `A.deps` contains all nodes A depends on
//!   for compilation/visibility. A child of N is implicitly visible to N
//!   (filesystem nesting), but the dep graph also admits cross-subtree
//!   edges (e.g. unrelated nodes both depending on a shared utility).
//!
//! The orchestrator owns the graph; the model builds it up via `decompose`
//! calls. Each node passes through stages (spec, iface, tests, impl, debug,
//! opt) tracked per-node so the scheduler can run independent stages in
//! parallel and dep-order anything that needs to compile.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub Uuid);

impl NodeId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// Build the WHOLE node tree in one shot — runs once, on the root
    /// only, before any other stages. The architect produces the
    /// project's structure (crates, modules, dep edges) but no spec
    /// content; later per-node stages flesh out individual nodes.
    Architect,
    /// Author the prose spec for this node (markdown).
    Spec,
    /// Author the public interface (`public.rs`).
    Iface,
    /// Author tests against the interface (`tests.rs`).
    Tests,
    /// Implement the private side (`private.rs`) so tests pass.
    Impl,
    /// Targeted fixup if `Impl` didn't make tests pass on the first try.
    Debug,
}

impl Stage {
    pub const ALL: [Stage; 6] = [
        Stage::Architect,
        Stage::Spec,
        Stage::Iface,
        Stage::Tests,
        Stage::Impl,
        Stage::Debug,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Stage::Architect => "architect",
            Stage::Spec => "spec",
            Stage::Iface => "iface",
            Stage::Tests => "tests",
            Stage::Impl => "impl",
            Stage::Debug => "debug",
        }
    }
}

impl std::fmt::Display for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageState {
    NotStarted,
    InProgress,
    Done,
    Failed,
}

impl Default for StageState {
    fn default() -> Self {
        StageState::NotStarted
    }
}

impl StageState {
    pub fn is_done(self) -> bool {
        matches!(self, StageState::Done)
    }
    pub fn is_terminal(self) -> bool {
        matches!(self, StageState::Done | StageState::Failed)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeStages {
    /// Architect stage. Only meaningful on the ROOT node — runs once and
    /// produces the entire tree. On every other node it stays
    /// `NotStarted` forever (`stage_is_ready` enforces this).
    #[serde(default)]
    pub architect: StageState,
    pub spec: StageState,
    pub iface: StageState,
    pub tests: StageState,
    #[serde(rename = "impl")]
    pub impl_: StageState,
    pub debug: StageState,
}

impl NodeStages {
    pub fn get(&self, stage: Stage) -> StageState {
        match stage {
            Stage::Architect => self.architect,
            Stage::Spec => self.spec,
            Stage::Iface => self.iface,
            Stage::Tests => self.tests,
            Stage::Impl => self.impl_,
            Stage::Debug => self.debug,
        }
    }

    pub fn set(&mut self, stage: Stage, value: StageState) {
        match stage {
            Stage::Architect => self.architect = value,
            Stage::Spec => self.spec = value,
            Stage::Iface => self.iface = value,
            Stage::Tests => self.tests = value,
            Stage::Impl => self.impl_ = value,
            Stage::Debug => self.debug = value,
        }
    }

    /// Reset every per-node-content stage (everything past architect) to
    /// `NotStarted`. Called when a node gains a new dep edge — its
    /// iface/tests/impl assumptions may have changed.
    pub fn reset_post_architect(&mut self) {
        self.spec = StageState::NotStarted;
        self.iface = StageState::NotStarted;
        self.tests = StageState::NotStarted;
        self.impl_ = StageState::NotStarted;
        self.debug = StageState::NotStarted;
    }
}

/// Reset stage `from` on `node` to `NotStarted`, cascade-reset all later
/// stages on the same node to `NotStarted`, and clear in-memory content
/// slots that are *first-written* by any of the reset stages. The caller
/// is expected to re-render after this so that cleared slots become
/// placeholder files on disk again — otherwise the previously-authored
/// downstream files would sit on disk while their producing stage is
/// shown as `NotStarted`, causing the next run of the producing (or
/// earlier) stage to see stale content (the iface-restart bug).
///
/// "First-written" slots per stage:
/// - `Spec`     → `spec_public_md`, `spec_private_md`
/// - `Iface`    → `public_rs`, `private_rs` (Iface authors stubs)
/// - `Tests`    → `tests_rs`
/// - `Impl`/`Debug` → none (both overwrite `private_rs`, which is
///   first-written by `Iface`)
///
/// Returns the list of (stage, prior_state) entries that actually changed.
pub fn reset_stage_and_cascade(node: &mut Node, from: Stage) -> Vec<(Stage, StageState)> {
    let to_reset: Vec<Stage> = Stage::ALL
        .iter()
        .copied()
        .skip_while(|s| *s != from)
        .collect();
    let mut changed = Vec::new();
    for s in &to_reset {
        let prev = node.stages.get(*s);
        if prev != StageState::NotStarted {
            node.stages.set(*s, StageState::NotStarted);
            changed.push((*s, prev));
        }
    }
    for s in &to_reset {
        match s {
            Stage::Architect => {}
            Stage::Spec => {
                node.spec_public_md = None;
                node.spec_private_md = None;
            }
            Stage::Iface => {
                node.public_rs = None;
                node.private_rs = None;
            }
            Stage::Tests => {
                node.tests_rs = None;
            }
            Stage::Impl | Stage::Debug => {}
        }
    }
    changed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    /// Module name (`snake_case`). Must be a valid Rust identifier.
    pub name: String,
    /// Parent in the filesystem tree. `None` only for the workspace root.
    pub parent: Option<NodeId>,
    /// Dep edges: nodes this node depends on for visibility/compile order.
    /// MUST not contain cycles when combined with everyone else's deps.
    pub deps: Vec<NodeId>,
    /// True if this node is a Cargo crate boundary. The root is always a
    /// crate (or a workspace). Otherwise, this node lives as a module of
    /// its parent's crate.
    pub crate_boundary: bool,
    /// Brief description of the abstraction. Used in prompt context.
    pub description: String,

    /// Authored content. `None` until the corresponding stage produces it.
    /// The spec is split into a public part (read by dependents and
    /// children — describes what the node does and exposes) and a
    /// private part (the writer's own implementation notes / rationale,
    /// not surfaced to other nodes).
    ///
    /// **Not persisted to JSON.** The authored content lives in the
    /// rendered files on disk (`src/<path>/public.rs`, `spec/<path>/public.md`,
    /// etc.) which are the source of truth and are committed to git.
    /// `graph::load` reads those files and populates these fields after
    /// loading the topology. Storing the content in both places (JSON +
    /// rendered files) doubled disk usage and bloated every commit;
    /// the `#[serde(skip_serializing)]` keeps JSON to topology + state.
    /// `default` lets old JSON files (with content) still deserialize —
    /// the in-memory content gets overwritten by what's on disk anyway.
    #[serde(default, skip_serializing)]
    pub spec_public_md: Option<String>,
    #[serde(default, skip_serializing)]
    pub spec_private_md: Option<String>,
    #[serde(default, skip_serializing)]
    pub public_rs: Option<String>,
    #[serde(default, skip_serializing)]
    pub private_rs: Option<String>,
    #[serde(default, skip_serializing)]
    pub tests_rs: Option<String>,

    pub stages: NodeStages,

    /// External crates.io deps declared at architect time. Only meaningful
    /// on the ROOT node. Renderer writes them into the workspace's
    /// `[workspace.dependencies]` so any member crate can pull them in
    /// via `name.workspace = true`. Empty on non-root nodes.
    #[serde(default)]
    pub external_crate_deps: Vec<ExternalCrateDep>,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// crates.io dependency, declared by the architect, rendered into
/// Cargo.toml at the workspace root. Member crates inherit via
/// `name.workspace = true`. Mirrors the architect's submission shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalCrateDep {
    pub name: String,
    /// Version requirement. If `None`, `"*"` is used at render time —
    /// good enough for cargo to resolve a recent crates.io release.
    #[serde(default)]
    pub version: Option<String>,
    /// Features to enable. Empty means default-only.
    #[serde(default)]
    pub features: Vec<String>,
    /// Operator note about why this dep was added.
    #[serde(default)]
    pub reason: String,
}

impl Node {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: NodeId::new(),
            name: name.into(),
            parent: None,
            deps: Vec::new(),
            crate_boundary: false,
            description: description.into(),
            spec_public_md: None,
            spec_private_md: None,
            public_rs: None,
            private_rs: None,
            tests_rs: None,
            stages: NodeStages::default(),
            external_crate_deps: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("parent node not found: {0}")]
    ParentNotFound(NodeId),
    #[error("node not found: {0}")]
    NodeNotFound(NodeId),
    #[error("self dependency on {0}")]
    SelfDep(NodeId),
    #[error("adding dep {from} → {to} would create a cycle")]
    WouldCycle { from: NodeId, to: NodeId },
    #[error("node name '{0}' already exists at sibling level under parent {1:?}")]
    DuplicateSiblingName(String, Option<NodeId>),
    #[error("invalid module name '{0}': must be a valid Rust identifier")]
    InvalidName(String),
    #[error("graph already has a root")]
    RootAlreadySet,
    #[error("graph has no root yet")]
    NoRoot,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeGraph {
    pub root: Option<NodeId>,
    pub nodes: IndexMap<NodeId, Node>,
}

impl NodeGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Find a node by exact name. If multiple exist with the same name (in
    /// different subtrees), returns the first one inserted.
    pub fn find_by_name(&self, name: &str) -> Option<&Node> {
        self.nodes.values().find(|n| n.name == name)
    }

    /// Insert the workspace root. Fails if a root already exists.
    pub fn insert_root(&mut self, mut node: Node) -> Result<NodeId, GraphError> {
        if self.root.is_some() {
            return Err(GraphError::RootAlreadySet);
        }
        validate_name(&node.name)?;
        node.parent = None;
        node.crate_boundary = true; // root is always a crate or workspace
        let id = node.id;
        self.root = Some(id);
        self.nodes.insert(id, node);
        Ok(id)
    }

    /// Add a child node under `parent`. Names must be unique among siblings
    /// of the same parent.
    pub fn add_child(
        &mut self,
        parent: NodeId,
        mut node: Node,
    ) -> Result<NodeId, GraphError> {
        if !self.nodes.contains_key(&parent) {
            return Err(GraphError::ParentNotFound(parent));
        }
        validate_name(&node.name)?;
        // Sibling-name uniqueness
        for n in self.nodes.values() {
            if n.parent == Some(parent) && n.name == node.name {
                return Err(GraphError::DuplicateSiblingName(node.name, Some(parent)));
            }
        }
        node.parent = Some(parent);
        let id = node.id;
        self.nodes.insert(id, node);
        Ok(id)
    }

    /// Add a dep edge `from → to`. Both must exist; cycles are rejected
    /// at BOTH the node level and the crate level. (A workspace where
    /// crate-A's nodes depend on crate-B's nodes and vice versa is a
    /// cargo-rejected crate cycle, even if the underlying node-level
    /// graph is acyclic — so we check both.)
    pub fn add_dep(&mut self, from: NodeId, to: NodeId) -> Result<(), GraphError> {
        if !self.nodes.contains_key(&from) {
            return Err(GraphError::NodeNotFound(from));
        }
        if !self.nodes.contains_key(&to) {
            return Err(GraphError::NodeNotFound(to));
        }
        if from == to {
            return Err(GraphError::SelfDep(from));
        }
        // Node-level cycle check.
        if self.dep_reaches(to, from) {
            return Err(GraphError::WouldCycle { from, to });
        }
        // Crate-level cycle check. If `from` and `to` are in different
        // crates, adding from→to creates a crate-level edge
        // crate(from)→crate(to). That's a cycle iff crate(to) already
        // reaches crate(from) via existing crate-level edges.
        let from_crate = self.containing_crate(from);
        let to_crate = self.containing_crate(to);
        if from_crate != to_crate
            && self.crate_reaches(to_crate, from_crate, Some((from, to)))
        {
            return Err(GraphError::WouldCycle { from, to });
        }
        let n = self.nodes.get_mut(&from).expect("checked");
        if !n.deps.contains(&to) {
            n.deps.push(to);
            n.updated_at = Utc::now();
        }
        Ok(())
    }

    /// The nearest crate-boundary ancestor of `node`, inclusive of the
    /// node itself. The root is always a crate boundary, so this always
    /// returns Some when the graph has a root.
    pub fn containing_crate(&self, node_id: NodeId) -> NodeId {
        let mut cur = Some(node_id);
        while let Some(id) = cur {
            let n = match self.nodes.get(&id) {
                Some(n) => n,
                None => break,
            };
            if n.crate_boundary {
                return id;
            }
            cur = n.parent;
        }
        // Fallback: the node itself, even though it didn't hit a
        // boundary. Shouldn't happen on a well-formed graph (root is
        // always a boundary).
        node_id
    }

    /// Does `start_crate` reach `target_crate` in the crate-level dep
    /// graph? `extra` optionally adds a hypothetical (from, to) dep edge
    /// to the search — used by `add_dep` to test "would this proposed
    /// edge create a cycle?".
    fn crate_reaches(
        &self,
        start_crate: NodeId,
        target_crate: NodeId,
        extra: Option<(NodeId, NodeId)>,
    ) -> bool {
        if start_crate == target_crate {
            return true;
        }
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut stack: Vec<NodeId> = vec![start_crate];
        while let Some(c) = stack.pop() {
            if !visited.insert(c) {
                continue;
            }
            if c == target_crate {
                return true;
            }
            // Collect crate-level edges out of `c`: for every node in c,
            // every dep that points outside c is a crate-level edge.
            for n in self.nodes.values() {
                if self.containing_crate(n.id) != c {
                    continue;
                }
                for d in &n.deps {
                    let dc = self.containing_crate(*d);
                    if dc != c {
                        stack.push(dc);
                    }
                }
            }
            // Hypothetical extra edge.
            if let Some((from, to)) = extra {
                if self.containing_crate(from) == c {
                    let to_c = self.containing_crate(to);
                    if to_c != c {
                        stack.push(to_c);
                    }
                }
            }
        }
        false
    }

    /// Does `start` reach `target` via dep edges (transitive closure)?
    pub fn dep_reaches(&self, start: NodeId, target: NodeId) -> bool {
        if start == target {
            return true;
        }
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut stack: Vec<NodeId> = vec![start];
        while let Some(n) = stack.pop() {
            if !visited.insert(n) {
                continue;
            }
            let Some(node) = self.nodes.get(&n) else {
                continue;
            };
            for d in &node.deps {
                if *d == target {
                    return true;
                }
                stack.push(*d);
            }
        }
        false
    }

    /// Set of nodes that transitively depend on `target` (i.e. nodes
    /// for which `dep_reaches(node, target)` is true). Includes
    /// `target` itself. Used by the engine's cascade-reset path: when
    /// a node gains a new dep, every node that ever depended on it
    /// needs its post-architect stages reset.
    pub fn reverse_dep_closure(&self, target: NodeId) -> HashSet<NodeId> {
        let mut out = HashSet::new();
        out.insert(target);
        // Build reverse edges once.
        let mut rev: std::collections::HashMap<NodeId, Vec<NodeId>> =
            std::collections::HashMap::new();
        for n in self.nodes.values() {
            for d in &n.deps {
                rev.entry(*d).or_default().push(n.id);
            }
        }
        let mut frontier = vec![target];
        while let Some(n) = frontier.pop() {
            if let Some(parents) = rev.get(&n) {
                for p in parents {
                    if out.insert(*p) {
                        frontier.push(*p);
                    }
                }
            }
        }
        out
    }

    /// Path of names from root to `id` (inclusive). For `root → frontend → router`
    /// returns `["root", "frontend", "router"]`. Returns `None` if `id` doesn't
    /// exist or its parent chain is broken.
    pub fn name_path(&self, id: NodeId) -> Option<Vec<&str>> {
        let mut chain: Vec<&str> = Vec::new();
        let mut current = Some(id);
        while let Some(c) = current {
            let n = self.nodes.get(&c)?;
            chain.push(n.name.as_str());
            current = n.parent;
        }
        chain.reverse();
        Some(chain)
    }

    /// Module path as a `crate::a::b::c` string. The root's name is omitted
    /// (Rust uses `crate::` for the crate root). Returns None on error.
    pub fn module_path_string(&self, id: NodeId) -> Option<String> {
        let path = self.name_path(id)?;
        if path.is_empty() {
            return None;
        }
        // Skip the root's name; everything below it is a module under `crate::`.
        let modules = &path[1..];
        if modules.is_empty() {
            Some("crate".to_string())
        } else {
            Some(format!("crate::{}", modules.join("::")))
        }
    }

    /// Topological order: deps come before dependents. Stable across runs:
    /// among nodes with equal in-degree, the original insertion order
    /// (from the graph's `IndexMap`) is preserved. Returns `None` only if
    /// the graph already has a cycle (which our `add_dep` prevents).
    pub fn topo_order(&self) -> Option<Vec<NodeId>> {
        // Standard Kahn, but using the ordered `IndexMap` for both the
        // in-degree map and the initial queue so the result is
        // deterministic.
        let ids: Vec<NodeId> = self.nodes.keys().copied().collect();
        let mut in_degree: HashMap<NodeId, usize> = self
            .nodes
            .iter()
            .map(|(id, n)| (*id, n.deps.len()))
            .collect();
        let mut queue: VecDeque<NodeId> = ids
            .iter()
            .copied()
            .filter(|id| in_degree.get(id).copied().unwrap_or(0) == 0)
            .collect();
        let mut order: Vec<NodeId> = Vec::with_capacity(self.nodes.len());
        while let Some(id) = queue.pop_front() {
            order.push(id);
            // For every node that lists `id` as a dep (in stable insertion
            // order), decrement its in-degree.
            for n in self.nodes.values() {
                if n.deps.contains(&id) {
                    let entry = in_degree.get_mut(&n.id).unwrap();
                    *entry -= 1;
                    if *entry == 0 {
                        queue.push_back(n.id);
                    }
                }
            }
        }
        if order.len() == self.nodes.len() {
            Some(order)
        } else {
            None
        }
    }

    /// Direct children of `parent` in stable insertion order.
    pub fn children_of(&self, parent: NodeId) -> Vec<&Node> {
        self.nodes
            .values()
            .filter(|n| n.parent == Some(parent))
            .collect()
    }

    /// All ancestors of `id`, root last (or first if `include_self=true`).
    pub fn ancestors(&self, id: NodeId, include_self: bool) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut current = if include_self {
            Some(id)
        } else {
            self.nodes.get(&id).and_then(|n| n.parent)
        };
        while let Some(c) = current {
            out.push(c);
            current = self.nodes.get(&c).and_then(|n| n.parent);
        }
        out
    }

    /// All transitive deps of `id` (the closure under `deps`). Does NOT
    /// include `id` itself.
    pub fn transitive_deps(&self, id: NodeId) -> Vec<NodeId> {
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut stack: Vec<NodeId> = self
            .nodes
            .get(&id)
            .map(|n| n.deps.clone())
            .unwrap_or_default();
        let mut out = Vec::new();
        while let Some(n) = stack.pop() {
            if !visited.insert(n) {
                continue;
            }
            out.push(n);
            if let Some(node) = self.nodes.get(&n) {
                for d in &node.deps {
                    stack.push(*d);
                }
            }
        }
        out
    }
}

fn validate_name(name: &str) -> Result<(), GraphError> {
    if name.is_empty() {
        return Err(GraphError::InvalidName(name.to_string()));
    }
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return Err(GraphError::InvalidName(name.to_string())),
    }
    for c in chars {
        if !(c.is_ascii_alphanumeric() || c == '_') {
            return Err(GraphError::InvalidName(name.to_string()));
        }
    }
    // Reject Rust keywords that are common naming pitfalls.
    if matches!(
        name,
        "crate"
            | "self"
            | "super"
            | "Self"
            | "use"
            | "mod"
            | "pub"
            | "fn"
            | "struct"
            | "enum"
            | "trait"
            | "impl"
            | "type"
            | "const"
            | "static"
            | "let"
            | "match"
            | "for"
            | "in"
            | "if"
            | "else"
            | "while"
            | "loop"
            | "break"
            | "continue"
            | "return"
            | "as"
            | "where"
            | "ref"
            | "move"
            | "mut"
            | "true"
            | "false"
            | "async"
            | "await"
            | "dyn"
    ) {
        return Err(GraphError::InvalidName(name.to_string()));
    }
    Ok(())
}

// --------------------------------------------------------------------------
// On-disk persistence.
// --------------------------------------------------------------------------
//
// The graph lives in `<workdir>/.bureau/`:
//   - `graph.json`       — topology pointer: root id + list of node files
//   - `nodes/<name>.json` — one file per node (full Node, pretty-printed)
//
// This makes the worktree's branch the source of truth for graph state:
// concurrent worktrees each mutate their own copy; landing rebases + ff-
// merges those changes onto main. No in-memory shared state to drift.

const BUREAU_DIR: &str = ".bureau";
const NODES_SUBDIR: &str = "nodes";
const GRAPH_FILE: &str = "graph.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GraphIndex {
    root: Option<NodeId>,
    /// Filenames inside `.bureau/nodes/` — `<name>.json` for each node.
    /// Order is the canonical insertion order so reads reconstruct it.
    node_files: Vec<String>,
}

fn bureau_dir(workdir: &Path) -> PathBuf {
    workdir.join(BUREAU_DIR)
}

fn nodes_dir(workdir: &Path) -> PathBuf {
    bureau_dir(workdir).join(NODES_SUBDIR)
}

fn index_path(workdir: &Path) -> PathBuf {
    bureau_dir(workdir).join(GRAPH_FILE)
}

fn node_file_path(workdir: &Path, name: &str) -> PathBuf {
    nodes_dir(workdir).join(format!("{name}.json"))
}

/// Load the graph's TOPOLOGY only (no slot content). Cheaper than `load`
/// when the caller only needs the node tree, dep edges, and stage
/// states — e.g. for `/api/state` polling, or for `pick_next_ready`
/// scheduling decisions. The Node content fields (`public_rs`, etc.)
/// stay at `None` — they're not in the JSON anymore so deserialization
/// produces None either way.
pub fn load_topology(workdir: &Path) -> Result<NodeGraph> {
    let idx_path = index_path(workdir);
    if !idx_path.exists() {
        return Ok(NodeGraph::new());
    }
    let idx_raw = std::fs::read_to_string(&idx_path)
        .with_context(|| format!("reading {}", idx_path.display()))?;
    let idx: GraphIndex = serde_json::from_str(&idx_raw)
        .with_context(|| format!("parsing {}", idx_path.display()))?;
    let mut g = NodeGraph::new();
    g.root = idx.root;
    for fname in &idx.node_files {
        let p = nodes_dir(workdir).join(fname);
        let raw = std::fs::read_to_string(&p)
            .with_context(|| format!("reading {}", p.display()))?;
        let n: Node = serde_json::from_str(&raw)
            .with_context(|| format!("parsing {}", p.display()))?;
        g.nodes.insert(n.id, n);
    }
    Ok(g)
}

/// Load the graph from disk. Returns an empty graph if `.bureau/graph.json`
/// doesn't exist (fresh workdir).
///
/// Loads topology + stage state from `.bureau/{graph,nodes/<name>}.json`,
/// then reads the rendered slot files (`src/<path>/public.rs`,
/// `spec/<path>/public.md`, etc.) to populate the in-memory content
/// fields. The rendered files are the source of truth for content; the
/// JSON only carries topology and stage state.
///
/// `layout` is needed because it determines the on-disk path of each
/// node's source directory (single-crate vs. workspace).
pub fn load(workdir: &Path, layout: crate::render::Layout) -> Result<NodeGraph> {
    let idx_path = index_path(workdir);
    if !idx_path.exists() {
        return Ok(NodeGraph::new());
    }
    let idx_raw = std::fs::read_to_string(&idx_path)
        .with_context(|| format!("reading {}", idx_path.display()))?;
    let idx: GraphIndex = serde_json::from_str(&idx_raw)
        .with_context(|| format!("parsing {}", idx_path.display()))?;
    let mut g = NodeGraph::new();
    g.root = idx.root;
    for fname in &idx.node_files {
        let p = nodes_dir(workdir).join(fname);
        let raw = std::fs::read_to_string(&p)
            .with_context(|| format!("reading {}", p.display()))?;
        // Old JSONs may carry content fields; `skip_serializing` only
        // affects writes. Deserialization works either way; we'll
        // overwrite the fields from disk below anyway.
        let mut n: Node = serde_json::from_str(&raw)
            .with_context(|| format!("parsing {}", p.display()))?;
        // Clear any stale content from the JSON so the on-disk-derived
        // values are authoritative.
        n.spec_public_md = None;
        n.spec_private_md = None;
        n.public_rs = None;
        n.private_rs = None;
        n.tests_rs = None;
        g.nodes.insert(n.id, n);
    }
    // Now populate content from rendered files. Done after all nodes
    // are inserted because path computation walks the parent chain.
    let ids: Vec<NodeId> = g.nodes.keys().copied().collect();
    for id in ids {
        populate_content_from_disk(workdir, &mut g, id, layout)?;
    }
    Ok(g)
}

/// Read the rendered slot files for one node and stuff their content
/// (when not a placeholder) into the in-memory `Node`. Called from
/// `load` after topology has been reconstructed.
fn populate_content_from_disk(
    workdir: &Path,
    graph: &mut NodeGraph,
    node_id: NodeId,
    layout: crate::render::Layout,
) -> Result<()> {
    // Compute paths against the immutable graph view, then mutate.
    let (src_dir, spec_dir) = {
        let Some(node) = graph.get(node_id) else {
            return Ok(());
        };
        (
            workdir.join(crate::render::node_src_dir(graph, node, layout)),
            workdir.join(crate::render::node_spec_dir(graph, node)),
        )
    };
    let public_rs = read_authored(
        &src_dir.join("public.rs"),
        crate::placeholders::is_placeholder_public_rs,
    );
    let private_rs = read_authored(
        &src_dir.join("private.rs"),
        crate::placeholders::is_placeholder_private_rs,
    );
    let tests_rs = read_authored(
        &src_dir.join("tests.rs"),
        crate::placeholders::is_placeholder_tests_rs,
    );
    let spec_public_md = read_authored(
        &spec_dir.join("public.md"),
        crate::placeholders::is_placeholder_public_md,
    );
    // private.md is only written when authored; absence = None.
    let spec_private_md = std::fs::read_to_string(spec_dir.join("private.md")).ok();
    if let Some(node) = graph.get_mut(node_id) {
        node.public_rs = public_rs;
        node.private_rs = private_rs;
        node.tests_rs = tests_rs;
        node.spec_public_md = spec_public_md;
        node.spec_private_md = spec_private_md;
    }
    Ok(())
}

/// Read a slot file. Returns `None` if the file is missing OR the
/// content is the framework's placeholder (which means "not yet
/// authored"). Returns `Some(content)` only when the model has
/// actually written something.
fn read_authored(path: &Path, is_placeholder: fn(&str) -> bool) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    if is_placeholder(&content) {
        None
    } else {
        Some(content)
    }
}

/// Save the graph to disk. Writes one JSON file per node into
/// `.bureau/nodes/<name>.json` plus the topology index at
/// `.bureau/graph.json`. Removes any stale per-node files (renamed or
/// deleted nodes).
///
/// All file writes go through `write_atomic` (write to `<path>.tmp`,
/// then rename). Without this, concurrent readers (`/api/state` and
/// `/api/graph` load_topology) can race with an in-flight write and
/// see a truncated / empty / partial file — load then unwraps to a
/// default empty NodeGraph and the UI flips to "Graph not yet
/// bootstrapped" mid-run.
pub fn save(workdir: &Path, g: &NodeGraph) -> Result<()> {
    let nodes_d = nodes_dir(workdir);
    std::fs::create_dir_all(&nodes_d)
        .with_context(|| format!("creating {}", nodes_d.display()))?;
    let mut node_files: Vec<String> = Vec::with_capacity(g.len());
    let mut keep: HashSet<String> = HashSet::new();
    for n in g.iter() {
        let fname = format!("{}.json", n.name);
        let p = node_file_path(workdir, &n.name);
        let raw = serde_json::to_string_pretty(n)
            .with_context(|| format!("serializing node {}", n.name))?;
        write_atomic(&p, raw.as_bytes())
            .with_context(|| format!("writing {}", p.display()))?;
        node_files.push(fname.clone());
        keep.insert(fname);
    }
    // Prune stale per-node files left over from renames or deletions.
    if let Ok(rd) = std::fs::read_dir(&nodes_d) {
        for entry in rd.flatten() {
            let name = entry.file_name();
            let s = name.to_string_lossy();
            if s.ends_with(".json") && !keep.contains(s.as_ref()) {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
    let idx = GraphIndex {
        root: g.root,
        node_files,
    };
    let idx_raw = serde_json::to_string_pretty(&idx).context("serializing graph index")?;
    let idx_path = index_path(workdir);
    write_atomic(&idx_path, idx_raw.as_bytes())
        .with_context(|| format!("writing {}", idx_path.display()))?;
    Ok(())
}

/// Write `contents` to `path` so that concurrent readers always see
/// either the OLD contents or the FULL NEW contents — never a partial
/// write. Implemented as write-to-`<path>.tmp` followed by atomic
/// rename onto `path`. On most filesystems (ext4, apfs, ntfs, tmpfs)
/// `rename` is atomic across an existing target; this gives us the
/// snapshot semantics that callers like `/api/state` rely on.
fn write_atomic(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    // Tmp filename in the same directory as the target so the rename
    // is intra-filesystem (cross-fs rename isn't atomic; it falls
    // back to copy-then-unlink which has the same race we're fixing).
    let mut tmp = path.to_path_buf();
    let fname = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    // pid + thread + nanos buys uniqueness if many writers fire on
    // the same path simultaneously; without it two writers can
    // clobber each other's .tmp file before either renames.
    let tag = format!(
        ".{}.{}.tmp",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    );
    tmp.set_file_name(format!("{fname}{tag}"));
    std::fs::write(&tmp, contents)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::Layout;

    fn node(name: &str) -> Node {
        Node::new(name, format!("{name} description"))
    }

    #[test]
    fn empty_graph_has_no_root() {
        let g = NodeGraph::new();
        assert!(g.is_empty());
        assert_eq!(g.root, None);
    }

    #[test]
    fn insert_root_succeeds_once() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("app")).unwrap();
        assert_eq!(g.root, Some(root));
        assert_eq!(g.len(), 1);
        let r = g.get(root).unwrap();
        assert_eq!(r.name, "app");
        assert!(r.crate_boundary, "root is always a crate boundary");
    }

    #[test]
    fn insert_root_twice_fails() {
        let mut g = NodeGraph::new();
        g.insert_root(node("app")).unwrap();
        let err = g.insert_root(node("other")).unwrap_err();
        assert!(matches!(err, GraphError::RootAlreadySet));
    }

    #[test]
    fn add_child_under_root() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("app")).unwrap();
        let c1 = g.add_child(root, node("frontend")).unwrap();
        let c2 = g.add_child(root, node("backend")).unwrap();
        assert_eq!(g.len(), 3);
        assert_eq!(g.get(c1).unwrap().parent, Some(root));
        assert_eq!(g.get(c2).unwrap().parent, Some(root));
    }

    #[test]
    fn add_child_to_missing_parent_fails() {
        let mut g = NodeGraph::new();
        let bogus = NodeId::new();
        let err = g.add_child(bogus, node("x")).unwrap_err();
        assert!(matches!(err, GraphError::ParentNotFound(_)));
    }

    #[test]
    fn duplicate_sibling_name_rejected() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("app")).unwrap();
        g.add_child(root, node("frontend")).unwrap();
        let err = g.add_child(root, node("frontend")).unwrap_err();
        assert!(matches!(err, GraphError::DuplicateSiblingName(_, _)));
    }

    #[test]
    fn same_name_in_different_subtrees_is_allowed() {
        // Could be undesirable in practice, but the graph itself permits
        // it — uniqueness is only enforced among siblings of the same
        // parent.
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("app")).unwrap();
        let a = g.add_child(root, node("a")).unwrap();
        let b = g.add_child(root, node("b")).unwrap();
        g.add_child(a, node("util")).unwrap();
        g.add_child(b, node("util")).unwrap();
        assert_eq!(g.len(), 5);
    }

    #[test]
    fn invalid_names_rejected() {
        let mut g = NodeGraph::new();
        for bad in ["", "1foo", "with-dash", "with space", "self", "crate", "mod"] {
            let err = g.insert_root(node(bad)).unwrap_err();
            assert!(matches!(err, GraphError::InvalidName(_)), "{bad}");
            // After insert_root fails, the graph is still empty, so we can
            // try the next one.
            assert!(g.is_empty());
        }
    }

    #[test]
    fn add_dep_basic() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("app")).unwrap();
        let a = g.add_child(root, node("a")).unwrap();
        let b = g.add_child(root, node("b")).unwrap();
        g.add_dep(a, b).unwrap();
        assert_eq!(g.get(a).unwrap().deps, vec![b]);
    }

    #[test]
    fn self_dep_rejected() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("app")).unwrap();
        let err = g.add_dep(root, root).unwrap_err();
        assert!(matches!(err, GraphError::SelfDep(_)));
    }

    #[test]
    fn cycle_rejected_direct() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("app")).unwrap();
        let a = g.add_child(root, node("a")).unwrap();
        let b = g.add_child(root, node("b")).unwrap();
        g.add_dep(a, b).unwrap();
        let err = g.add_dep(b, a).unwrap_err();
        assert!(matches!(err, GraphError::WouldCycle { .. }));
    }

    #[test]
    fn cycle_rejected_transitive() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("app")).unwrap();
        let a = g.add_child(root, node("a")).unwrap();
        let b = g.add_child(root, node("b")).unwrap();
        let c = g.add_child(root, node("c")).unwrap();
        g.add_dep(a, b).unwrap();
        g.add_dep(b, c).unwrap();
        // c → a would close the cycle a → b → c → a
        let err = g.add_dep(c, a).unwrap_err();
        assert!(matches!(err, GraphError::WouldCycle { .. }));
    }

    #[test]
    fn duplicate_dep_is_idempotent() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("app")).unwrap();
        let a = g.add_child(root, node("a")).unwrap();
        let b = g.add_child(root, node("b")).unwrap();
        g.add_dep(a, b).unwrap();
        g.add_dep(a, b).unwrap();
        assert_eq!(g.get(a).unwrap().deps, vec![b]);
    }

    #[test]
    fn dep_reaches() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("app")).unwrap();
        let a = g.add_child(root, node("a")).unwrap();
        let b = g.add_child(root, node("b")).unwrap();
        let c = g.add_child(root, node("c")).unwrap();
        g.add_dep(a, b).unwrap();
        g.add_dep(b, c).unwrap();
        assert!(g.dep_reaches(a, b));
        assert!(g.dep_reaches(a, c));
        assert!(g.dep_reaches(b, c));
        assert!(!g.dep_reaches(c, a));
        assert!(!g.dep_reaches(b, a));
        assert!(g.dep_reaches(a, a)); // reflexive on identity
    }

    #[test]
    fn add_dep_rejects_cross_crate_cycle() {
        // Three sibling crates X, Y, Z. Children x1 ∈ X, y1 ∈ Y, z1 ∈ Z.
        // Build x1→y1 and y1→z1 — fine. Now z1→x1 should be rejected:
        // crate-level cycle X→Y→Z→X even though node-level deps are
        // acyclic.
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("ws")).unwrap();
        let mk_crate = |g: &mut NodeGraph, name: &str| {
            let mut n = node(name);
            n.crate_boundary = true;
            g.add_child(root, n).unwrap()
        };
        let cx = mk_crate(&mut g, "X");
        let cy = mk_crate(&mut g, "Y");
        let cz = mk_crate(&mut g, "Z");
        let x1 = g.add_child(cx, node("x1")).unwrap();
        let y1 = g.add_child(cy, node("y1")).unwrap();
        let z1 = g.add_child(cz, node("z1")).unwrap();
        g.add_dep(x1, y1).unwrap(); // X → Y
        g.add_dep(y1, z1).unwrap(); // Y → Z
        // Z → X would close the cycle.
        let err = g.add_dep(z1, x1).unwrap_err();
        assert!(matches!(err, GraphError::WouldCycle { .. }));
        // Verify the rejected edge wasn't applied.
        assert!(!g.get(z1).unwrap().deps.contains(&x1));
    }

    #[test]
    fn add_dep_allows_intra_crate_dep_that_would_be_cross_crate_cycle() {
        // Within ONE crate, dep_reaches handles cycles already (existing
        // test); adding a sibling dep where both nodes share a crate
        // doesn't trigger the crate-level check.
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("ws")).unwrap();
        let a = g.add_child(root, node("a")).unwrap();
        let b = g.add_child(root, node("b")).unwrap();
        // Both inside root's crate (root is the only crate boundary).
        g.add_dep(a, b).unwrap();
        // Cycle would be at node level, caught by dep_reaches.
        let err = g.add_dep(b, a).unwrap_err();
        assert!(matches!(err, GraphError::WouldCycle { .. }));
    }

    #[test]
    fn name_path_simple() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("app")).unwrap();
        let f = g.add_child(root, node("frontend")).unwrap();
        let r = g.add_child(f, node("router")).unwrap();
        assert_eq!(g.name_path(root).unwrap(), vec!["app"]);
        assert_eq!(g.name_path(f).unwrap(), vec!["app", "frontend"]);
        assert_eq!(g.name_path(r).unwrap(), vec!["app", "frontend", "router"]);
    }

    #[test]
    fn module_path_string() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("app")).unwrap();
        let f = g.add_child(root, node("frontend")).unwrap();
        let r = g.add_child(f, node("router")).unwrap();
        assert_eq!(g.module_path_string(root).unwrap(), "crate");
        assert_eq!(g.module_path_string(f).unwrap(), "crate::frontend");
        assert_eq!(
            g.module_path_string(r).unwrap(),
            "crate::frontend::router"
        );
    }

    #[test]
    fn topo_order_respects_deps() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("app")).unwrap();
        let a = g.add_child(root, node("a")).unwrap();
        let b = g.add_child(root, node("b")).unwrap();
        let c = g.add_child(root, node("c")).unwrap();
        // a depends on b; b depends on c
        g.add_dep(a, b).unwrap();
        g.add_dep(b, c).unwrap();
        let order = g.topo_order().unwrap();
        let pos = |id| order.iter().position(|x| *x == id).unwrap();
        // c must come before b which must come before a
        assert!(pos(c) < pos(b));
        assert!(pos(b) < pos(a));
    }

    #[test]
    fn topo_order_independent_nodes_appear() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("app")).unwrap();
        let _a = g.add_child(root, node("a")).unwrap();
        let _b = g.add_child(root, node("b")).unwrap();
        let _c = g.add_child(root, node("c")).unwrap();
        let order = g.topo_order().unwrap();
        assert_eq!(order.len(), 4); // root + 3 children
    }

    #[test]
    fn ancestors_walks_up() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("app")).unwrap();
        let f = g.add_child(root, node("frontend")).unwrap();
        let r = g.add_child(f, node("router")).unwrap();
        assert_eq!(g.ancestors(r, false), vec![f, root]);
        assert_eq!(g.ancestors(r, true), vec![r, f, root]);
        assert_eq!(g.ancestors(root, false), Vec::<NodeId>::new());
    }

    #[test]
    fn transitive_deps() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("app")).unwrap();
        let a = g.add_child(root, node("a")).unwrap();
        let b = g.add_child(root, node("b")).unwrap();
        let c = g.add_child(root, node("c")).unwrap();
        let d = g.add_child(root, node("d")).unwrap();
        // a → b → c, a → d (so a's transitive deps are {b, c, d})
        g.add_dep(a, b).unwrap();
        g.add_dep(a, d).unwrap();
        g.add_dep(b, c).unwrap();
        let mut td = g.transitive_deps(a);
        td.sort_by_key(|n| g.get(*n).unwrap().name.clone());
        let names: Vec<&str> = td.iter().map(|id| g.get(*id).unwrap().name.as_str()).collect();
        assert_eq!(names, vec!["b", "c", "d"]);
    }

    #[test]
    fn find_by_name() {
        let mut g = NodeGraph::new();
        let _root = g.insert_root(node("app")).unwrap();
        assert_eq!(g.find_by_name("app").unwrap().name, "app");
        assert!(g.find_by_name("missing").is_none());
    }

    #[test]
    fn stage_state_default_is_not_started() {
        let s = NodeStages::default();
        assert_eq!(s.spec, StageState::NotStarted);
        assert_eq!(s.iface, StageState::NotStarted);
    }

    #[test]
    fn stage_state_get_set_round_trip() {
        let mut s = NodeStages::default();
        for stage in Stage::ALL {
            s.set(stage, StageState::Done);
            assert_eq!(s.get(stage), StageState::Done);
        }
    }

    #[test]
    fn graph_serializes_and_round_trips_through_json() {
        // JSON now carries topology and stage state only — content
        // fields are skip_serialized, so they don't round-trip through
        // pure serde (only through the disk-backed `load` / `render`).
        // This test pins the topology + state aspect.
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("app")).unwrap();
        let a = g.add_child(root, node("a")).unwrap();
        let b = g.add_child(root, node("b")).unwrap();
        g.add_dep(a, b).unwrap();
        g.get_mut(root).unwrap().stages.spec = StageState::Done;
        let json = serde_json::to_string(&g).unwrap();
        let g2: NodeGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(g2.len(), 3);
        assert_eq!(g2.get(root).unwrap().stages.spec, StageState::Done);
        assert_eq!(g2.get(a).unwrap().deps, vec![b]);
    }

    #[test]
    fn load_on_empty_workdir_returns_empty_graph() {
        let tmp = tempfile::tempdir().unwrap();
        let g = load(tmp.path(), Layout::SingleCrate).unwrap();
        assert_eq!(g.len(), 0);
        assert!(g.root.is_none());
    }

    #[test]
    fn save_then_load_round_trips() {
        // `save` persists topology + stage state to JSON; content fields
        // are no longer in JSON — they live in the rendered files on
        // disk and `load` reads them from there. So this test goes
        // through `render_graph` (which calls `save` internally and
        // also writes the rendered files) to exercise the full
        // round-trip.
        let tmp = tempfile::tempdir().unwrap();
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("app")).unwrap();
        let a = g.add_child(root, node("alpha")).unwrap();
        let _b = g.add_child(root, node("beta")).unwrap();
        g.add_dep(a, root).unwrap();
        g.get_mut(a).unwrap().spec_public_md = Some("# alpha spec".into());
        g.get_mut(a).unwrap().stages.spec = StageState::Done;

        crate::render::render_graph(tmp.path(), &g, Layout::SingleCrate).unwrap();
        // Per-node JSON files exist (topology + state).
        assert!(tmp.path().join(".bureau/nodes/app.json").exists());
        assert!(tmp.path().join(".bureau/nodes/alpha.json").exists());
        assert!(tmp.path().join(".bureau/nodes/beta.json").exists());
        assert!(tmp.path().join(".bureau/graph.json").exists());

        let g2 = load(tmp.path(), Layout::SingleCrate).unwrap();
        assert_eq!(g2.len(), 3);
        assert_eq!(g2.root, Some(root));
        // Content round-trips through the rendered files on disk.
        assert_eq!(
            g2.get(a).unwrap().spec_public_md.as_deref(),
            Some("# alpha spec"),
            "content should come back from the rendered spec/<alpha>/public.md"
        );
        assert_eq!(g2.get(a).unwrap().stages.spec, StageState::Done);
    }

    #[test]
    fn load_treats_placeholders_as_unauthored() {
        // A node whose iface stage hasn't run yet has placeholder files
        // on disk. `load` should map those back to `None`, not the
        // placeholder text.
        let tmp = tempfile::tempdir().unwrap();
        let mut g = NodeGraph::new();
        let _root = g.insert_root(node("app")).unwrap();
        crate::render::render_graph(tmp.path(), &g, Layout::SingleCrate).unwrap();
        let g2 = load(tmp.path(), Layout::SingleCrate).unwrap();
        let root = g2.root.unwrap();
        let n = g2.get(root).unwrap();
        assert!(n.public_rs.is_none(), "placeholder public.rs should load as None");
        assert!(n.private_rs.is_none());
        assert!(n.tests_rs.is_none());
        assert!(n.spec_public_md.is_none());
        assert!(n.spec_private_md.is_none());
    }

    #[test]
    fn save_does_not_write_content_into_json() {
        // The node JSON file should contain topology + state, NOT the
        // verbatim slot contents (which live in the rendered files).
        let tmp = tempfile::tempdir().unwrap();
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("app")).unwrap();
        g.get_mut(root).unwrap().spec_public_md = Some("# top secret\n".into());
        g.get_mut(root).unwrap().public_rs = Some("pub trait Top {}\n".into());
        save(tmp.path(), &g).unwrap();
        let raw = std::fs::read_to_string(tmp.path().join(".bureau/nodes/app.json")).unwrap();
        assert!(
            !raw.contains("top secret") && !raw.contains("pub trait Top"),
            "node JSON must NOT carry slot content: {raw}"
        );
    }

    #[test]
    fn save_prunes_stale_node_files() {
        let tmp = tempfile::tempdir().unwrap();
        // Initial save with two nodes.
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("app")).unwrap();
        let _x = g.add_child(root, node("xx")).unwrap();
        save(tmp.path(), &g).unwrap();
        assert!(tmp.path().join(".bureau/nodes/xx.json").exists());

        // Build a fresh graph without `xx`; save should remove its file.
        let mut g2 = NodeGraph::new();
        let _ = g2.insert_root(node("app")).unwrap();
        save(tmp.path(), &g2).unwrap();
        assert!(!tmp.path().join(".bureau/nodes/xx.json").exists());
        assert!(tmp.path().join(".bureau/nodes/app.json").exists());
    }

    #[test]
    fn reset_cascade_from_iface_wipes_tests_and_iface_outputs() {
        // The exact bug from the user's report: Iface gets reset on
        // restart, but Tests is already authored. The cascade should
        // also reset Tests + clear `tests_rs` (and `public_rs` /
        // `private_rs` because Iface itself is being reset).
        let mut n = node("foo");
        n.stages.spec = StageState::Done;
        n.stages.iface = StageState::InProgress;
        n.stages.tests = StageState::Done;
        n.stages.impl_ = StageState::NotStarted;
        n.spec_public_md = Some("# spec\n".into());
        n.public_rs = Some("pub trait T { fn f(&self); }\n".into());
        n.private_rs = Some("impl T for X { fn f(&self) { todo!() } }\n".into());
        n.tests_rs = Some("#[test] fn t() {}\n".into());

        let changed = reset_stage_and_cascade(&mut n, Stage::Iface);

        // Only previously-non-NotStarted stages count as changed.
        let changed_stages: Vec<Stage> = changed.iter().map(|(s, _)| *s).collect();
        assert_eq!(changed_stages, vec![Stage::Iface, Stage::Tests]);
        assert_eq!(n.stages.iface, StageState::NotStarted);
        assert_eq!(n.stages.tests, StageState::NotStarted);
        assert_eq!(n.stages.impl_, StageState::NotStarted);
        assert_eq!(n.stages.debug, StageState::NotStarted);
        // Iface stage is being reset → its first-written slots cleared.
        assert!(n.public_rs.is_none());
        assert!(n.private_rs.is_none());
        // Tests stage cascaded → tests_rs cleared.
        assert!(n.tests_rs.is_none());
        // Earlier stage's content is preserved.
        assert!(n.spec_public_md.is_some());
        // Spec stage stays Done — only stages >= `from` are touched.
        assert_eq!(n.stages.spec, StageState::Done);
    }

    #[test]
    fn reset_cascade_from_impl_keeps_iface_outputs() {
        // Reset Impl: only Impl and Debug states change; private_rs is
        // NOT cleared because its first-writer is Iface (which stays
        // Done). The Iface stub remains so the workspace compiles
        // while Impl is being re-authored.
        let mut n = node("foo");
        n.stages.iface = StageState::Done;
        n.stages.tests = StageState::Done;
        n.stages.impl_ = StageState::InProgress;
        n.public_rs = Some("pub trait T { fn f(&self); }\n".into());
        n.private_rs = Some("impl T for X { fn f(&self) { 42 } }\n".into());
        n.tests_rs = Some("#[test] fn t() {}\n".into());

        reset_stage_and_cascade(&mut n, Stage::Impl);

        assert_eq!(n.stages.iface, StageState::Done);
        assert_eq!(n.stages.tests, StageState::Done);
        assert_eq!(n.stages.impl_, StageState::NotStarted);
        assert!(n.public_rs.is_some(), "iface output preserved");
        assert!(n.private_rs.is_some(), "iface stub preserved (Impl re-writes on next run)");
        assert!(n.tests_rs.is_some(), "tests preserved");
    }

    #[test]
    fn reset_cascade_from_spec_wipes_everything_downstream() {
        let mut n = node("foo");
        n.stages.spec = StageState::InProgress;
        n.stages.iface = StageState::Done;
        n.stages.tests = StageState::Done;
        n.stages.impl_ = StageState::Done;
        n.spec_public_md = Some("# spec\n".into());
        n.spec_private_md = Some("# notes\n".into());
        n.public_rs = Some("pub trait T { fn f(&self); }\n".into());
        n.private_rs = Some("impl T for X { fn f(&self) { 42 } }\n".into());
        n.tests_rs = Some("#[test] fn t() {}\n".into());

        reset_stage_and_cascade(&mut n, Stage::Spec);

        assert!(n.spec_public_md.is_none());
        assert!(n.spec_private_md.is_none());
        assert!(n.public_rs.is_none());
        assert!(n.private_rs.is_none());
        assert!(n.tests_rs.is_none());
    }
}
