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

use chrono::{DateTime, Utc};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
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
    /// Optional polish.
    Opt,
}

impl Stage {
    pub const ALL: [Stage; 6] = [
        Stage::Spec,
        Stage::Iface,
        Stage::Tests,
        Stage::Impl,
        Stage::Debug,
        Stage::Opt,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Stage::Spec => "spec",
            Stage::Iface => "iface",
            Stage::Tests => "tests",
            Stage::Impl => "impl",
            Stage::Debug => "debug",
            Stage::Opt => "opt",
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
    Skipped,
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
        matches!(
            self,
            StageState::Done | StageState::Failed | StageState::Skipped
        )
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeStages {
    pub spec: StageState,
    pub iface: StageState,
    pub tests: StageState,
    #[serde(rename = "impl")]
    pub impl_: StageState,
    pub debug: StageState,
    pub opt: StageState,
}

impl NodeStages {
    pub fn get(&self, stage: Stage) -> StageState {
        match stage {
            Stage::Spec => self.spec,
            Stage::Iface => self.iface,
            Stage::Tests => self.tests,
            Stage::Impl => self.impl_,
            Stage::Debug => self.debug,
            Stage::Opt => self.opt,
        }
    }

    pub fn set(&mut self, stage: Stage, value: StageState) {
        match stage {
            Stage::Spec => self.spec = value,
            Stage::Iface => self.iface = value,
            Stage::Tests => self.tests = value,
            Stage::Impl => self.impl_ = value,
            Stage::Debug => self.debug = value,
            Stage::Opt => self.opt = value,
        }
    }
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
    pub spec_md: Option<String>,
    pub public_rs: Option<String>,
    pub private_rs: Option<String>,
    pub tests_rs: Option<String>,

    pub stages: NodeStages,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
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
            spec_md: None,
            public_rs: None,
            private_rs: None,
            tests_rs: None,
            stages: NodeStages::default(),
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

    /// Add a dep edge `from → to`. Both must exist; cycles are rejected.
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
        // Cycle check: would adding from → to create a cycle? A cycle exists
        // iff `to` already reaches `from` via existing dep edges.
        if self.dep_reaches(to, from) {
            return Err(GraphError::WouldCycle { from, to });
        }
        let n = self.nodes.get_mut(&from).expect("checked");
        if !n.deps.contains(&to) {
            n.deps.push(to);
            n.updated_at = Utc::now();
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut g = NodeGraph::new();
        let root = g.insert_root(node("app")).unwrap();
        let a = g.add_child(root, node("a")).unwrap();
        let b = g.add_child(root, node("b")).unwrap();
        g.add_dep(a, b).unwrap();
        g.get_mut(root).unwrap().spec_md = Some("# spec".to_string());
        let json = serde_json::to_string(&g).unwrap();
        let g2: NodeGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(g2.len(), 3);
        assert_eq!(g2.get(root).unwrap().spec_md.as_deref(), Some("# spec"));
        assert_eq!(g2.get(a).unwrap().deps, vec![b]);
    }
}
