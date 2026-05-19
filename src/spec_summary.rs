//! LLM-summarized ancestor specs, cached in memory.
//!
//! Each node's context bundle includes prose from its ancestor chain
//! (each ancestor's first-paragraph brief, plus the immediate parent's
//! full public spec). For deep trees with long specs, this is the
//! single biggest non-Rust contributor to context bloat.
//!
//! This module replaces the ancestor-spec text with a model-summarized
//! version. Cached by content hash, so:
//!  - The same (node, content_version) is summarized AT MOST ONCE.
//!  - A spec revision invalidates only the affected entry.
//!  - The cache lives only in memory — engine restart re-summarizes.
//!    That's intentional: the cost is small (one LLM call per spec
//!    revision) and avoiding on-disk state keeps the engine's
//!    persistence story simple.
//!
//! Summaries are STRICTLY an optimization. If the summarizer fails or
//! the cache misses, callers fall back to the framework's existing
//! first-paragraph brief — never blocked on this layer.
//!
//! Note on cost / latency: prewarm is fired BEFORE the per-stage LLM
//! call. The prewarm fans out in parallel across all uncached ancestors,
//! so total latency is `max(individual summarization)` ≈ a few seconds
//! per uncached ancestor on a small model. Pick a cheap+fast model
//! via `models.summary` in config.

use crate::graph::NodeId;
use anyhow::Result;
use async_trait::async_trait;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

/// Summarizer interface — one method, async, takes a content string.
/// The engine wires a concrete `OpenRouterSummarizer` for production;
/// tests can substitute a deterministic mock.
#[async_trait]
pub trait Summarizer: Send + Sync {
    async fn summarize(&self, content: &str) -> Result<String>;
}

/// In-memory cache of ancestor-spec summaries keyed by content hash.
///
/// The key intentionally does NOT include the node_id — same content
/// produces the same summary regardless of which node it came from,
/// and conflating "this node's spec" with "this content" lets us
/// share cache hits when an ancestor's spec hasn't actually changed
/// even after a node is moved/renamed.
pub struct SpecSummaryCache {
    inner: Mutex<HashMap<u64, String>>,
    summarizer: Arc<dyn Summarizer>,
}

impl SpecSummaryCache {
    pub fn new(summarizer: Arc<dyn Summarizer>) -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            summarizer,
        }
    }

    /// Look up a cached summary by content. Returns `None` on miss —
    /// caller should fall back to the framework's brief.
    pub fn get(&self, content: &str) -> Option<String> {
        self.inner.lock().get(&hash_content(content)).cloned()
    }

    /// Pre-warm the cache for every `(node_id, content)` pair that
    /// isn't already cached. Fans out in parallel; awaits all. Errors
    /// from individual summarizations are LOGGED, not propagated —
    /// the cache stays empty for that entry and the caller falls back
    /// to the brief. Returns the count of summaries actually computed
    /// (cache misses), for visibility.
    pub async fn prewarm(&self, pairs: &[(NodeId, String)]) -> usize {
        // Collect uncached entries first under the lock, then run
        // their summarization calls without holding any lock. Dedup
        // by hash WITHIN the batch too — two pairs with identical
        // content (e.g. two ancestors whose spec text accidentally
        // matches) only need to be summarized once.
        let to_summarize: Vec<(u64, String)> = {
            let cache = self.inner.lock();
            let mut seen = std::collections::HashSet::new();
            pairs
                .iter()
                .filter_map(|(_id, content)| {
                    let h = hash_content(content);
                    if cache.contains_key(&h) || !seen.insert(h) {
                        None
                    } else {
                        Some((h, content.clone()))
                    }
                })
                .collect()
        };
        if to_summarize.is_empty() {
            return 0;
        }
        let summarizer = self.summarizer.clone();
        let mut tasks = Vec::with_capacity(to_summarize.len());
        for (h, content) in to_summarize {
            let s = summarizer.clone();
            tasks.push(tokio::spawn(async move {
                let result = s.summarize(&content).await;
                (h, result)
            }));
        }
        let mut count = 0usize;
        for t in tasks {
            match t.await {
                Ok((h, Ok(summary))) => {
                    self.inner.lock().insert(h, summary);
                    count += 1;
                }
                Ok((_, Err(e))) => {
                    tracing::warn!("spec summarization failed (will fall back to brief): {e:#}");
                }
                Err(je) => {
                    tracing::warn!("spec summarizer task join error: {je}");
                }
            }
        }
        count
    }
}

fn hash_content(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// Production summarizer: calls a small LLM via rig + openrouter, no
/// tools, no streaming. The summary prompt is fixed and aimed at
/// compressing spec markdown into ~5-10 lines of plain text covering
/// what the node does, what it exposes, and what it explicitly
/// excludes.
pub struct OpenRouterSummarizer {
    client: rig::providers::openrouter::Client,
    model: String,
    max_tokens: u64,
}

impl OpenRouterSummarizer {
    pub fn new(
        client: rig::providers::openrouter::Client,
        model: String,
        max_tokens: u64,
    ) -> Self {
        Self {
            client,
            model,
            max_tokens,
        }
    }
}

#[async_trait]
impl Summarizer for OpenRouterSummarizer {
    async fn summarize(&self, content: &str) -> Result<String> {
        use rig::client::CompletionClient;
        use rig::completion::Prompt;
        let preamble = "You compress software specifications into compact summaries. \
            Goal: 5-10 short lines, plain prose, no markdown headings. Keep the \
            spec's named abstractions, invariants, and out-of-scope items. Drop \
            rationale, history, and prose padding. The reader is another LLM \
            that will use this as background context for an unrelated downstream \
            task — clarity and brevity beat completeness.";
        let user = format!(
            "Summarize this spec into 5-10 lines:\n\n---\n{content}\n---\n\nSummary:"
        );
        let resp = self
            .client
            .agent(&self.model)
            .preamble(preamble)
            .max_tokens(self.max_tokens)
            .temperature(0.0)
            .build()
            .prompt(&user)
            .await?;
        Ok(resp.to_string().trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic mock: returns "SUMMARY: {first_50_chars}".
    /// Tracks invocation count via an AtomicUsize so tests can verify
    /// cache behavior.
    struct MockSummarizer {
        calls: std::sync::atomic::AtomicUsize,
    }
    #[async_trait]
    impl Summarizer for MockSummarizer {
        async fn summarize(&self, content: &str) -> Result<String> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let take: String = content.chars().take(50).collect();
            Ok(format!("SUMMARY: {take}"))
        }
    }

    fn fresh_cache() -> (Arc<SpecSummaryCache>, Arc<MockSummarizer>) {
        let mock = Arc::new(MockSummarizer {
            calls: std::sync::atomic::AtomicUsize::new(0),
        });
        let cache = Arc::new(SpecSummaryCache::new(mock.clone()));
        (cache, mock)
    }

    #[tokio::test]
    async fn prewarm_summarizes_uncached() {
        let (cache, mock) = fresh_cache();
        let pairs = vec![
            (NodeId(uuid::Uuid::new_v4()), "spec A".into()),
            (NodeId(uuid::Uuid::new_v4()), "spec B".into()),
        ];
        let n = cache.prewarm(&pairs).await;
        assert_eq!(n, 2);
        assert_eq!(mock.calls.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert!(cache.get("spec A").unwrap().contains("SUMMARY: spec A"));
        assert!(cache.get("spec B").unwrap().contains("SUMMARY: spec B"));
    }

    #[tokio::test]
    async fn prewarm_skips_cached_entries() {
        let (cache, mock) = fresh_cache();
        let id = NodeId(uuid::Uuid::new_v4());
        cache.prewarm(&[(id, "spec X".into())]).await;
        assert_eq!(mock.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        // Second prewarm with the same content → no new calls.
        cache.prewarm(&[(id, "spec X".into())]).await;
        assert_eq!(mock.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cache_keyed_by_content_not_node_id() {
        // Two different node_ids with IDENTICAL content → one summary
        // call. Spec content drift is what matters, not node identity.
        let (cache, mock) = fresh_cache();
        let same_content = "same spec text";
        cache
            .prewarm(&[
                (NodeId(uuid::Uuid::new_v4()), same_content.into()),
                (NodeId(uuid::Uuid::new_v4()), same_content.into()),
            ])
            .await;
        assert_eq!(mock.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn content_change_misses_cache_again() {
        let (cache, mock) = fresh_cache();
        let id = NodeId(uuid::Uuid::new_v4());
        cache.prewarm(&[(id, "v1".into())]).await;
        cache.prewarm(&[(id, "v2".into())]).await;
        // Different content → two summaries.
        assert_eq!(mock.calls.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert!(cache.get("v1").unwrap().contains("SUMMARY: v1"));
        assert!(cache.get("v2").unwrap().contains("SUMMARY: v2"));
    }

    #[tokio::test]
    async fn get_returns_none_on_miss() {
        let (cache, _mock) = fresh_cache();
        assert!(cache.get("never summarized").is_none());
    }
}
