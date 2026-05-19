//! Search the crates.io registry by keyword.
//!
//! Backs the `search_crates` tool. The architect uses this to verify
//! that a crate name actually exists before declaring it in
//! `external_crate_deps` — the original bug here was the architect
//! declaring `md-4` (no such crate; the real one is `md4`), which
//! made the cargo gate fail with no actionable diagnostic.
//!
//! crates.io's public API is documented at
//! <https://crates.io/data-access#api>. Anonymous access is fine for
//! search; rate-limiting is per-IP, conservative on our end is
//! sensible.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// One hit from a crates.io search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrateSummary {
    pub name: String,
    pub description: Option<String>,
    pub max_stable_version: Option<String>,
    pub max_version: String,
    pub recent_downloads: Option<u64>,
}

/// Search crates.io for `query`. Returns at most `limit` results,
/// sorted by relevance (crates.io's default ordering).
pub async fn search(query: &str, limit: usize) -> Result<Vec<CrateSummary>> {
    // crates.io's search endpoint: `/api/v1/crates?q=<query>&per_page=<n>`.
    // Cap per_page at 25 — anything more is rarely useful and we don't
    // want to encourage the model to drown in low-quality matches.
    let per_page = limit.clamp(1, 25);
    let url = format!(
        "https://crates.io/api/v1/crates?q={}&per_page={per_page}",
        urlencoding::encode(query)
    );
    let client = http_client()?;
    let resp = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?
        .error_for_status()
        .with_context(|| format!("crates.io search returned an error status for {url}"))?;
    #[derive(Deserialize)]
    struct SearchResp {
        crates: Vec<CrateSummary>,
    }
    let parsed: SearchResp = resp
        .json()
        .await
        .context("parsing crates.io search response")?;
    Ok(parsed.crates)
}

/// Per-query outcome from a batched search.
#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub query: String,
    /// `Some(...)` on success — the per-query hit list (may be empty
    /// if crates.io found nothing). `None` when `error` is set.
    pub crates: Option<Vec<CrateSummary>>,
    /// Set if this query failed (network, 5xx, deserialization).
    /// The remaining queries in the batch still get their results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Run several searches in parallel. One bad query doesn't fail the
/// batch — each result carries its own success / error. This is what
/// `search_crates` exposes to the model so it can scatter-gather
/// "verify all these names at once" in one tool call.
pub async fn search_many(queries: Vec<String>, limit: usize) -> Vec<SearchHit> {
    let tasks: Vec<_> = queries
        .into_iter()
        .map(|q| async move {
            let q_for_result = q.clone();
            match search(&q, limit).await {
                Ok(crates) => SearchHit {
                    query: q_for_result,
                    crates: Some(crates),
                    error: None,
                },
                Err(e) => SearchHit {
                    query: q_for_result,
                    crates: None,
                    error: Some(format!("{e:#}")),
                },
            }
        })
        .collect();
    futures::future::join_all(tasks).await
}

/// crates.io requires a `User-Agent` header on every request; without
/// one they 403. The string identifies the framework + a contact URL
/// (per crates.io's published policy).
fn http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent("bureau-rs (https://github.com/graydon/bureau-rs)")
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .context("building reqwest client")
}

#[cfg(test)]
mod tests {
    // Network-touching tests are off by default — set BUREAU_NET_TESTS=1
    // to run them locally. Most builds run on machines that can't
    // reach the public internet (CI sandboxes, the codespace's
    // network policies, etc.); the suite SHOULD pass with zero net.

    fn net_enabled() -> bool {
        std::env::var("BUREAU_NET_TESTS").is_ok()
    }

    #[tokio::test]
    async fn search_returns_md4_for_md_query() {
        if !net_enabled() {
            return;
        }
        let hits = super::search("md4", 5).await.unwrap();
        assert!(
            hits.iter().any(|c| c.name == "md4" || c.name == "md-4"),
            "expected an md4-shaped crate in: {hits:#?}"
        );
    }

    #[tokio::test]
    async fn search_caps_results() {
        if !net_enabled() {
            return;
        }
        let hits = super::search("serde", 3).await.unwrap();
        assert!(hits.len() <= 3);
    }
}
