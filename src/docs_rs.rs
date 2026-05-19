//! Fetch and clean docs.rs HTML pages into plain markdown for the model.
//!
//! Two entry points:
//!
//! - [`crate_root`] — fetch the crate's docs.rs landing page. Use this
//!   to learn what a crate does at a glance (top-level docs, the list
//!   of public items linked from the crate root).
//! - [`item_page`] — fetch a specific item page (e.g. a struct's docs
//!   with all its methods). Use this when the model has identified a
//!   specific type/trait/function it cares about and wants the full
//!   surface.
//!
//! The HTML is run through [`rs_trafilatura::extract`] which strips
//! navigation chrome and returns markdown (preferred) or plain text.
//! That's small enough to feed into context bundles without drowning
//! the model in boilerplate.

use anyhow::{Context, Result};

/// Markdown (or plain text fallback) extracted from one docs.rs page,
/// plus the URL it came from.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DocPage {
    pub url: String,
    pub title: Option<String>,
    pub content: String,
}

/// Request for one batched-fetch entry: a crate name + optional
/// version + optional sub-path. When `path` is empty / None, the
/// crate's landing page is fetched; otherwise the specific item page.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct DocsRequest {
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
    /// Sub-path beyond the crate root, e.g.
    /// `"block_api/struct.Md4Core.html"`. Defaults to the crate root.
    #[serde(default)]
    pub path: Option<String>,
}

/// Per-request outcome from a batched fetch. Mirrors `DocsRequest`'s
/// keys for echoing the input back, plus exactly one of `page` /
/// `error` so the caller can dispatch on success.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DocsHit {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<DocPage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Run several docs-page fetches in parallel. One bad request doesn't
/// fail the batch — each entry carries its own page or error. Used by
/// the `crate_docs` and `docs_lookup` tools to let the model
/// scatter-gather across many crates / items at once.
pub async fn fetch_many(requests: Vec<DocsRequest>) -> Vec<DocsHit> {
    let tasks: Vec<_> = requests
        .into_iter()
        .map(|req| async move {
            let name = req.name.clone();
            let version = req.version.clone();
            let path = req.path.clone();
            let result = match &path {
                Some(p) if !p.is_empty() => {
                    item_page(&req.name, req.version.as_deref(), p).await
                }
                _ => crate_root(&req.name, req.version.as_deref()).await,
            };
            match result {
                Ok(page) => DocsHit {
                    name,
                    version,
                    path,
                    page: Some(page),
                    error: None,
                },
                Err(e) => DocsHit {
                    name,
                    version,
                    path,
                    page: None,
                    error: Some(format!("{e:#}")),
                },
            }
        })
        .collect();
    futures::future::join_all(tasks).await
}

/// Fetch the crate's landing page on docs.rs and return its extracted
/// content. `version` defaults to `"latest"` if `None`.
pub async fn crate_root(name: &str, version: Option<&str>) -> Result<DocPage> {
    let v = version.unwrap_or("latest");
    let url = format!("https://docs.rs/{name}/{v}/{name}/");
    fetch_and_extract(&url).await
}

/// Fetch a specific item page on docs.rs. `path` is everything beyond
/// the crate root: e.g. `"block_api/struct.Md4Core.html"` for
/// `https://docs.rs/md4/latest/md4/block_api/struct.Md4Core.html`.
pub async fn item_page(name: &str, version: Option<&str>, path: &str) -> Result<DocPage> {
    let v = version.unwrap_or("latest");
    // Strip leading slashes to keep the URL well-formed.
    let trimmed = path.trim_start_matches('/');
    let url = format!("https://docs.rs/{name}/{v}/{name}/{trimmed}");
    fetch_and_extract(&url).await
}

async fn fetch_and_extract(url: &str) -> Result<DocPage> {
    let client = http_client()?;
    let resp = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    // Don't error on 404 in error_for_status — instead surface a clean
    // diagnostic message. docs.rs 404s for typos and missing-version
    // refs; the model needs to know which it was.
    if !resp.status().is_success() {
        let status = resp.status();
        return Err(anyhow::anyhow!(
            "docs.rs returned HTTP {} for {url} — \
             check the crate name and (if specified) version",
            status
        ));
    }
    let html = resp.text().await.context("reading docs.rs body")?;
    let extracted = rs_trafilatura::extract(&html)
        .map_err(|e| anyhow::anyhow!("trafilatura extract: {e}"))?;
    // Prefer markdown — it preserves headings + code blocks, which is
    // the structural info the model needs to map types to traits to
    // methods. Fall back to plain text if trafilatura didn't produce
    // markdown (some pages don't).
    let content = extracted
        .content_markdown
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(extracted.content_text);
    Ok(DocPage {
        url: url.to_string(),
        title: extracted.metadata.title,
        content,
    })
}

fn http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent("bureau-rs (https://github.com/graydon/bureau-rs)")
        .timeout(std::time::Duration::from_secs(20))
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .context("building reqwest client")
}

#[cfg(test)]
mod tests {
    fn net_enabled() -> bool {
        std::env::var("BUREAU_NET_TESTS").is_ok()
    }

    #[tokio::test]
    async fn crate_root_fetches_md4_landing_page() {
        if !net_enabled() {
            return;
        }
        let page = super::crate_root("md4", None).await.unwrap();
        assert!(page.url.contains("docs.rs/md4/latest/md4"));
        // The md4 crate's docs mention its hashing/digest role.
        let lower = page.content.to_lowercase();
        assert!(lower.contains("md4") || lower.contains("digest"));
    }

    #[tokio::test]
    async fn item_page_fetches_specific_struct() {
        if !net_enabled() {
            return;
        }
        let page = super::item_page("md4", None, "block_api/struct.Md4Core.html")
            .await
            .unwrap();
        assert!(page.url.contains("Md4Core"));
    }

    #[tokio::test]
    async fn nonexistent_crate_returns_clear_error() {
        if !net_enabled() {
            return;
        }
        let err = super::crate_root("this-crate-does-not-exist-xyz123", None)
            .await
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("HTTP") && msg.contains("404"), "got: {msg}");
    }
}
