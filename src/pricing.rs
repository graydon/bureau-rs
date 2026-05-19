//! Live pricing from OpenRouter.
//!
//! Fetched once at startup from `/api/v1/models`. The endpoint returns
//! per-token prices as decimal strings (e.g. `"0.000015"`); we normalize
//! to dollars per 1M tokens so the rest of the engine can do the
//! arithmetic in familiar units.
//!
//! Falls back to a small built-in table if the fetch fails (offline /
//! tests / network blip) — the orchestrator must not stall waiting on
//! prices.
//!
//! OpenRouter's response also carries per-model `input_cache_read` and
//! `input_cache_write` rates when the model supports caching. We use
//! those directly instead of the previously hardcoded Claude-specific
//! 0.1x / 1.25x multipliers — those numbers don't apply to other
//! providers (OpenAI's cache-read is 0.5x; DeepSeek's is 0.1x; etc.).

use serde::Deserialize;
use std::collections::HashMap;

/// Per-model prices in **dollars per 1M tokens** (the unit the rest of
/// the engine uses). All fields are absolute prices, not multipliers.
#[derive(Debug, Clone, Default)]
pub struct ModelPrice {
    pub input: f64,
    pub output: f64,
    /// Price paid when reading from prompt cache. If the provider/model
    /// doesn't support caching this stays 0.0 and the cost code falls
    /// back to charging `input` for cached tokens.
    pub cache_read: f64,
    /// Surcharge paid when writing to prompt cache (first time the
    /// prefix is seen). 0.0 if unsupported.
    pub cache_write: f64,
}

/// Pricing table keyed by OpenRouter's model id (e.g.
/// `"anthropic/claude-sonnet-4"`). Lookup tolerates partial prefix
/// matches so config models like `"anthropic/claude-sonnet-4"` resolve
/// even if OpenRouter returns a versioned id like
/// `"anthropic/claude-sonnet-4:beta"`.
#[derive(Debug, Clone, Default)]
pub struct PriceTable {
    by_id: HashMap<String, ModelPrice>,
}

impl PriceTable {
    /// Look up the price for a model. Falls back to substring match for
    /// versioned ids, then to a small built-in table of common
    /// patterns. Returns `None` if nothing matches — caller should use
    /// a conservative default.
    pub fn get(&self, model: &str) -> Option<ModelPrice> {
        if let Some(p) = self.by_id.get(model) {
            return Some(p.clone());
        }
        // Versioned id fallback: try stripping `:variant` suffix and
        // any leading `openrouter/` prefix.
        let stripped = model.split(':').next().unwrap_or(model);
        if let Some(p) = self.by_id.get(stripped) {
            return Some(p.clone());
        }
        // Last resort: substring contains. OpenRouter ids look like
        // `vendor/model[:variant]`; user config sometimes specifies
        // just the trailing part.
        for (k, v) in &self.by_id {
            if k.ends_with(model) || model.ends_with(k.as_str()) {
                return Some(v.clone());
            }
        }
        None
    }

    /// How many distinct models we know prices for.
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
    #[serde(default)]
    pricing: Option<PricingFields>,
}

/// Per-token prices as strings (OpenRouter's wire format). Missing or
/// non-parseable fields default to 0.0.
#[derive(Debug, Deserialize)]
struct PricingFields {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    completion: Option<String>,
    #[serde(default)]
    input_cache_read: Option<String>,
    #[serde(default)]
    input_cache_write: Option<String>,
}

fn parse_per_token(s: &Option<String>) -> f64 {
    // OpenRouter wire format is "dollars per token" as decimal string.
    // Multiply by 1e6 to get our internal "dollars per 1M tokens".
    s.as_deref()
        .and_then(|x| x.parse::<f64>().ok())
        .unwrap_or(0.0)
        * 1_000_000.0
}

/// Fetch the live price table from OpenRouter. Times out after 5s and
/// returns an empty table on any failure — callers should treat
/// `is_empty()` as "use built-in fallback prices".
pub async fn fetch(base_url: Option<&str>) -> PriceTable {
    let url = base_url
        .map(|b| format!("{}/models", b.trim_end_matches('/')))
        .unwrap_or_else(|| "https://openrouter.ai/api/v1/models".to_string());
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("pricing: client build failed: {e}; using fallback");
            return PriceTable::default();
        }
    };
    let resp = match client.get(&url).send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("pricing: GET {url} failed: {e}; using fallback");
            return PriceTable::default();
        }
    };
    if !resp.status().is_success() {
        tracing::warn!(
            "pricing: GET {url} returned {}; using fallback",
            resp.status()
        );
        return PriceTable::default();
    }
    let body: ModelsResponse = match resp.json().await {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!("pricing: parse {url}: {e}; using fallback");
            return PriceTable::default();
        }
    };
    let mut by_id: HashMap<String, ModelPrice> = HashMap::with_capacity(body.data.len());
    for m in body.data {
        let Some(p) = m.pricing else { continue };
        by_id.insert(
            m.id,
            ModelPrice {
                input: parse_per_token(&p.prompt),
                output: parse_per_token(&p.completion),
                cache_read: parse_per_token(&p.input_cache_read),
                cache_write: parse_per_token(&p.input_cache_write),
            },
        );
    }
    tracing::info!("pricing: loaded {} models from OpenRouter", by_id.len());
    PriceTable { by_id }
}

/// Built-in fallback when the live fetch fails or returns nothing for a
/// given model. Substring-matched against the lowercased model id —
/// this is the pre-existing behaviour from before we had live prices,
/// preserved as a safety net so a network blip doesn't blow up the
/// budget heuristic.
pub fn fallback_price(model: &str) -> ModelPrice {
    let m = model.to_ascii_lowercase();
    let (input, output) = if m.contains("opus") {
        (15.0, 75.0)
    } else if m.contains("sonnet") {
        (3.0, 15.0)
    } else if m.contains("haiku") {
        (1.0, 5.0)
    } else if m.contains("gpt-4o-mini") || m.contains("4o-mini") {
        (0.15, 0.6)
    } else if m.contains("gpt-4o") {
        (2.5, 10.0)
    } else if m.contains("gpt-5-mini") {
        (0.25, 2.0)
    } else if m.contains("qwen3-coder") {
        (0.2, 0.8)
    } else if m.contains("nemotron") {
        (0.4, 1.6)
    } else if m.contains("deepseek") {
        (0.3, 1.2)
    } else {
        // Conservative mid-tier guess. Bias high so the budget cap
        // triggers earlier rather than overshoots when we have no idea.
        (3.0, 15.0)
    };
    // Pre-cache-era fallback: assume Claude-style 0.1x cache-read and
    // 1.25x cache-write multipliers. Wrong for non-Anthropic providers,
    // but better than charging cache tokens at full price.
    ModelPrice {
        input,
        output,
        cache_read: input * 0.1,
        cache_write: input * 1.25,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_per_token_handles_missing_and_malformed() {
        assert_eq!(parse_per_token(&None), 0.0);
        assert_eq!(parse_per_token(&Some("".into())), 0.0);
        assert_eq!(parse_per_token(&Some("not-a-number".into())), 0.0);
        // 0.000015 / token == $15 / 1M tokens
        assert!((parse_per_token(&Some("0.000015".into())) - 15.0).abs() < 1e-9);
    }

    #[test]
    fn pricetable_get_falls_back_through_variants() {
        let mut by_id = HashMap::new();
        by_id.insert(
            "anthropic/claude-sonnet-4".to_string(),
            ModelPrice {
                input: 3.0,
                output: 15.0,
                cache_read: 0.3,
                cache_write: 3.75,
            },
        );
        let t = PriceTable { by_id };
        // Exact match.
        assert!((t.get("anthropic/claude-sonnet-4").unwrap().input - 3.0).abs() < 1e-9);
        // Versioned id strips the `:variant` suffix.
        assert!(t.get("anthropic/claude-sonnet-4:beta").is_some());
        // Substring suffix-match.
        assert!(t.get("claude-sonnet-4").is_some());
        // Total miss.
        assert!(t.get("does-not-exist").is_none());
    }

    #[test]
    fn fallback_price_recognizes_common_models() {
        let opus = fallback_price("anthropic/claude-opus-4");
        assert!((opus.input - 15.0).abs() < 1e-9);
        let sonnet = fallback_price("anthropic/claude-sonnet-4");
        assert!((sonnet.input - 3.0).abs() < 1e-9);
        let unknown = fallback_price("some/unknown-model");
        // Conservative high default ensures budget is respected.
        assert!(unknown.input >= 1.0);
    }
}
