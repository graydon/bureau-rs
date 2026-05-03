use crate::artifact;
use anyhow::{Context, Result, anyhow};
use indexmap::IndexMap;
use std::path::Path;
use toml::Value as TomlValue;

/// Merge two Cargo.toml files. `[dependencies]` is merged by union; conflicts
/// (same key, different version) result in an error. All other top-level
/// tables: `b` overlays `a`.
pub fn merge_cargo_toml(a: &str, b: &str) -> Result<String> {
    let a_val: TomlValue = toml::from_str(a).context("parsing left Cargo.toml")?;
    let b_val: TomlValue = toml::from_str(b).context("parsing right Cargo.toml")?;

    let mut out = a_val.clone();
    let a_table = a_val
        .as_table()
        .ok_or_else(|| anyhow!("Cargo.toml a is not a table"))?;
    let b_table = b_val
        .as_table()
        .ok_or_else(|| anyhow!("Cargo.toml b is not a table"))?;
    let out_table = out.as_table_mut().unwrap();

    for (key, b_value) in b_table.iter() {
        if matches!(
            key.as_str(),
            "dependencies" | "dev-dependencies" | "build-dependencies"
        ) {
            let a_deps = a_table
                .get(key)
                .and_then(|v| v.as_table())
                .cloned()
                .unwrap_or_default();
            let b_deps = b_value
                .as_table()
                .ok_or_else(|| anyhow!("[{}] is not a table in b", key))?;
            let mut merged = a_deps.clone();
            for (dep, ver) in b_deps.iter() {
                match merged.get(dep) {
                    None => {
                        merged.insert(dep.clone(), ver.clone());
                    }
                    Some(existing) if existing == ver => {}
                    Some(existing) => {
                        return Err(anyhow!(
                            "conflict in [{key}]: '{dep}' = {existing} vs {ver}"
                        ));
                    }
                }
            }
            out_table.insert(key.clone(), TomlValue::Table(merged));
        } else if !out_table.contains_key(key) {
            out_table.insert(key.clone(), b_value.clone());
        }
        // else: keep `a`'s value for non-dependency tables (idempotent overlay)
    }

    Ok(toml::to_string_pretty(&out)?)
}

/// Try to merge two versions of a file using our content-aware drivers.
/// Returns Some(merged) on success; None means we don't know how to merge
/// this file type.
pub fn try_content_merge(path: &Path, a: &str, b: &str) -> Result<Option<String>> {
    if path.file_name().and_then(|f| f.to_str()) == Some("Cargo.toml") {
        return Ok(Some(merge_cargo_toml(a, b)?));
    }
    if let Some(name) = path.file_name().and_then(|f| f.to_str()) {
        if matches!(name, "lib.rs" | "main.rs" | "mod.rs") {
            // For top-level mod-decl files, attempt mod-decl union.
            if a == b {
                return Ok(Some(a.to_string()));
            }
            return Ok(Some(artifact::merge_mod_declarations(a, b)?));
        }
    }
    Ok(None)
}

/// Three-way file merge using content-aware drivers when possible.
/// `_base` is provided for future improvement (currently we drop to a/b
/// union).
pub fn three_way_merge(
    path: &Path,
    _base: Option<&str>,
    ours: &str,
    theirs: &str,
) -> Result<MergeOutcome> {
    if ours == theirs {
        return Ok(MergeOutcome::Clean(ours.to_string()));
    }
    if let Some(merged) = try_content_merge(path, ours, theirs)? {
        return Ok(MergeOutcome::Clean(merged));
    }
    Ok(MergeOutcome::Conflict {
        ours: ours.to_string(),
        theirs: theirs.to_string(),
    })
}

#[derive(Debug, Clone)]
pub enum MergeOutcome {
    Clean(String),
    Conflict { ours: String, theirs: String },
}

/// Read [dependencies] keys from a Cargo.toml string.
pub fn read_dependencies(content: &str) -> Result<IndexMap<String, TomlValue>> {
    let v: TomlValue = toml::from_str(content)?;
    let mut out = IndexMap::new();
    if let Some(deps) = v.get("dependencies").and_then(|d| d.as_table()) {
        for (k, vv) in deps.iter() {
            out.insert(k.clone(), vv.clone());
        }
    }
    Ok(out)
}
