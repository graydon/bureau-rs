use anyhow::{Context, Result, anyhow};
use std::path::Path;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;

pub fn parse_rust(content: &str) -> Result<syn::File> {
    syn::parse_file(content).map_err(|e| {
        let span = e.span().start();
        let line = span.line;
        let col = span.column;
        let msg = format!("{e}");
        if line > 0 {
            let snippet = error_snippet(content, line, col);
            anyhow!("Rust syntax error at line {line}, column {col}: {msg}\n\n{snippet}")
        } else {
            anyhow!("Rust syntax error: {msg}")
        }
    })
}

pub fn validate_rust(path: &Path, content: &str) -> Result<()> {
    if !is_rust_file(path) {
        return Ok(());
    }
    parse_rust(content).map(|_| ()).with_context(|| {
        format!("invalid Rust syntax in {}", path.display())
    })
}

/// Render a small snippet showing the line with the error and a caret. Lines
/// are 1-indexed; columns are 0-indexed in syn's output. We shift the caret
/// by 1 so it aligns with the visible character.
fn error_snippet(content: &str, line: usize, col: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();
    if line == 0 || line > lines.len() {
        return String::new();
    }
    let lo = line.saturating_sub(2).max(1);
    let hi = (line + 1).min(lines.len());
    let mut out = String::new();
    for n in lo..=hi {
        let prefix = format!("{:>4} | ", n);
        out.push_str(&prefix);
        out.push_str(lines[n - 1]);
        out.push('\n');
        if n == line {
            // Caret line, aligned under the offending column.
            let pad = " ".repeat(prefix.len() + col);
            out.push_str(&pad);
            out.push_str("^\n");
        }
    }
    out
}

pub fn is_rust_file(path: &Path) -> bool {
    matches!(path.extension().and_then(|e| e.to_str()), Some("rs"))
}

/// Replace function bodies that aren't `todo!()` or `unimplemented!()` with `todo!()`.
/// Used to enforce Interface phase invariants.
pub fn stub_function_bodies(content: &str) -> Result<(String, Vec<String>)> {
    let mut file = parse_rust(content)?;
    let mut visitor = StubVisitor::default();
    visitor.visit_file_mut(&mut file);
    let formatted = prettyplease::unparse(&file);
    Ok((formatted, visitor.warnings))
}

#[derive(Default)]
struct StubVisitor {
    warnings: Vec<String>,
}

impl StubVisitor {
    fn body_is_stub(block: &syn::Block) -> bool {
        if block.stmts.len() != 1 {
            return false;
        }
        let s = &block.stmts[0];
        // Match `todo!()` / `unimplemented!()` as the sole macro statement
        if let syn::Stmt::Macro(m) = s {
            let path = &m.mac.path;
            return path_is(path, "todo") || path_is(path, "unimplemented");
        }
        if let syn::Stmt::Expr(syn::Expr::Macro(m), _) = s {
            let path = &m.mac.path;
            return path_is(path, "todo") || path_is(path, "unimplemented");
        }
        false
    }
}

fn path_is(path: &syn::Path, name: &str) -> bool {
    path.segments.len() == 1 && path.segments[0].ident == name
}

impl VisitMut for StubVisitor {
    fn visit_item_fn_mut(&mut self, node: &mut syn::ItemFn) {
        if !Self::body_is_stub(&node.block) {
            self.warnings.push(format!(
                "fn {} had body replaced with todo!()",
                node.sig.ident
            ));
            node.block = syn::parse_quote!({ todo!() });
        }
        syn::visit_mut::visit_item_fn_mut(self, node);
    }

    fn visit_impl_item_fn_mut(&mut self, node: &mut syn::ImplItemFn) {
        if !Self::body_is_stub(&node.block) {
            self.warnings.push(format!(
                "impl fn {} had body replaced with todo!()",
                node.sig.ident
            ));
            node.block = syn::parse_quote!({ todo!() });
        }
        syn::visit_mut::visit_impl_item_fn_mut(self, node);
    }
}

/// Extract public function/method signatures from a Rust file.
/// Used to detect signature changes in the Implementation phase.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PublicSignatures {
    pub items: Vec<String>,
}

impl PublicSignatures {
    pub fn from_source(content: &str) -> Result<Self> {
        let file = parse_rust(content)?;
        let mut v = SigCollector::default();
        v.visit_file(&file);
        v.items.sort();
        Ok(Self { items: v.items })
    }
}

#[derive(Default)]
struct SigCollector {
    items: Vec<String>,
}

impl<'ast> Visit<'ast> for SigCollector {
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        if matches!(node.vis, syn::Visibility::Public(_)) {
            self.items
                .push(format!("fn {} {}", node.sig.ident, sig_repr(&node.sig)));
        }
        syn::visit::visit_item_fn(self, node);
    }
    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        if matches!(node.vis, syn::Visibility::Public(_)) {
            self.items
                .push(format!("fn {} {}", node.sig.ident, sig_repr(&node.sig)));
        }
        syn::visit::visit_impl_item_fn(self, node);
    }
    fn visit_trait_item_fn(&mut self, node: &'ast syn::TraitItemFn) {
        self.items
            .push(format!("fn {} {}", node.sig.ident, sig_repr(&node.sig)));
        syn::visit::visit_trait_item_fn(self, node);
    }
    fn visit_item_struct(&mut self, node: &'ast syn::ItemStruct) {
        if matches!(node.vis, syn::Visibility::Public(_)) {
            self.items.push(format!("struct {}", node.ident));
        }
    }
    fn visit_item_enum(&mut self, node: &'ast syn::ItemEnum) {
        if matches!(node.vis, syn::Visibility::Public(_)) {
            self.items.push(format!("enum {}", node.ident));
        }
    }
    fn visit_item_trait(&mut self, node: &'ast syn::ItemTrait) {
        if matches!(node.vis, syn::Visibility::Public(_)) {
            self.items.push(format!("trait {}", node.ident));
        }
        syn::visit::visit_item_trait(self, node);
    }
}

fn sig_repr(sig: &syn::Signature) -> String {
    use quote::ToTokens;
    let mut t = proc_macro2::TokenStream::new();
    sig.inputs.to_tokens(&mut t);
    let inputs = t.to_string();
    let output = match &sig.output {
        syn::ReturnType::Default => String::new(),
        syn::ReturnType::Type(_, ty) => format!(" -> {}", ty.to_token_stream()),
    };
    format!("({}){}", inputs, output)
}

/// Merge two `mod foo;` lists into a deduplicated, sorted list.
pub fn merge_mod_declarations(a: &str, b: &str) -> Result<String> {
    let mut a_file = parse_rust(a)?;
    let b_file = parse_rust(b)?;
    let mut existing: Vec<String> = a_file
        .items
        .iter()
        .filter_map(item_mod_name)
        .map(|s| s.to_string())
        .collect();
    let mut to_add = Vec::new();
    for it in &b_file.items {
        if let Some(name) = item_mod_name(it) {
            let n_str = name.to_string();
            if !existing.iter().any(|n| *n == n_str) {
                existing.push(n_str);
                to_add.push(it.clone());
            }
        }
    }
    a_file.items.extend(to_add);
    Ok(prettyplease::unparse(&a_file))
}

fn item_mod_name(it: &syn::Item) -> Option<&syn::Ident> {
    if let syn::Item::Mod(m) = it {
        Some(&m.ident)
    } else {
        None
    }
}

/// Replace the body of a named function in a Rust source file.
pub fn replace_fn_body(content: &str, fn_name: &str, new_body: &str) -> Result<String> {
    let mut file = parse_rust(content)?;
    let new_block: syn::Block = syn::parse_str(&format!("{{ {} }}", new_body))
        .with_context(|| format!("parsing new body for fn {}", fn_name))?;
    let mut replacer = ReplaceFn {
        target: fn_name.to_string(),
        new_block,
        replaced: false,
    };
    replacer.visit_file_mut(&mut file);
    if !replacer.replaced {
        return Err(anyhow!("fn {} not found", fn_name));
    }
    Ok(prettyplease::unparse(&file))
}

struct ReplaceFn {
    target: String,
    new_block: syn::Block,
    replaced: bool,
}

impl VisitMut for ReplaceFn {
    fn visit_item_fn_mut(&mut self, node: &mut syn::ItemFn) {
        if node.sig.ident == self.target {
            node.block = Box::new(self.new_block.clone());
            self.replaced = true;
        }
    }
    fn visit_impl_item_fn_mut(&mut self, node: &mut syn::ImplItemFn) {
        if node.sig.ident == self.target {
            node.block = self.new_block.clone();
            self.replaced = true;
        }
    }
}
