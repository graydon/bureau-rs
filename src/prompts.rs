//! Prompt text constants and builders.
//!
//! The long-form prompt text lives as markdown files in `src/prompts/`
//! and is baked into the binary via `include_str!`. This module loads
//! the relevant template, applies any `{var}` substitutions, and returns
//! the rendered string. The engine module stays focused on orchestrator
//! flow.

use crate::graph::Stage;
use crate::tools::{PromptLimits, Role};

/// Minimal template substitution: replaces each `{key}` occurrence in
/// `template` with `value`. Used to interpolate runtime limits (e.g.
/// `{max_file}`, `{max_spec}`) into prompts loaded from `.md` files via
/// `include_str!`. Not a full templating language — just a search-and-
/// replace, so authors should pick placeholder names that won't appear
/// elsewhere in the prose.
fn render(template: &str, vars: &[(&str, &str)]) -> String {
    let mut s = template.to_string();
    for (k, v) in vars {
        let placeholder = format!("{{{k}}}");
        s = s.replace(&placeholder, v);
    }
    s
}


pub(crate) fn truncate_args_for_display(args: &str, max: usize) -> String {
    if args.len() <= max {
        return args.to_string();
    }
    let mut end = max;
    while end > 0 && !args.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}…  [TRUNCATED — {} bytes total; the args you sent are not echoed back in full]",
        &args[..end],
        args.len()
    )
}

/// Build the focused system prompt for a forced-retry attempt. Lists the
/// unresolved failures with truncated args and tells the model exactly
/// what to do.
pub(crate) fn retry_preamble(
    role: Role,
    stage: Stage,
    failures: &[(String, String, String)],
    attempt: u32,
    remaining: u32,
    args_cap: usize,
) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "# RETRY · {stage} · {role:?} (attempt {attempt}, {remaining} retries remaining after this)\n\n"
    ));
    s.push_str(
        "Your previous turn left tool calls in a FAILED state. The framework will not \
         accept this stage as complete until each is either retried successfully or \
         explicitly abandoned with a reason in your final message. Process every \
         failure below.\n\n",
    );
    s.push_str(
        "For each failed call you must do ONE of:\n\
         1. **Retry** the same tool with corrected arguments. Read the error message — \
            it tells you exactly what's wrong.\n\
         2. **Abandon** the call and explain in your end-of-turn message why it's not \
            actually needed (one sentence per abandoned call).\n\n",
    );
    s.push_str(
        "Note on truncated args: the args you sent are shown only as a stub for \
         identification. For `submit_*` tools, do NOT try to reconstruct the truncated \
         text from the stub — re-derive the full content from the spec / dep ifaces / \
         tests in the context document, then submit it fresh.\n\n",
    );
    s.push_str("## Failed calls to address\n\n");
    for (i, (tool, args, err)) in failures.iter().enumerate() {
        let args_display = truncate_args_for_display(args, args_cap);
        s.push_str(&format!(
            "{}. **`{}`** — error: {}\n   args: `{}`\n\n",
            i + 1,
            tool,
            err,
            args_display
        ));
    }
    s
}
/// Extract a one-line description from a markdown problem statement: the
/// first non-blank, non-heading paragraph (joined to a single line, trimmed
/// to ~200 chars). Falls back to the first non-blank line.
pub(crate) fn problem_first_paragraph(md: &str) -> String {
    let mut buf = String::new();
    for line in md.lines() {
        let t = line.trim();
        if t.is_empty() {
            if !buf.is_empty() {
                break;
            }
            continue;
        }
        if t.starts_with('#') {
            // Skip headings until we hit prose.
            continue;
        }
        if !buf.is_empty() {
            buf.push(' ');
        }
        buf.push_str(t);
    }
    if buf.is_empty() {
        // Fallback: first non-blank line, even if it's a heading.
        for line in md.lines() {
            let t = line.trim().trim_start_matches('#').trim();
            if !t.is_empty() {
                buf = t.to_string();
                break;
            }
        }
    }
    if buf.is_empty() {
        return "Project root.".to_string();
    }
    if buf.len() > 200 {
        let mut end = 197;
        while !buf.is_char_boundary(end) {
            end -= 1;
        }
        buf.truncate(end);
        buf.push_str("...");
    }
    buf
}

pub(crate) fn role_user_prompt(stage: Stage, role: Role) -> String {
    match (stage, role) {
        (s, Role::Writer) => format!(
            "Do the {s} stage for this node using the slot-filler tool(s). End with a one-line \
             summary."
        ),
        (s, Role::Critic) => format!(
            "Critique the writer's {s}-stage output. Call submit_critique exactly once with a \
             concrete `issues` list (empty list = nothing to fix, the framework will skip the \
             reviser and judge)."
        ),
        (s, Role::Reviser) => format!(
            "Address each critic point for the {s} stage. End with a one-line summary of the changes."
        ),
        (s, Role::Judge) => format!(
            "Verify the reviser addressed each critic point for the {s} stage. Call \
             submit_verdict exactly once."
        ),
        (_, Role::QuickFixer) => "Fix the compile / test errors listed in the system prompt. \
             Use read_file / write_file / write_file_range / apply_patch, re-check with the \
             cargo_* tool, and stop as soon as the gate passes."
            .to_string(),
    }
}
pub(crate) fn judge_block(stage: Stage) -> String {
    let upper = match stage {
        Stage::Architect => "ARCHITECT",
        Stage::Spec => "SPEC",
        Stage::Iface => "IFACE",
        Stage::Tests => "TESTS",
        Stage::Impl => "IMPL",
        Stage::Debug => "DEBUG",
    };
    let cargo_tool = match stage {
        Stage::Iface => "`cargo_check`",
        Stage::Tests => "`cargo_test_no_run`",
        Stage::Impl | Stage::Debug => "`cargo_test`",
        Stage::Architect | Stage::Spec => "(no cargo gate)",
    };
    render(
        include_str!("prompts/judge.md"),
        &[("UPPER", upper), ("cargo_tool", cargo_tool)],
    )
}
pub(crate) fn quickfix_preamble(stage: Stage) -> String {
    let gate = match stage {
        Stage::Iface => "`cargo_check`",
        Stage::Tests => "`cargo_check` and `cargo_test_no_run`",
        Stage::Impl | Stage::Debug => "`cargo_check` and `cargo_test`",
        Stage::Architect | Stage::Spec => "(no gate)",
    };
    render(
        include_str!("prompts/quickfix.md"),
        &[("stage", stage.as_str()), ("gate", gate)],
    )
}
/// The universal preamble — stable across every call in the entire run.
/// Used as the SYSTEM prompt so it caches across nodes, stages, and
/// roles. The per-(stage,role) block lives in the user prompt instead,
/// see `role_block()`.
pub(crate) fn universal_preamble() -> &'static str {
    include_str!("prompts/common.md")
}

/// The per-(stage,role) preamble block. Lives in the USER prompt, after
/// the context document, so node-stable context bytes prefix this and
/// the prompt cache survives across stages on the same node up to the
/// point where this block starts to differ.
pub(crate) fn role_block(stage: Stage, role: Role, limits: PromptLimits) -> String {
    let max_file = limits.max_file_lines.to_string();
    let max_spec = limits.max_spec_section_lines.to_string();
    let vars = &[
        ("max_file", max_file.as_str()),
        ("max_spec", max_spec.as_str()),
    ];

    match (stage, role) {
        // ---- ARCHITECT ----
        (Stage::Architect, Role::Writer) => {
            render(include_str!("prompts/architect_writer.md"), vars)
        }
        (Stage::Architect, _) => {
            include_str!("prompts/architect_nonwriter.md").to_string()
        }

        // ---- SPEC ----
        (Stage::Spec, Role::Writer) => render(include_str!("prompts/spec_writer.md"), vars),
        (Stage::Spec, Role::Reviser) => render(include_str!("prompts/spec_reviser.md"), vars),
        (Stage::Spec, Role::Critic) => include_str!("prompts/spec_critic.md").to_string(),
        (Stage::Spec, Role::Judge) => judge_block(Stage::Spec),

        // ---- IFACE ----
        (Stage::Iface, Role::Writer) | (Stage::Iface, Role::Reviser) => {
            render(include_str!("prompts/iface_writer.md"), vars)
        }
        (Stage::Iface, Role::Critic) => include_str!("prompts/iface_critic.md").to_string(),
        (Stage::Iface, Role::Judge) => judge_block(Stage::Iface),

        // ---- TESTS ----
        (Stage::Tests, Role::Writer) | (Stage::Tests, Role::Reviser) => {
            render(include_str!("prompts/tests_writer.md"), vars)
        }
        (Stage::Tests, Role::Critic) => include_str!("prompts/tests_critic.md").to_string(),
        (Stage::Tests, Role::Judge) => judge_block(Stage::Tests),

        // ---- IMPL ----
        (Stage::Impl, Role::Writer) | (Stage::Impl, Role::Reviser) => {
            render(include_str!("prompts/impl_writer.md"), vars)
        }
        (Stage::Impl, Role::Critic) => include_str!("prompts/impl_critic.md").to_string(),
        (Stage::Impl, Role::Judge) => judge_block(Stage::Impl),

        // ---- DEBUG ----
        (Stage::Debug, Role::Writer) | (Stage::Debug, Role::Reviser) => {
            render(include_str!("prompts/debug_writer.md"), vars)
        }
        (Stage::Debug, Role::Critic) => include_str!("prompts/debug_critic.md").to_string(),
        (Stage::Debug, Role::Judge) => judge_block(Stage::Debug),

        // QuickFixer — same preamble for every stage; the specific gate
        // and the errors to address come from the cycle context block.
        (_, Role::QuickFixer) => quickfix_preamble(stage),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_limits() -> PromptLimits {
        PromptLimits {
            max_file_lines: 600,
            max_spec_section_lines: 800,
        }
    }

    #[test]
    fn problem_first_paragraph_skips_headings() {
        let md = "# Problem: A samba-equivalent server\n\n\
                  Build a Rust workspace that reimplements a substantial subset of \
                  Samba — SMB/CIFS file serving, NetBIOS, etc.\n\n\
                  More text.";
        let p = problem_first_paragraph(md);
        assert!(p.starts_with("Build a Rust workspace"), "got: {p}");
        assert!(!p.contains('#'));
    }

    #[test]
    fn problem_first_paragraph_truncates_long_text() {
        let md = format!("Lorem ipsum {}", "dolor sit amet ".repeat(40));
        let p = problem_first_paragraph(&md);
        assert!(p.len() <= 200, "len was {}", p.len());
        assert!(p.ends_with("..."));
    }

    #[test]
    fn problem_first_paragraph_falls_back_to_heading_when_no_prose() {
        let p = problem_first_paragraph("# Just a Title\n");
        assert_eq!(p, "Just a Title");
    }

    #[test]
    fn problem_first_paragraph_handles_empty_input() {
        assert_eq!(problem_first_paragraph(""), "Project root.");
        assert_eq!(problem_first_paragraph("   \n   \n"), "Project root.");
    }

    #[test]
    fn role_preamble_iface_actor_forbids_mod_and_directs_to_super_public() {
        let block = role_block(Stage::Iface, Role::Writer, test_limits());
        assert!(
            block.contains("`mod`") && block.to_lowercase().contains("forbidden"),
            "iface actor block should mention `mod` is forbidden: {block}"
        );
        assert!(
            block.contains("super::public"),
            "should direct to super::public"
        );
        // The snake_case reminder lives in the universal preamble (system
        // prompt) rather than the per-(stage,role) block. After the
        // cache-friendly split, those are two separate functions.
        assert!(
            universal_preamble().contains("snake_case"),
            "universal preamble should remind about snake_case node names"
        );
    }

    #[test]
    fn role_preamble_spec_actor_pushes_decompose_for_large_missions() {
        let p = role_block(Stage::Spec, Role::Writer, test_limits());
        assert!(
            p.contains("decompose"),
            "spec actor preamble must mention decompose"
        );
        assert!(
            p.to_lowercase().contains("snake_case"),
            "spec actor preamble must mention snake_case names"
        );
    }

    #[test]
    fn role_preamble_interpolates_limits_from_config() {
        let limits = PromptLimits {
            max_file_lines: 777,
            max_spec_section_lines: 999,
        };
        let iface = role_block(Stage::Iface, Role::Writer, limits);
        assert!(
            iface.contains("777"),
            "iface actor preamble should mention max_file_lines: {iface}"
        );
        let spec = role_block(Stage::Spec, Role::Writer, limits);
        assert!(
            spec.contains("999"),
            "spec actor preamble should mention max_spec_section_lines: {spec}"
        );
    }

    #[test]
    fn role_preamble_universal_rules_no_longer_dump_all_tool_names() {
        let p = role_block(Stage::Impl, Role::Writer, test_limits());
        assert!(
            !p.contains("submit_spec_public") && !p.contains("submit_spec_private"),
            "impl writer should not see spec tools in its preamble: {p}"
        );
        assert!(
            !p.contains("submit_verdict"),
            "impl writer should not see submit_verdict in its preamble: {p}"
        );
    }

    #[test]
    fn spec_reviser_warns_against_changelog_in_spec_body() {
        let p = role_block(Stage::Spec, Role::Reviser, test_limits());
        let lc = p.to_lowercase();
        assert!(
            lc.contains("changelog") || lc.contains("meta-narrative") || lc.contains("clean spec"),
            "spec reviser preamble must call out 'no changelog/meta-narrative': {p}"
        );
    }
}
