# Rust Bureau

This is a further experimental development of https://github.com/graydon/bureau 

TL;DR: a rust program that talks to LLM inference endpoints directly and uses them to synthesize
another rust program (of your direction), but with a lot of opinions and constraints imposed,
trying to explore the design space of "custom harnesses".

## Questions to explore

The point is to explore/learn a few things simultaneously:

### Q1. Do LLMs do better at waterfall-style development than humans?

The hypothesis here is that maybe waterfall -- highly structured, top-down, specs-interfaces-tests-code -- might actually be suited to the strengths and weaknesses of LLMs:

  - Strengths
    - Patience, persistence, speed: they don't mind writing/refining a program 4 or 5 times, and you as an operator won't really care about the time that takes because it's still 100x faster than if humans did it.
    - Controllable variability in performance: you can literally ask for a smarter one to do an difficult/guidance task and a dumber one to do a rote/subordinate task. Might lend itself to hierarchical delegation.
    - Good attention to detail _in context_: up to a threshold, they can remain very sharp about what they notice and can just "eyeball" things that humans literally can't because our working memory is worse. So refining or implementing specs, looking for inconsistencies, writing tests for long lists of cases can all be done quite well (until the context breaks down).
  - Weaknesses
    - Performance collapse past context limits. Argues for two sub-aspects:
      - Structured context / information hiding for each call: give model a specific thing to do and everything it should need to know to do it, and _nothing else_. Benefits from modularity, explicit dependencies.
      - Hierarchical decomposition: do a high level thing, then split it into pieces and expand each one _in separate contexts_, and then repeat.
    - Hallucination / plausible-but-wrong output. Argues for multiple passes and revisions that look for inconsistencies, contradictions, missing content, pointless content, etc. Easier to schedule such tasks when each phase has defniite inputs, outputs, expectations, relationships to other existing artifacts.

### Q2. Do specialized/custom harnesses have advantages over general ones?

A specialized harness in this case means we're not running claude/codex/gemini/opencode/pi/whatever, we're
just calling LLM inference endpoints directly, with context and prompts we choose, in the order we choose.

  - Pro:
    - Cost! If we're being charged frontier lab API rates anyway, no discount for using fronier lab harness, might as well use openrouter and cheaper models when possible.
    - Speed! Cheaper models can also be much much faster models. You can call cerebras or groq hardware.
    - More control over details:
      - Context sent (skip "compaction", just assemble what's needed / start new contexts as needed)
      - Models used for each task, schedule of escalation/supervision between models and tasks
      - Set of tools available, specialization to a given task
      - Semantics of merging/reconciling data from multiple calls
      - Semantics of retries, rollbacks, data dependencies, data consistency model
    - Faster/simpler iteration and parallelism, don't need to start a whole 3rd party harness to "launch an agent"
    - More insight into what's going on, easy to add debug views of everything
  - Con:
    - Cheaper models are kinda dim-witted and you might need the frontier ones anyways
    - Ignoring mountains of work done by frontier harnesses in care and feeding of smart models
    - Have to write your own stupid plumbing bits (retries, backoff, loop-detection, context assembly)
      - Mid-level frameworks do some of this, eg. https://rig.rs/
    - Intentionally-non-general agents with intentionally-limited tools can find themselves unable to accomplish tasks
    
### Q3. Are there advantags to PL-specific tools and harnesses?

This program _only_ builds Rust programs, and it only builds a specific shape of Rust program.

  - The LLM
    - Describes a conceptual high-level tree of modules
    - Reads/writes/modifies individual spec-items or code-items as requested
    - Reads/responds to error messages
  - The harness
    - Builds the tree into a set of spec and source files deterministically
    - Structures the source files in terms of crates, modules, interfaces, tests, implementations
    - Limits what types of items can go in each file and which modules can see which others
    - Schedules work to match the PL's visibility/dependency structure
    - Runs all the build and test commands, gathers all the diagnostics
    - Presents tools to summarize/read/write/replace/patch individual items in individual modules
  - Why is this better?
    - Enforces information hiding (minimize context size and tokens, increase max project scale)
    - Enforces test-before-impl (avoid test-the-impl / impl-the-test misbehaviour)
    - Allows "perfect" parallel scheudling (semantic-dependency order)
    - Allows "perfect" consistency model (no concurrent-edit conflicts)
    - Avoids wasting time and tokens with predictable tool-calls
 
 ## Evaluation

 So far in evaluation terms I think:
 
  - 1 is a maybe. More structure might help, might just send the models off on long pointless writing quests. The biggest thing to try to attenuate is their _productivity_. They love generating output, which is frequently a liability. Getting them to generate less is easy enough; but if you want them to generate less-locally while generating more-globally (i.e. build a large system, but don't build a baroque pile of spaghetti) it's harder. This remains hard to steer and hard to evaluate.

  - 2 is a clearer yes. Still might not be a definite yes, but it seems like writing-your-own brings a _lot_ of benefits, and more to the point it's not clear to me that _not_ writing your own avoids a lot of the problems. Frontier lab harnesses are also buggy garbage, so if your own is buggy garbage too, but cheaper and faster and easier to debug and control .. is that really much worse?
  
  - 3 seems like it's also a probably-yes, though somewhat connected to 1. I think if you limit the tooling to PL-specific you likely _need_ to limit the development structure somewhat, just because there's such an open-ended set of things you can do with a PL in the most general case. But this experiment has limited both very strictly, partly for experimental reasons and partly because it's easier than trying to support all possible uses for PL-specific tools. But I did not, for example, embed `rust-analyzer`; I just used `syn`.
