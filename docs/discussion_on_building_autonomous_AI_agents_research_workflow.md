# Question
I am using AI coding agents such as OpenAI Codex GPT 5.2 and Claude Code in the VS code to work on various research projects (not software engineering or coding to build apps. I am a researcher at a university). But I feel like I am not harnessing the full potential of agentic research and coding where multiple agents work in parallel tacking different tasks of the research while coordinating with each other effectively and efficiently to avoid any contradiction, or clash, or conflict. For this purpose apparently orchestration is important. I want to automate this entire process as much as possible so that minimum intervention can happen only at critical points of the job.

Take the attached research plan I have. This is an end to end research plan for a research project I am about to start. Currently I have not started anything except this plan. Now I want to create an AI agent system (using the same infrastructure where I use claude code and Codex GPT 5.2 inside VS Code, it is all membership based, not API call). Currently my workflow is sequential. I tackle one task at a time, the AI agent finishes the task, I review or another AI agent review it, then carry out revision until it is fully implemented. We then move to the next task. 

Then I started wondering: Can we improve this workflow, carry out different tasks parallel concurrently with multiple agents working simultaneously, coordinating with each other to avoid any collusion/conflict/clash with minimum intervention from me? It seems like there are people or organizations have already developed such workflow to drastically speed up the process while having great results. 

Another example is that the agents stop working when I call it a day and go home. Can we setup a system where the agents continue work on different tasks of a job continuously even when I am not in front of the screen? What kind of setup would achieve that? For instance, for the research plan attached, let's say data collection is a big job with many tasks. How can I setup a system where multiple AI agents work on all different tasks within this job in parallel while coordinating with each other continuously and stop only when the job is finished or faced challenges that cannot overcome on their own, and it needs human intervention? The data analysis can be another job with many tasks and multiple AI agents work on those tasks simiultaneously again. 

We are not trying to invent the wheel. I believe there are already frameworks that we can utilize, if not then we can build one for myself. The bigggest challenge for me is to coordinate those agents while they work on different tasks without changing or interrupting or inverting other agents work. For instance, for the data collection job of the research plan attached, how can we have a system for coordination such that agents work on different tasks of this data collection will work on their assigned tasks continuously autonomously while coordinating with each other to have a compatible, integrated data collected as in the plan? 

As I said people might have created such workflows, there might be github repositories we can utilize or if needed we can create one for myself. The difference for me is that I am a researcher, not programmer or software engineer who builds apps, softwares, etc. I want to create a system/workflow for doing research, end to end, using multiple AI agents concurrently, continuously until the job is finished. For instance, one of the resources below provide details about how they created an explorer from scratch using long running coding agents without intervention. The agents worked non-stop for about a week to finish the task. They also provided the Github repo for that. 

The ultimate goal is to have an autonomous agentic research workflow (kind AI scientists team) where multiple agents working on a research project (empirical, mathematical modeling) in parallel continuously while coordinating with each other effectively to avoid any redundancy, any clash, or any other potential issue might rise. I don't like my current workflow, in which I constantly babysit each agent, launch tasks manually one by one sequentially. 

Please do a thorough and deep search first to understand what is available out there given the context and goal I described above. Then think critically about my use cases and my goal. Only after you have done thorough research, think through the task, then provide what you come up with. We need a system, a workflow to run multiple agents concurrently, in parallel, continuously with minimum intervention and effective coordination to finish a job of a research project, or even finish the first round of the entire research project. This job consists of multiple tasks, maybe dozens or hundreds. The end goal is to drastically improve my current efficiency in terms of how I have been using those AI agents while maintaining quality.

Take all the time you have. I really drastically improve my workflow and productivity. I don't like what I am currently doing. 

Some resources that might help:
1. https://simonwillison.net/2025/Oct/5/parallel-coding-agents/
2. https://steipete.me/posts/just-talk-to-it
3. https://steipete.me/posts/2025/shipping-at-inference-speed
4. https://cursor.com/blog/agent-best-practices
5. https://cursor.com/blog/scaling-agents

# Answer:

## Reality check first: what you want is possible, but not the way most people imagine

### 1) “Multiple agents editing the same repo at the same time” is a trap

If agents share a workspace and freely edit overlapping files, you will get:

* merge conflicts (easy),
* *definition drift* (harder), and
* contradictory “truths” (hardest: e.g., two different STR formulas, two different rollup inclusion rules).

Cursor’s own scaling write-up basically says they learned this the hard way: shared coordination mechanisms (e.g., a shared file + locks) became bottlenecks and failure points; they moved toward clearer hierarchy/roles and simpler systems with explicit “judge” evaluation. ([Cursor][1])

### 2) The proven pattern is **isolation + integration**

You want the same operating model that makes large engineering teams work:

* Each agent works in isolation (branch/worktree/environment).
* A supervisor assigns tasks and controls interfaces.
* A reviewer/judge validates outputs against tests/acceptance criteria.
* Only then do you merge/integrate.

That’s how Simon Willison runs “parallel coding agents” too: separate checkouts/worktrees, parallel tasks, then integrate. ([Simon Willison’s Weblog][2])

### 3) “Agents run when I’m away” requires an always-on host

If your laptop sleeps, the agents stop. Period.
To work overnight/weekend you need one of:

* a desktop/workstation that stays on,
* a remote server/VM you SSH into (VS Code Remote SSH), and you run agents inside `tmux` or system services, or
* a cloud-agent platform (Cursor cloud agents, etc.) that runs remotely and notifies you with PRs. ([Cursor][3])

---

## What exists right now that matches your constraints (subscription CLI/VS Code, not API coding)

### A. **AGENTS.md**: the “team contract” file that stops agent chaos

This is not a toy. It’s emerging as a standard way to tell agents how to behave in a repo: build steps, conventions, PR rules, safety boundaries, and—critically—*what they own vs don’t own*. ([AGENTS][4])

The *best* example for your purpose is the FastRender browser experiment repo, which used an `AGENTS.md` with:

* explicit workstreams (“pick one workstream; work can proceed in parallel”),
* “non-negotiables,”
* resource limits and safety rules,
* regression/testing philosophy,
* and strict scope boundaries. ([GitHub][5])

This is exactly the kind of coordination artifact your research repo needs.

### B. Claude Code can run headless and be script-orchestrated

Claude Code supports non-interactive, programmatic runs via `claude -p`, with:

* `--allowedTools` to auto-approve tools (so it doesn’t pause),
* `--output-format json` for structured outputs (so a supervisor script can parse status),
* `--continue` / `--resume` using session IDs (so long-running workflows can persist). ([Claude Code][6])

This is a key enabler for “agents keep working when I’m not there.”

### C. Codex CLI supports scripting / automation

Codex CLI is positioned as a local coding agent, included with ChatGPT plans, and explicitly supports scripting via an `exec` command (“Automate repeatable workflows by scripting Codex with the exec command”). ([OpenAI Developers][7])

So you can run Codex non-interactively as a worker in a larger orchestrated system.

### D. Real orchestration frameworks that work with CLI agents (not just API frameworks)

#### 1) **AWS CLI Agent Orchestrator (CAO)** (open source)

This is the closest thing to what you described *as a productized workflow*:

* hierarchical supervisor + specialized worker agents,
* workers run in isolated `tmux` sessions,
* explicit orchestration patterns: “handoff” (sync), “assign” (async parallel), and “send message” (communication),
* scheduled runs (cron-like) so workflows run unattended. ([Amazon Web Services, Inc.][8])

It supports Claude Code as a provider today; Codex CLI support is on their roadmap. ([Amazon Web Services, Inc.][8])

#### 2) **myclaude** (open source) — specifically “Claude Code + Codex + Gemini” multi-backend workflow

This repo is directly aimed at your exact situation:

* Orchestrator role: Claude Code (planning/context/verification)
* Executor role: a wrapper that can run Codex/Claude/Gemini backends
* It even shows a **parallel task format** for concurrent execution. ([GitHub][9])

(Important: it’s AGPL-licensed; that’s fine for personal/internal use, but check implications if you plan to redistribute.)

#### 3) **sow (System of Work)** (Claude Code oriented)

This one is an opinionated workflow system:

* 5-phase lifecycle (Discovery → Design → Implementation → Review → Finalize),
* “one project per branch,”
* “multi-session concurrency via git worktrees,”
* “zero-context resumability” (resume from disk state). ([GitHub][10])

#### 4) Cursor’s own “cloud agents” model (if you’re willing to use Cursor)

Cursor describes agents that can:

* clone your repo,
* create a branch,
* open a PR,
* and notify you—so you can review from your phone. ([Cursor][3])

That’s basically “agents keep working when you go home” as a managed service.

---

## The coordination model that will actually solve your “clash/conflict/contradiction” problem

You don’t need agents to “talk to each other more.” You need **mechanisms** that make conflicts structurally unlikely.

Here are the coordination primitives that work in practice:

### 1) **Workstream partitioning + file ownership**

Borrow the FastRender approach: define workstreams where *each workstream owns specific directories + artifacts*, and explicitly “does NOT own” others. ([GitHub][5])

For your research plan, natural workstreams are:

* **W0 Protocol/Definitions (owner: “Research Lead / Protocol Agent”)**
  Owns: STR definition, inclusion criteria, key dates, unit conventions.
* **W1 Off-chain ETL (growthepie, L2Beat, Blobscan)**
  Owns: ingestion scripts, raw snapshotting, schemas for off-chain pulls.
* **W2 On-chain ETL (Ethereum L1 extraction)**
  Owns: BigQuery/SQL, decoding blobs, fee decomposition.
* **W3 Attribution registry (rollup address mapping)**
  Owns: `rollup_registry_v*.csv/json`, change log, evidence links.
* **W4 Metric construction + unit tests**
  Owns: STR computation functions, accounting identities.
* **W5 Validation/reconciliation report**
  Owns: cross-source comparisons, error bands, “trusted series” rules.
* **W6 Empirical analysis + figures**
  Owns: regressions, break tests, plots.
* **W7 Policy counterfactual (EIP-7918 module)**
  Owns: counterfactual fee series + sensitivity.
* **W8 Writing/deck**
  Owns: report text, deck, narrative.

This aligns almost one-to-one with your plan’s phases 0–7. 

### 2) **Isolation by default: branches/worktrees per task**

Every agent gets:

* its own git branch,
* ideally its own git worktree (separate directory) so they cannot step on each other locally.

Willison’s parallel-agent workflow leans heavily on that isolation. ([Simon Willison’s Weblog][2])
`sow` also explicitly uses git worktrees to enable concurrency safely. ([GitHub][10])

### 3) **A “single source of truth” for definitions**

In research, contradictions usually come from definition drift. Your repo should have a *single canonical*:

* `protocol/definitions.yaml` (STR formula(s), units, denominators, date cutoffs),
* `protocol/inclusion.yaml` (rollup list rules),
* `registry/rollup_registry.csv` (attribution map, versioned),
* `contracts/schemas/panel_schema_str_v1.yaml` (analysis dataset schema).

Workers don’t edit those casually; they propose changes via PR, and the Protocol Owner (human or protocol-agent) must approve.

### 4) **Automated gates, not manual babysitting**

You want minimum intervention. That means:

* every task has **acceptance tests**,
* a reviewer/judge agent runs them,
* merges happen only when tests pass.

Cursor’s “judge” idea is exactly this: separate “do work” from “verify work.” ([Cursor][1])

### 5) **Communication is via artifacts, not chat**

Chat between agents is not a reliable coordination substrate. The reliable substrate is:

* PRs,
* task status files,
* changelogs,
* schemas,
* tests,
* and immutable snapshots of data.

That’s also why AGENTS.md exists: it’s durable instruction state. ([AGENTS][4])

---

## A concrete system/workflow for *your* research plan

### Goal: “AI scientists team” that runs continuously with minimal intervention

Here’s the blueprint that gets you there without inventing new science.

## Layer 0: Repo structure (the “operating system” for the agents)

Create a repo where outputs are first-class:

```
research/
  AGENTS.md
  protocol/
    protocol.md
    definitions.yaml
    inclusion.yaml
  tasks/
    backlog.yaml
    T001_protocol_lock.md
    T010_etl_growthepie.md
    T020_etl_l2beat.md
    T030_etl_blobscan.md
    T040_etl_onchain.md
    T050_rollup_registry_v1.md
    T060_metric_str.md
    T070_validation.md
    T080_analysis_trends.md
    T090_counterfactual_7918.md
    T100_report_draft.md
  registry/
    rollup_registry_v1.csv
    rollup_registry_CHANGELOG.md
    sources.md
  data/
    raw/        # immutable snapshots (date-stamped)
    staging/
    processed/
  etl/
    growthepie/
    l2beat/
    blobscan/
    onchain/
  analysis/
    notebooks/
    src/
    tests/
  reports/
    figures/
    paper/
    deck/
  Makefile (or task runner)
  .github/workflows/ (optional CI)
```

Key idea: agents don’t “remember” your plan; they read it from the repo every run.

## Layer 1: AGENTS.md (your coordination contract)

You should adopt the same concept FastRender uses: repo-wide rules + workstream docs, plus explicit “what counts” and “stop conditions.” ([GitHub][5])

AGENTS.md is broadly supported across tools and can be nested per directory. ([AGENTS][4])

I’d implement **nested AGENTS.md**:

* root `AGENTS.md`: global rules
* `etl/AGENTS.md`: ETL conventions
* `analysis/AGENTS.md`: stats conventions
* `registry/AGENTS.md`: attribution rules

This sharply reduces clashes.

## Layer 2: Orchestration engine (pick one of these)

### Option 1 (best fit if you want “real orchestration” today): **CAO + Claude Code workers**

Use CAO for:

* spawning multiple Claude Code sessions in tmux,
* assigning tasks asynchronously,
* scheduling runs (overnight),
* messaging workers, and
* keeping a supervisor as the “brain.” ([Amazon Web Services, Inc.][8])

Codex can still be used, but likely as a tool invoked by a Claude worker (until CAO adds native Codex support).

### Option 2 (best fit if you explicitly want Claude+Codex division of labor): **myclaude**

myclaude is explicitly built around:

* Claude Code as orchestrator,
* Codex/Claude/Gemini as execution backends,
* parallel execution spec. ([GitHub][9])

This is closest to “Claude coordinates, Codex codes” as a system.

### Option 3 (lowest friction, immediate payoff): **DIY parallelism with git worktrees**

This is Willison’s approach: multiple agents in parallel, each in its own checkout/worktree. ([Simon Willison’s Weblog][2])

It won’t be “autonomous while away” unless you keep the machine on + use tmux, but it will immediately 2–4× your throughput with minimal setup.

---

## Layer 3: The “job controller” logic (how tasks flow without you babysitting)

You want a loop like this:

1. **Supervisor reads `tasks/backlog.yaml`** and selects ready tasks (dependencies satisfied).
2. For each ready task:

   * create a branch/worktree,
   * assign to a worker agent with a strict scope (directories + outputs),
   * provide only necessary context (task file + relevant protocol/definitions).
3. Worker produces outputs + updates task status + opens PR (or commits to branch).
4. Reviewer/judge agent:

   * runs `make test`, validation scripts, schema checks,
   * checks task acceptance criteria,
   * either marks “done” or “blocked with reason.”
5. Supervisor merges “done” tasks and re-plans.

CAO is literally designed for this type of pattern (supervisor/worker, tmux isolation, scheduled runs). ([Amazon Web Services, Inc.][8])
Cursor’s scaling notes argue for this sort of hierarchy and separation of roles. ([Cursor][1])

---

## How this maps to your plan: think in “parallelizable bundles”

Below is a concrete decomposition of your plan into parallel tasks that won’t clash.

### Bundle A: Protocol lock (W0) — fast, must be first, blocks nothing else

* Task: write `protocol/protocol.md` and `protocol/definitions.yaml`
* Acceptance: STR formula(s) + units (ETH-native primary) + rollup inclusion rule are explicit and frozen. 

While this is being drafted, you can already run Bundle B in parallel (ETL scaffolding) as long as schema is treated as provisional.

### Bundle B: Off-chain ETL (W1) — 3 agents, zero file overlap

* Agent B1: `etl/growthepie/` + snapshotter
* Agent B2: `etl/l2beat/` + snapshotter
* Agent B3: `etl/blobscan/` + snapshotter

Acceptance:

* each produces a dated snapshot in `data/raw/<source>/<YYYY-MM-DD>/...`
* each produces a normalized parquet in `data/staging/...`
* schema checks pass

Your plan explicitly relies on these three sources. 

### Bundle C: On-chain ETL (W2) — separate agent, deeper engineering

* Build SQL/python extraction for rollup-related L1 tx fees and blob fee fields post-4844.
* Output: daily aggregates + tx-level table keyed by tx hash (for audit).

Acceptance:

* reproducible query scripts + documented assumptions
* produces daily table with required fields (base fee burn, blob burn, tips decomposition) consistent with your formulas. 

### Bundle D: Attribution registry (W3) — can run in parallel with ETL

* Build `registry/rollup_registry_v1.csv` plus changelog.
* Use evidence links per entry.
* Acceptance: ≥90% of blob-carrying tx spend attributable post-Dencun (your plan’s criterion). 

This is where contradictions typically happen. The fix is **versioning + evidence + review gates**.

### Bundle E: Metric construction + tests (W4) — parallel once some data exists

* Build `analysis/src/metrics.py` (or similar) with:

  * STR,
  * burn vs tips decomposition,
  * identities as unit tests.
* Acceptance:

  * accounting identity tests pass,
  * output matches known totals within tolerance.

### Bundle F: Validation & reconciliation (W5) — a “judge” workflow

* Compare RentPaid across:

  * growthepie vs L2Beat vs on-chain series
* Output: `reports/validation.md` with error bands.

This is exactly where your “dashboard science” skepticism belongs. 

### Bundle G/H: Analysis + counterfactual + writing (W6/W7/W8) — can overlap after dataset stabilizes

* analysis agent runs regressions + figures,
* policy agent implements EIP-7918 counterfactual module,
* writer agent drafts report/deck based on figures.

---

## Continuous operation: how to run 24/7 with minimal intervention

### The simplest reliable setup

* Put the repo on an always-on machine (or remote VM).
* Run orchestration in `tmux`.
* Run a scheduled “tick” every N minutes to:

  * pick tasks,
  * run workers headless,
  * run judge checks,
  * update status.

CAO explicitly uses tmux-based session isolation and supports scheduled runs. ([Amazon Web Services, Inc.][8])

### Headless workers that won’t stop at a prompt

For Claude Code:

* `claude -p` runs non-interactively.
* `--allowedTools` removes permission prompts.
* `--output-format json` makes it machine-parseable.
* `--resume` / `--continue` enables long-running sessions. ([Claude Code][6])

For Codex:

* Codex CLI docs explicitly mention scripting via the `exec` command. ([OpenAI Developers][7])

### Minimal intervention points (define them explicitly)

Agents should stop and escalate only for:

* credentials/secrets needed,
* ambiguous *definition* change requests (anything in `protocol/` or `registry/`),
* failed validation tests with non-obvious resolution,
* requests to broaden scope (“should we include non-Ethereum DA?” etc),
* publication-quality narrative choices.

Everything else is “keep going.”

---

## The biggest quality lever: define “what counts” and “what does not count”

FastRender’s AGENTS.md is ruthless about this: avoid “vanity work,” avoid endless harness improvements, focus on measurable outcomes, and treat timeouts/hangs as bugs. ([GitHub][5])

For your project, the analogous “what counts” list should be:

**Counts**

* a new data source integrated with immutable snapshots + schema validation
* a new metric implemented with unit tests + accounting identity checks
* attribution registry improvements with evidence + coverage metric improvement
* a validation delta explained or resolved
* a figure generated from reproducible code

**Does NOT count**

* “refactoring” without improving reproducibility/validation
* adding a new dataset without snapshotting + schema + provenance
* any analysis result not tied back to locked definitions
* narrative writing before validation is done

This single change prevents agents from thrashing.

---

## A starter AGENTS.md tailored to your research plan (copy/paste)

```md
# AGENTS.md — Research Agent Operating Rules

## Mission
Implement the end-to-end research plan in /protocol/protocol.md:
data acquisition → attribution → metric construction → validation → analysis → report.

## Non-negotiables (coordination + research integrity)
1) Do not change metric definitions or inclusion rules silently.
   - Canonical definitions live in /protocol/definitions.yaml and /protocol/inclusion.yaml.
   - Propose changes via PR and explain why.

2) All third-party data pulls must be snapshotted immutably.
   - Write to /data/raw/<source>/<YYYY-MM-DD>/...
   - Never overwrite prior snapshots.

3) Every computed dataset must have a schema and tests.
   - Schemas in /schemas/
   - Tests in /analysis/tests/

4) Workstream ownership is strict:
   - ETL agents: only edit /etl/<source>/ and /data/ pipeline code
   - Registry agent: only edit /registry/
   - Metrics agent: only edit /analysis/src/metrics* and tests
   - Validation agent: only edit /reports/validation*
   - Writing agent: only edit /reports/paper and /reports/deck

5) Integration happens via PRs only. No direct commits to main.

## Commands
- Run unit tests: `make test`
- Run data validation: `make validate`
- Build dataset: `make build-panel`
- Generate figures: `make figures`

## Stop conditions (escalate to human)
- Missing credentials / access to BigQuery or APIs
- Conflicting definitions between sources
- Attribution ambiguity that materially changes results
- Validation fails with no clear fix
- Any requested change to /protocol/*

## Output expectations
Every task must end with:
- What changed (files)
- How to reproduce
- What tests ran and results
- Any remaining risks / assumptions
```

This is exactly the kind of durable coordination artifact AGENTS.md is meant for. ([AGENTS][4])

---

## What I would do if I were optimizing *your* workflow this week

### Day 1: Get parallelism immediately (no new framework)

1. Create 3–5 git worktrees (one per workstream) and open each in a separate VS Code window.
2. Put one agent per window.
3. Add the AGENTS.md + repo structure above.
4. Run with one supervisor window that only edits `tasks/` + `protocol/`.

This alone usually yields a 2–4× speedup with little risk. ([Simon Willison’s Weblog][2])

### Day 2–3: Add “overnight autonomy”

Pick **one** orchestrator path:

* If you want robust orchestration and can live with Claude-only workers: **CAO**. ([Amazon Web Services, Inc.][8])
* If you want Claude orchestrating and Codex executing in parallel: **myclaude**. ([GitHub][9])

Then:

* move runs to an always-on box + tmux,
* schedule periodic runs,
* enforce judge gates (`make test`, `make validate`) before merge.

### The “don’t kid yourself” constraint

Running “hundreds of agents for a week” like the Cursor browser experiment is not a normal individual setup; it’s an engineering+infra effort with strong contracts, heavy automation, and lots of evaluation machinery. Cursor themselves emphasize hierarchy, roles, and simplicity when scaling. ([Cursor][1])
You can absolutely get big wins without that scale by making coordination *structural* (branches, ownership, tests, snapshots).

---

## References (your preferred link format, in a code block)

```text
[Willison 2025](https://simonwillison.net/2025/Oct/5/parallel-coding-agents/ "Parallel coding agents")
[Steinberger 2025](https://steipete.me/posts/just-talk-to-it "Just Talk To It")
[Cursor 2025](https://cursor.com/blog/agent-best-practices "Agent best practices")
[Cursor 2025](https://cursor.com/blog/scaling-agents "Scaling agents")
[Anthropic 2026](https://code.claude.com/docs/en/headless "Run Claude Code programmatically")
[OpenAI 2026](https://developers.openai.com/codex/cli/ "Codex CLI")
[AWS 2025](https://aws.amazon.com/blogs/opensource/introducing-cli-agent-orchestrator-transforming-developer-cli-tools-into-a-multi-agent-powerhouse/ "Introducing CLI Agent Orchestrator")
[awslabs 2026](https://github.com/awslabs/cli-agent-orchestrator/ "cli-agent-orchestrator")
[agentsmd 2026](https://agents.md/ "AGENTS.md")
[wilsonzlin 2026](https://github.com/wilsonzlin/fastrender/blob/main/AGENTS.md "FastRender AGENTS.md")
[cexll 2026](https://github.com/cexll/myclaude "myclaude")
[jmgilman 2026](https://github.com/jmgilman/sow "sow")
```

---

If you want the fastest path to “minimum babysitting,” the single highest-leverage move is: **turn your research plan into a repo-native task system + AGENTS.md contracts + automated gates**, then run **CAO or myclaude** on an always-on host. Everything else is secondary.

[1]: https://cursor.com/blog/scaling-agents "Scaling long-running autonomous coding · Cursor"
[2]: https://simonwillison.net/2025/Oct/5/parallel-coding-agents/ "Embracing the parallel coding agent lifestyle"
[3]: https://cursor.com/blog/agent-best-practices "Best practices for coding with agents · Cursor"
[4]: https://agents.md/ "https://agents.md/"
[5]: https://github.com/wilsonzlin/fastrender/blob/main/AGENTS.md?raw=true "https://github.com/wilsonzlin/fastrender/blob/main/AGENTS.md?raw=true"
[6]: https://code.claude.com/docs/en/headless "https://code.claude.com/docs/en/headless"
[7]: https://developers.openai.com/codex/cli/ "https://developers.openai.com/codex/cli/"
[8]: https://aws.amazon.com/blogs/opensource/introducing-cli-agent-orchestrator-transforming-developer-cli-tools-into-a-multi-agent-powerhouse/ "https://aws.amazon.com/blogs/opensource/introducing-cli-agent-orchestrator-transforming-developer-cli-tools-into-a-multi-agent-powerhouse/"
[9]: https://github.com/cexll/myclaude "https://github.com/cexll/myclaude"
[10]: https://github.com/jmgilman/sow "https://github.com/jmgilman/sow"

---

# Follow-up Question
Two of my colleague who have been using Ai agent for research provided the following two responses when I presented them with the same question. Please read through them thoroughly, evaluate their practicality. Then evaluate critically if they contain anything new or novel (must be practical) that you have not considered and can be integrated into what you suggested to enhance and improve the approach you suggested. We want to get the best of them to improve what you suggested. We want to have a practical robust model that will get the job done, done with high quality. One thing to note that my membership is max and pro membership for both Claude code and OpenAI codex so that I rarely hit the usage limit, as one of the following responses mentioned. 

<response1>
This is a solved problem in the domain of **"Agentic Workflows,"** but to achieve it without becoming a full-time software engineer, you need to shift your mental model.

You are currently operating in a **"Chat-Based"** model (you talk, bot answers). To achieve your goal, you must move to a **"Filesystem-Based"** model (agents read files, do work, update files).

Below is the architecture for an **Autonomous Research Swarm** that fits your constraints (VS Code, Research focus).

### 1. The Core Constraint: Membership vs. Parallelism

Before building, we must address a critical bottleneck. You want to run multiple agents **in parallel** using your **Membership** (Claude Pro / ChatGPT Plus).

* **The Reality:** Membership accounts have strict "Rate Limits" (e.g., 50 messages every few hours). If you launch 3 agents at once, they will hit this limit in roughly 15 minutes, and your entire swarm will freeze.
* **The Solution:** You have two paths.
* **Path A (Membership-Based):** You must run agents **Sequentially** (Agent 1 finishes  triggers Agent 2). This is fully autonomous (runs while you sleep) but not parallel.
* **Path B (API-Based):** To run **Parallel** agents (true simultaneous work), you must switch the "Worker" agents to **API Keys** (Pay-As-You-Go). For text-based research, this is often cheaper than expected ($5–$15/month for heavy use), but it is the only way to bypass the "50 message" cap.



*The guide below assumes **Path B** for the "Parallel" requirement, but I will note how to do Path A.*

---

### 2. The Architecture: "The Digital Lab Bench"

To coordinate agents without you "babysitting" them, we replace your chat window with a **Master Coordination File**.

* **The Manager:** A `KANBAN.md` file in your project root. This is the "Brain."
* **The Workers:** Independent instances of **Claude Code (CLI)** or **Roo Code** running in background terminals.
* **The Environment:** **GitHub Codespaces** (or a Remote Server). This solves the "stopping when I go home" problem.

### 3. Step-by-Step Implementation

#### Phase 1: The Infrastructure (Solving "24/7 Operation")

Agents running on your laptop die when the laptop sleeps.

1. **Move to the Cloud:** Push your research plan to a private GitHub repository.
2. **Launch Codespaces:** On your GitHub repo page, click **Code  Create Codespace**.
* *Why:* This opens VS Code in your browser. It runs on a cloud server. You can close the tab, turn off your computer, and the terminal processes inside it **keep running**.



#### Phase 2: The Coordinator (The "Kanban" File)

Create a file named `RESEARCH_STATUS.md` in the root. This is the only way agents "talk" to each other.

**Copy this template:**

```markdown
# 🧪 Research Swarm Status: Ethereum L1 Take Rate

## 🚦 Global Status
**Current Phase:** Phase 1 (Data Collection)
**Active Agents:** 3

## 📋 Task Queue
**Instructions for Agents:**
1. Pick a task marked `[TODO]`.
2. Change mark to `[IN PROGRESS - AgentName]`.
3. Save outputs to your specific folder (e.g., `data/raw/growthpie/`).
4. When file is saved and verified, mark task `[DONE]`.

### Phase 1: Data Inventory
- [ ] **Task 1.1:** Fetch daily L2 fees/rent from Growthepie (Jan 2022-Present). Save to `data/raw/growthpie.csv`. (Agent: Data-Alpha)
- [ ] **Task 1.2:** Scrape L2Beat cost breakdown series. Save to `data/raw/l2beat.csv`. (Agent: Data-Beta)
- [ ] **Task 1.3:** Query Blobscan API for post-4844 blob metrics. (Agent: Data-Gamma)

### Phase 2: Analysis (Wait for Phase 1 DONE)
- [ ] **Task 2.1:** Create `rollup_registry.json` mapping L1 addresses to L2 names.
- [ ] **Task 2.2:** Compute "Settlement Take Rate" (STR) time series.

```

#### Phase 3: Launching the Swarm (The "Terminal Grid")

You cannot run parallel agents in the VS Code sidebar. You must use the **Terminal** with a tool called `tmux` (Terminal Multiplexer) which allows multiple shells to run at once.

1. **Open Terminal** in VS Code (Ctrl+`).
2. **Start the Lab:** Type `tmux new -s research_lab`.
3. **Split the Room:** Press `Ctrl+B`, then `"` (quote key) to split the screen. Do it again to get 3 panes.

**Pane 1: Agent Alpha (GrowthPie Specialist)**
Run this command (using Claude Code CLI):

```bash
claude -p "You are Agent Alpha. Read RESEARCH_STATUS.md. Your goal is Task 1.1. Write a Python script to fetch the data. Save it. Update the status file. Loop until the task is DONE." --dangerously-skip-permissions

```

* *Note: `--dangerously-skip-permissions` enables the "Autonomous" mode where it doesn't wait for your approval on every command.*

**Pane 2: Agent Beta (L2Beat Specialist)**

```bash
claude -p "You are Agent Beta. Read RESEARCH_STATUS.md. Your goal is Task 1.2. Execute independently. Save to 'data/raw/l2beat.csv'. Do not overwrite other files." --dangerously-skip-permissions

```

**Pane 3: Agent Gamma (Blobscan Specialist)**

```bash
claude -p "You are Agent Gamma. Read RESEARCH_STATUS.md. Your goal is Task 1.3. Query Blobscan API..." --dangerously-skip-permissions

```

#### Phase 4: The Handoff (Go Home)

1. **Detach:** Press `Ctrl+B`, then `d`. The terminal screen vanishes, but the agents are **still running**.
2. **Close VS Code:** You can now leave.
3. **Resume:** When you return, open VS Code/Codespaces, open the terminal, and type `tmux attach`. You will see exactly what they did while you were away.

---

### 4. Avoiding "Clashes" (Conflict Resolution)

In your prompt, you asked how to prevent them from interfering with each other. The answer is **Directory Isolation**.

Do not let agents work in the same folder. Structure your project like this:

```text
/project_root
  ├── RESEARCH_STATUS.md
  ├── /data
  │    ├── /agent_alpha_workspace/  <-- Only Alpha writes here
  │    ├── /agent_beta_workspace/   <-- Only Beta writes here
  │    └── /merged/                 <-- Only the "Manager" merges here

```

Explicitly instruct Agent Alpha: *"You are only allowed to write files inside `/data/agent_alpha_workspace`."* This makes it impossible for them to overwrite each other's work.

### 5. Summary: Your New Workflow

| Feature | Old Workflow | New "Agent Swarm" Workflow |
| --- | --- | --- |
| **Coordination** | You manually assigning tasks | **`RESEARCH_STATUS.md`** (Agents self-assign) |
| **Parallelism** | One task at a time | **Split Terminal (`tmux`)** running 3 CLI agents |
| **Uptime** | Stops when laptop sleeps | **GitHub Codespaces** (Runs 24/7 on cloud) |
| **Permissions** | "Approve" button clicking | **`--dangerously-skip-permissions`** (Full Autonomy) |

### Recommended Tools (2025 Stack)

Since you want to use existing frameworks:

1. **For the Agents:** **Claude Code (CLI)** or **Roo Code** (VS Code Extension with "Autonomous Mode").
2. **For the Environment:** **GitHub Codespaces**.
3. **For Complexity:** If you eventually want to write code, look at **CrewAI** (Python framework), but the method above requires **zero Python code** for the infrastructure—just terminal commands.
</response1>

<response2>
# Autonomous Multi-Agent Research Workflow System
## A Comprehensive Guide for Academic Researchers

*Tailored for empirical research workflows using Claude Code and OpenAI Codex*

---

## Executive Summary

Based on extensive research into current multi-agent orchestration practices, I've synthesized a practical system for running parallel AI agents on research projects. This guide provides:

1. **Architecture patterns** that actually work (planner-worker model)
2. **Tool recommendations** for your existing Claude Code + Codex setup
3. **Concrete implementation** for your Ethereum L1/L2 research project
4. **Infrastructure setup** for continuous, overnight agent execution

The key insight from my research: **the planner-worker architecture outperforms flat peer coordination**. Cursor's team ran agents for weeks building a browser from scratch (1M+ lines of code) using this pattern. The failed approaches used locks and democratic coordination - they became bottlenecks.

---

## Part 1: The Core Architecture That Works

### The Planner-Worker Model (Proven at Scale)

From Cursor's research running hundreds of agents for weeks:

```
┌─────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR                             │
│     (You or a meta-agent that monitors overall progress)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         PLANNER(S)                              │
│  - Explores codebase/data, creates task breakdown               │
│  - Can spawn sub-planners for specific areas                    │
│  - Does NOT execute - only plans and coordinates                │
│  - Outputs: Task specs in markdown with clear success criteria  │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│    WORKER 1     │  │    WORKER 2     │  │    WORKER N     │
│  (Claude Code)  │  │    (Codex)      │  │  (Claude Code)  │
│                 │  │                 │  │                 │
│ - Isolated task │  │ - Isolated task │  │ - Isolated task │
│ - Own worktree  │  │ - Own worktree  │  │ - Own worktree  │
│ - No coord with │  │ - No coord with │  │ - No coord with │
│   other workers │  │   other workers │  │   other workers │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          JUDGE                                  │
│  - Reviews completed work against success criteria              │
│  - Decides: merge, reject, or iterate                          │
│  - Triggers next planning cycle if needed                       │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Works Better Than Democratic Coordination

| Approach | Problem | Outcome |
|----------|---------|---------|
| **Flat/Democratic** | Agents avoid hard tasks, make small safe changes | Work churns without progress |
| **Lock-based** | Bottleneck; agents hold locks too long | 20 agents → throughput of 2-3 |
| **Planner-Worker** | Clear hierarchy, workers don't coordinate | Scales to 100s of agents |

---

## Part 2: Implementation for Your Research Project

### Mapping Your Research Plan to the Architecture

Your Ethereum L1/L2 research has 7 phases. Here's how to parallelize them:

```
PHASE 0: Protocol Lock (Sequential - You + 1 Agent)
├── Cannot parallelize - sets foundations for everything else
└── Output: Protocol document, data dictionary, analysis checklist

PHASE 1: Data Collection (HIGH PARALLELISM - 4-6 Agents)
├── Agent 1: growthepie API extraction → ETL pipeline
├── Agent 2: L2Beat cost series extraction → cross-check data
├── Agent 3: Blobscan API → blob usage metrics
├── Agent 4: BigQuery L1 data → raw chain extraction
├── Agent 5: Data validation scripts → automated checks
└── Coordinator: Schema harmonization (runs after others)

PHASE 2: Attribution Map (MEDIUM PARALLELISM - 2-3 Agents)
├── Agent 1: Build rollup registry from Blobscan metadata
├── Agent 2: L2Beat tracking context extraction
└── Agent 3: Official rollup docs scraping + validation

PHASE 3: Metric Construction (MEDIUM PARALLELISM - 2-3 Agents)
├── Agent 1: STR computation module
├── Agent 2: Burn/tips decomposition module
└── Agent 3: Unit tests for metric functions

PHASE 4: Validation (HIGH PARALLELISM - 3-4 Agents)
├── Agent 1: RentPaid triangulation (growthepie vs L2Beat)
├── Agent 2: On-chain validation (computed vs reported)
├── Agent 3: Fees cross-check (top 5 rollups)
└── Agent 4: Validation report generation

PHASE 5: Analysis (MEDIUM PARALLELISM - 3-4 Agents)
├── Agent 1: Trend tests (Mann-Kendall, OLS with HAC)
├── Agent 2: Structural break tests (Chow, Bai-Perron)
├── Agent 3: Elasticity regressions
└── Agent 4: Visualization generation

PHASE 6: Counterfactual (SEQUENTIAL with parallel sensitivity)
├── Main Agent: EIP-7918 mechanism implementation
└── Parallel: Sensitivity analysis across parameters

PHASE 7: Synthesis (LOW PARALLELISM - 2 Agents)
├── Agent 1: Technical report + LaTeX
└── Agent 2: Stakeholder deck
```

### Coordination Files Structure

Create a shared coordination system using files (not real-time communication):

```
project-root/
├── .orchestrator/
│   ├── master-plan.md           # Overall research plan (you maintain)
│   ├── active-tasks/            # Currently assigned tasks
│   │   ├── task-001-growthepie.md
│   │   ├── task-002-l2beat.md
│   │   └── ...
│   ├── completed-tasks/         # Done tasks with outputs
│   ├── blocked-tasks/           # Tasks needing human input
│   ├── state.json               # Machine-readable state
│   └── handoff-notes/           # Cross-agent context
├── data/
│   ├── raw/                     # Immutable data snapshots
│   ├── processed/               # Transformed data
│   └── schemas/                 # Data dictionaries
├── src/
│   ├── etl/                     # Data collection code
│   ├── analysis/                # Analysis scripts
│   └── validation/              # Validation modules
├── docs/
│   ├── phase-1-data.md          # Living documentation
│   ├── phase-2-attribution.md
│   └── ...
└── AGENTS.md                    # Instructions for all agents
```

### The AGENTS.md File (Critical for Coordination)

```markdown
# Research Project: Ethereum L1/L2 Take Rate Analysis

## Project Overview
This research tests whether Ethereum L1 captures durable economic rent from 
L2 scaling using Settlement Take Rate (STR) as the primary metric.

## Coordination Protocol

### Before Starting Any Task
1. Check `.orchestrator/active-tasks/` for your assigned task
2. Read the task specification completely
3. Check `.orchestrator/state.json` for dependencies
4. If dependencies not met, wait or notify in blocked-tasks

### During Work
1. Work ONLY on your assigned task - do not touch other areas
2. Commit frequently with descriptive messages
3. Update your task file's status section every ~30 minutes
4. If stuck >1 hour, write detailed blockers and move to blocked-tasks

### After Completing Task
1. Run all tests for your module
2. Update task status to "pending_review"
3. Write handoff notes if others depend on your output
4. Do NOT start new tasks without orchestrator assignment

## Code Standards
- Python 3.11+ with type hints
- Pandas for data manipulation
- pytest for testing (>90% coverage for core metrics)
- All data fetches must be idempotent and cacheable
- Log all API calls with timestamps

## Data Standards
- All timestamps in UTC
- Amounts in native units (ETH, wei) first, USD conversion secondary
- Snapshot all API responses before transformation
- Version all data artifacts with dates

## Communication
- NO real-time coordination between workers
- All coordination happens through files in .orchestrator/
- Questions go to blocked-tasks with @human tag
- Critical findings go to handoff-notes immediately
```

---

## Part 3: Tool Setup and Configuration

### Option A: tmux + Persistent Sessions (Recommended for Overnight Runs)

This is how Peter Steinberger and Simon Willison run parallel agents continuously.

**Setup:**

```bash
# Install tmux
brew install tmux  # macOS
# or: sudo apt install tmux  # Ubuntu

# Create named session for your project
tmux new-session -s eth-research -d

# Create windows for each agent
tmux new-window -t eth-research -n "planner"
tmux new-window -t eth-research -n "worker-1"
tmux new-window -t eth-research -n "worker-2"
tmux new-window -t eth-research -n "worker-3"
tmux new-window -t eth-research -n "worker-4"

# In each window, start an agent
# Window "worker-1":
cd ~/projects/eth-research && codex --full-auto

# Window "worker-2":
cd ~/projects/eth-research && claude

# Detach and agents keep running
# Ctrl+B then D

# Reattach later from anywhere
tmux attach -t eth-research
```

**Recommended tmux config (~/.tmux.conf):**

```bash
# Use Ctrl+A as prefix (easier than Ctrl+B)
set -g prefix C-a
unbind C-b

# Enable mouse
set -g mouse on

# Longer history
set -g history-limit 50000

# Status bar with agent names
set -g status-interval 5
set -g status-left "[#S] "
set -g status-right "#H %H:%M"

# Visual activity alerts
setw -g monitor-activity on
set -g visual-activity on
```

### Option B: Named Tmux Manager (ntm) - Purpose-Built for Multi-Agent

From my research, this tool is specifically designed for your use case:

```bash
# Install ntm
curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/ntm/main/install.sh | bash

# Create project with multiple agents
ntm init eth-research --cc=3 --cod=2
# Creates: 3 Claude Code agents + 2 Codex agents

# Send tasks to specific agent types
ntm send eth-research --cc "Implement growthepie API ETL pipeline"
ntm send eth-research --cod "Create L2Beat data extraction module"

# Broadcast to all agents of a type
ntm send eth-research --cc "Run all tests and report coverage"

# Monitor all agents
ntm dashboard eth-research

# Agents persist across disconnects
ntm attach eth-research
```

### Option C: Code Conductor (GitHub-Native Orchestration)

If you prefer GitHub Issues as your task tracker:

```bash
# Install Code Conductor
curl -fsSL https://raw.githubusercontent.com/ryanmac/code-conductor/main/conductor-init.sh | bash

# Label issues for agent work
gh issue create --label "conductor:task" --title "Implement growthepie ETL"

# Start workers (each claims tasks from GitHub)
./conductor start data-worker-1
./conductor start data-worker-2
./conductor start analysis-worker

# Workers automatically:
# - Claim tasks (GitHub Issues)
# - Create isolated worktrees
# - Implement features
# - Open pull requests
# - Move to next task
```

### Option D: Async Cloud Agents (Codex Web / Cursor Cloud)

For truly hands-off overnight execution:

**Codex Web (OpenAI):**
1. Go to chatgpt.com/codex
2. Connect your GitHub repository
3. Submit tasks via web interface or from your phone
4. Codex works in isolated cloud sandbox
5. Creates PR when done
6. You review in the morning

**Cursor Cloud Agents:**
1. Start tasks from cursor.com/agents
2. Each runs in remote sandbox
3. Check progress from web/mobile
4. Merge PRs when ready

---

## Part 4: Concrete Implementation Plan for Your Research

### Week 1: Infrastructure Setup

**Day 1-2: Repository Structure**
```bash
# Create project structure
mkdir -p eth-l1-l2-research/{.orchestrator/{active-tasks,completed-tasks,blocked-tasks,handoff-notes},data/{raw,processed,schemas},src/{etl,analysis,validation},docs,tests}

# Initialize git
cd eth-l1-l2-research
git init

# Create AGENTS.md (copy from above)
# Create .orchestrator/master-plan.md (copy your research plan)
```

**Day 3-4: Phase 0 - Protocol Lock (You + 1 Agent)**
```
Task: Finalize protocol document
- Exact STR formula with edge case handling
- Rollup inclusion criteria (top N by fees, settlement to Ethereum)
- Data source priority hierarchy
- Validation tolerance thresholds
```

**Day 5-7: Test Parallel Setup**
- Start 2-3 agents on simple parallel tasks
- Verify they don't interfere
- Test coordination file system
- Establish your monitoring routine

### Week 2-3: Phase 1 - Data Collection (4-6 Agents in Parallel)

**Launch Configuration:**
```bash
# Terminal layout (tmux)
# ┌─────────────┬─────────────┬─────────────┐
# │ Planner     │ Worker 1    │ Worker 2    │
# │ (Codex)     │ (Claude)    │ (Codex)     │
# ├─────────────┼─────────────┼─────────────┤
# │ Worker 3    │ Worker 4    │ Monitor     │
# │ (Claude)    │ (Claude)    │ (you)       │
# └─────────────┴─────────────┴─────────────┘
```

**Task Assignments:**

| Worker | Task | Input | Output | Success Criteria |
|--------|------|-------|--------|-----------------|
| 1 | growthepie ETL | API docs | `etl/growthepie.py` | ≥95% day coverage |
| 2 | L2Beat ETL | API docs | `etl/l2beat.py` | Cost series aligned |
| 3 | Blobscan ETL | API docs | `etl/blobscan.py` | Blob metrics daily |
| 4 | BigQuery extraction | Schemas | `etl/bigquery.py` | Block-level data |

**Task File Template (.orchestrator/active-tasks/task-001.md):**
```markdown
# Task: growthepie API ETL Pipeline

## Assignment
- Worker: worker-1
- Assigned: 2026-01-20 09:00 UTC
- Deadline: 2026-01-21 18:00 UTC

## Specification
Create an ETL pipeline that:
1. Fetches daily fundamentals from growthepie API
2. Extracts: fees, rent_paid, profit, txcount by rollup
3. Handles pagination and rate limits
4. Caches raw responses before transformation
5. Outputs: parquet files in data/raw/growthepie/

## Dependencies
- data/schemas/growthepie_schema.json (created in Phase 0)

## Success Criteria
- [ ] ≥95% day coverage for top 10 rollups
- [ ] All tests pass
- [ ] Code coverage >90%
- [ ] Raw API responses snapshotted

## Status
- [x] Assigned
- [x] In Progress (updated: 2026-01-20 14:30)
- [ ] Testing
- [ ] Review
- [ ] Complete

## Notes
(Agent updates this section)
2026-01-20 10:00: Started API exploration
2026-01-20 14:30: Basic extraction working, pagination issues
```

### Overnight Continuous Execution Pattern

**Setup (before you leave):**
```bash
# Start tmux session
tmux new-session -s overnight -d

# Start agents with queued tasks
tmux send-keys -t overnight:0 "cd ~/eth-research && codex" Enter
tmux send-keys -t overnight:0 "Queue: Complete task-001, then task-002, then task-003" Enter

tmux new-window -t overnight
tmux send-keys -t overnight:1 "cd ~/eth-research && claude" Enter
tmux send-keys -t overnight:1 "Work through tasks in .orchestrator/active-tasks/ assigned to worker-2" Enter

# Add more windows/agents as needed

# Detach - agents keep running
tmux detach
```

**Morning Check-In:**
```bash
# Reattach
tmux attach -t overnight

# Check status
cat .orchestrator/state.json
ls -la .orchestrator/completed-tasks/
ls -la .orchestrator/blocked-tasks/

# Review PRs if using git flow
gh pr list

# Check for any blocked tasks needing human input
cat .orchestrator/blocked-tasks/*.md
```

---

## Part 5: Quality Gates and Human Intervention Points

### Automated Quality Gates (Agents Run These)

```python
# quality_gates.py
def phase_1_gate():
    """Quality gate for data collection phase"""
    checks = {
        "coverage": check_day_coverage() >= 0.95,
        "schema_valid": validate_schemas(),
        "no_duplicates": check_no_duplicates(),
        "timestamps_utc": check_timestamps_utc(),
        "amounts_valid": check_amounts_non_negative()
    }
    return all(checks.values()), checks

def phase_4_gate():
    """Quality gate for validation phase"""
    checks = {
        "triangulation_pass": abs(growthepie - l2beat) / l2beat < 0.10,
        "onchain_match": onchain_vs_reported_match(),
        "no_missing_data": missing_data_pct < 0.05
    }
    return all(checks.values()), checks
```

### Human Intervention Points (You Must Review)

| Phase | Intervention Point | Decision Required |
|-------|-------------------|-------------------|
| 0 | Protocol sign-off | Are definitions correct? |
| 1 | Data source discrepancies | Which source is authoritative? |
| 2 | Attribution ambiguities | How to handle shared infra? |
| 3 | Metric edge cases | How to handle missing data? |
| 4 | Validation failures | Accept or investigate? |
| 5 | Unexpected results | Real or data error? |
| 7 | Final outputs | Ready for submission? |

---

## Part 6: GitHub Repositories and Tools

### Recommended Tools (from my research)

| Tool | Purpose | URL |
|------|---------|-----|
| **ntm** | Named Tmux Manager for multi-agent | github.com/Dicklesworthstone/ntm |
| **Code Conductor** | GitHub-native agent orchestration | github.com/ryanmac/code-conductor |
| **Claude Flow** | Multi-agent swarm orchestration | github.com/ruvnet/claude-flow |
| **async-code** | Parallel Claude/Codex with web UI | github.com/ObservedObserver/async-code |
| **myclaude** | Multi-agent workflow framework | github.com/cexll/myclaude |
| **Claude-Code-Workflow** | JSON-driven multi-agent framework | github.com/catlog22/Claude-Code-Workflow |

### Agent Configuration Files

**~/.codex/config.toml (Codex):**
```toml
model = "gpt-5.2-codex"
model_reasoning_effort = "xhigh"  # Recommended for swarm automation
tool_output_token_limit = 25000
model_auto_compact_token_limit = 233000

[features]
ghost_commit = false
unified_exec = true
apply_patch_freeform = true
web_search_request = true
skills = true
shell_snapshot = true

[projects."/path/to/eth-research"]
trust_level = "trusted"
```

**CLAUDE.md (for Claude Code):**
```markdown
# Claude Code Configuration

## Model Preferences
- Default to thorough reading before editing
- Commit atomically (only files you edited)
- Run tests after each significant change

## Project-Specific Instructions
- This is an academic research project on Ethereum economics
- Quality > Speed
- Always include type hints
- Document all assumptions
- Log all external API calls

## Coordination
- Check .orchestrator/active-tasks/ for assignments
- Update task status files regularly
- Do not modify files outside your task scope
- Write to handoff-notes/ when creating outputs others need
```

---

## Part 7: Monitoring and Debugging

### Real-Time Monitoring Setup

```bash
# Create monitoring pane in tmux
tmux split-window -h

# Watch for changes
watch -n 5 'cat .orchestrator/state.json | jq .'

# Or use a simple monitoring script
while true; do
    clear
    echo "=== Active Tasks ==="
    ls .orchestrator/active-tasks/
    echo ""
    echo "=== Completed ==="
    ls .orchestrator/completed-tasks/ | wc -l
    echo ""
    echo "=== Blocked ==="
    cat .orchestrator/blocked-tasks/*.md 2>/dev/null || echo "None"
    sleep 30
done
```

### Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Agent stuck | No progress >1 hour | Stop agent, check logs, simplify task |
| Merge conflicts | Multiple agents edit same file | Better task decomposition, use worktrees |
| Context overflow | Agent forgets earlier work | Shorter tasks, clear handoff notes |
| API rate limits | ETL failures | Add caching, exponential backoff |
| Drift | Agent goes off-task | Clearer task specs, more checkpoints |

---

## Part 8: Estimated Timeline and Effort

### With Multi-Agent Setup (Your New Workflow)

| Phase | Duration | Agents | Your Involvement |
|-------|----------|--------|------------------|
| 0: Protocol | 3 days | 1 | High (defining specs) |
| 1: Data | 5-7 days | 4-6 | Low (monitoring) |
| 2: Attribution | 3-4 days | 2-3 | Medium (decisions) |
| 3: Metrics | 3-4 days | 2-3 | Low (reviewing) |
| 4: Validation | 4-5 days | 3-4 | Medium (decisions) |
| 5: Analysis | 5-7 days | 3-4 | Medium (interpreting) |
| 6: Counterfactual | 3-4 days | 2 | Low |
| 7: Synthesis | 5-7 days | 2 | High (finalizing) |

**Total: ~5-7 weeks** (vs. 12-16 weeks sequential)

### vs. Your Current Sequential Workflow

| Aspect | Current | Multi-Agent |
|--------|---------|-------------|
| Time to completion | 12-16 weeks | 5-7 weeks |
| Your daily involvement | 8+ hours | 2-4 hours |
| Overnight progress | None | Continuous |
| Parallelism | 1 task | 4-6 tasks |
| Review burden | Per-change | Per-phase |

---

## Quick Start Checklist

- [ ] Install tmux: `brew install tmux`
- [ ] Create project structure (see Part 4)
- [ ] Write AGENTS.md with project specifics
- [ ] Set up coordination file system
- [ ] Test with 2 agents on simple parallel tasks
- [ ] Establish morning/evening check-in routine
- [ ] Define quality gates for each phase
- [ ] Start Phase 0 (protocol lock) - cannot parallelize
- [ ] Launch Phase 1 with 4-6 agents

---

## Final Recommendations

1. **Start conservative**: 2-3 agents before scaling to 6+
2. **Task decomposition is key**: Smaller, isolated tasks = better parallelism
3. **Trust the planner-worker model**: Don't let workers coordinate with each other
4. **Invest in AGENTS.md**: Clear instructions prevent most coordination failures
5. **Use overnight runs**: Your research can progress while you sleep
6. **Review in batches**: Don't review every change; review phase completions
7. **Fail fast**: If an agent is stuck >1 hour, intervene

The combination of Claude Code and Codex gives you the best of both worlds:
- **Codex**: Better at reading lots of code before acting, stays focused longer
- **Claude Code**: Better at nuanced decisions, writing, and interpretation

Use Codex for implementation-heavy tasks (ETL, analysis code) and Claude for planning and synthesis tasks.

---

*This guide synthesizes insights from Simon Willison, Peter Steinberger, Cursor's research team, and dozens of open-source multi-agent frameworks. The planner-worker architecture is proven at scale - Cursor built a browser from scratch with it.*
</response2>

# Answer
## Bottom line

1. **Response 2 is directionally the right architecture** (planner → workers → judge). It matches Cursor’s recent “hundreds of concurrent agents” lessons: **shared-file self-coordination + locks bottleneck and fails; hierarchy works**. ([Cursor][1])
2. **Response 1 has several practical gems** (tmux, “filesystem-based” coordination, Codespaces/remote runtime, directory isolation), but its core coordination model (“agents self-assign tasks from a shared Kanban brain file”) is *exactly* the pattern Cursor reports failing at scale. ([Cursor][1])
3. **The “membership can’t do parallel” claim in Response 1 is outdated/overstated.** With ChatGPT Pro, Codex publishes ranges like **300–1500 local messages per 5-hour window** (and **50–400 cloud tasks per 5 hours**)—so small-to-medium parallelism is realistic. ([OpenAI Developers][2])
4. The best integrated approach for you:

   * **File-based orchestration** (good from Response 1)
   * **Planner–Worker–Judge loop** (good from Response 2, strongly supported by Cursor’s experiments)
   * **Git worktrees/PRs as concurrency control** (Response 2’s best practical trick)
   * **Remote/persistent runtime** (Response 1’s “go home, keep running” idea—*with a correction about Codespaces auto-timeouts*) ([GitHub Docs][3])

---

## Evaluation: Response 1

### What’s practical and worth keeping

* **“Filesystem-based” instead of chat-based** is correct. Shared state must live in the repo (task specs, protocol, schemas, logs). That’s how you get continuity and reduce “context amnesia.”
* **tmux** is a very practical mechanism for “agents keep working while I’m away.” Detach/reattach is the simplest persistence layer for terminal agents.
* **Directory isolation** (each agent writes only to its sandbox) is a helpful *first-order* anti-clobber tactic.
* **Codespaces (or other remote dev VM)** is a legitimate solution to “laptop sleeps → agents die,” *with an important caveat below*. ([GitHub Docs][3])
* Highlighting the **`--dangerously-skip-permissions`** mode is relevant, but it must be paired with sandboxing/deny rules. Anthropic explicitly documents permission/sandbox settings and even how to disable bypass mode. ([Claude Code][4])

### What’s weak / misleading

* **The “membership vs parallelism” bottleneck framing is not reliable.**

  * For Codex, OpenAI publishes high Pro-tier usage ranges that support real parallelism. ([OpenAI Developers][2])
  * For Claude Code, Anthropic introduced weekly limits for heavy users; Max tiers can still be large, but it’s not “infinite swarm.” ([TechCrunch][5])
    So: parallel is feasible; “hundreds of agents for a week nonstop” is unlikely on consumer memberships.
* **The “agents self-assign tasks from a single shared file” is the wrong coordination primitive for your goal.** Cursor tried almost exactly that (shared file + locking), and it created lock bottlenecks, brittleness, and risk-avoidant agents that churn on small tasks. ([Cursor][1])
  For 2–3 agents you might limp along. For 5–8 agents (what you actually want), it becomes messy quickly.
* **Codespaces aren’t magically 24/7 by default.**
  GitHub explicitly says Codespaces **time out after inactivity** (default 30 minutes) and stop running. Closing the browser tab doesn’t stop it, but inactivity does. ([GitHub Docs][3])
  If your agents go “quiet” for long stretches, your codespace may shut down mid-run unless you configure timeouts appropriately.

### Net assessment

Response 1 is a good “how to run multiple terminal sessions” guide, but its coordination model is fragile, and its membership-limit claim is too sweeping.

---

## Evaluation: Response 2

### What’s practical and worth keeping (this is the core)

* **Planner–Worker–Judge architecture** is not just plausible—it’s currently the best-documented pattern for long-running, parallel agents. Cursor reports:

  * Shared-file peer coordination + locks bottlenecked badly
  * Flat “everyone equal” agents became risk-averse and churned
  * Separating roles (planners make tasks, workers execute, judge resets cycles) scaled much better ([Cursor][1])
* **Git worktrees/branches per worker** is an excellent practical mechanism to avoid “agents inverting each other’s work.” It’s real concurrency control, not wishful thinking.
* **Task specs with success criteria** + **quality gates** is exactly how you turn agent output into something you can trust at scale.
* It maps your research plan phases to parallel workstreams in a sensible way (Phase 1 high parallelism, Phase 0 protocol lock mostly sequential, etc.). That aligns with your plan structure. 

### What’s overconfident / needs correction

* The tool list is a grab bag. Some are useful; some are likely overkill:

  * **`ntm`**: plausibly useful if you already want tmux but hate manual pane management. ([GitHub][6])
  * **`async-code`**: useful if you want a web UI + containerized tasks, but adds Docker/infra overhead. ([GitHub][7])
  * **`code-conductor`**: interesting GitHub-native orchestration with worktrees; still early-stage and oriented to shipping PRs, so treat it as experimental. ([GitHub][8])
  * **`claude-flow`**: it’s huge, alpha, and loaded with “enterprise swarm intelligence” claims. For a solo academic trying to get a project done, this is *probably* negative ROI initially. ([GitHub][9])
* The timeline estimates (“5–7 weeks vs 12–16”) read like marketing. Real speedups depend on:

  * how cleanly tasks can be interface-separated
  * how many “definition/attribution” decisions require your judgment (your plan flags these risks explicitly) 

### Net assessment

Response 2 contains the right architecture and several high-leverage implementation details. It just needs pruning and “research-grade” guardrails.

---

## The genuinely new, practical things in their responses that are worth integrating

Here are the pieces that *materially improve* a robust system (beyond “run more agents”):

1. **Response 2’s Planner–Worker–Judge loop is not optional** if you want parallelism without chaos. Cursor’s Jan 2026 write-up is direct evidence that “agents self-coordinate in a shared file” fails and hierarchy works. ([Cursor][1])
2. **Response 1’s remote + tmux detach flow is the simplest “work while you sleep” mechanism**, but you must account for Codespaces auto-timeout behavior. ([GitHub Docs][3])
3. **“Dangerously skip permissions” needs explicit sandboxing and deny rules** (not just “turn it on”). Anthropic documents permission rules + sandbox settings and even how to disable bypass mode org-wide. ([Claude Code][4])
4. **Use Git worktrees as the default isolation mechanism** (Response 2) rather than “separate folders.” Folder isolation prevents clobbering files; worktrees prevent *semantic* code clashes and make integration auditable.
5. **Off-the-shelf tmux orchestration tools** (like `ntm`) can reduce your “operator overhead,” but only if you’re comfortable trusting and maintaining them. ([GitHub][6])

---

## The integrated “best version” I’d run in your constraints

This is the **minimum-complexity** system that still matches what actually works at scale.

### Core design principles

1. **Repo is the shared memory.** Everything important must be written to disk: task specs, protocol lock, schemas, logs, decisions.
2. **Workers do not coordinate with each other.** This is key. Workers coordinate *only* via task specs + artifacts they produce. Cursor explicitly found that peer coordination creates bottlenecks and risk aversion. ([Cursor][1])
3. **All parallelism is “interface parallelism.”** You only parallelize tasks that can be separated by contracts: file paths, schemas, function signatures, and success criteria.
4. **Judge gate keeps quality.** The judge runs deterministic checks (tests, data validation) and triggers “fresh starts” to prevent long-run drift—another lesson Cursor calls out. ([Cursor][1])

---

## What I would change in both colleagues’ designs

### Replace the shared Kanban file “brain” with task files + a planner

* Response 1 proposes agents self-assign from one file. Cursor says this fails. ([Cursor][1])
* Instead:

  * Planner creates **task specs** (`.orchestrator/tasks/*.md`) with dependencies and success criteria.
  * Planner assigns tasks explicitly to workers (or assigns from a queue with very simple rules).
  * Workers *never modify* the global queue except to update their own task file status.

### Prefer “worktrees/PRs” over “shared folders” as your primary anti-conflict mechanism

* Directory isolation is necessary for data artifacts, but code conflicts are best handled by git branches + CI + PR review.

### Build “contracts first” for the research plan

Your plan already implies contracts:

* STR formula definitions (Phase 0)
* data dictionary + rollup registry schema (Phase 0/2)
* idempotent ETL outputs + deterministic rebuild (Phase 1)
  If these are not locked, agents will do “metric shopping” and your outputs won’t integrate. 

---

## Concrete workflow for your STR research plan

Your plan has phases 0–7. Here’s how to run it as a swarm without babysitting. 

### Roles

* **Planner (1 agent):** reads the repo, decomposes work, writes task specs, assigns tasks, watches for blocked tasks.
* **Workers (3–6 agents):** each works one task in one worktree, produces artifacts, opens PR.
* **Judge (1 agent or you):** runs quality gates + reviews PRs against success criteria; approves merge or sends back for fixes.

Cursor explicitly reports this separation solved coordination problems at scale. ([Cursor][1])

### Phase 0: “Protocol lock” (mostly sequential)

Do this first. Don’t parallelize definitions. This matches your plan: avoid metric drift. 

Tasks a single agent can do:

* Protocol doc (exact STR formula + edge cases)
* Data dictionary + schemas
* Rollup inclusion rules

### Phase 1: Data inventory + acquisition (high parallelism)

This is where the swarm earns its keep.

Parallelizable worker tasks (each in separate worktree):

* **Worker A:** `growthepie` ETL module + snapshotting strategy
* **Worker B:** L2Beat costs ETL + reconciliation notes
* **Worker C:** Blobscan ETL + rollup labeling utilities integration
* **Worker D:** L1 raw data extraction choice + scaffold (BigQuery vs node)
* **Worker E (optional):** automated validation scripts (coverage, duplicates, schema checks)

All of these can run in parallel if Phase 0 produced stable schemas + output contracts. 

### Phase 2: Rollup attribution map (medium parallelism)

Parallelize *inputs*, not the registry itself:

* Worker A: collect candidate addresses from Blobscan metadata
* Worker B: scrape/compile addresses from L2Beat / official docs
* Planner/Judge merges into a single versioned registry + changelog

### Phases 3–5: Metric construction + validation + analysis

You can parallelize by module:

* Metric functions + unit tests
* Validation triangulation report
* Trend / break / elasticity analysis scripts
* Figure generation scripts

### Phase 6: Counterfactual

Mostly sequential on core mechanism implementation, with parallel sensitivity runs after.

---

## “Work while I sleep” setup: what actually works

### Option 1: Remote VM or Codespaces + tmux (simple, effective)

* tmux keeps processes alive after you disconnect.
* Codespaces keeps running after you close the tab, **but it will stop after inactivity unless you change the timeout**. ([GitHub Docs][3])

If you use Codespaces, you must:

* set a longer timeout, and/or
* accept that truly idle periods may stop the agents.

### Option 2: Use Codex cloud tasks for true unattended work (still membership-based)

OpenAI’s Codex is explicitly designed as a cloud agent that can run multiple tasks in parallel in sandboxes and propose PRs. ([OpenAI][10])
If your goal is “keep working while I’m away,” Codex cloud tasks are *almost purpose-built* for that.

Even if you prefer VS Code for active work, you can offload well-scoped tasks to Codex cloud tasks overnight and review PRs in the morning.

### Reality check on quotas (since your colleague brought it up)

* Codex Pro publishes “local messages” and “cloud tasks” ranges per 5-hour window. ([OpenAI Developers][2])
* Claude Code has weekly limits for Pro/Max tiers (Max tiers are larger, but not infinite). ([TechCrunch][5])

Given you said you rarely hit limits, you can likely run a small swarm (3–6 workers). But “hundreds of agents for a week” is *unlikely* on consumer plans.

---

## Safety + autonomy: how to use “skip permissions” without being reckless

If you want unattended operation, you’ll be tempted to allow everything. Don’t, unless you’re in a sandbox.

Anthropic documents:

* allow/ask/deny permission rules
* sandboxing options
* and how to disable bypass mode entirely (`disableBypassPermissionsMode`) ([Claude Code][4])

Practical guidance:

* **Run full autonomy only inside an isolated environment** (Codespaces/VM/devcontainer) where the filesystem contains only the repo.
* Deny access to secrets and home dirs explicitly.
* Keep “dangerous” commands gated (or disallowed).

---

## My integrated recommendation: what to do next (minimal operator overhead)

If you want “minimum babysitting,” don’t start by installing heavy orchestration frameworks. Start with the smallest thing that can work:

1. **Put your plan into the repo as immutable reference** (already exists). 
2. Add:

   * `AGENTS.md` (global rules)
   * `.orchestrator/` with task files
   * `docs/protocol.md` and `data/schemas/*`
   * `scripts/quality_gates.py` and a `make gate` command
3. **Use git worktrees** so each agent has a workspace, and require PRs for merges.
4. Run:

   * 1 planner agent (generates/updates tasks)
   * 2–4 workers (execute tasks)
   * 1 judge agent (runs gates + reviews PRs)
5. Schedule “fresh start” cycles: after each merge (or every N hours), restart workers from the updated task spec to prevent drift. Cursor explicitly says fresh starts help with long-running drift. ([Cursor][1])

---

## Optional tooling: which of Response 2’s extra tools are actually worth it?

* **`ntm`**: If you already buy tmux as the backbone, `ntm` can reduce manual pane/session management. It’s explicitly positioned as a multi-agent tmux manager. ([GitHub][6])
  Caveat: it’s a third-party install script—inspect before running.
* **`async-code`**: If you want a lightweight “task dashboard” and container isolation for each run, it’s plausible. But you’re now operating Docker + a web app. ([GitHub][7])
* **`code-conductor`**: Interesting if you want GitHub Issues → agents claim tasks → PRs automatically, with worktrees. But it’s oriented to feature shipping and is still early-stage. ([GitHub][8])
* **`claude-flow`**: I’d skip initially. It’s big, alpha, and adds a lot of orchestration machinery that you’ll end up maintaining. ([GitHub][9])

---

## I also generated a ready-to-use repo skeleton

To make this concrete, I created a **minimal “research swarm template”** zip with:

* `.orchestrator/` task queue
* `AGENTS.md`
* a task template + a few Phase 0/Phase 1 example tasks
* stub quality gates + Makefile

[Download the template](sandbox:/mnt/data/research_swarm_template.zip)

(You can drop it into a new GitHub repo or inside your current repo and adapt the task files to your exact plan.)

---

## References in your preferred link format

```text
[Lin 2026](https://cursor.com/blog/scaling-agents "Scaling long-running autonomous coding")
[GitHub 2026](https://docs.github.com/en/codespaces/about-codespaces/understanding-the-codespace-lifecycle "Understanding the codespace lifecycle")
[OpenAI 2026](https://developers.openai.com/codex/pricing/ "Codex Pricing")
[OpenAI 2026](https://developers.openai.com/codex/cli/ "Codex CLI")
[OpenAI 2025](https://openai.com/index/introducing-codex/ "Introducing Codex")
[Anthropic 2026](https://code.claude.com/docs/en/settings "Claude Code settings")
```

[1]: https://cursor.com/blog/scaling-agents "https://cursor.com/blog/scaling-agents"
[2]: https://developers.openai.com/codex/pricing/ "https://developers.openai.com/codex/pricing/"
[3]: https://docs.github.com/en/codespaces/about-codespaces/understanding-the-codespace-lifecycle "https://docs.github.com/en/codespaces/about-codespaces/understanding-the-codespace-lifecycle"
[4]: https://code.claude.com/docs/en/settings "https://code.claude.com/docs/en/settings"
[5]: https://techcrunch.com/2025/07/28/anthropic-unveils-new-rate-limits-to-curb-claude-code-power-users/ "https://techcrunch.com/2025/07/28/anthropic-unveils-new-rate-limits-to-curb-claude-code-power-users/"
[6]: https://github.com/Dicklesworthstone/ntm "https://github.com/Dicklesworthstone/ntm"
[7]: https://github.com/ObservedObserver/async-code "https://github.com/ObservedObserver/async-code"
[8]: https://github.com/ryanmac/code-conductor "https://github.com/ryanmac/code-conductor"
[9]: https://github.com/ruvnet/claude-flow "https://github.com/ruvnet/claude-flow"
[10]: https://openai.com/index/introducing-codex/ "https://openai.com/index/introducing-codex/"
