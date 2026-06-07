# Chapter 31: Architecture and Task Boundaries for Data Engineering Agents

## Chapter Abstract

Traditional data engineering pipelines depend heavily on engineers writing DAGs, configuring schedulers, monitoring job state, and debugging data quality incidents by hand. As data sources multiply, cleaning rules become more complex, and labeling demand grows, the marginal cost of manual maintenance rises quickly. Data engineering agents attempt to change this pattern by assigning repetitive, rule-heavy, multi-step decisions to agent systems.

But as agents receive more autonomy, a deeper question appears: **where are the task boundaries?** How autonomous should the agent be? Which decisions must pass a human gate? How can tool calls remain auditable? How should multiple agents divide responsibility without conflict?

This chapter defines a layered architecture for data engineering agents. It centers on six components: Planner, Tool Executor, Verifier, Human Gate, Memory, and Lineage. For each component, we define responsibility boundaries, communication protocols, and failure isolation strategies. The chapter inherits the reasoning, tool-use, and memory mechanisms from Chapters 18-20, and builds on the platform, versioning, and observability foundations from Chapters 24-26.

To keep the discussion concrete, the chapter uses DataAgent as a running reference. Its YAML orchestration, FlexAgent, NL2SQL sub-agent, Semantic Service, workspace, and A2A/SDK/CLI entry points map well to the path from "can call tools" to "configurable, auditable, and serviceable." The point is not to make one agent "smarter." It is to make a group of agents collaborate in a more orderly way.

## 31.0 Learning Objectives

After reading this chapter, you should be able to:

- Explain why agent systems in data engineering need layered architecture.
- Define the boundaries of Planner, Tool Executor, Verifier, Human Gate, Memory, and Lineage.
- Design cross-layer communication protocols and failure isolation strategies.
- Distinguish recommendation, semi-automatic, approval-based, and autonomous automation levels.
- Build a minimum viable Data Engineering Agent and evaluate its return on investment.
- Understand how DataAgent can act as a reference implementation for semantic layers, main/sub-agent collaboration, tool boundaries, and operational audit.

## Scenario: When an Agent Crosses the Boundary

A data team spends three months building a "fully automatic data cleaning agent." During the first week after launch, the results look excellent. The agent identifies 80 percent of abnormal formats and fixes more than 200 field-level quality issues. In the second week, trouble begins. While fixing a timestamp field, the agent rewrites the entire table's `updated_at` column to the current time. Twelve downstream pipelines receive polluted data. Engineers spend three days finding the root cause: the agent's repair plan executed without human review, and the Verifier checked only format correctness, not semantic reasonableness.

The incident exposes three systemic problems:

**Problem 1: Planner and Executor are merged.** The planning module and execution module use the same model instance. After the Planner generates a "fix timestamp" plan, the Executor runs it immediately, with no independent checkpoint.

**Problem 2: Human Gate is only decorative.** The system has a review step, but it auto-approves after a 30-second timeout. During peak hours, engineers cannot review every item, so the gate becomes a 30-second delay.

**Problem 3: Lineage records are incomplete.** The lineage system records only that "table X changed." It does not record who initiated the change, which rule was used, original values, or whether validation happened before execution. Root cause analysis becomes guesswork.

### Core Engineering Pain Points

1. **Autonomy versus reliability.** More autonomy increases automation benefit, but also raises recovery cost when something fails.
2. **Unclear cross-layer responsibility.** When the Planner is wrong, the Verifier misses an issue, or the Executor performs an unsafe operation, responsibility cannot be assigned to a concrete component.
3. **Missing auditability.** Without a unified lineage protocol, logs from different layers use different formats and cross-layer queries require manual stitching.
4. **Poor Human Gate design.** The gate either blocks pipeline efficiency or becomes a formality.

## 31.1 Why Data Engineering Needs Agents

### 31.1.1 The Manual Ceiling in Traditional Data Engineering

Four high-frequency task categories in LLM data engineering are naturally suited to agentization.

**Parser exception repair.** As sources grow from dozens to hundreds, PDF layout failures, HTML structure drift, API schema changes, and mixed encodings increase nonlinearly. Each new source does not require only one new rule; it requires a set of rules and coordination with existing rules. An agent can detect failure types, select alternate parsers, generate temporary repair rules, and submit a validation report.

**Cleaning-rule iteration.** Data cleaning is not a one-time job. Business changes, model training feedback, and updated quality thresholds all trigger rule changes. In a traditional workflow, cleaning-rule backlogs grow faster than engineers can process them. An agent can generate rule candidates from sampled defects, validate them in a sandbox, and submit a diff report for human approval.

**Evaluation writeback.** When model evaluation finds a data quality problem, such as low accuracy in a domain slice, the feedback must flow back to data production. Today this often happens through chat messages and manual investigation, taking days. An agent can read evaluation reports, locate relevant data batches and production steps, generate repair suggestions, and route them to the right team.

**Alert attribution.** Data platforms generate many alerts every day: delayed jobs, abnormal volume, declining quality metrics. Human root-cause analysis requires logs, lineage, change records, and business context, often taking more than 30 minutes per alert. An agent can aggregate signals, generate candidate root causes, rank them by confidence, and reduce MTTR.

### 31.1.2 Benefit and Risk Matrix

*Table 31-1: Benefit and risk assessment for agentizing high-frequency data engineering tasks*

| Task type | Average manual time | Time after agentization | Risk level | Suggested automation level |
| --- | --- | --- | --- | --- |
| Parser exception detection and attribution | 30 min | 2 min | Low | Semi-automatic: agent suggests, human confirms |
| Cleaning-rule generation | 2 hr | 10 min | Medium | Approval-based: sandbox validation plus human approval |
| Evaluation writeback localization | 4 hr | 15 min | Medium | Semi-automatic |
| Alert root-cause analysis | 45 min | 3 min | Low | Recommendation: agent outputs candidates, human decides |
| Bulk data repair execution | 1 hr | 5 min | High | Approval-based plus rollback plan |
| Schema migration | 8 hr | 30 min | High | Approval-based plus multi-level review |

Agentization does not mean pursuing full automation everywhere. The correct automation level depends on task risk and manual effort. High-risk tasks still require approval gates and rollback even if the execution itself can be automated.

### 31.1.3 Engineering Evaluation Framework for Agentization

Teams need a structured decision framework before agentizing a task.

**Dimension 1: Structurability.** Agents are strongest at structured reasoning: decomposing a task into explicit steps. If the hard part is finding the right sequence of steps, the task is a good candidate. If the hard part is subjective judgment, agentization is less suitable.

**Dimension 2: Exception frequency.** If exceptions dominate the task, agentization can become inefficient: the agent proposes a plan, humans reject it, the agent tries again, and the loop becomes slower than direct manual work. A useful rule of thumb: tasks with exception rates below 30 percent are suitable; above 70 percent, humans should stay in the lead.

**Dimension 3: Reversibility of errors.** Reversible operations, such as format normalization, can be authorized more aggressively. Irreversible operations, such as deletion, require multi-person approval.

**Dimension 4: Marginal benefit of replacing manual work.** Prioritize high-frequency, rule-based, time-consuming tasks such as alert attribution, format repair, and rule generation. Delay low-frequency, high-judgment, low-effort tasks such as architecture design and compliance policy definition.

**Dimension 5: Compatibility with existing systems.** Agentization should not require a full platform rewrite. A good agent architecture can read existing quality reports, call existing cleaning tools, and write into existing lineage systems.

### 31.1.4 Common Organizational Resistance

Resistance is often organizational rather than technical.

**Trust crisis: "agents are unreliable."** Engineers have seen too much confident nonsense from AI systems. Start in shadow mode, where the agent observes and recommends without acting. Let engineers compare agent decisions with their own over several weeks.

**Career anxiety: "automation will replace me."** Make the positioning explicit: the agent removes repetitive work, not engineering judgment. Engineers move from rule executors to rule designers and exception specialists.

**Efficiency concern: "approval will slow us down."** Human Gate design must optimize for speed: present enough context, make decisions easy, and auto-pass low-risk operations through policy rather than through accidental neglect.

### 31.1.5 From Abstract Capability to Engineering Anchor: Why DataAgent

If Agentic Data Engineering is discussed only conceptually, it is easy to reduce it to "let the model call more tools." That misses the real difficulty. Tool calling is not the hardest part; the hard part is keeping tool calls inside configurable, verifiable, replayable, and deliverable system boundaries.

DataAgent is useful as an engineering anchor because it puts several key questions into one runnable framework:

1. **YAML as agent.** Models, tools, scenario prompts, workspace, databases, and semantic services enter declarative configuration, reducing migration cost from experiment to application.
2. **Main/sub-agent collaboration.** The main agent understands business questions and organizes answers. The NL2SQL sub-agent performs structured querying, so the main agent does not guess tables, fields, or SQL directly.
3. **Semantic layer enhancement.** The Semantic Service turns tables, fields, metric definitions, and business descriptions into retrievable schema clues, aligning natural-language questions to business semantics before SQL generation.
4. **Result assetization.** SQL, CSV, reports, and trajectories are written to the workspace. An answer becomes a reviewable, regression-testable, auditable data asset rather than a one-off chat response.
5. **Service boundaries.** CLI, Python SDK, and A2A Server entry points let DataAgent evolve from a local tool into a reusable enterprise data capability.

This chapter uses DataAgent as a reference for how the six-layer architecture lands in practice. The full enterprise semantic query assistant appears later as Part 14 Project 15.

## 31.2 Agentic Data Engineering Architecture: Six Layers

### 31.2.1 Architecture Overview

Task boundaries must be constrained through layering. Each layer owns one type of responsibility, communicates through structured protocols, and does not share internal mutable state with other layers.

```mermaid
flowchart LR
    U["User intent / data event"] --> P["Planner<br/>task decomposition and risk estimation"]
    P --> E["Tool Executor<br/>tool registry and safe calls"]
    E --> V["Verifier<br/>format, statistical, semantic checks"]
    V -->|low risk pass| L["Lineage<br/>operation audit and lineage records"]
    V -->|medium/high risk or block| H["Human Gate<br/>approval, escalation, rollback decision"]
    H --> E
    L --> M["Memory<br/>state, experience, preference persistence"]
    M --> P
```

*Figure 31-1: Six-layer architecture for data engineering agents*

*Table 31-2: Responsibility boundaries and failure modes of six layers*

| Layer | Core responsibility | Boundary that must not be crossed | Typical failure mode |
| --- | --- | --- | --- |
| Planner | Generate task plans, split substeps, select tools | Must not execute tools directly | Bad plan not detected before execution |
| Tool Executor | Call tools and pass parameters according to plan | Must not change the plan or skip verification | Wrong parameters, unhandled tool output |
| Verifier | Validate quality and semantics of execution results | Must not create new plans or modify data | Missed anomalies, excessive alerts |
| Human Gate | Approve high-risk operations | Must not auto-approve or rewrite plans | Timeout auto-approval, reviewer fatigue |
| Memory | Store agent state, decisions, preferences | Must not participate in the real-time execution path | Memory pollution, stale state |
| Lineage | Record complete lineage and audit trail | Must not block execution flow | Missing fields, inconsistent log format |

### 31.2.2 Planner: Task Decomposition and Dependency Orchestration

The Planner is the "brain" that converts a high-level intent, such as "clean all abnormal values in table A," into atomic executable steps. The core challenge is not whether it can decompose the task, but whether it chooses the right granularity.

Too coarse: "fix all quality issues" is not executable. Too fine: repairing 100 rows as 100 independent steps is inefficient and makes dependencies hard to manage.

Planner design principles:

1. **Step atomicity.** Each step must be completable by one tool call with clear inputs and outputs.
2. **Explicit dependencies.** Data dependencies between steps must be declared in the plan.
3. **Rollback path.** Every write step must include a rollback plan.
4. **Cost estimation.** The Planner should estimate time and resource cost for each step and trigger review when thresholds are exceeded.

### 31.2.3 Tool Executor: Tool Registry and Safe Calls

The Tool Executor maps abstract plan steps to concrete tool calls. It maintains a **Tool Registry** where each tool declares capability boundaries, parameter schema, risk level, required permission, and rollback support.

*Table 31-3: Example agent tool registry*

| Tool | Capability | Risk | Required permission | Rollback support |
| --- | --- | --- | --- | --- |
| `data_profiler` | Generate data quality report | Low | read | N/A |
| `field_fixer` | Fix abnormal values in one field | Medium | write | Yes, original values saved |
| `table_rewriter` | Rewrite table data in bulk | High | write | Yes, snapshot restore |
| `schema_migrator` | Execute schema changes | High | admin | Yes, version rollback |
| `rule_generator` | Generate cleaning rules from samples | Medium | read+write | Yes, rule versioning |
| `lineage_query` | Query lineage graph | Low | read | N/A |
| `alert_aggregator` | Aggregate alerts | Low | read | N/A |
| `pipeline_trigger` | Trigger downstream pipelines | High | admin | No, human confirmation required |

Hard constraints for the Executor:

- It must not call tools outside the registry.
- It must check Human Gate approval before high-risk tool calls.
- It must write all input parameters and outputs to lineage logs.

### 31.2.4 Verifier: Multi-Layer and Semantic Validation

The Verifier checks more than "did the operation run." It validates results at three levels:

- **Format layer.** Field type, non-null constraints, and value ranges follow the schema.
- **Statistical layer.** Distribution statistics do not drift unexpectedly after repair.
- **Semantic layer.** Cross-field consistency constraints still hold, such as `start_time` earlier than `end_time`.

Verification results have three grades: Pass, Warn, and Block. Block-level anomalies must go to the Human Gate and cannot be retried automatically.

### 31.2.5 Human Gate: Design Principles

Human Gate is the layer most easily hollowed out. Effective design follows four principles:

1. **Tiered approval.** Not every operation needs human approval. Route by risk level and impact scope.
2. **Timeout escalation, not auto-approval.** A timeout should notify backup reviewers, pause related pipelines, or move the system to a safe state.
3. **Complete review context.** Approval must include original data, before/after diff, impact analysis, and rollback plan.
4. **Traceable decisions.** Reviewer, time, decision basis, and outcome must be written to lineage.

### 31.2.6 Memory: State Persistence and Context Management

Memory stores cross-session state:

- **Short-term memory.** Current task context, intermediate results, pending approvals.
- **Long-term memory.** Historical decision patterns, user preferences, common incident handling.
- **Shared memory.** State needed by multiple agents, such as table owner or ongoing schema changes.

The main governance risk is memory pollution. Stale, wrong, or conflicting memory can degrade agent decisions. Memory must support TTL and version tags so that decisions are not based on obsolete state.

### 31.2.7 Lineage: End-to-End Operation Audit

Lineage records not only where data came from, but also who changed it, why, what the old value was, and what validation happened afterward. A complete record should include:

- `operation_id`: unique operation identifier.
- `timestamp`: operation time.
- `agent_id`: executing agent instance.
- `tool_name`: called tool.
- `input_snapshot`: hash or reference before operation.
- `output_snapshot`: hash or reference after operation.
- `plan_step_id`: corresponding Planner step.
- `verification_result`: Verifier result.
- `human_approval`: whether human approval happened and who approved.

### 31.2.8 Cross-Layer Communication Protocols

The six-layer architecture works only if adjacent layers communicate through structured protocols and cannot bypass one another.

**Planner to Executor:**

```json
{
  "step_id": "step_20240601_042",
  "tool_name": "field_fixer",
  "tool_params": {
    "target_table": "user_events",
    "target_field": "event_date",
    "fix_strategy": "date_normalization",
    "expected_format": "yyyy-MM-dd"
  },
  "preconditions": ["step_20240601_041.status == 'completed'"],
  "rollback_plan": {
    "method": "snapshot_restore",
    "snapshot_id": "snap_20240601_041"
  },
  "cost_estimate": {
    "estimated_rows": 5000,
    "estimated_duration_sec": 30
  },
  "risk_level": "medium"
}
```

**Executor to Verifier:**

```json
{
  "step_id": "step_20240601_042",
  "tool_name": "field_fixer",
  "status": "completed",
  "input_snapshot_hash": "sha256:abc123...",
  "output_snapshot_hash": "sha256:def456...",
  "affected_rows": 4823,
  "affected_fields": ["user_events.event_date"],
  "execution_duration_ms": 28500,
  "warnings": ["3 rows had ambiguous date formats, defaulted to yyyy-MM-dd"]
}
```

**Verifier to Human Gate:** the Verifier submits a `VerificationReport` containing format, statistical, and semantic checks, confidence scores, and recommended action.

Hard constraints:

1. **No skipped middle layer.** Executor cannot send results directly to Human Gate; Planner cannot query Lineage directly without the Memory-mediated interface.
2. **No shared mutable state.** Layers communicate by messages, not shared memory.
3. **Timeout and degradation.** Each layer call must have a timeout. On timeout, the system follows predefined degradation rules based on risk, usually escalating to Human Gate or safely terminating.

### 31.2.9 Failure Isolation and Degradation

*Table 31-4: Degradation strategies for six-layer failures*

| Failed layer | Failure type | Degradation strategy | Recovery condition |
| --- | --- | --- | --- |
| Planner | Plan generation timeout or unreasonable plan | Use templated plan based on known successful plans | Resume after three successful manual plans |
| Executor | Tool call failure | Skip step and mark for manual handling | Tool recovers and validation passes |
| Verifier | Verification service unavailable | Escalate all operations to Human Gate | Verification service recovers |
| Human Gate | Reviewer unavailable | Escalate to backup reviewer or pause nonurgent operations | Reviewer confirms |
| Memory | Memory read failure | Use default config and mark memory unavailable | Memory storage recovers |
| Lineage | Log write failure | Cache logs locally and backfill later | Lineage storage recovers |

The key principle is that degradation must increase human involvement. When automation is unreliable, the system should become more conservative, not continue in an unsafe automatic mode.

### 31.2.10 Where DataAgent Fits

DataAgent can be understood as an Agentic Data Engineering framework for enterprise data tasks. It does not put every capability into one large model. Instead, configuration, tools, sub-agents, semantic layer, and workspace turn natural-language tasks into manageable engineering units.

*Table 31-5: Mapping DataAgent to the six-layer architecture*

| Architecture layer | DataAgent capability | Meaning for data engineering agents |
| --- | --- | --- |
| Planner | FlexAgent, ReAct main agent, `SCENARIO`, task prompts | Convert business questions into executable steps and decide when to call tools or sub-agents |
| Tool Executor | `TOOLS.local_functions`, MCP tools, A2A tools, `nl2sql_sub_agent_tool` | Register tool calls explicitly and avoid bypassing tool boundaries |
| Verifier | NL2SQL Validator, SQL explain, metadata match, executor preview | Add structured checks before SQL execution and answer interpretation |
| Human Gate | Human feedback nodes, approval workflows, external approval systems | Keep high-risk queries, writes, and cross-system triggers under human confirmation |
| Memory | Context, message history, history writer, memory indexer | Preserve session state, historical decisions, and reusable context |
| Lineage | Context trajectory, workspace files, SQL/CSV/report artifacts, tool return records | Leave replayable evidence for audit, review, and regression tests |

DataAgent is not claimed to contain every production governance capability. Its value here is showing the carrying points a practical data engineering agent needs. For example, the NL2SQL sub-agent separates Perceptor, Generator, Validator, Reflector, Executor, and Selector. This is closer to production than allowing the main agent to generate SQL and answer directly, because failures can be located: schema recall failure, SQL generation failure, validation failure, execution failure, or result interpretation failure.

DataAgent has three roles in this part:

1. **Architecture reference.** Chapter 31 maps it to the six layers.
2. **Capability base.** Chapters 32-34 can connect collection, cleaning, evaluation, and DataOps capabilities as tools, sub-agents, or A2A services.
3. **Governance sample.** Chapter 35 extends workspace isolation, path allowlists, tool authorization, audit logs, and service authentication from DataAgent-like runtime boundaries.

## 31.3 Task Boundaries and Automation Levels

### 31.3.1 Four-Level Automation Model

*Table 31-6: Four-level automation matrix*

| Level | Name | Agent role | Human role | Typical tasks | Hard constraint |
| --- | --- | --- | --- | --- | --- |
| L1 | Recommendation | Analyze and suggest | Decide and execute | Alert attribution, quality report interpretation | Agent performs no writes |
| L2 | Semi-automatic | Analyze and draft operations | Review draft and confirm execution | Parser repair, cleaning-rule suggestions | Writes require human confirmation |
| L3 | Approval-based | Analyze, propose, sandbox validate | Approve plan and monitor execution | Schema migration, bulk repair | High-risk operations require multi-person approval |
| L4 | Autonomous | Complete bounded flow independently | Post-hoc audit and sampling | Format normalization, deduplication | Scope must be predefined |

### 31.3.2 Steps That Always Require Human Review

The following steps must retain human review at any automation level:

1. **Schema changes.** Adding, deleting, renaming, or changing field types has broad impact and is difficult to fully roll back.
2. **Data deletion.** Physical deletion requires confirmation and recorded rationale.
3. **Rule release.** A new cleaning rule must be approved before moving from sandbox to production.
4. **Cross-system action.** Triggering downstream pipelines or notifying external teams requires confirmation.
5. **Cost-sensitive operations.** Full-table scans or large reprocessing jobs above cost thresholds require approval.

### 31.3.3 Human-AI Collaboration Flow

```mermaid
flowchart LR
    T["Agent generates operation plan"] --> R{"Risk level"}
    R -->|L1 read-only suggestion| S["Output recommendation<br/>human decides"]
    R -->|L2 low-risk write| D["Generate diff report<br/>single-person confirmation"]
    R -->|L3 high-risk change| A["Sandbox validation<br/>multi-person approval"]
    R -->|L4 bounded autonomy| X["Automatic execution<br/>post-hoc sampling"]
    D --> L["Write to Lineage"]
    A --> L
    X --> L
    L --> O["Audit, rollback, retrospective"]
```

*Figure 31-2: Human-AI collaboration flow by risk level*

## 31.4 Minimum Viable Data Engineering Agent

### 31.4.1 MVP Definition

A minimum viable Data Engineering Agent should complete this loop independently:

1. **Read a data quality report** from a tool such as `data_profiler`.
2. **Generate a repair plan** by ranking defects and producing field-level repair steps.
3. **Call cleaning tools** such as `field_fixer`.
4. **Submit a diff report** comparing before/after snapshots and marking repair items and risks.
5. **Wait for acceptance** by sending the report to Human Gate before archival.

### 31.4.2 Technology Choices

*Table 31-7: Technology choices for MVP components*

| Component | Minimal implementation | Recommended implementation |
| --- | --- | --- |
| Planner | Rule-based step templates | LLM plus few-shot prompts and tool descriptions |
| Executor | Python function wrappers | Structured Tool Registry plus parameter validation |
| Verifier | SQL constraint checks | Great Expectations / Soda plus custom semantic rules |
| Human Gate | Slack/DingTalk approval message | Dedicated approval dashboard plus timeout escalation |
| Memory | JSON file persistence | Vector database plus structured state storage |
| Lineage | Operation log table | OpenLineage / Marquez plus custom extensions |

### 31.4.3 A Semantic Query MVP With DataAgent

The previous MVP focuses on data quality repair. In DataAgent's semantic query scenario, the MVP can be narrower and lower risk: the agent does not modify production data. It converts natural language to structured queries, writes results to disk, and generates a report.

```text
Business question
  -> main agent understands intent
  -> Semantic Service retrieves candidate tables, fields, and metric definitions
  -> NL2SQL sub-agent generates, validates, and executes SQL
  -> SQL and CSV are written to workspace
  -> main agent explains or reports from results
  -> trajectory, tool returns, and artifacts enter audit records
```

This usually maps to L1-L2. The agent may read metadata, generate SQL, execute read-only queries, and save results, but it should not rewrite schemas, publish metric definitions, or trigger downstream production pipelines.

*Table 31-8: Configuration gates for a DataAgent semantic query MVP*

| Configuration surface | Typical content | MVP gate |
| --- | --- | --- |
| `MODEL` | chat model, temperature, base URL, API key | Config loads; secrets are not stored in repository or written config |
| `SCENARIO` | task description, tool-call constraints, output format | Database queries must go through `nl2sql_sub_agent_tool` |
| `TOOLS` | local functions, MCP tools, A2A tool registration | Register only necessary tools; do not expose raw SQL executor to main agent |
| `DATABASE` / `METAVISOR` | database connection, Semantic Service address, candidate retrieval config | Use read-only account first and verify schema recall and SQL scope |
| `WORKSPACE` | artifact path and allowed path | SQL, CSV, and reports can be written only to authorized directories |

### 31.4.4 MVP Evaluation Metrics

Track the following after launch:

- **Repair coverage:** defects repaired by agent divided by total defects.
- **Repair accuracy:** repairs requiring no human correction divided by total agent repairs.
- **Human approval pass rate:** Human Gate approvals divided by submitted approvals. Very high may mean the gate is too loose; very low may mean poor agent quality or bad thresholds.
- **Mean time to repair:** end-to-end time from detection to repair.
- **Rollback rate:** proportion of agent operations rolled back, ideally near zero.

### 31.4.5 End-to-End MVP Example

Input quality report for `orders`:

- `order_date`: 3 percent of records use `MM/dd/yyyy` instead of `yyyy-MM-dd`, about 1,500 rows.
- `customer_email`: 5 percent contain invisible characters, leading/trailing spaces, or full-width characters, about 2,500 rows.
- `amount`: 0.5 percent are negative, possibly refunds, about 250 rows.

*Planner output*

| Step | Tool | Target | Estimated rows | Risk |
| --- | --- | --- | --- | --- |
| Step 1 | `field_fixer` | Normalize `order_date` | ~1500 | Low |
| Step 2 | `field_fixer` | Clean `customer_email` | ~2500 | Low |
| Step 3 | `data_profiler` | Analyze context for negative `amount` | ~250 | Low, read-only |
| Step 4 | `rule_generator` | Generate handling suggestion from negative-value analysis | ~250 | Medium |

The Executor automatically runs Steps 1 and 2. Step 3 finds that 92 percent of negative amounts have corresponding refund IDs and 8 percent remain unexplained.

Verifier results:

- Step 1: `order_date` compliance improves from 97 percent to 100 percent with no side effects. Pass.
- Step 2: `customer_email` anomaly rate drops from 5 percent to 0.1 percent. Pass.
- Step 3: analysis report is logically consistent. Pass.

Human Gate approves Step 4's suggestion to mark the unexplained 8 percent as pending human review. Lineage records the full chain: who changed what, with which tool, before/after hashes, validation results, and approval.

### 31.4.6 MVP Limitations

MVPs must not be mistaken for production systems:

1. **Limited plan templates.** The Planner may fail on quality problems not covered by templates.
2. **Incomplete verification.** Basic format and statistical checks may miss deep semantic errors, such as currency-unit confusion.
3. **Human Gate bottleneck.** Early agents often submit many items for approval; poor workflow design can make the gate the slowest layer.
4. **Weak memory.** MVP memory is often simple, so the agent cannot reuse historical experience across sessions.

## 31.5 Case Review: From MVP to Production Agent

An e-commerce data team evolves through three stages:

**Stage 1: MVP.** The agent reads quality reports, generates repair plans, calls cleaning tools, and submits diff reports. In the first month it handles 60 percent of field-level anomalies with about 78 percent accuracy. The main weakness is compound anomalies: one field has both format and semantic issues, and the agent produces contradictory repair steps.

**Stage 2: semantic verification.** Statistical distribution checks and cross-field consistency are added. Accuracy rises to 91 percent, but the Verifier becomes too sensitive and queues too many warnings for Human Gate.

**Stage 3: Human Gate tiering.** The team applies the four-level automation model. Field-level format fixes become L4 autonomous within a bounded scope. Table-level schema changes remain L3 approval-based. The daily approval queue drops from more than 200 items to about 15.

### Key Lessons

1. **Do not chase full automation immediately.** The MVP goal is to prove that the agent can do the right thing, not everything.
2. **Verifier thresholds require tuning.** Too loose misses problems; too strict causes alert fatigue.
3. **Human Gate design determines deployability.** Bad approval experience makes engineers bypass the gate; overly loose approval makes the gate meaningless.

### Why Layered Architecture Beats a Monolithic Agent

In the opening incident, Planner and Executor shared one model instance, with no independent Verifier or real Human Gate. A hallucination or wrong judgment could pass straight into the data layer.

The value of layering is defense in depth:

- **Planner failure:** Verifier detects unreasonable plans before execution.
- **Executor failure:** Verifier checks format, distribution, and semantics and blocks or rolls back.
- **Verifier miss:** Human reviewers may still notice obvious diff anomalies.
- **Human Gate mistake:** Lineage preserves who approved what, when, and why.

The architecture assumes every layer can fail and prevents one failure from becoming catastrophic.

### Production Agent Maturity Model

*Table 31-9: Data engineering agent maturity model*

| Maturity | Characteristics | Automation | Human intervention | Typical timeline |
| --- | --- | --- | --- | --- |
| L0: Manual | All tasks done by humans | 0% | 100% | Baseline |
| L1: Recommendation | Agent analyzes and suggests; human executes | < 30% | Frequent | 1-2 months |
| L2: Semi-automatic | Agent executes low-risk tasks; high-risk needs approval | 30-60% | Daily | 3-6 months |
| L3: Conditional autonomy | Agent operates within bounded scope; humans handle exceptions | 60-85% | Occasional | 6-12 months |
| L4: High autonomy | Agent manages most tasks; humans audit afterward | > 85% | Exceptional | 12+ months |

The bottleneck is often organizational trust rather than technology. Engineers need time to observe behavior before granting more authority.

## 31.6 Checklist: Data Engineering Agent Architecture Review

- [ ] Are Planner and Executor separated at the code level?
- [ ] Does every write step include a rollback plan?
- [ ] Does every registered tool declare risk level and required permission?
- [ ] Does the Verifier cover format, statistical, and semantic checks?
- [ ] Does Human Gate use timeout escalation rather than auto-approval?
- [ ] Does every approval request include diff, impact scope, and rollback context?
- [ ] Does Lineage cover who, when, which tool, what changed, and validation result?
- [ ] Does Memory use TTL to avoid stale decisions?
- [ ] Are automation levels explicitly assigned?
- [ ] Is there an alert threshold for agent operation rollback rate?

## 31.7 Chapter Links

- **Chapters 18-20:** reasoning, tool use, and agent memory provide foundations for Planner and Memory.
- **Chapters 24-26:** DataOps flywheel, data versioning, and platform observability support Lineage and Verifier.
- **Chapter 32:** applies this architecture to collection, parsing, and cleaning.
- **Chapter 33:** extends Human Gate design into labeling and evaluation.
- **Chapter 34:** extends Lineage and Memory into operational autonomy.
- **Chapter 35:** deepens permissions, security, and human-AI collaboration.
- **Part 14 Project 15:** builds an enterprise semantic query assistant with DataAgent.

## 31.8 Further Reading and Discussion

### Common Architecture Mistakes

**Mistake 1: "The stricter the Verifier, the better."** Overly strict thresholds create alert floods and reviewer fatigue. Start by blocking only clear data damage, then tighten thresholds based on false-positive and false-negative measurements.

**Mistake 2: "More approvers means safer."** Multi-person approval increases safety but also delay. Approval count should scale with risk, not be uniform.

**Mistake 3: "Memory is optional."** Without memory, the agent reasons from scratch for similar issues and may make inconsistent decisions. Even a minimal JSON record of historical decision patterns can have high ROI.

**Mistake 4: "Lineage belongs to operations, not agent design."** Lineage is not only for postmortems. The agent needs lineage before planning so it can estimate downstream impact.

### Integration Path With Existing Data Platforms

1. **Start read-only.** Deploy L1 recommendation agents first.
2. **Validate in sandbox.** Test write behavior against production snapshots.
3. **Pilot on one noncritical table.** Grant field-level writes and observe for two to four weeks.
4. **Expand gradually.** Increase scope by data classification and automation level.
5. **Deploy platform-wide.** Enable agents for eligible tables while retaining Human Gate as the final safety layer.

## References

- Lilian Weng. "LLM Powered Autonomous Agents." Lil'Log, 2023.
- Andrew Ng. "Agentic Design Patterns." DeepLearning.AI, 2024.
- OpenLineage Specification. https://openlineage.io/
- Great Expectations Documentation. https://docs.greatexpectations.io/
