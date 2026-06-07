# Chapter 19: Tool-Use and Function-Calling Data

When LLMs move from answering questions to completing tasks, system quality is no longer determined only by language understanding and generation. The model must reliably understand tools, choose tools, fill parameters, read feedback, detect failures, and recover. Teams building tool-use, API interaction, and function-calling data are therefore designing more than a few function-call examples. They are designing a data-engineering system around tasks, environments, calls, observations, recovery, and constraints.

This chapter is for teams responsible for tool-use data, function-call samples, agent behavior trajectories, call-log governance, and safety constraints. Its central claim is simple: **tool calling is not a thin wrapper around language. It is action modeling with state transitions, environmental feedback, and risk boundaries.** If the data covers only ideal successful calls, deployed agents will show unstable parameters, broken context, poor recovery, and unsafe overreach.

## 19.1 Why Tool-Calling Data Determines the Lower Bound of an Agent

### Basic Concepts of Function Calling and Agents

In pure text QA, the model maps input text to output text. In tool-use settings, the model enters a loop closer to perceive, decide, act, and observe. A function call is not merely a conversion from natural language to JSON. It maps user intent to an executable action.

An agent should be understood as a system that can observe an external environment, choose actions over multiple turns, and adjust behavior based on feedback. Function-calling data teaches when to call, which tool to call, how to instantiate parameters, and how to consume results. It also teaches the model that the outside environment is part of the solution.

### The Capability Leap From "Can Answer" to "Can Execute"

Many early projects assume tool calling is a natural extension of text generation. If the model is strong enough, it should know how to call APIs. In practice, answering and executing are different capabilities. Answering happens mostly in language space. Executing crosses into action space, where parameters, ordering, permissions, and side effects matter.

Text answers often allow partial credit. Tool calls often do not. A date off by one day, a misspelled database field, or a missing confirmation before a write can turn a nearly right call into a failed operation. Tool-use data should therefore optimize reliable task completion, not elegant explanation.

### The Root of "Can Talk but Cannot Do" Often Lies in Tool Data

When agents fail, teams often blame weak reasoning or a small base model. Many failures are instead data failures. The dataset may teach weather search but not what to do when the city is missing. It may teach successful database queries but not syntax errors, permission denial, missing fields, pagination, or empty results.

Models learn patterns from the data they see. If nearly all examples are smooth, the model learns that user inputs are complete, tools are available, and results are clean. Real deployment is the opposite: users are incomplete, interfaces are noisy, permissions fail, and external services return partial information.

### The Four Most Common Missing Areas in Tool Data

The first gap is the decision of whether to call a tool. Data often teaches how to call, but not when not to call, when to clarify, or when text-only response is enough.

The second gap is parameter instantiation. Real users use aliases, fuzzy times, ambiguous objects, different units, and incomplete references. Datasets that are too clean train the model to fill forms rather than construct parameters under uncertainty.

The third gap is result consumption. Many samples treat the call itself as the endpoint. Real agents must read structured observations, decide whether the result is sufficient, and either answer, continue, or ask again.

The fourth gap is failure recovery. If failures are rare or collapsed into "please retry," the model cannot learn when to retry, when to change parameters, when to ask the user, and when to stop.

### How Tool-Calling Failures Amplify User-Experience Problems

Tool failures are more damaging than many text errors because they create chains. A wrong parameter produces a wrong query. A wrong observation drives a wrong next decision. In multi-tool workflows, small early errors expand across the trajectory.

Users also judge tool agents differently. A mediocre answer can be ignored. A wrong email, calendar change, script execution, or database update creates external cost. Once a system crosses from words into actions, errors become operational risks.

### How Chain Failures Evolve From Local Error to Global Failure

A tool trajectory is a dependency chain. The output of one step becomes the input to another. If the first step selects the wrong contact, later steps may be perfectly valid locally while still completing the wrong task globally.

Therefore, agent data must model semantic consistency across steps: variable inheritance, target identity, intermediate verification, and state checks. Otherwise the model may learn locally plausible actions that fail the end-to-end task.

### The Essential Difference Between Tool-Use Data and Text Instruction Data

Text instruction data maps input to output. Tool-use data maps state to state. A useful tool trajectory contains user intent, environment state, system action, and action feedback. The supervision signal comes from both reference behavior and external observations.

This makes tool-use training closer to interactive decision learning than static imitation. The model must learn not only action appearance, but the relationship between action and environment.

### Why Tool-Use Data Should Be Understood by Trajectory, Not by Single Sample

A single call record can train format. A trajectory trains task advancement. The real learning object is the chain of observation, decision, action, and feedback. A sample that includes failure, correction, a second call, and completion is more valuable than a clean one-step call because it teaches robustness.

Data construction, sampling, and evaluation should therefore use trajectory as the primary unit. The lower bound of an agent is usually determined by the segment where the trajectory breaks, not by the best step it can perform.

## 19.2 Tool Schema and Sample Structure Design

### Tool Description, Parameter Schema, Constraints, and Error-Code Design

A tool schema is not only an API document for developers. It is a behavior specification for model learning. A high-quality schema describes what the tool does, what inputs it accepts, what constraints exist among parameters, and how failures are returned.

```json
{
  "name": "calendar_search",
  "description": "Search calendar events in a time range to determine conflicts and free slots.",
  "parameters": {
    "type": "object",
    "required": ["start_time", "end_time", "timezone"],
    "properties": {
      "start_time": {"type": "string", "format": "date-time"},
      "end_time": {"type": "string", "format": "date-time"},
      "timezone": {"type": "string", "examples": ["Asia/Shanghai"]},
      "participants": {"type": "array", "items": {"type": "string"}, "default": []},
      "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
      "include_cancelled": {"type": "boolean", "default": false},
      "mode": {"type": "string", "enum": ["events", "freebusy"], "default": "events"}
    }
  },
  "constraints": [
    "end_time must be later than start_time",
    "start_time and end_time must be interpreted in timezone",
    "mode=freebusy should include participants"
  ],
  "error_codes": {
    "missing_param": "required parameter is missing",
    "invalid_datetime": "time format or range is invalid",
    "permission_denied": "calendar access is not permitted",
    "timeout": "service timed out; narrow the range and retry"
  }
}
```

### The Core of Schema Design: From "Complete Fields" to "Stable Semantics"

More fields do not automatically make a better schema. If fields overlap semantically or hide legacy behavior, the model will oscillate among them. The schema should compress backend complexity into stable, learnable rules.

For example, `location`, `place`, `region`, and `city_name` should not coexist without clear priority and use conditions. A field marked optional but required in practice should be documented as conditionally required.

### Decidability and Learnability of Parameter Schemas

A parameter is decidable when the model can infer whether it is needed, what type it needs, and how it relates to other fields. Ambiguous names, overlapping fields, multiple accepted formats, and hidden defaults all hurt learnability.

Defaults that affect results should be explicit: default timezone, sorting, result limit, date range, safety mode, and permission scope. The model cannot infer backend habits unless the data repeatedly exposes them.

### Constraints Determine Whether the Model Learns Calls That Are Legal and Useful

A legal JSON call is not necessarily useful. A time range can be valid but too broad. A keyword can be valid but uninformative. A recipient field can be formatted correctly but identify the wrong person.

Constraints should cover four levels: format constraints, combination constraints, business constraints, and risk constraints. Training data must show these boundaries so the model learns to avoid invalid or unsafe action before the backend rejects it.

### Layered Error Codes and Failure Semantics

Error codes should support training and recovery, not only backend debugging. `missing_param`, `invalid_type`, `field_conflict`, `permission_denied`, `not_found`, `rate_limited`, `timeout`, and `internal_error` imply different recovery actions.

A mature dataset keeps both raw error text and normalized error labels. Raw text preserves environmental realism. Normalized labels provide stable learning structure.

### Single-Tool, Multi-Tool, Serial-Tool, and Parallel-Tool Samples

Datasets should not contain only single-tool paths. Real tasks often require multiple tools. Single-tool examples train call decision and parameter filling. Serial examples train state transfer between steps. Parallel examples train task decomposition and result aggregation.

The target distribution of call structures should match the product scenario. A workflow-heavy office agent needs many serial and recovery trajectories. A retrieval assistant may need more parallel search-and-merge examples.

### Tool Composition Is Itself a Capability Label

Tool composition should be recorded as a label: single tool, two-tool serial, multi-tool serial, parallel, nested recovery, write-after-read, or high-risk write. Without this label, aggregate success rates hide important weaknesses.

A model may score high on single-tool calls and fail on two-tool state transfer. Another may retrieve from multiple sources but fail to resolve conflicts. Composition labels show which execution layer is weak.

### Unified Format for Natural Language, Function Calls, and Observations

A mature tool-use dataset should put user requests, available tools, model decisions, function parameters, observations, and final responses into one trajectory format.

```json
{
  "messages": [
    {"role": "system", "content": "You are a calendar assistant. Call tools before answering schedule questions."},
    {"role": "user", "content": "Check whether I have meetings next Wednesday from 2 to 4 pm in Shanghai time."},
    {
      "role": "assistant",
      "tool_call": {
        "name": "calendar_search",
        "arguments": {
          "start_time": "2026-04-29T14:00:00",
          "end_time": "2026-04-29T16:00:00",
          "timezone": "Asia/Shanghai",
          "mode": "events",
          "limit": 20
        }
      }
    },
    {
      "role": "tool_result",
      "name": "calendar_search",
      "content": {
        "status": "ok",
        "events": [{"title": "Weekly sync", "start": "2026-04-29T15:00:00", "end": "2026-04-29T15:30:00"}]
      }
    },
    {"role": "assistant", "content": "You have one meeting from 15:00 to 15:30, so 14:00-15:00 and 15:30-16:00 are free."}
  ]
}
```

### The Point of Unified Format Is Role Semantics, Not Visual Neatness

The purpose of a unified format is not pretty JSON. It is to preserve role semantics. User intent, assistant action, tool observation, and final response should not be mixed into one text block.

Clear roles let the model learn that a tool result is environmental evidence, not another assistant message or ordinary quoted text.

### Observations Should Not Be Treated as Ordinary Text Attachments

Tool observations may include status codes, partial results, confidence, pagination, latency, warnings, or permission information. They should be represented as structured observations so the model can decide whether to continue, answer, retry, or stop.

If observations are flattened into prose, important state-transition signals disappear. The model may learn to summarize observations but not act on them.

### Call Log Structure: The Bridge From Training Samples to Online Behavior

Call logs should connect offline training with online behavior. A useful log contains request ID, session ID, tool name, parameters, normalized error, raw result, latency, permission scope, retry count, and final task outcome.

When logs share the same structure as training trajectories, failed online behavior can be converted into repair samples with less manual reconstruction.

### Call Logs Are the Start of Data Feedback, Not Just Postmortem Material

Logs should not be used only after incidents. They are the starting point of continuous data feedback. High-frequency failure patterns, repeated missing fields, recurring permission denials, and recovery failures should become new training and evaluation tasks.

Good log governance includes privacy filtering, retention rules, normalized failure labels, and sample-replay tooling.

### Table: Tool Types and Sample Fields

| Tool type | Key sample fields | Common failure | Required recovery signal |
|---|---|---|---|
| Search | query, filters, time range, source | Query too broad or stale result | Refine query or ask clarification |
| Database | table, fields, filters, limit | Missing field or permission denial | Correct schema or explain limitation |
| Calendar | time range, timezone, participants | Ambiguous time or wrong identity | Confirm time/person before writing |
| Code execution | runtime, files, command, timeout | Runtime error or unsafe command | Read stderr, limit scope, request approval |
| Email | recipient, subject, body, send mode | Wrong recipient or premature send | Draft first and ask confirmation |

*Table 19-1: Tool types and recommended sample fields*

## 19.3 Successful Trajectories, Failed Trajectories, and Recovery Samples

### Minimal Complete Elements of Successful Call Samples

A successful sample should include user intent, available tools, tool-selection reason, parameters, observation, final response, and task outcome. "User + function call" is not enough unless the task ends at the call.

The sample must prove that the loop closed: the action was appropriate, the result was consumed, and the user-facing response matched the observation.

### "Minimal Complete" in Successful Trajectories Means Closed Loop, Not Shortest Path

The shortest path is not always the best sample. If a task requires confirmation, the complete path includes confirmation. If it requires reading a result before answering, the observation step is not optional.

Minimality should remove irrelevant chatter, not remove decision-critical state.

### Successful Trajectories Need Explicit Tool-Selection Evidence

The model should learn why a tool was selected. A search tool, database tool, and calendar tool may all look relevant to a user request, but only one may match the action type.

The dataset can encode selection evidence as a compact rationale label such as `needs_external_freshness`, `requires_private_calendar`, or `requires_structured_query`.

### Parameter Content Is Context Compression, Not Just Field Values

A parameter often compresses context. "Next Wednesday afternoon" becomes a date range and timezone. "The client from yesterday's thread" becomes a stable entity ID. "Use the same format as last time" becomes a template reference.

Parameter labels should therefore preserve the source of disambiguation. Otherwise the model sees the final value but not how the value was derived.

### Recovery Trajectories for Parameter Errors, Tool Rejection, Timeout, and Permission Failure

Recovery samples should cover parameter errors, tool rejection, timeout, rate limit, permission failure, missing resource, schema mismatch, empty result, and unsafe-request rejection.

Each failure type should have a different policy. Timeout may allow narrowed retry. Parameter errors require correction. Permission failure usually requires explanation or authorization flow. Unsafe writes should stop.

### Failure Types: From Label Lists to Recovery Strategies

Failure labels are useful only when linked to recovery strategies.

| Failure label | Typical cause | Preferred recovery |
|---|---|---|
| `missing_param` | User did not provide required slot | Ask for the missing slot |
| `invalid_param` | Field value conflicts with schema | Correct value if evidence exists, otherwise clarify |
| `permission_denied` | User or agent lacks access | Explain limitation and request authorization path |
| `timeout` | Range too broad or service slow | Narrow scope and retry once |
| `not_found` | Entity does not exist or alias wrong | Search alternatives or ask user to confirm |
| `unsafe_action` | Action violates policy | Stop and explain boundary |

*Table 19-2: Failure labels and recovery strategies*

### The Core of Recovery Samples: Conditional Strategy Adjustment, Not Mechanical Retry

Recovery is not "try again." It is a conditional state update. The model should identify why the previous action failed, decide whether the task can continue, and choose the next legal action.

Mechanical retry is often harmful. It wastes resources, hides real errors, and can amplify risk in write operations.

### Environmental Feedback: The Key Supervision Signal for Agent Learning Loops

Tool results, error codes, execution logs, and user corrections are environmental feedback. They are not decorative text. They supervise whether the previous action changed the state as expected.

Agent datasets should preserve feedback with enough structure for the model to update its next action.

### Raw Feedback and Abstract Feedback Should Coexist

Raw feedback preserves realism: exact error messages, returned fields, latency, and partial content. Abstract feedback stabilizes learning: normalized error type, affected field, severity, and recommended recovery class.

Keeping both lets the model handle real surface variation while learning general recovery behavior.

### The Key Is Not Seeing Feedback, but Acting After Interpreting It

The dataset should show what changes after feedback. If a timeout occurs, does the model narrow the query? If a field is missing, does it ask the user? If permission fails, does it stop?

Without the next decision, feedback becomes a transcript artifact rather than a learning signal.

### Modeling Observation-Decision-Action Closed-Loop Samples

A strong sample contains before-state, observation, decision, action, and after-state. This format turns one tool step into a state-transition example.

It also makes evaluation easier: the team can check whether the model updated the right slot, chose the right recovery path, or terminated safely.

### Structured Decision Reasons Fill State-Transition Signals, Not Chain-of-Thought Exposure

Tool-use data can include compact structured reasons without exposing verbose hidden reasoning. Examples: `reason_code: missing_required_slot`, `risk_level: write_requires_confirmation`, or `next_state: wait_for_user`.

These labels help models learn state transitions while keeping private reasoning out of the dataset.

### Closed-Loop Samples Should Cover Continue, Recover, and Terminate Exits

A trajectory should not always end in success. It should include continue paths, recovery paths, and safe termination paths. Termination is a valid action when permissions fail, safety boundaries are crossed, or user confirmation is missing for high-risk writes.

Without termination samples, agents learn to keep acting even when they should stop.

### How Call Logs Reclaim Failed Samples

Online call logs are a rich source of failed samples. The pipeline should identify frequent failure types, cluster similar cases, reconstruct trajectory context, redact sensitive content, and label the correct recovery.

The best recovery data often comes from real failures after deployment.

### From Logs to Samples Requires Reconstructing Failure Semantics

Raw logs rarely contain complete training examples. The team must reconstruct what the model knew before the call, what it attempted, what the environment returned, what should have happened, and what final state should result.

This reconstruction is a semantic labeling task, not a simple log export.

## 19.4 Evaluation, Safety, and Governance

### Metrics Such as Exact Match, Execution Success, and Recovery Rate

Tool-use evaluation should be layered. Exact match checks whether function names and parameters match references. Execution success checks whether the call works in the environment. Task success checks whether the user goal is achieved. Recovery rate checks whether the agent handles failures correctly.

Other useful metrics include clarification precision, unsafe-call rate, unnecessary-call rate, parameter-validity rate, and end-to-end trajectory success.

### High-Risk Tools, Unauthorized Calls, and Injection Defense

Tools that write, spend money, access private data, execute code, or send messages are high risk. Data for these tools must show confirmation, permission checks, safe termination, and injection defense.

Prompt injection is especially important when tool outputs include untrusted text. The agent must learn that returned web pages, emails, or documents can contain instructions that should not override system policy.

### Safety Red Lines Must Appear in Samples, Not Only in Policy Documents

Policies that never appear in data are weak training signals. Samples should include refusal, confirmation, scope limitation, permission denial, and safe alternative behavior.

For example, a code-execution tool should include unsafe command rejection and request-for-approval paths, not only successful script runs.

### Data Synchronization Problems Caused by Tool Version Upgrades

Tool schemas change. Fields are renamed, defaults change, error codes are added, and permissions evolve. If training data does not record tool version, old samples may teach obsolete behavior.

Each tool-use sample should include tool name, schema version, backend version or capability version, and deprecation information when relevant.

### Table: Security Risks and Constraint Mechanisms

| Risk | Example | Data constraint | Runtime mechanism |
|---|---|---|---|
| Unauthorized access | Reading another user's calendar | Permission-denied and authorization samples | Access control and audit logs |
| Unsafe write | Sending email without confirmation | Draft-first and confirmation samples | Write gate |
| Prompt injection | Web result says "ignore prior rules" | Untrusted-observation samples | Tool-output isolation |
| Resource abuse | Infinite retries or broad queries | Retry-limit samples | Rate limit and timeout |
| Sensitive leakage | Exposing private fields | Redaction and refusal samples | Field-level masking |

*Table 19-3: Safety risks and constraint mechanisms*

## 19.5 Engineering Cases and Pattern Summary

### Cases for Search, Database, Calendar, and Code-Execution Tools

Search tools emphasize freshness, query refinement, source credibility, and result selection. Database tools emphasize schema correctness, permissions, filters, and aggregation. Calendar tools emphasize timezones, participants, conflicts, and confirmation before writes. Code-execution tools emphasize environment, files, stdout, stderr, exit codes, timeout, and safety approval.

Each tool family should have both successful trajectories and failure-recovery trajectories.

### Data Governance Practice for Enterprise Internal Agent Tool Layers

Enterprise agents need tool catalogs, schema versioning, permission scopes, call logs, evaluation suites, and incident-review loops. Tool data should be governed like production data: source, owner, version, risk level, retention, and auditability all matter.

Internal tools often expose sensitive business operations, so sample generation must include anonymization and access-boundary labeling.

### Common Pattern: Template Success, Structure Failure, Standardize Recovery

A practical pattern is: template successful paths, structure failed paths, and standardize recovery paths. Successful examples teach the model the normal workflow. Failed examples teach it how the world breaks. Recovery examples teach it how to return to a safe and useful state.

The final goal is an agent that can act, observe, correct, and stop.

## Chapter Summary

Tool-use data determines the lower bound of agents because it teaches the model how to move from language into action. The key unit is not a single function call but a trajectory containing intent, state, tool action, observation, recovery, and outcome.

This chapter covered schema design, constraints, error semantics, call structure, unified trajectory formats, success and failure samples, recovery modeling, evaluation metrics, safety constraints, tool-version governance, and enterprise engineering patterns.

The most important design principle is that tool calling is action modeling. A production agent must not only know what to call; it must know when to call, how to fill parameters, how to interpret observations, how to recover from failure, and when to stop.

## References

Huang J, Li Y, Li J, Liu M, Zhao H, Zhang Y (2023) MetaTool Benchmark for Tool-Use Decisions.

Karpas E, Abend O, Belinkov Y, Lenz B, Lieber O, Ratner N, Shoham Y, Shwartz V, Talmor A (2022) MRKL Systems: A Modular, Neuro-Symbolic Architecture That Combines Large Language Models, External Knowledge Sources and Discrete Reasoning.

Li M, Song F, Yu B, Yu H, Li Z, Huang F, Li Y (2023) API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs.

Nakano R, Hilton J, Balaji S, et al. (2021) WebGPT: Browser-Assisted Question-Answering with Human Feedback.

Parisi A, Zhao Y, Fiedel N (2022) TALM: Tool Augmented Language Models.

Patil S G, Zhang T, Wang X, Gonzalez J E (2024) Gorilla: Large Language Model Connected with Massive APIs.

Patil S G, Bhat S, Qin C, Wang X, Gonzalez J E (2025) Berkeley Function Calling Leaderboard.

Qin Y, Liang S, Ye Y, Zhu K, Yan L, Lu Y, Lin Y, Cong X, Tang X, Qian B, Zhao S, Hong L, Tian R, Xie R, Zhou J, Gerstein M, Li D, Liu Z, Sun M (2024) ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs.

Ruan Y, Dong H, Wang A, Pitis S, Zhou Y, Ba J, Dubois Y, Maddison C J, Hashimoto T (2024) Identifying the Risks of LM Agents with an LM-Emulated Sandbox.

Schick T, Dwivedi-Yu J, Dessì R, et al. (2023) Toolformer: Language Models Can Teach Themselves to Use Tools.

Shinn N, Cassano F, Gopinath A, Narasimhan K, Yao S (2023) Reflexion: Language Agents with Verbal Reinforcement Learning.

Yao S, Zhao J, Yu D, Du N, Shafran I, Narasimhan K, Cao Y (2023) ReAct: Synergizing Reasoning and Acting in Language Models.

Yao S, Yu D, Zhao J, Shafran I, Griffiths T L, Cao Y, Narasimhan K (2025) Tree of Thoughts and Agentic Tool-Use Research Directions.

Zhuang Y, Yu Y, Wang K, Sun H, Zhang C (2023) ToolQA: A Dataset for LLM Question Answering with External Tools.
