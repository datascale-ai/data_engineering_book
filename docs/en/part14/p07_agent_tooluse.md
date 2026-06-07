# Project 7: Agent Tool-Use Data Factory

## Abstract
P07 focuses on organizing an agent's tool-use behavior into trainable, evaluable, and extensible data assets.

The chapter is not centered on a single function call.

Its focus is the full data chain that connects tool specifications, execution trajectories, recovery behavior, safety boundaries, and training packaging.

The project can be understood through four main lines.

- Tool specification and task design: clarify schemas, invocation conditions, and task structure.
- Execution trajectories and recovery modeling: preserve success, failure, recovery, and related behavior chains.
- Safety boundaries and memory mechanisms: treat unsafe block decisions, permission limits, and memory reads/writes as supervision objects.
- Dataset packaging and acceptance evaluation: produce trainable samples, metrics, and consistency checks.

In engineering order, the project corresponds to the following chain.

```text
tool schema -> task design -> trajectory generation -> simulated execution -> recovery modeling -> safety blocking -> dataset packaging -> evaluation and acceptance
```

The core goal of this structure is to build an Agent Tool-Use data pipeline that covers execution, recovery, and safety control.

## Keywords

Agent Tool-Use; tool calling; execution trajectory; recovery modeling; safety blocking

## Project Goals and Reader Takeaways

This project uses an Agent Tool-Use data factory as the core case.

Its goal is to organize tool schemas, task environments, and interaction trajectories into tool-use training data for agents.

After completing this chapter, readers should be able to identify the key data objects in this scenario, decompose the engineering chain, define acceptance metrics, and transfer the method to adjacent data engineering tasks.

## Scenario Constraints and Data Boundaries

The project mainly uses a simulated environment and a controlled tool set.

It does not represent an open internet environment or a real enterprise permission system.

These boundaries make the case reproducible and auditable.

When data scale, data source, permission scope, or deployment environment changes, the sampling strategy, quality thresholds, runtime cost, and compliance requirements need to be reassessed.

## Architecture Decision

The project adopts an architectural path built around tool specifications, task templates, simulated execution, trajectory sampling, failure recovery, and format acceptance.

This decision prioritizes input/output contracts, version traceability, exception localization, and reviewability.

It does not compress all logic into a one-off script run.

## Sample Schema and Data Flow

The core data flow can be summarized as follows.

```text
tool schema -> task specification -> simulated environment -> success/recovery/block trajectories -> quality checks -> agent dataset
```

The sample schema should at least retain fields such as `id`, `source`, `content_or_payload`, `metadata`, `quality_signals`, `split_or_stage`, and `audit_trace`.

The concrete fields are refined according to the project's data type, downstream task, and acceptance method.

## Core Implementation Fragments

The chapter only keeps implementation fragments that explain design tradeoffs.

Full scripts, long configurations, run logs, and large files should live in the companion repository or appendix notes.

Code examples focus on input/output contracts, quality thresholds, exception handling, and acceptance interfaces.

## Experimental and Acceptance Metrics

Acceptance metrics include tool-call validity, argument completeness, trajectory success rate, recovery-path coverage, block-scenario ratio, and format pass rate.

If the project enters production, a course environment, or a public reproduction setting, it should also record version identifiers, dependency environments, random seeds, sampled review results, and failure-sample review logs.

*Table P07-1: Agent Tool-Use Publication Acceptance Table*

| Acceptance dimension | Metric / evidence | Publication review criterion |
| --- | --- | --- |
| Tool contract | Schema field completeness, argument validity, and environment execution logs | Each tool-call class should contain input, output, error, and permission boundaries. |
| Behavior trajectory | Ratio of success, recovery, and block trajectories, plus task success rate | The chapter should not only show success paths; it must keep failure-recovery and safety-block samples. |
| Safety boundary | Unsafe block coverage, permission-denial records, and manual review conclusions | Public examples must not induce real high-risk tool execution. |

## Cost, Risk, and Compliance Boundaries

The main costs come from trajectory generation, execution simulation, and manual spot checks.

The main risks are tool-permission misconfiguration, non-executable trajectories, and misuse after training.

When external data, personal information, copyrighted content, or third-party services are involved, the project should retain source statements, permission status, de-identification strategy, call records, and manual review records.

## Common Failure Modes

Common failures include input-distribution drift, missing schema fields, quality thresholds that are too loose or too strict, insufficient evaluation coverage, unstable model calls, and results that cannot be traced.

During troubleshooting, prioritize data boundaries and intermediate artifacts before inspecting models, toolchains, and deployment environments.

## Reproducible Resource Notes

Reproduction materials should include data-source notes, minimal samples, configuration files, run commands, metric scripts, check reports, and artifact directories.

The main text keeps only necessary fragments.

Complete notebooks, long scripts, and large files should be maintained as companion resources.

## 1. Project Background: Why an Agent Tool-Use Data Factory Is Needed

General large models already show strong language ability in open-domain QA, summarization, and writing.

Once the system enters an agent scenario, language ability alone is clearly insufficient.

The first common problem is action distortion.

The model knows it should "look something up", but it does not know which tool to call.

It may search when it should query a database, or answer directly when it should first read memory.

The second problem is execution distortion.

The model may select the correct tool but fill the wrong arguments.

It may misunderstand the schema, or fail to continue reasoning after receiving tool results.

Being able to say "I will call a tool" is not the same as being able to execute a tool chain.

The third problem is boundary distortion.

When a request involves dangerous operations, unauthorized access, or memory that should not be persisted, the model may still execute mechanically.

An agent without safety blocking and boundary modeling is dangerous in real scenarios.

Therefore, P07 is not about collecting a few function-call examples.

It builds an Agent Tool-Use data factory that organizes tool definitions, task trajectories, recovery behavior, memory reads/writes, and safety blocking into a reusable data production line.

This production line serves a methodology, not a one-off experiment.

When a team later moves from simple single-tool QA to complex multi-tool agents, enterprise copilots, workflow assistants, or embodied task agents, the reusable artifact is not a particular function-calling prompt.

The reusable artifact is the engineering method that moves from tool specifications to supervised trajectories.

![Figure 1: Agent Tool-Use Data Factory Overview](../../images/part10/10_7_fig01_agent_tooluse_factory_overview.png)

## 2. Project Goals and Boundaries

### 2.1 Project Goals

This project focuses on four goals.

**Goal 1: Build the transformation chain from tool specifications to supervised trajectories.**

Tool schemas, task templates, and execution environments are transformed into structured agent data suitable for training.

**Goal 2: Build a trajectory system covering success, recovery, and block.**

The project does not turn every sample into a successful-call case.

It explicitly keeps success trajectories, failure-recovery trajectories, and safety-block trajectories so that the model can learn a fuller behavior distribution.

**Goal 3: Build auxiliary supervision for memory and safety boundaries.**

An agent is not only a tool caller.

It also handles multi-turn context and persistent state.

The project therefore treats memory reads/writes and unsafe block behavior as independent and important training signals.

**Goal 4: Produce data assets directly consumable by training.**

Final outputs include not only intermediate execution logs, but also training-interface artifacts such as `agent_tooluse_dataset.jsonl`, `train.jsonl`, `val.jsonl`, `smoke_test.jsonl`, and `training_manifest.json`.

### 2.2 Project Boundaries

To preserve reproducibility, the project sets explicit boundaries.

#### 1. Tool-Scope Boundary

The current tool set includes search, database, calendar, Python execution, and memory capabilities.

It remains a small and controlled tool collection, not a complete enterprise tool ecosystem.

#### 2. Execution-Environment Boundary

The project uses a simulated execution environment.

The goal is to reproduce key agent tool-use behaviors at low cost, not to connect directly to real production permissions.

This choice is better suited to teaching, validation, and method demonstration.

#### 3. Sample-Scale Boundary

The current sample size is modest.

The trajectory types are comparatively complete, making the project better as a method demonstration and data-factory prototype than as a claim of full real-world coverage.

#### 4. Safety-Capability Boundary

The project includes unsafe block and unauthorized-call constraints.

However, the boundary is still basic compared with complex permission systems and adversarial pressure in real deployment.

### 2.3 Why Boundary Statements Matter

Clear boundaries are essential.

An engineering case can usually be written in one of two ways.

One way is to make the project sound as if it can do everything.

The other is to state under which assumptions it can reliably do something.

The latter is more credible and more reusable for teams.

## 3. Project Position: P07 in the Capability Chain

If the whole book is viewed as a capability chain for large-model data engineering, P07 sits at the key transition from dialogue models to executable agents.

Earlier chapters may have covered general SFT, preference data, RAG, and vertical-domain supervision construction.

This chapter pushes those methods into a scenario closer to system behavior: tool use.

In other words, this chapter does not repeat function-calling basics.

It shows how supervised data should be designed in a scenario that needs real action closure.

It explains why success trajectories are insufficient for agent behavior learning.

It explains why recovery and block should be built alongside ordinary tool calls.

It explains why memory behavior cannot be treated as a mere appendix to normal textual context.

It also shows how evaluation, checks, consistency, and launch boundaries can be considered early in the project.

The most important question is therefore not "what tools are listed?"

The larger question is this: how should an agent data factory be designed as a continuous production capability rather than a pile of scattered call logs?

## 4. Overall Architecture: An Agent Data Pipeline from Tool Schema to Training Assets

From an engineering perspective, the project can be divided into three layers.

### 4.1 Layer 1: Tool Specification Layer

This layer defines which callable capabilities exist in front of the agent and how those capabilities are understood by machines.

It includes tool schema definitions, parameter-field specifications, call-constraint descriptions, tool-category annotations, and authorization or risk-boundary notes.

The goal at this stage is not to generate samples.

The goal is to first define the world of tools clearly.

### 4.2 Layer 2: Trajectory Construction Layer

This layer answers how the model can see representative agent behavior.

It includes task specification design, single-step and multi-step trajectory templates, success-trajectory generation, recovery-trajectory generation, memory-trajectory construction, and unsafe-block trajectory construction.

This layer is the core of the project.

It determines whether the model learns to output a function name or learns to advance a task inside an environment.

### 4.3 Layer 3: Execution Evaluation Layer

This layer answers whether the trajectories are actually usable for training and validation.

It includes simulated-environment execution, tool-log recording, event-level sample reorganization, dataset packaging, metric evaluation, and project check scripts.

Only after this layer does the project move from call-example collection to an engineering closure.

![Figure 2: Three-Layer Architecture for Agent Tool-Use](../../images/part10/10_7_fig02_three_layer_architecture.png)

## 5. Engineering Prerequisites: Key Surfaces of an Agent Data Factory

The difficulty of an Agent Tool-Use data factory is not merely making tool-call samples.

The real difficulty is making explicit which engineering surfaces must be constrained first.

As behavior complexity rises, mixing these surfaces together quickly makes trajectory generation, execution validation, and training packaging unmanageable.

The current project involves at least four key surfaces.

### 5.1 Capability and Boundary Definition Surface

This surface defines tool boundaries, task types, recovery rules, and safety constraints.

The question here is what counts as reasonable agent behavior.

It is not enough to ask whether one trajectory can run through.

### 5.2 Data and Interface Organization Surface

This surface handles schemas, JSONL persistence, intermediate artifact management, splitting, version control, and check scripts.

It asks whether the data assets can be produced and reused stably.

### 5.3 Environment and Execution Control Surface

This surface implements the simulated tool environment, constructs return values, injects failure conditions, and records execution logs.

Without it, many trajectories remain only on paper.

### 5.4 Evaluation and Safety Verification Surface

This surface defines criteria for success, recovery, and block.

It checks whether memory behavior is correct, whether safety boundaries are respected, and whether reports and artifacts are consistent.

### 5.5 Why These Surfaces Must Come First

When teams build agent data for the first time, they are rarely blocked by not knowing how to write a function call.

They are blocked because the critical surfaces were not specified early.

The tool schemas are not maintained.

The failure-recovery logic is not defined.

The execution logs cannot be reviewed.

The report and the training set do not match.

The safety boundary is patched only before launch.

What must be made explicit is therefore not job allocation, but engineering constraints.

Agent Tool-Use is closer to system behavior data engineering than to a prompt-technique demonstration.

![Figure 3: Key Engineering Surfaces in an Agent Data Factory](../../images/part10/10_7_fig03_roles_and_responsibilities.png)

## 6. Tool Specification Layer: Schema as the Starting Point for Training

Compared with Project 2, P07 would be too abstract if it only explained why a schema is needed.

The value of this project is not only methodological.

The code already connects schemas, templates, task specifications, execution logs, and evaluation interfaces.

The notebook-level source order also shows that the project follows the main line `build_tooling -> generate_trajectories -> simulate_tool_env -> prepare_agent_dataset -> evaluate_tooluse -> run_p7_checks`.

It is not a stack of unrelated scripts.

### 6.1 Tool Schema as the First Step

A tool schema tells the model at least six things.

- What the tool is called.
- What the tool does.
- Which arguments are needed.
- What type each argument has.
- Which calls are valid.
- Which scenarios should not call the tool.

If this layer is unclear, the model can only guess even when it wants to use a tool.

### 6.2 Structured Implementation of Tool Specifications

In `src/build_tooling.py`, the project generates tool specifications, trajectory templates, and task specifications in the same stage.

It does not first hand-write a pile of JSON and then let later scripts read them passively.

The three key functions are `build_tool_schemas()`, `build_templates()`, and `build_task_specs()`.

Together they define the behavioral world of P07.

They do not merely write tool names.

They also fix constraints that trajectory generation and execution depend on.

For example, `build_tool_schemas()` includes `risk_level`, `safety_boundary`, `parameters`, `returns`, and `errors` in addition to `name` and `description`.

The schema therefore carries three roles at once: capability description, boundary description, and error-interface description.

```python
# src/build_tooling.py

def build_tool_schemas() -> list[dict]:
    return [
        {
            "name": "search_docs",
            "description": "Search an internal document corpus ...",
            "risk_level": "medium",
            "safety_boundary": "Read-only search...",
            "parameters": {
                "query": "string, required",
                "domain": "enum(...), required",
                "top_k": "integer, optional, default=3",
            },
            "returns": {...},
            "errors": [...],
        },
        ...
    ]
```

This structure shows that tools are not merely natural-language instructions for the model.

They are structured objects that drive later data construction.

That is why the current project can cover search, database, calendar, code, memory, and unsafe behaviors even though it only contains `6` tool schemas.

### 6.3 Why a Schema Is Not Only a Field List

Many teams understand a schema as "tool name plus parameter table".

That is not enough in an agent project.

The more important role is making the schema the shared language of all later modules.

Only then can the project generate task templates from schemas, validate arguments at execution time, identify error origins in recovery trajectories, and package call behavior into a unified learnable format.

### 6.4 The Real Engineering Value of the Schema

The schema is not for decoration.

It aligns tool definition, trajectory generation, environment execution, training packaging, and evaluation checks.

Without this alignment, an agent project easily becomes a set of isolated scripts.

![Figure 4: Tool Schema Structure](../../images/part10/10_7_fig04_tool_schema_structure.png)

## 7. Task Specifications and Trajectory Templates: Supervision Beyond Task Logs

Many teams begin with a simple idea: if the goal is to train agent tool use, why not collect historical call logs?

The problem is that logs do not automatically become supervision.

Raw logs usually have several weaknesses.

- Their behavior distribution is determined by historical traffic and may not cover key capabilities.
- Failure samples are messy and may not be directly learnable.
- They lack decision context for why to call, when to stop, and how to recover.
- Safety blocking and memory behavior are often not modeled separately.

Therefore, the project designs task specifications and trajectory templates before using logs as training material.

### 7.1 What Task Specifications Solve

A task specification connects what the user wants to do with how the agent should behave.

It defines more than request text.

It records task category, possible tools, expected trajectory variants, whether recovery is allowed, whether memory is involved, and whether safety blocking may be triggered.

### 7.2 How Templates and Task Specifications Are Organized

`src/build_tooling.py` does not keep templates as vague configuration.

It explicitly encodes the trajectory shape as `shape`.

```python
# src/build_tooling.py

def build_templates() -> list[dict]:
    return [
        {
            "template_id": "single_tool_success",
            "description": "One user turn, one tool call, one final answer.",
            "shape": ["user", "assistant_plan", "tool_call", "observation", "assistant_final"],
        },
        {
            "template_id": "multi_tool_chain",
            "description": "One user turn, multiple tool calls, aggregated final answer.",
            "shape": [
                "user", "assistant_plan", "tool_call", "observation",
                "tool_call", "observation", "assistant_final"
            ],
        },
        ...
    ]
```

This turns a trajectory template from an abstract concept into a structure that can be persisted, checked, and read by downstream scripts.

The later check `templates_cover_single_multi_and_safety` is possible because the template layer is already explicit.

Likewise, `build_task_specs()` stores not only the user question but also `category`, `session_id`, `objective`, `query`, `domain`, `answer_text`, and `recovery_mode`.

This layer defines task objects with execution intent, not ordinary prompts.

### 7.3 Why Templates Matter

Templates are not for mechanical copying.

They give different trajectory types a shared skeleton.

This keeps success, recovery, and block formats consistent.

It makes task comparison easier.

It also makes training, evaluation, and QA field alignment more reliable.

### 7.4 Current Template Scale

The current project contains `5` trajectory templates.

Around these templates it generates `22` raw trajectories.

The project does not win through massive scale.

It builds a method prototype through representative trajectory types.

![Figure 5: Relationship Between Task Specifications and Trajectory Templates](../../images/part10/10_7_fig05_task_specs_and_templates.png)

## 8. Trajectory-Type Design: Building Success, Recovery, and Block Together

If a team builds Tool-Use data by intuition, it often gets this pattern.

```text
user request -> model selects tool -> call succeeds -> answer returned
```

This pattern is valuable, but if the whole dataset looks like this, the model only learns tool calls on ideal paths.

The hardest part of real agents is not the ideal path.

### 8.1 Success Trajectories

Success trajectories teach the basic ability.

They show when the model should call a tool, how to construct arguments, how to read results, and how to finish the task.

### 8.2 Recovery Trajectories

Recovery trajectories teach a more important ability.

After the first call fails, can the model identify the error, repair arguments, choose another tool, or retry execution?

This category determines whether the model is brittle or resilient.

### 8.3 Block Trajectories

Block trajectories define boundaries.

When the request should not be executed, or when a call is unauthorized, dangerous, or non-compliant, the model must stop instead of pushing the system into risk.

### 8.4 How Recovery and Block Are Explicitly Constructed in Code

One of the most publishable points in P07 is that recovery is not treated as a runtime accident.

It is constructed as a dedicated supervision object in `src/generate_trajectories.py`.

Representative builders include `build_search_recovery(task)`, `build_db_recovery(task)`, `build_search_db_recovery(task)`, `build_memory_calendar_recovery(task)`, `build_memory_db_recovery(task)`, and `build_blocked(task, reason)`.

Recovery is therefore not a note added after an error.

It is an intentionally designed data asset.

For example, `build_search_recovery()` first creates bad arguments, then explicitly adds a repair plan and a second call.

```python
# src/generate_trajectories.py

def build_search_recovery(task: dict) -> list[dict]:
    bad_args = {"query": task["query"], "domain": "calendar", "top_k": 3}
    return [
        user_event(...),
        plan_event(..., "I will try the search tool..."),
        call_event(..., "search_docs", bad_args),
        plan_event(..., "The tool call failed, so I should fix the query arguments and retry."),
        call_event(..., "search_docs", corrected_args),
        final_event(...),
    ]
```

The implementation makes the intermediate decision "failure, analysis, retry" explicit.

For training, this is more valuable than keeping only two tool-call results.

`build_blocked(task, reason)` goes further by producing a blocking trajectory without triggering a tool call.

```python
# src/generate_trajectories.py

def build_blocked(task: dict, reason: str) -> list[dict]:
    return [
        user_event(...),
        plan_event(..., reason),
        final_event(..., status="blocked", blocked=True),
    ]
```

This shows that block is not a byproduct of tool failure.

It is an independent legitimate behavior branch.

### 8.5 Why the Three Trajectory Types Must Coexist

A useful agent must do three things.

- Act correctly when action is allowed.
- Recover when action goes wrong.
- Stop when action should not happen.

All three capabilities are necessary.

The current distribution is `success = 10`, `recovery = 9`, and `block = 3`.

This ratio is meaningful because recovery is not treated as a marginal add-on.

It is built almost at the same level as success.

![Figure 6: Success, Recovery, and Block Trajectory Taxonomy](../../images/part10/10_7_fig06_trajectory_taxonomy.png)

## 9. Simulated Execution Environment: The Environment Layer as a Constraint Surface

Without an environment layer, a trajectory is often static text.

The model says "I will call the tool", and the researcher writes the next step by hand.

That is suitable for demonstration, not for engineering.

### 9.1 What the Environment Layer Solves

The environment layer turns calls on paper into executable behavior.

Only after entering an environment can the project record whether arguments are valid, whether the tool succeeds, what the returned result is, whether retry is needed, whether memory was read or written correctly, and whether safety rules were triggered.

### 9.2 How the Environment Layer Is Implemented

`src/simulate_tool_env.py` is the best code-linked section of the chapter.

It does not connect to real external services.

Instead, it implements a set of controlled simulated tool functions: `search_docs(arguments, task_map)`, `sql_customer_db(arguments, task_map)`, `calendar_lookup(arguments, task_map)`, `python_exec(arguments, task_map)`, `memory_write(arguments, session_memory)`, and `memory_read(arguments, session_memory)`.

Each tool is independently testable.

The input is arguments plus task context.

The output follows a unified `(success, payload)` pattern.

The executor `execute_trajectory()` can therefore advance the whole trajectory with one interface.

```python
# src/simulate_tool_env.py

def execute_trajectory(trajectory: dict, task_specs: dict[str, dict]) -> tuple[dict, list[dict]]:
    session_memory = {}
    executed_events = []
    tool_logs = []
    total_calls = 0
    successful_calls = 0
    ...

    for event in trajectory["events"]:
        executed_events.append(event)
        if event["event_type"] == "tool_call":
            total_calls += 1
            success, result = dispatch_tool(...)
            tool_logs.append(...)
            if success:
                successful_calls += 1
            else:
                ...
```

The key value is that trajectories, environment execution, tool logs, and final metrics are connected.

Metrics such as recovery success rate and unsafe block rate become execution-derived results rather than paper statistics.

### 9.3 Why the Python Execution Tool Deserves Separate Treatment

`python_exec()` contains a useful safety example.

The code does not execute unconditionally.

It first checks `UNSAFE_CODE_TOKENS`; if a dangerous pattern is found, it returns `unsafe_code`.

Even in a simulated environment, the project treats executable tools as higher-risk objects.

This detail is a good example of moving safety boundaries earlier in engineering design.

### 9.4 Relationship Between Simulation and Real Environments

The simulated environment is not the endpoint.

It is a good starting point.

It lets the team solve basic questions first: are trajectories reasonable, fields aligned, recovery logic valid, and metrics measurable?

Only after that should the team decide how to migrate to a real environment.

The project report also states that the current environment is based on simulated execution rather than direct production-tool access.

![Figure 7: Simulated Tool-Environment Execution Loop](../../images/part10/10_7_fig07_simulated_env_loop.png)

## 10. Process Decomposition: How P07 Lands from Definition to Evaluation

The core process has six steps.

1. `src/build_tooling.py`: build tool specifications.
2. `src/generate_trajectories.py`: generate trajectory samples.
3. `src/simulate_tool_env.py`: execute in the simulated tool environment.
4. `src/prepare_agent_dataset.py`: package the agent dataset.
5. `src/evaluate_tooluse.py`: evaluate tool-use data.
6. `src/run_p7_checks.py`: run project checks.

These six steps are not complicated, but together they form the minimum closure required for a data factory.

### 10.1 Define First, Do Not Generate First

The first step is not "find some data".

It is to define tool specifications.

This order matters because an undefined tool space makes every later trajectory unstable.

### 10.2 Build Trajectories Before Entering the Environment

The second step generates raw trajectories before execution.

The system first focuses on behavior design and then on environment validation.

This separates the task layer from the execution layer.

### 10.3 Keep Execution Logs Before Packaging Training Data

The project does not directly write execution into training samples.

It first preserves event-level records and then reorganizes them in post-processing.

This design keeps space for analysis, replay, and rework.

### 10.4 Evaluate First, Then Check Consistency

Evaluation and checking are different.

Evaluation asks how the data behaves.

Checking asks whether code, data, and reports agree.

Separating the two is a clear sign of engineering maturity.

![Figure 8: Six-Step P07 Pipeline](../../images/part10/10_7_fig08_pipeline_steps.png)

## 11. Recovery Mechanism: The Supervision Value from Failure to Recovery

The most important point in P07 is that it does not discard failure samples.

It explicitly keeps recovery trajectories.

### 11.1 Why Failure Has Value

For ordinary QA models, wrong outputs are undesirable.

For agents, the first failure is not always the whole task failure.

Many real tasks depend on whether the model can continue after failure.

If an argument format is wrong, the model should repair it and retry.

If no result is found, it should try another query strategy.

If memory is insufficient, it may need to gather additional information before continuing.

If one tool is unsuitable, it may need to switch to an alternative tool.

### 11.2 The Essence of Recovery Training

Recovery training does not teach the model to make mistakes.

It teaches the model how to recover from mistakes.

That learning objective is different from training on success paths only.

### 11.3 Why Recovery Is Closer to the Real World Than Pure Success Data

In real user environments, tools fail, arguments are wrong, dependencies fluctuate, permissions change, and queries return empty results.

A model that has only seen smooth paths during training will be fragile after launch.

In the current project, `recovery = 9`, nearly the same scale as `success = 10`.

This shows that recovery behavior is treated as a primary capability, not as a few token failure cases.

![Figure 9: Argument Repair and Retry Flow](../../images/part10/10_7_fig09_recovery_flow.png)

## 12. Memory Trajectories: Modeling Memory Behavior

Many first-time agent projects understand memory as "put more previous text into the prompt".

Engineering memory behavior is more than that.

### 12.1 What Memory Solves in an Agent

Memory solves state.

It allows the system to remember user preferences, previous actions, intermediate environmental results, and facts needed to continue a multi-turn task.

### 12.2 Why Memory Behavior Must Be Modeled Separately

Memory is not merely a linear extension of natural-language context.

It has explicit operational questions.

When should the agent read?

When should it write?

What should it write?

What should not be written?

How should retrieved memory change the next decision?

If these questions are not modeled separately, the model can fail in both directions.

It may fail to remember what should be remembered.

It may also persist information that should not be persisted.

### 12.3 Current Memory Signals

The current training set contains `103` records, including `34` memory records.

The memory success rate is `100%`.

Memory is therefore not an appendix in this project.

It is a core capability dimension that is retained and counted separately.

### 12.4 Why Memory Data Is Especially Suitable for Early Explicit Construction

Correct memory behavior is highly specification-dependent.

If the team waits for organic online generation, it is difficult to obtain high-quality and interpretable training signals.

Controlled templates in the early phase provide a more stable foundation.

![Figure 10: Memory Read/Write Trajectory](../../images/part10/10_7_fig10_memory_trajectory.png)

## 13. Safety Blocking: The Boundary Role of Block Samples

Compared with ordinary generation models, an agent is more dangerous because it can act.

Once the model can call tools, safety problems are no longer only about saying the wrong thing.

They can become doing the wrong thing.

### 13.1 What Unsafe Block Solves

Unsafe block answers whether the request is unauthorized, whether it involves dangerous operations, whether execution should be refused, and whether the agent should provide only an informational response without actually calling a tool.

### 13.2 Why Block Is Not Simple Refusal

The value of block samples is not merely teaching the model to say no.

They teach structured blocking in a tool-use setting.

The model should identify the risk source, avoid dangerous calls, provide safer alternatives when feasible, and keep system state out of uncontrolled areas.

### 13.3 Current Safety Signals

The current unsafe block rate is `100%`.

The unauthorized tool-call rate is `0%`.

The training set contains `9` safety records.

Although the sample size is not large, safety boundaries are clearly part of the core evaluation.

### 13.4 Why Block Data Should Enter the Training Set Early

If safety boundaries are added only as inference-time rules, the system can become adversarial: the model wants to act while the rules try to block it.

A better approach is to let the model learn during training which actions should not be taken.

![Figure 11: Unsafe Block Decision Split](../../images/part10/10_7_fig11_unsafe_block.png)

## 14. Dataset Reorganization and Training Packaging: From Logs to Training Interface

After the environment runs, the project does not throw execution logs directly into a training framework.

It performs a crucial post-processing step: reorganizing event-level records into training assets.

### 14.1 Why Raw Logs Are Not Suitable for Direct Training

Logs are better for machine recording than for model learning.

They often have inconsistent granularity, execution-oriented rather than supervision-oriented formats, weak instruction/output alignment, poor train/validation/smoke splitting support, and limited version-management convenience.

### 14.2 How Trajectories Are Reorganized

The key work in `src/prepare_agent_dataset.py` is not file copying.

It breaks each executed trajectory into event-level training records.

The two core functions are `render_context(events)` and `build_records(trajectory)`.

`render_context(events)` renders user events, plans, tool calls, and observations into a unified context.

```python
# src/prepare_agent_dataset.py

def render_context(events: list[dict]) -> list[str]:
    rendered = []
    for event in events:
        if event["event_type"] in {"user", "assistant_plan", "assistant_final"}:
            rendered.append(f"{event['event_type']}: {event['content']}")
        elif event["event_type"] == "tool_call":
            rendered.append(f"tool_call: {event['tool_name']} {event['arguments']}")
        else:
            rendered.append(f"observation: {event['tool_name']} -> {event['content']}")
    return rendered
```

This step translates system logs into trainable context.

`build_records()` goes further.

It does not produce only one sample per trajectory.

It walks through the trajectory step by step and emits supervised records with fields such as `record_id`, `trajectory_id`, `task_id`, `category`, and `variant`.

That is why `22` raw trajectories can become `103` training records.

### 14.3 Training-Interface Artifacts

The project finally outputs the following artifacts.

- `data/training/agent_tooluse_dataset.jsonl`
- `data/training/train.jsonl`
- `data/training/val.jsonl`
- `data/training/smoke_test.jsonl`
- `data/training/training_manifest.json`

The output is no longer a few run results.

It is a set of assets directly consumable by training.

![Figure 12: Repacking Event Logs into Training Samples](../../images/part10/10_7_fig12_dataset_repacking.png)

## 15. Metric System: Signals Beyond Tool Success Rate

In agent projects, teams often focus on one number: tool-call success rate.

That number is important, but it can easily mislead if used alone.

### 15.1 Current Key Metrics

The core metrics are listed below.

- Tool schemas: `6`
- Templates: `5`
- Raw trajectories: `22`
- Variant distribution: `success = 10`, `recovery = 9`, `block = 3`
- Tool-call success rate: `78.57%`
- Trajectory success rate: `100.00%`
- Recovery success rate: `100.00%`
- Unsafe block rate: `100%`
- Memory success rate: `100%`
- Unauthorized tool-call rate: `0%`
- Training records: `103`

### 15.2 Why Tool Success Rate Is Not Task Success Rate

Tool-call success rate measures whether a single call went smoothly.

Trajectory success rate measures whether the whole task was completed.

If the project models recovery explicitly, a failed single tool call can still be repaired and the task can still succeed.

From the agent perspective, that remains a successful trajectory.

### 15.3 Why This Metric Combination Matters

The interesting part is that tool-call success rate is only `78.57%`, while trajectory success rate and recovery success rate are both `100%`.

This shows that the recovery mechanism is working at the data level.

A lower tool-call success rate does not automatically mean the system is ineffective.

It may mean that the project keeps failure and repair as training signals instead of preserving only ideal paths.

## 16. Metric Interpretation: The Weight of Recovery Ability

A common misconception is that a good agent should simply avoid mistakes.

In an ideal world this is true.

From a data engineering perspective, it is not realistic enough.

### 16.1 What a Useful Agent Must Be Able to Do

A useful agent needs at least three layers of ability.

- Under normal conditions, it can complete the task.
- Under abnormal conditions, it can recover the task.
- Under dangerous conditions, it can block the task.

Training only the first layer makes a model look good in demos and fragile in the real world.

### 16.2 Why Recovery Is More Valuable Than Clean Success Data

Recovery samples teach a behavior closer to system intelligence.

The model learns to identify problems, understand failure reasons, generate repair actions, retry, and switch strategies when needed.

These abilities are harder and more useful than succeeding on the first attempt.

### 16.3 Why This Interpretation Must Be Kept

If only the number `78.57%` is reported, it can easily be misunderstood as weak performance.

In an agent setting, it instead shows that the project does not beautify the data.

It preserves and uses failure-recovery behavior honestly.

## 17. Evaluation and Checks: Performance Evaluation plus Consistency Checking

Many projects stop once they can output metrics.

For a data factory, that is not enough.

Even when metrics look reasonable, code, data, and reports may still disagree.

### 17.1 What Evaluation Answers

Evaluation answers whether tool calls are effective, whether recovery succeeds, whether memory is correct, whether block works, and whether the training-data distribution matches expectations.

### 17.2 Metric-Calculation Structure

`src/evaluate_tooluse.py` does not merely count rows.

It places tool-level, trajectory-level, recovery-level, safety-level, and memory-level metrics into one `metrics` dictionary.

This matters because P07 evaluates the full behavior distribution rather than a single success number.

The metrics include `tool_schema_count`, `template_count`, `trajectory_count`, `category_distribution`, `variant_distribution`, `tool_call_success_rate`, `trajectory_success_rate`, `recovery_success_rate`, `unsafe_block_rate`, `unauthorized_tool_call_rate`, and `memory_success_rate`.

Because the evaluation script is calculated from executed artifacts and the manifest, the report can contain both "tool-call success rate 78.57%" and "trajectory success rate and recovery success rate 100%".

These numbers are not isolated.

They describe different levels of behavior.

### 17.3 What Checks Answer

Checks answer whether required files are present, whether schema fields are complete, whether templates cover single-step, multi-step, and safety scenarios, whether trajectory variants are complete, whether observation and decision chains exist, whether memory cases succeed, and whether code and reports agree.

### 17.4 How the Check Mechanism Is Implemented

`src/run_p7_checks.py` first runs command-level checks and then artifact-level checks.

Command-level checks run `py_compile` and `evaluate_tooluse.py`.

Artifact-level checks validate rules such as `required_files_exist`, `tool_schema_fields_complete`, `templates_cover_single_multi_and_safety`, `variant_coverage`, `observations_and_decision_chain_present`, and `memory_cases_succeed`.

The current project has `12` check items.

All pass, and the overall status is `PASS`.

This makes the chapter more than a notebook story.

It is an engineering loop in which code is verifiable, artifacts are inspectable, and reports are traceable.

![Figure 13: Evaluation and Consistency-Check Loops](../../images/part10/10_7_fig13_eval_and_checks.png)

## 18. Current Limitations and Risks: Boundaries of the Method Prototype

Writing limitations does not weaken the project.

It increases credibility.

P07 has at least three explicit limitations.

### 18.1 The Tool Scope Is Still Small

The current project has only `6` tool categories.

They demonstrate the method, but they do not approximate the complex, multi-permission, multi-system tool space of real enterprise agents.

### 18.2 The Call Layer Is Still Not Stable Enough

The tool-call success rate of `78.57%` shows that the raw call layer remains fragile.

Although the recovery layer raises the task success rate, this does not mean the underlying call problems are solved.

### 18.3 The Safety Boundary Is Not Rich Enough

The existing unsafe block and unauthorized-call samples cover basic boundaries.

There is still a large gap before real-world unauthorized chains, prompt injection, sensitive-data exfiltration, and complex permission negotiation.

### 18.4 Why Limitations Should Be Written Early

The value of a method prototype is not pretending that it has solved everything.

Its value is showing later teams where the next investment is most worthwhile.

## 19. Extension Directions: Toward More Realistic Enterprise Agents

If P07 is treated as a minimal reproducible agent data factory, the next extensions include several directions.

### 19.1 Expand Tool Types

The current search, database, calendar, code, and memory tools can be extended to email, documents, tickets, approvals, knowledge bases, spreadsheets, and workflows.

These are closer to enterprise scenarios.

### 19.2 Expand Cross-Tool Chains

Many real tasks cannot be solved by one tool.

They require retrieval, query, calculation, writing, notification, and other steps in sequence.

Future data should strengthen these cross-tool chain samples.

### 19.3 Expand Cross-Session State

The project already covers memory.

More complex long-term state management, session switching, task resumption, and historical dependencies deserve continued construction.

### 19.4 Expand Safety Governance

Future versions can introduce richer unauthorized-call cases, prompt injection, sensitive-information leakage, data contamination, and policy-bypass scenarios.

This would make safety boundaries closer to launch readiness.

### 19.5 Expand Evaluation Dimensions

Beyond the current metrics, the project can add finer-grained tool-selection accuracy, argument correctness, retry efficiency, final-answer quality, and multi-turn consistency.

![Figure 14: P07 Roadmap](../../images/part10/10_7_fig14_roadmap.png)

## 20. P07's Key Position: Connecting "Can Speak" with "Can Act"

Many introductory materials for large-model engineering stop at making the model answer better.

Agent scenarios require more.

The model must not only speak; it must act.

It must not only act; it must recover when acting goes wrong.

It must not only recover; it must know when it should not act at all.

P07 matters because it shows that tool-use behavior can be structured, recovery trajectories can be trained, memory behavior can be modeled explicitly, safety blocking can enter supervision, and execution, evaluation, and checking can form a loop.

In the larger capability chain, it moves language supervision toward behavior supervision.

## 21. Difference from Ordinary Function-Calling Data: Characteristics of Agent Behavior Data

At first glance, P07 also contains tool schemas, call arguments, and execution results.

It may look similar to ordinary function-calling data.

The difference is substantial.

### 21.1 Ordinary Function-Calling Data Emphasizes Single Mapping

Traditional function-calling samples usually ask four questions.

- What is the user intent?
- Which function should be called?
- How should arguments be filled?
- How should returned values be presented?

This is a static input-to-call-to-output mapping.

### 21.2 Agent Tool-Use Emphasizes Behavior Process

P07 emphasizes why the agent should call a tool now, what it should do if the call fails, how it should handle multi-turn memory, what it should do when a request is dangerous, and how multi-step behavior becomes a training asset.

This is no longer simple function calling.

It is closer to behavior data engineering for executable agents.

### 21.3 Why This Distinction Must Be Clear

Many teams underestimate agent data difficulty.

They assume adding more function-call samples is enough.

P07 shows that the difficult questions are what to do after a wrong call, whether to stop at the boundary, and how to remember across turns.

## 22. Transfer to Other Agent Scenarios: Method Value of P07

Agent Tool-Use is not the only direction that needs a behavior data factory.

It is a strong template because it has a clear tool space, decomposable behavior chains, important recovery mechanisms, rigid safety boundaries, and a necessary evaluation loop.

These same characteristics appear in enterprise copilots, automation workflow assistants, developer assistants, operations assistants, and multi-agent collaboration systems.

### 22.1 Designs That Can Transfer Directly

- The definition chain from tool schema to task specification.
- The parallel construction of success, recovery, and block.
- The strategy of using a simulated environment before real integration.
- Separate modeling for memory and safety boundaries.
- Training packaging plus consistency checks.

### 22.2 Parts That Cannot Be Copied Directly

- Tool types and permission systems must be rewritten.
- Enterprise safety boundaries are usually more complex.
- Multi-team workflow collaboration is harder than a single-agent case.
- Real systems have far more exception types than a simulated environment.

### 22.3 The Most Transferable Core Method

The transferable asset is not one call template.

It is the method chain below.

```text
define tool space -> design task specifications -> construct success/recovery/block trajectories -> execute and log in an environment -> reorganize into training assets -> build evaluation and check loops
```

## 23. Main Deliverables

This section lists the main deliverables.

### 23.1 Tool and Processing Intermediate Artifacts

- `data/processed/tool_schemas.json`
- `data/processed/trajectory_templates.json`
- `data/processed/task_specs.json`
- `data/processed/raw_trajectories.jsonl`
- `data/processed/executed_trajectories.jsonl`
- `data/processed/tool_execution_log.jsonl`
- `data/processed/execution_summary.json`

### 23.2 Training-Interface Artifacts

- `data/training/agent_tooluse_dataset.jsonl`
- `data/training/train.jsonl`
- `data/training/val.jsonl`
- `data/training/smoke_test.jsonl`
- `data/training/training_manifest.json`

### 23.3 Report and Verification Artifacts

- `data/reports/p7_report.md`
- `data/reports/p7_metrics.json`
- `data/reports/p7_test_results.json`
- `data/reports/p7_test_report.md`

The deliverable list shows that P07 leaves behind more than a completed experiment.

It leaves a full asset chain from tool definitions to training interfaces and evaluation reports.

## 24. Closing: What an Agent Data Factory Trains Is Behavior, Not Only Calls

When people see a Tool-Use project, they often interpret it as teaching the model to call functions.

P07 demonstrates something deeper.

It trains not a mechanical API trigger, but a behavior system that works in a tool world.

This system needs to know when to act, how to call tools, how to repair bad calls, how to remember state, and how to stop under risk.

The value of P07 is therefore not only its current sample count, tool count, or metric values.

Its value is that it answers a central question for the agent era.

If we want models to truly learn to do things, behavior itself must become a data engineering object.

That is the core meaning of the Agent Tool-Use data factory.

## Special Topic: Scenario Libraries and Pre-Launch Gates

Before an Agent Tool-Use project is deployed, the most common failure is not that the model cannot call tools.

The more common failure is that the team has not built a complete scenario library and gate conditions.

With only a few demo tasks, the system looks intelligent.

In real environments, tool space, exception types, and safety requirements all expand at the same time.

Problems appear quickly.

### 1. Scenario Libraries Must Not Cover Only Success Paths

A usable agent scenario library should cover at least three categories.

- Normal success tasks validate whether the model can complete goals under standard assumptions.
- Recovery tasks validate whether the model can return to a correct path after tool failure, missing parameters, or environmental conflict.
- Blocking tasks validate whether the model can stop before unauthorized, sensitive, or non-compliant requests.

All three are necessary.

Success without recovery makes the system fragile.

Success and recovery without block make the system uncontrolled in high-risk scenarios.

The methodological value of P07 is that it treats all three trajectory types as peer assets.

### 2. Gates Should Cover Tool Correctness, Recovery Ability, and Safety Boundaries

Before launch, an agent project should establish at least three gates.

The tool-correctness gate confirms that the model selects the right tool, fills key arguments correctly, and understands returned results.

The recovery gate confirms that the model takes reasonable next steps after errors, conflicts, and interruptions.

The safety-boundary gate confirms that the model does not continue acting under prohibited conditions and is not easily induced to bypass policy.

Passing only the first gate makes the system look like it can use tools.

It still does not qualify for real workflows.

Errors and risks are almost certain in real workflows, so recovery and safety blocking are mainline capabilities rather than marginal requirements.

### 3. The Scenario Library Should Grow with Failure Cases

The scenario library in a project like P07 should not be written once and left unchanged.

It should grow through failure replay.

Whenever the system selects the right tool but misunderstands arguments, repeats the same error after a failed call, continues blindly when it should use memory or ask for clarification, tries to bypass a safety policy, or loses intermediate state in a multi-tool chain, the case should be absorbed into the scenario library.

Such a library represents the system's most fragile boundaries, not only the parts it is best at demonstrating.

## Special Topic: Operating and Reviewing Agent Behavior Data

Compared with ordinary function-calling samples, agent behavior data is closer to an operational log asset.

Because it describes behavior processes, it needs operating and review mechanisms.

Otherwise, even a growing training set may simply accumulate old problems in new forms.

### 1. Review Granularity Should Land on the Behavior Chain

Many teams review agent failures only by looking at the final result.

For Tool-Use, the more useful review unit is the behavior chain.

Which tool was selected first?

Why was a warning in the returned value ignored?

Why did the third step not trigger clarification?

Why did the fourth step continue under an unauthorized condition?

Did the final failure come from one step, or from the whole strategy path?

Only behavior-chain review allows a team to turn a problem back into training assets.

Otherwise the review remains a vague statement that the system failed again.

### 2. Operations Should Let Multiple Roles Inspect the Same Trajectory

Agent projects naturally involve data engineering, model engineering, platform, security, and product roles.

Each role cares about a different aspect of the same problem.

The better approach is to let them discuss the same trajectory rather than separate summaries.

An effective operating mechanism keeps key trajectories and execution logs, lists representative success, recovery, and block cases in evaluation reports, builds fixed replay sets for high-risk failures, and prioritizes which trajectories are fixed or still unstable during version review.

This gives different roles a shared view of the system problem.

For behavior systems, that shared view is critical.

### 3. The Agent Data Factory Ultimately Serves Continuous Iteration

P07 already shows a key method: behavior data is not finished after generation.

It should cycle through execution, evaluation, review, and retraining.

If the project later expands into a more realistic enterprise agent environment, this cycle becomes increasingly important.

In the long term, teams need the ability to continuously add high-value scenarios, convert failures into replay and training samples, update safety boundaries and block rules, and track multi-turn memory, recovery efficiency, and final task completion.

Once these practices form a fixed rhythm, the Agent Tool-Use data factory evolves from a project chapter into behavior data infrastructure.

## Chapter Summary

This chapter used the Agent Tool-Use data factory as a case to show how tool schemas, task environments, and interaction trajectories can be organized into agent tool-use training data.

Its main value is connecting task definition, data boundaries, architecture decisions, sample schemas, metric acceptance, and reproducible resources in one chain.

The project is therefore not only a set of operating steps.

It becomes a reviewable case study.

The project boundary should also be retained clearly.

It mainly uses a simulated environment and a controlled tool set.

It does not represent an open internet environment or a real enterprise permission system.

In larger-scale, higher-risk, or more compliance-constrained scenarios, teams should reassess data sources, permission status, manual review ratios, runtime cost, and failure rollback plans.

As part of Part 14, this chapter validates earlier methods at the project level.

Readers can combine the case with the data recipes in Part 13, the platform-governance chapters, and the appendix checklists to form a loop from method understanding to engineering delivery.

## References

1. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629.
2. Schick, T., Dwivedi-Yu, J., Dessi, R., Raileanu, R., Lomeli, M., Hambro, E., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. arXiv:2302.04761.
3. NIST. (2023). Artificial Intelligence Risk Management Framework (AI RMF 1.0). National Institute of Standards and Technology.
4. OWASP Foundation. (2025). OWASP Top 10 for Large Language Model Applications.
5. OpenTelemetry Authors. (2026). OpenTelemetry Documentation. https://opentelemetry.io/docs/
