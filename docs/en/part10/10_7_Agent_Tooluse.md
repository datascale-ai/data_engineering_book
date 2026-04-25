# Project 7: Agent Tool-Use Data Factory

## Overview of this chapter

P07 focuses on organizing the Agent’s tool usage behavior into trainable, evaluable, and scalable data assets. The focus of the chapter is not on individual function calls, but on the complete data chain between tool specifications, execution traces, recovery behaviors, safety boundaries and training encapsulation.

This chapter can be understood according to four main lines:

* Tool specification and task design: clarify the schema, calling conditions and task structure.
* Execution trajectory and recovery modeling: retain different types of behavior chains such as success, failure, recovery, etc.
* Security boundary and memory mechanism: Incorporate unsafe blocks, permission restrictions and memory reading and writing into supervision objects.
* Data encapsulation and evaluation acceptance: forming trainable samples, verification indicators and inspection mechanisms.

If read in engineering order, this chapter corresponds to a complete link:

**Tool schema -> Task design -> Trajectory generation -> Simulation execution -> Recovery modeling -> Security blocking -> Data encapsulation -> Evaluation and acceptance**

The core goal of this structure is to build an Agent Tool-Use data pipeline that can cover execution, recovery and security control.

---

## 1. Project background: The necessity of Agent Tool-Use data factory

General-purpose large models have demonstrated strong language capabilities in tasks such as open-domain question answering, summarization, and writing. However, once they enter the Agent scenario, language capabilities alone are obviously not enough.

The most common problems fall into three categories.

The first category is **motion distortion**. The model knows that it should "check it out", but it doesn't know which tool to call, or it should search the database instead, or it should read the memory first but answer directly.

The second category is **execution distortion**. Although the model selects the right tool, it fills in the wrong parameters, or does not understand the tool schema, or does not continue to reason after getting the returned results. This means that saying "I want to call the tool" does not mean that the tool chain will actually be executed.

The third category is **boundary distortion**. Models may still execute mechanically when user requests involve dangerous operations, unauthorized access, or memory that should not be persisted. An Agent without security blocking and boundary modeling is very dangerous in real scenarios.

Therefore, the goal of P07 is not to simply collect some function call examples, but to build an **Agent Tool-Use data factory** to organize tool definitions, task trajectories, recovery behaviors, memory reading and writing, and security blocking into a reusable data production line.

This production line serves not a one-time experiment, but a methodology:

> When the team needs to migrate from simple single-tool Q&A to complex multi-tool Agent, enterprise Copilot, workflow assistant and embodied task agent in the future, what can be truly reused is not a certain function call prompt, but this set of engineering methods "from tool specification to supervision track".

![Figure 1: Agent Tool-Use Data Factory Overview](../../images/part10/10_7_fig01_agent_tooluse_factory_overview.png)

---

## 2. Project goals and boundaries

### 2.1 Project Goals

This project focuses on the following four goals.

**Goal 1: Establish a transformation link from tool specification to supervision trajectory. **
That is, the tool schema, task template and execution environment are converted into structured Agent data suitable for training.

**Goal 2: Establish a trajectory system covering success, recovery, and block. **
This project does not uniformly make all samples into "successful call cases", but clearly retains success trajectories, failure recovery trajectories and safety blocking trajectories, allowing the model to learn a more complete behavior distribution.

**Goal 3: Establish an auxiliary supervision layer for memory and security boundaries. **
Agents are not just tool invokers, they are also involved in multiple rounds of context and persistent state management. Therefore, the project regards memory reading and writing and unsafe block as independent and important training signal construction.

**Goal 4: Form data assets that can be directly consumed by the training side. **
The final output includes not only intermediate execution logs, but also training interface layer products such as `agent_tooluse_dataset.jsonl`, `train.jsonl`, `val.jsonl`, `smoke_test.jsonl`, and `training_manifest.json`.

### 2.2 Project Boundaries

In order to maintain project reproducibility, this project explicitly sets several boundaries.

#### 1) Tool range boundary

The current scope of tools includes capabilities such as search, database, calendar, Python execution, and memory, but it is still a smaller, controllable collection of tools rather than a complete enterprise-level tool ecosystem.

#### 2) Execution environment boundary

This project uses a **simulated execution environment**, with the goal of reproducing key behaviors in Agent tool calls at low cost, rather than directly accessing real production permissions. This is better suited for teaching, validation, and demonstration of methods.

#### 3) Sample size boundary

The total number of samples in the current project is not large, but the trajectory types are relatively complete. It is more suitable as a method demonstration and factory prototype, rather than claiming to have covered all Agent behaviors in the real world.

#### 4) Security capability boundary

The project has incorporated unsafe block and unauthorized call constraints, but the relevant boundaries are still relatively basic, and there is still a significant gap between the complex permission system and offensive and defensive pressure in real online scenarios.

### 2.3 The role of boundary description

It is very important to write clear boundaries. Because there are usually only two ways to write a project case:

* One is to write the project so that "everything can be done";
* The other is to write the project as "what can be done stably and under what conditions".

The latter is obviously more credible and more suitable for reuse by the team.

---

## 3. Project positioning: P07’s capability chain position

If the whole book is regarded as a large model data engineering capability chain, then P07 is at the key position of "from conversation model to executable agent".

Previous chapters may have discussed methodologies such as general SFT, preference data, RAG, vertical domain supervision constructs, etc. The value of this chapter is to push these methods further into a scenario closer to system behavior: **Tool usage**.

In other words, this chapter does not teach the basics of function calling again, but shows:

* How to design supervision data in a scene that requires real action closed loop;
* Why the success trajectory is not enough to support Agent behavior learning;
* Why recovery and block need to be constructed in parallel with ordinary tool calls;
* Why memory behavior cannot be treated as an adjunct to ordinary text context;
* How to take assessment, inspection, consistency and go-live boundaries into account early in the project.

In this sense, the most important thing about this chapter is not a "tool list" but answering a larger question:

> How should the Agent Data Factory be designed as a set of continuous production capabilities instead of a pile of scattered call logs?

---

## 4. Overall architecture: Agent data pipeline from tool schema to training assets

From an engineering perspective, this project can be broken down into three floors.

### 4.1 The first layer: Tool specification layer

This layer solves the problem of "what callable capabilities are available to the Agent and how these capabilities are understood by the machine." Mainly include:

* Tool schema definition
* Parameter field specifications
* Call constraint description
* Tool category label
* Description of authorization and risk boundaries

The goal of this step is not to generate samples, but to clearly define the tool world first.

### 4.2 The second layer: trajectory construction layer

This layer solves "how to let the model see representative Agent behavior." Mainly include:

* Task specification design
* Single-step and multi-step trajectory templates
* success trajectory generation
* recovery trajectory generation
* memory trajectory construction
* unsafe block trajectory construction

This step is the core part of the entire project, because it determines whether the model learns "a model that outputs a function name" or "an Agent that advances tasks in the environment."

### 4.3 The third layer: execution evaluation layer

What this layer solves is "whether these trajectories can really be used for training and verification." Mainly include:

* Simulation environment execution
* Tool logging
* Event-level sample reorganization
*Dataset encapsulation
*Indicator evaluation
* Project check script

At this point, the project has changed from "calling sample collection" to "engineering closed loop".

![Figure 2: Agent Tool-Use three-layer architecture diagram](../../images/part10/10_7_fig02_three_layer_architecture.png)

---

## 5. Project pre-processing: key aspects of Agent data factory

The difficulty of the Agent Tool-Use data factory is not just to "make the tool call sample", but to first clearly write down which engineering aspects need to be explicitly constrained. As behavioral complexity increases, if these key aspects are mixed together, the subsequent trajectory generation, execution verification and training encapsulation will quickly get out of control.

The current project involves at least the following four key aspects.

### 5.1 Capability and boundary definition surface

This layer is responsible for defining tool boundaries, task types, recovery rules, and security constraints. The first thing to answer here is: what is "reasonable Agent behavior", rather than just caring about whether a certain trajectory can run through.

### 5.2 Data and interface organization

This layer is responsible for schema, JSONL disk placement, intermediate product management, segmentation, version control and inspection scripts. It focuses on whether data assets can be stably produced and reused.

### 5.3 Environment and execution control plane

This layer is responsible for implementing the simulation tool environment, constructing return results, injecting failure conditions and recording execution logs. Without this layer, many trajectories can only remain on paper.

### 5.4 Evaluation and Security Verification Surface

This layer is responsible for defining the criteria for success, recovery, and block, checking whether the memory behavior is correct, evaluating whether the security boundaries are observed, and ensuring that the report is consistent with the product.

### 5.5 Pre-positioning of key aspects

Because when many teams are working on Agent data for the first time, what really gets stuck is not "not knowing how to write function calls", but rather the failure to write down these key prepositions clearly, resulting in:

* The tool schema is not maintained;
* No one has defined the failure recovery logic;
* The execution log cannot be reviewed;
* The report and training set do not match each other;
* The security boundary depends entirely on pre-launch patches.

Therefore, it is not the division of labor that needs to be explicitly written down, but the engineering constraints themselves. **Agent Tool-Use is more like system behavior data engineering than prompt word skill demonstration. **

![Figure 3: Schematic diagram of key engineering aspects of Agent Data Factory](../../images/part10/10_7_fig03_roles_and_responsibilities.png)

---

## 6. Tool specification layer: schema as the starting point for training

Compared with 10-2, if this chapter of P07 only writes "Why schema is needed", it will appear abstract. Because the value of this project lies not only in the methodology, but also in the fact that the schema, templates, task specifications, execution logs and evaluation interfaces have been completely connected in the code**. It can also be seen from the source code sequence of notebook expansion that the entire project is organized according to the main line of `build_tooling -> generate_trajectories -> simulate_tool_env -> prepare_agent_dataset -> evaluate_tooluse -> run_p7_checks`, rather than a stack of scattered scripts.

### 6.1 Tool schema as a first step

The tool schema determines that the model must know at least the following things:

* What is the tool called;
* What it does;
* What parameters are required;
* What type is the parameter?
* Which calls are legal;
* Which scenes should not be called.

If this layer is not clearly defined, even if the model "want to use tools", it can only rely on fuzzy guesses to call it.

### 6.2 Structured implementation of tool specifications

In `src/build_tooling.py`, the project generates tool specifications, trajectory templates and task specifications at the same stage, instead of hand-writing a bunch of JSON and then passively reading it by subsequent scripts. The three most critical functions here are:

* `build_tool_schemas()`: Generate tool definition;
* `build_templates()`: Generate trajectory template;
* `build_task_specs()`: Generate task specifications.

The combination of these three functions is actually the "behavioral world definition layer" of P07. It does not simply write out the tool name, but fixes the constraints on which subsequent trajectory generation and execution depend. For example, in addition to `name` and `description`, `build_tool_schemas()` also provides `risk_level`, `safety_boundary`, `parameters`, `returns` and `errors`, which makes the schema simultaneously assume the three roles of **capability description, boundary description and error interface description**.

A highly summarized code form is as follows:

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

This structural description shows that the project does not regard the tool as "a natural language description for the model", but as a set of structured objects that can drive subsequent data construction. This also explains why although the current project only has `6` tool schema, it can already cover multiple types of behavior boundaries such as search, db, calendar, code, memory, and unsafe.

### 6.3 Why schema is not just a list of fields

Many people understand schema as "tool name + parameter list", but in the Agent project, this is not enough. What's more important is to make schema a common language for all subsequent modules. Only in this way can the project later be able to:

* Automatically generate task templates based on schema;
* Verify whether the parameters are compliant during execution;
* Determine where the error comes from in the recovery trace;
* Unify the calling behavior into a learnable format during training.

### 6.4 The true value of schema in engineering

The schema is not to look good, but to align the layers of "tool definition - trajectory generation - environment execution - training encapsulation - evaluation and inspection". Without this level of alignment, an Agent project can easily become a bunch of siled scripts.

![Figure 4: Tool schema structure diagram](../../images/part10/10_7_fig04_tool_schema_structure.png)

---

## 7. Task specifications and trajectory templates: Supervision structure beyond task logs

Many teams will initially think: Since the goal is to train the use of Agent tools, wouldn't it be enough to collect some historical call logs? But the reality is that logs are not naturally equal to surveillance data.

Because the original log usually has several problems:

* Behavior distribution is determined by historical traffic and does not necessarily cover key capabilities;
* Failure samples are messy and may not be directly learned;
* Lack of decision-making context of "why to call", "when to give up" and "how to recover";
* Security blocking and memory behavior are often not modeled separately.

Therefore, this project does not directly use log training, but first designs **task specifications and trajectory templates**.

### 7.1 What problem does the task specification solve?

The role of task specifications is to connect "what the user wants to do" and "how the Agent should behave". It defines not just the request text, but also:

*Task category;
*Tools that may be involved;
* Expected trajectory variants;
* Whether to allow recovery;
* Whether memory is involved;
* Whether it is possible to trigger a security block.

### 7.2 Organization of templates and task specifications

`src/build_tooling.py` does not write the template as an abstract configuration, but directly encodes the template shape explicitly into `shape`. For example:

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

This way of writing changes the "trajectory template" from an abstract concept into a structure that can be directly placed on the disk, directly inspected, and directly read by downstream. The reason why `run_p7_checks.py` can check `templates_cover_single_multi_and_safety` later is because the template layer has been explicitly structured.

Similarly, `build_task_specs()` not only saves user questions, but also saves fields such as `category`, `session_id`, `objective`, `query`, `domain`, `answer_text`, `recovery_mode`, etc. In other words, what this layer defines is not an ordinary prompt, but a "task object with execution intention".

### 7.3 Why templates are important

Templates are not intended for mechanical reproduction, but for a unified skeleton for different trajectory types. The benefits of doing this are:

* Success, recovery, and block can keep the format consistent;
* Easier comparison between different tasks;
* It is easier to align fields in subsequent training and evaluation;
* QA can locate problems faster.

### 7.4 Template size of the current project

The current project contains `5` trajectory templates and generates `22` raw trajectories around these templates. This shows that the project does not rely on massive data to win, but relies on the representativeness of trajectory types to build method models.

![Figure 5: Relationship diagram between task specifications and trajectory template](../../images/part10/10_7_fig05_task_specs_and_templates.png)

---

## 8. Trajectory type design: success, recovery, block parallel construction

If a team is asked to do Tool-Use data intuitively, the easiest data set to obtain is often like this:

> User makes request -> Model selection tool -> Call successful -> Return answer

Such samples are certainly valuable, but if the entire data set looks like this, all the model ends up learning is "tool calls on the ideal path." The most difficult part of the real Agent is that it is not on the ideal path.

### 8.1 success trajectory

The success track solves the most basic problems: when should the model call tools, how to construct parameters, how to read results, and how to complete tasks. This is the entry-level capability layer of Agent.

### 8.2 recovery trajectory

The recovery trajectory solves a more critical issue: when the first call fails, can the model identify the error, correct parameters, reselect tools, or retry execution. This type of sample directly determines whether the model is a fragile system that "stops when an error occurs".

### 8.3 block trajectory

The block trajectory solves the boundary problem: when the request itself should not be executed, or when the tool call is overreaching, dangerous, or non-compliant, can the model stop, instead of continuing to push the system to the risk area?

### 8.4 How are recovery and block explicitly constructed in the code?

The most worthwhile thing about P07 is that it does not regard recovery as an accidental phenomenon during runtime, but directly writes recovery as a special trajectory constructor in `src/generate_trajectories.py`. For example:

* `build_search_recovery(task)`
* `build_db_recovery(task)`
* `build_search_db_recovery(task)`
* `build_memory_calendar_recovery(task)`
* `build_memory_db_recovery(task)`
* `build_blocked(task, reason)`

This means that recovery is not a “note after error” but an object of supervision that is consciously designed and produced.

For example, the structure of `build_search_recovery()` first deliberately constructs a bad parameter, and then explicitly adds the repair plan and the second call:

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

This implementation explicitly writes out the intermediate decision-making of "failure-analysis-retry". This is more valuable for training than just retaining the results of two tool calls.

And `build_blocked(task, reason)` goes one step further, it directly generates blocking traces that do not trigger tool calls:

```python
# src/generate_trajectories.py

def build_blocked(task: dict, reason: str) -> list[dict]:
    return [
        user_event(...),
        plan_event(..., reason),
        final_event(..., status="blocked", blocked=True),
    ]
```

This shows that block is not a by-product of "tool call failure", but an independent branch of legitimate behavior.

### 8.5 Why do three types of trajectories exist at the same time?

Because a truly usable Agent not only does things, but also must:

* Do it right when you can;
* Fix it when you make a mistake;
* Stop when you shouldn't.

These three types of abilities are indispensable.

The variant distribution of the current project is: `success = 10`, `recovery = 9`, `block = 3`. This set of proportions is very representative, because it shows that the project does not treat recovery as a scrap, but puts it in a position that is almost as important as success.

![Figure 6: success/recovery/block trajectory layered diagram](../../images/part10/10_7_fig06_trajectory_taxonomy.png)

---

## 9. Simulate execution environment: the environment layer serves as a constraint surface

Without the environment layer, the so-called trajectory is often just static text: the model says "I want to call the tool", and then the researcher writes out the next step manually. This approach is suitable for demonstrations, not engineering.

### 9.1 What problems does the environment layer solve?

The role of the environment layer is to turn "calls on paper" into "executable behaviors". Only by entering the environment can the project be truly recorded:

* Whether the parameters are legal;
* Whether the tool returns successfully;
* What is the return result;
* Whether the next step should be retried;
* Whether memory is read and written correctly;
* Whether the security rule is triggered.

### 9.2 How is the environment layer implemented in the code?

`src/simulate_tool_env.py` is the most suitable place for this chapter to be discussed in combination with code. It does not connect to real external services, but first implements a set of controllable simulation tool functions:

* `search_docs(arguments, task_map)`
* `sql_customer_db(arguments, task_map)`
* `calendar_lookup(arguments, task_map)`
* `python_exec(arguments, task_map)`
* `memory_write(arguments, session_memory)`
* `memory_read(arguments, session_memory)`

This split is very clear: each tool is an independently testable function, the input is arguments and task context, and the output is unified into a binary result like `(success, payload)`. So the subsequent executor `execute_trajectory()` can use the unified interface to gradually advance the entire trajectory.

A high-level execution framework is as follows:

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

The key value of this logic is that the project truly connects the trajectory, environment, tool logs and final indicators. In this way, indicators such as "recovery success rate" and "unsafe block rate" are not paper statistics, but real results after execution.

### 9.3 Why is the Python execution tool worth writing separately?

There is a good security example in `python_exec()`: the code is not directly executed unconditionally, but first checks `UNSAFE_CODE_TOKENS`, and returns `unsafe_code` if it hits a dangerous pattern. This shows that even in the simulation environment, the project has regarded "executable tools" as a higher-risk type of object, rather than ordinary functions. Such code details are very suitable to be written in the final draft as an example of "how to move the safety boundary forward in engineering".

### 9.4 The relationship between simulated environment and real environment

A simulation environment is not the end, but it is a good starting point. It allows the team to first solve basic issues such as "whether the trajectory is reasonable, whether the fields are aligned, whether the recovery logic is established, and whether the indicators can be evaluated" before deciding how to migrate to the real environment. The overall project report also clearly states that the current environment is dominated by simulated execution rather than direct connection to real production tools.

![Figure 7: Simulation tool environment execution closed-loop diagram](../../images/part10/10_7_fig07_simulated_env_loop.png)

---

## 10. Process dismantling: how P07 was gradually implemented from definition to evaluation

The core process of the current project can be summarized in six steps.

1. `src/build_tooling.py`: Build tool specification
2. `src/generate_trajectories.py`: Generate trajectory samples
3. `src/simulate_tool_env.py`: Simulation tool environment execution
4. `src/prepare_agent_dataset.py`: Encapsulate Agent data set
5. `src/evaluate_tooluse.py`: Assessment tool usage data
6. `src/run_p7_checks.py`: Project inspection

These six steps are not complicated, but they just correspond to the minimum closed loop required for a complete data factory.

### 10.1 Define first, not generate

The first step of the project is not to "find some data first", but to define the tool specifications first. This order is critical because once the tool space is not explicitly defined, all trajectories generated subsequently may be based on a shaky foundation.

### 10.2 Create the trajectory first, then enter the environment

The second step of the project is to generate the original trajectory instead of entering execution from the beginning. This shows that the system first focuses on "behavioral design" and then enters "environmental verification", which helps to separate the task layer and execution layer.

### 10.3 Execute the log first, then encapsulate the training set

The project does not directly write the execution process into training samples, but first leaves event-level records and then reorganizes them into data sets in post-processing. This design is important because it leaves room for analysis, playback, and rework.

### 10.4 Evaluate first, then check for consistency

Assessment is not the same as inspection. Assessment answers "how well it performed" and inspection answers "whether code, data, and reports are consistent." Separating the two is a clear sign of engineering maturity.

![Figure 8: P07 six-step pipeline diagram](../../images/part10/10_7_fig08_pipeline_steps.png)

---

## 11. Recovery mechanism: supervision value from failure to recovery

The most noteworthy point of P07 is that it does not simply discard failed samples, but explicitly retains the recovery track.

### 11.1 Why Failure is Valuable

For ordinary question and answer models, error output is of course undesirable; but for the Agent, "the first failure" does not mean "the entire task failed." The key to many real-world tasks lies precisely in whether the model can continue to advance after failure.

For example:

* The parameter format is wrong. Can you correct it and try again?
* If no results can be found, can you change the query method?
* The memory I read is not sufficient. Can I fill in the information before continuing?
* If a certain tool is not applicable, can I switch to an alternative tool?

### 11.2 Recovery The essence of training

The essence of recovery training is not to teach the model to "make mistakes", but to teach the model "how to recover from errors." This has a completely different learning goal than training only successful paths.

### 11.3 Why recovery is closer to the real world than success

In a real user environment, tools will fail, parameters will be wrong, dependencies will jitter, permissions will change, and queries will be empty. If a model has only seen smooth paths during training, it will be very fragile once it goes online.

In the current project, `recovery = 9` is almost of the same magnitude as `success = 10`. This shows that Data Factory regards recovery behavior as its main capability, rather than "filling in a few failure cases to make sense."

![Figure 9: Parameter repair and retry flow chart](../../images/part10/10_7_fig09_recovery_flow.png)

---

## 12. Memory trace: modeling of memory behavior

When many people work as an agent for the first time, they simply interpret memory as "putting together more of the previous text." But the real memory behavior in engineering is much more than that.

### 12.1 What problems does memory solve in Agent?

Memory solves the state problem. It enables the system to:

* Remember user preferences;
* Remember previously performed actions;
* Remember existing intermediate results in the environment;
* Continue to advance based on past information in multiple rounds of missions.

### 12.2 Why model memory behavior separately?

Because memory is not a linear extension of ordinary natural language context, it contains more explicit operability:

* When to read;
* When to write;
* What to write;
* What not to write;
* How it affects subsequent decisions after reading it.

If these are not modeled separately, the model is prone to errors on both ends: either information that should be remembered is not remembered, or information that should not be persisted is included.

### 12.3 Memory signal of the current project

The current training set has a total of `103` records, of which memory records are `34`, and the memory success rate is `100%`. This shows that memory is not an accessory item in the project, but a core capability dimension that is explicitly retained and counted separately.

### 12.4 Why memory data is particularly suitable for early explicit construction

Because the correct behavior of memory is often very specification dependent. If it is completely left to online natural generation, it will be difficult to obtain high-quality, interpretable training signals. On the contrary, early explicit construction through controlled templates makes it easier to establish a stable foundation.

![Figure 10: Memory read and write trajectory diagram](../../images/part10/10_7_fig10_memory_trajectory.png)

---

## 13. Security blocking: the boundary function of block samples

Compared with ordinary generative models, one of the more dangerous aspects of Agent is that it can really "hands on". Once the model has the ability to call tools, the security issue is no longer just "saying the wrong thing", but may become "doing the wrong thing".

### 13.1 What problems does unsafe block solve in the project?

What unsafe block solves is:

* Whether the request is ultra vires;
* Whether dangerous operations are involved;
* Whether execution should be refused;
* Whether we should just do an informational response without actually calling the tool.

### 13.2 Why block is not equal to "simple refusal to answer"

The value of block samples is not just to let the model learn to say "no", but to let it learn to do **structured blocking** in tool usage scenarios:

* Identify sources of risk;
* Does not trigger dangerous calls;
* Provide safer alternative instructions where feasible;
* Prevent system status from entering an uncontrolled area.

### 13.3 Safety signals of the current project

The current unsafe block rate is `100%`, the unauthorized tool call rate is `0%`, and the safety records in the training set are `9`. This shows that despite the small sample size, the project has clearly incorporated safety boundaries into the core assessment.

### 13.4 Why block data should enter the training set early

Because if the safety boundary is only supplemented by rules on the inference side, it is easy for a confrontational state of "the model wants to do it, but the rules are blocking it". A better approach is to let the model learn what not to do during training.

![Figure 11: Unsafe block decision-making flow diagram](../../images/part10/10_7_fig11_unsafe_block.png)

---

## 14. Data reorganization and training encapsulation: log to training interface

After running the environment, the project did not directly throw the execution log to the training framework as it is, but did a very critical post-processing step: **reorganize event-level records into training assets**.

### 14.1 Why original logs are not suitable for direct training

Because logs are more suitable for machine recording, they are not necessarily suitable for model learning. Raw logs usually:

* The granularity is not uniform;
* The format favors execution rather than supervision;
* Lack of explicit instruction / output alignment;
* Not conducive to train / val / smoke segmentation;
* Inconvenient for subsequent version management.

### 14.2 Implementation of trajectory reorganization

The key to `src/prepare_agent_dataset.py` is not to simply copy files, but to split the entire trajectory into event-level training records. The two core functions here are:

* `render_context(events)`: Render users, plans, tool calls, and observation results into a unified context;
* `build_records(trajectory)`: Based on the trajectory after execution, training records are gradually generated.

For example, `render_context()` will rewrite different events into readable text:

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

This step is very similar to translating "system log" into "training consumable context".

`build_records()` goes one step further. It does not only produce one sample per track, but continuously produces supervision records with fields such as `record_id`, `trajectory_id`, `task_id`, `category`, `variant` along the steps. This is why although the final training set only has `22` original trajectories, it can form `103` training records.

### 14.3 Training interface layer products

The final output of the project is:

* `data/training/agent_tooluse_dataset.jsonl`
* `data/training/train.jsonl`
* `data/training/val.jsonl`
* `data/training/smoke_test.jsonl`
* `data/training/training_manifest.json`

This shows that the output of the project is no longer "several running results", but a set of assets that can be directly consumed by the training side.

![Figure 12: Reorganization diagram from event log to training sample](../../images/part10/10_7_fig12_dataset_repacking.png)

---

## 15. Indicator system: signals other than tool success rate

When working on Agent projects, many teams tend to focus on one number: tool invocation success rate. This indicator is certainly important, but if you only look at it, it is easy to misjudge the entire project.

### 15.1 Key indicators of current projects

Core indicators of current projects include:

* Tool schema: `6`
* Number of templates: `5`
*Original track: `22`
* Variant distribution: `success = 10`, `recovery = 9`, `block = 3`
* Tool call success rate: `78.57%`
* Trajectory success rate: `100.00%`
* Recovery success rate: `100.00%`
* unsafe block rate: `100%`
* memory success rate: `100%`
* Unauthorized tool call rate: `0%`
* Number of training records: `103`

### 15.2 Why tool success rate is not equal to task success rate

The tool call success rate measures "whether a single call went smoothly", but the trajectory success rate measures "whether the entire task was completed." If the project explicitly models recovery, then a single tool call failure is repaired and the task is finally completed, which is still a success from the Agent's perspective.

### 15.3 Why does this set of indicators have engineering significance?

The most interesting thing about this set of indicators is that the tool call success rate is only `78.57%`, but the trajectory success rate and recovery success rate both reach `100%`. This just shows that the recovery mechanism has already played a role in the data layer.

A low tool success rate here does not automatically mean that the system is ineffective, but may mean that the project does incorporate failure and repair into the training signal, rather than only retaining the ideal path.

---

## 16. Indicator Interpretation: Weight of Recovery Capability

A very common misunderstanding is that a good agent should try not to make mistakes. In an ideal world, of course, this would be the case, but from a data engineering perspective, this goal is unrealistic.

### 16.1 What capabilities should a truly usable Agent have?

A truly usable Agent requires at least three levels of capabilities:

* Level 1: Can complete tasks under normal circumstances;
*Level 2: Tasks can be resumed under abnormal circumstances;
*Level 3: Can block tasks in dangerous situations.

If you only train the first layer, the model will look beautiful in the demo, but will be very fragile in the real world.

### 16.2 Why recovery is more precious than “pure success data”

Because recovery samples allow the model to learn a behavior closer to system intelligence:

* Identify problems;
* Understand the reasons for failure;
* Generate repair actions;
* Try again;
* Switch strategies when necessary.

These abilities are far more difficult and more realistic than "successful the first time".

### 16.3 Why this level of interpretation must be retained

If only numbers were reported, 78.57% could easily be misinterpreted as a “low” result. But once it is put back into the Agent scene, it shows that the project is not beautifying the data, but is faithfully retaining and utilizing failure recovery behavior.

---

## 17. Assessment and Inspection: Performance Assessment and Consistency Check

Many projects end when they can outperform the indicators, but in the context of data factory, this is not enough. Because even if the metrics look reasonable, the code, data, and reports may not be consistent.

### 17.1 What questions does the assessment answer?

The assessment answers:

* Whether the overall tool call is valid;
* Whether recovery is successful;
* Is the memory correct?
* Whether block is effective;
* Whether the training data distribution is as expected.

### 17.2 Structure of indicator calculation

`src/evaluate_tooluse.py` does not simply count the number of items, but puts the tool layer, track layer, recovery layer, security layer and memory layer indicators in the same `metrics` dictionary for unified output. This is a good point to point out in the chapter, because it reflects that the evaluation object of P07 is not a single success, but the complete behavior distribution.

Judging from the source code structure, the indicators include at least:

* `tool_schema_count`
* `template_count`
* `trajectory_count`
* `category_distribution`
* `variant_distribution`
* `tool_call_success_rate`
* `trajectory_success_rate`
* `recovery_success_rate`
* `unsafe_block_rate`
* `unauthorized_tool_call_rate`
* `memory_success_rate`

It is precisely because the evaluation script is calculated based on the execution product and manifest that the combination of "tool call success rate 78.57%" and "trajectory success rate and recovery success rate are both 100%" appears in the current report, instead of just reporting an isolated number.

### 17.3 Check what questions are answered

Check the answer is:

* Whether the necessary documents are complete;
* Whether the tool schema field is complete;
* Whether the template covers single-step, multi-step and safety scenarios;
* Whether the trajectory variant is complete;
* Whether the observation and decision-making chain exists;
* Whether the memory related case is successful;
* Whether the code and report match.

### 17.4 How to implement the inspection mechanism

`src/run_p7_checks.py` Perform command-level checks first, then data/product-level checks. In the command-level inspection, `py_compile` and `evaluate_tooluse.py` are run directly; in the data-level inspection, rules such as `required_files_exist`, `tool_schema_fields_complete`, `templates_cover_single_multi_and_safety`, `variant_coverage`, `observations_and_decision_chain_present`, `memory_cases_succeed` are verified one by one. The current project has a total of `12` inspection items, all of which passed, and the overall status is `PASS`.

This step is very important, because it makes this chapter not just "a notebook to tell a story", but "a closed engineering loop with verifiable code, verifiable products, and traceable reports."

![Figure 13: Evaluation and inspection double closed loop diagram](../../images/part10/10_7_fig13_eval_and_checks.png)

---

## 18. Limitations and Risks of Current Projects: Boundaries of Method Samples

Writing about limitations does not weaken the project, it increases its credibility. P07 currently has at least three clear limitations.

### 18.1 Tool scope is still small

The current tool type is only the `6` class, which can display methods, but it is not enough to approximate the complex, multi-authority, and multi-system coupling tool space in real enterprise agents.

### 18.2 The calling layer itself is still not stable enough

The tool call success rate `78.57%` indicates that the original call layer is still vulnerable. Although the recovery layer has brought the task success rate back, this does not mean that the underlying calling problem has been solved.

### 18.3 Security boundaries are not rich enough

The existing unsafe block and unauthorized call samples have covered the most basic boundaries, but there is still a lot of room for unauthorized links, prompt injection, sensitive data outgoing and complex permission negotiation in the real world.

### 18.4 Why limitations should be written out in advance

Because the real value of a method model lies not in pretending that it has solved everything, but in letting latecomers know where the next step is most worth investing.

---

## 19. Expansion direction: towards a more realistic enterprise agent

If P07 is regarded as a minimally reproducible Agent data factory, then the next expansion directions include at least the following categories.

### 19.1 Extended tool types

From the current basic tools such as search, db, calendar, code, and memory, it has been further expanded to include email, documents, work orders, approvals, knowledge bases, forms, workflows, and other capabilities that are closer to the real enterprise scenarios.

### 19.2 Extending cross-tool links

Many real tasks are not completed by a single tool, but require multi-step collaboration such as retrieval, query, calculation, writing, and notification. In the future, we can focus on enhancing this type of cross-tool link samples.

### 19.3 Extending cross-session state

The project currently covers memory, but more complex long-term state management, session switching, task recovery and historical dependencies are still worthy of continued construction.

### 19.4 Extending Security Governance

In the future, more scenarios for unauthorized calls, prompt injection, sensitive information leakage, data pollution, and policy bypass can be introduced to make the security boundary truly close to the pre-launch requirements.

### 19.5 Expanding evaluation dimensions

In addition to the current metrics, more fine-grained tool selection accuracy, parameter correctness, retry efficiency, final answer quality, and multi-round consistency metrics can be added.

![Figure 14: P07 subsequent evolution roadmap](../../images/part10/10_7_fig14_roadmap.png)

---

## 20. The key position of P07: the ability layer that connects “being able to speak” and “being able to do”

In many tutorials, large model engineering still remains at the step of "making the model answer more decent". But the Agent scenario puts forward a higher requirement: the model must not only be able to speak, but also be able to do it; not only must be able to do it, but it must also be able to repair it when it makes a mistake; not only it must be able to repair it, but it must also know when it should not be done at all.

This is the significance of P07. It is not to prove that you are a mature enterprise agent, but to explain:

* Tool usage behavior can be structured;
* The recovery trajectory can be trained;
* Memory behavior can be modeled explicitly;
* Security blocking can enter the supervision layer;
* Execution, evaluation and inspection can form a closed loop.

This makes it play a connecting role in the overall capability chain: it advances "language supervision" to "behavior supervision".

---

## 21. Differences from ordinary function call data: Characteristics of Agent behavioral data

On the surface, P07 also contains tool schema, call parameters and execution results, so some people may think that it is not essentially different from common function calling data. But in fact the two are very different.

### 21.1 Ordinary function call data emphasizes single mapping

Traditional function calling samples usually concern themselves with:

* What is the user intention;
* Which function should be called;
* How to fill in parameters;
* How the return value is presented.

This is a static mapping of "input -> call -> output".

### 21.2 Agent Tool-Use puts more emphasis on the behavioral process

P07 puts more emphasis on:

* Why you should call the tool now;
* What to do if the call fails;
* What to do if multiple rounds of memorization are needed;
* What to do if the request is dangerous;
* How to precipitate multi-step behaviors into training assets.

This is no longer a simple function calling, but closer to "behavioral data engineering of executable agents".

### 21.3 Why this distinction must be clearly stated in the chapter

Because many teams underestimate the difficulty of Agent data and think that adding a few more function call samples is enough. P07 just illustrates: The real difficulty is not "whether it can be adjusted", but "what to do if the adjustment is wrong, whether it can stop when it is time to stop, and whether it can remember when crossing the wheel."

---

## 22. Migrating to other Agent scenarios: the value of P07 method model

Agent Tool-Use is not the only direction that requires behavioral data factories, but it is a very good example. The reason is that it has the following characteristics at the same time:

* The tool space is clear;
* The behavior chain can be disassembled;
* The recovery mechanism is important;
* Safety boundary rigidity;
* Evaluate closed loop necessary.

These features actually also exist in enterprise Copilot, automated workflow assistants, development assistants, operation and maintenance assistants, and multi-agent collaboration systems.

### 22.1 Which designs can be directly migrated

* Definition link from tool schema to task specification;
* success / recovery / block parallel construction ideas;
* The strategy of accessing the simulated environment first and then accessing the real environment;
* The practice of modeling memory and security boundaries separately;
* Training encapsulation and inspection closed loop.

### 22.2 Which parts cannot be copied directly?

* Tool types and permission systems must be rewritten;
* Security boundaries in enterprise scenarios are usually more complex;
* Multi-team collaboration workflow is more difficult than single-agent scenarios;
* Real systems have many more types of exceptions than simulated environments.

### 22.3 The most portable core method

What can really be migrated is not a certain calling template, but this method chain:

> Define the tool space -> Design task specifications -> Construct success/recovery/block trajectories -> Execute and record in the environment -> Reorganize into training assets -> Establish a closed loop of evaluation and inspection.

---

## 23. List of major deliverables

A list of the main deliverables is given here.

### 23.1 Tools and Processing Intermediates

* `data/processed/tool_schemas.json`
* `data/processed/trajectory_templates.json`
* `data/processed/task_specs.json`
* `data/processed/raw_trajectories.jsonl`
* `data/processed/executed_trajectories.jsonl`
* `data/processed/tool_execution_log.jsonl`
* `data/processed/execution_summary.json`

### 23.2 Training interface products

* `data/training/agent_tooluse_dataset.jsonl`
* `data/training/train.jsonl`
* `data/training/val.jsonl`
* `data/training/smoke_test.jsonl`
* `data/training/training_manifest.json`

### 23.3 Reporting and Verification Products

* `data/reports/p7_report.md`
* `data/reports/p7_metrics.json`
* `data/reports/p7_test_results.json`
* `data/reports/p7_test_report.md`

This list of deliverables shows that what P07 finally settles is not a "run through an experiment", but a complete set of assets from tool definition to training interface to evaluation report.

---

## 24. Conclusion: What Agent Data Factory really wants to train is not just calling actions, but behavioral capabilities.

When many people see the Tool-Use project, their first reaction is to understand it as "let the model learn to call functions." But what P07 shows is actually something deeper:

It trains not a mechanical API trigger, but a behavioral system that works in a world of tools. The system needs to know:

* Under what circumstances should action be taken;
* How to call during action;
* How to fix the wrong adjustment;
* How to remember when there is a state;
* How to stop when there is risk.

In this sense, the value of P07 does not just lie in how many samples, tools, and indicators it has now, but in that it clearly answers a question that is very important to the Agent era:

> If we want the model to truly learn to "do things", we must turn the behavior itself into a data engineering object.

This is exactly what Agent Tool-Use Data Factory is all about.

---

## Special topic: Scene library construction and pre-launch access control

Before the Agent Tool-Use project is actually implemented, the most likely problem is not "whether the model can adjust the tool", but "whether the team has completed the scene library and access control conditions." If there are only a few demonstration tasks, the system looks smart; once it enters the real environment, the tool space, exception types, and safety requirements are simultaneously magnified, and problems are quickly exposed.

### 1. The scene library cannot only cover the successful path

A usable Agent scene library should cover at least three types of content at the same time:

* Regular successful tasks to verify whether the model can complete the goal under standard conditions;
* Recovery tasks to verify whether the model can get back on track when tools fail, parameters are missing, or environmental conflicts occur;
* Blocking tasks to verify whether the model can be stopped before unauthorized, sensitive or non-compliant requests.

These three types of tasks are indispensable. With only success and no recovery, the system will appear fragile in real environments; with only success and recovery and no block, the system will get out of control in high-risk scenarios. P07 The most valuable part of the current method is that these three trajectories have been regarded as parallel assets, rather than treating recovery and interruption as auxiliary samples.

### 2. Access control should cover tool correctness, recovery capabilities and security boundaries

Before the Agent project goes online, it is worth establishing at least three access controls:

* Tool correctness access control, confirm that the model will select the right tool, fill in the key parameters, and understand the returned results;
* Recovery capability access control to confirm that the model can take the next reasonable step after errors, conflicts and interruptions;
* Security boundary access control, confirming that the model will not continue to act under prohibited conditions, nor can it be easily induced to bypass the policy.

If the system only passes the first gate, it looks like it "knows how to use the tools", but in fact it is not qualified to enter the real process. Because errors and risks will almost certainly occur in real processes, recovery capabilities and security blocking capabilities are not just peripheral requirements, but main chain capabilities.

### 3. The scenario library needs to be continuously updated with failure cases

P07 The scene library of this type of project should not be written all at once, but should continue to grow with failure replay. Whenever the system encounters the following problems in a certain type of scenario, it is worth absorbing it into the scenario library:

* The tool is selected correctly, but the parameters are misunderstood;
* Repeat the same mistake after the tool call fails;
* Continue blindly when you should move to a memory or clarification step;
* Although the security policy is clearly triggered, it is still trying to find a bypass path;
* Intermediate states are lost during multi-tool collaboration, resulting in subsequent behavioral drift.

The scene library constructed in this way can truly represent the most fragile boundaries of the system, rather than only the parts of the system that are best at displaying.

---

## Special topic: Operation and review mechanism of Agent behavior data

Compared with ordinary function call samples, Agent behavior data is more like an "operation log asset". Since it describes a behavioral process, operations and review mechanisms are needed for continuous correction. Otherwise, even if the training set continues to grow, the team may just change the old problem and continue to accumulate it.

### 1. The granularity of the review should fall on the "behavior chain"

When many teams review Agent problems, they are used to looking only at the final result: whether the task succeeded or failed. But for Tool-Use, the more effective review granularity is usually the behavior chain itself. For example:

* Why did you choose this tool in the first step;
* Why is the warning in the return value ignored in the second step;
* Clarification of why the third step is not triggered;
* Why step 4 continues to be executed even under ultra vires conditions;
* Is the final failure due to a single step error or because the entire strategy path is unreasonable?

Only when the review granularity falls on the behavior chain can the team really have the opportunity to turn the problem back to the training assets instead of staying in the vague description of "I didn't do it right this time."

### 2. The operating mechanism should allow multiple roles to follow the same trajectory

The Agent project naturally involves multiple roles: data engineering, model engineering, platform, security, and product. Different characters have different concerns about the same issue, but it's better to have them all discuss the same track rather than looking at their own summaries.

A more efficient way to operate is usually to:

* Unified retention of key tracks and execution logs;
* Explicitly list representative success, recovery and block cases in the evaluation report;
* Establish fixed replay sets for high-risk failure scenarios;
* Prioritize discussing "which tracks have been fixed and which tracks are still unstable" during version review.

The value of this is that although different roles have different focuses, they can form the same picture of system problems. For behavioral systems, this shared view is very important.

### 3. Agent Data Factory ultimately requires continuous service iteration

P07 has currently demonstrated a very key methodology: behavioral data does not end when it is generated, but needs to be continuously cycled through execution, evaluation, review and retraining. If we continue to expand to a more realistic enterprise agent environment in the future, this cycle will become increasingly important.

In the long run, what the team needs to settle down most is not a single template, but the following capabilities:

* Able to continuously add high-value scenes;
* Can continuously convert failure cases into replay and training samples;
* Ability to continuously update security boundaries and blocking rules;
* Can continuously track multiple rounds of memory, recovery efficiency and final task completion rate.

As soon as these few things start to form a regular rhythm, the Agent Tool-Use data factory will no longer be just a project chapter, but will gradually evolve into a real behavioral data infrastructure.

