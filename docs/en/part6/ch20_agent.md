# Chapter 20: Agent Memory and Multi-Turn Interaction Data

In long-horizon, multi-turn, task-oriented agent data engineering, the hard part is not making single-turn QA longer. The hard part is making the model preserve task identity, state consistency, and behavioral stability across turns. A useful agent must remember long-term user preferences, track intermediate task progress, handle local dependencies in the current context, and continue correctly after interruption, switching, and recovery.

This chapter treats memory as a data object that can be structured, trained, tested, and governed. Multi-turn interaction is not casual chatting; it is a state-transition process around task progress. We discuss the nature of multi-turn trajectories, session segmentation and state representation, memory write/update/recall, replay testing and failure analysis, and data patterns for AI assistants, customer-service agents, and office agents.

## 20.1 Why Multi-Turn Trajectories Cannot Be Reduced to "A Few More Rounds of Dialogue"

### State Dependency, Role Drift, and Task Interruption in Multi-Turn Dialogue

Single-turn data asks whether the input is understood and the output is correct. Multi-turn data asks whether the current turn remains coherent with all previous turns as part of one task process. Each response is a local action inside a state chain.

Many systems can "talk for a long time" in demos while still failing tasks. They preserve language continuity but not task continuity. Language continuity means the conversation reads smoothly. Task continuity means every action respects prior constraints, current progress, and future goals.

State dependency includes parameter dependency, stage dependency, and result dependency. A date confirmed in turn one may determine a tool parameter in turn six. A task may not legally proceed until a prior confirmation is complete. A tool observation may force the agent to update the plan.

Role drift is another risk. A task executor may slowly become a general explainer. A cautious workflow agent may start making unsupported assumptions. Multi-turn samples should label not only whether a turn sounds right, but whether it preserves the correct role: executor, coordinator, confirmer, explainer, waiter, or recovery actor.

Task interruption makes the problem harder. Users switch topics, insert new tasks, pause work, and return hours later. A mature dataset must include interruption, switching, suspension, and recovery, not only smooth linear completion.

### Differences Among Long-Term Memory, Short-Term Memory, and Context Window

Context window, short-term memory, and long-term memory are different layers.

The context window is the text currently visible to the model. It answers what the model can see, not what should be kept. A larger window does not automatically produce better consistency because it also contains irrelevant dialogue, failed attempts, and noise.

Short-term memory is the current task working set. It stores confirmed slots, unfinished steps, recent tool failures, current blockers, and waiting conditions. It serves the active thread and should decay when the task ends.

Long-term memory stores information that remains useful across sessions and tasks: stable preferences, recurring work patterns, project background, and durable constraints. It should be extracted, summarized, conflict-resolved, and time-aware. It should not become a raw history warehouse.

If these layers are confused, systems remember temporary facts forever and forget stable preferences after the current window closes.

### Why Multi-Turn Supervision Is Not Only "Next Reply"

Predicting the next reply is necessary but insufficient. Task-oriented agents must also learn current state, memory write decisions, whether to call tools, whether to wait, whether to switch threads, and where to resume.

Multi-turn supervision should include three layers: reply supervision, state supervision, and process supervision. Reply supervision checks the local text. State supervision checks how the internal state changes. Process supervision checks whether the action creates a sustainable causal chain for later recovery.

Without state and process supervision, long conversations remain long text samples rather than agent trajectories.

### Common Incorrect Labeling Patterns in Multi-Turn Data

The first mistake is concatenating a dialogue into one long sample without labeling state changes. This preserves order but loses the important events: information collection, action execution, waiting, branching, recovery, and thread switching.

The second mistake is labeling only the final outcome. A booking task may succeed after confirmations, conflict handling, user correction, and fallback. If only the final success is preserved, the model cannot learn how to repair.

The third mistake is treating user feedback as ordinary conversation. "Not that file," "move it to next week," or "use the previous format" are state-update signals.

The fourth mistake is missing decay and invalidation. Not every mentioned fact stays true. Temporary intent, one-time preference, denied information, and superseded variables need expiration or override rules.

The fifth mistake is treating memory writes as invisible system implementation. Memory write, update, and recall should appear as trainable events.

The sixth mistake is treating mixed-task dialogue as one thread. Real users run parallel tasks. Without thread IDs and active-thread labels, models learn to merge unrelated states.

## 20.2 Session Segmentation and State Representation

### Dividing Session, Episode, and Task Thread

A `session` is a continuous interaction container. It is time-oriented and can contain several topics or tasks.

An `episode` is a task-oriented process with a start, progress path, and completion, suspension, or failure. It is the best unit for multi-turn training and replay evaluation.

A `task_thread` is a finer task line inside an episode or session. It is useful when users interleave tasks, such as editing a resume, replying to an email, and scheduling a meeting in the same conversation.

Session answers "what happened in one continuous interaction." Episode answers "what task process is this." Thread answers "which task line is active and what state does it hold."

### Session Segmentation Should Center on State Continuity

Segmentation should follow state continuity rather than turn count or time gap. If the next turn must inherit task variables from the previous turn, it belongs to the same state trajectory. If it shares only long-term user preference but not current task variables, it is probably a new episode or thread.

Useful segmentation signals include new goal, task completion, thread suspension, old-task recovery, tool failure, re-planning, and explicit user correction.

### Structured Representation of Dialogue State, User Preference, and Task Progress

Task-oriented agents need structured state rather than only natural-language context.

Dialogue-state fields describe current control position: active thread, pending confirmations, last action result, waiting-for-user, waiting-for-tool, and conflict flags.

User-preference fields describe stable behavior-shaping choices: output format, default language, tone, tool preference, approval habit, naming convention, or risk boundary.

Task-progress fields describe what has been completed, what remains, what is blocked, and what the last valid action was. They are essential for "continue where we left off."

These fields should not be merged into one undifferentiated dictionary. They update, decay, and override differently.

### Minimal Closed-Loop Unit of State Representation

A trainable multi-turn step needs a minimal closed loop: before-state, trigger, decision action, observation when present, and after-state.

```json
{
  "episode_id": "ep_0012",
  "thread_id": "t_calendar",
  "before_state": {
    "active_state": "collecting_info",
    "pending_slots": ["timezone"],
    "confirmed": {"date": "2026-04-29", "duration_min": 90}
  },
  "trigger": {"type": "user_message", "text": "Use Shanghai time."},
  "action": {
    "type": "tool_call",
    "name": "calendar_search",
    "arguments": {
      "start_time": "2026-04-29T13:00:00",
      "end_time": "2026-04-29T18:00:00",
      "timezone": "Asia/Shanghai"
    }
  },
  "after_state": {
    "active_state": "waiting_for_user_confirmation",
    "pending_slots": ["choose_slot"],
    "last_action_result": "found_conflicts"
  }
}
```

This format makes replay possible. The team can inspect where state first drifted, which memory was written incorrectly, and which recovery step failed.

### Data Modeling for Interruption, Switching, and Recovery

Interruption samples should record the task node where interruption happened, unfinished state, frozen fields, and fields that must be revalidated during recovery. Prices, inventory, time windows, and external document versions may age while a task is suspended.

Switching samples should show the old thread entering suspended state and the new thread receiving its own state container. Long-term preferences can be shared, but temporary parameters must not leak.

Recovery samples should show how the agent reconstructs executable state. Recovery is not "repeat the last message." It is reactivation with unfinished slots, recent errors, validated assumptions, and next-step options.

![Figure 20-1: Multi-turn agent state transition graph](../../images/part6/图20_1_zh.png)

*Figure 20-1: Multi-turn agent state transition graph*

## 20.3 Memory Write, Update, and Recall

### What Should Enter Memory and What Should Stay Only in Context

The first rule of memory is: remember only information that will be useful in the future and is safe to retain. More memory is not automatically better. Many systems fail because they remember too much, too early, and too long.

Long-term memory is appropriate for stable, reusable, clearly bounded information, such as preferred output format, recurring workflow, durable project background, and stable constraints.

Short-term context is appropriate for current-task variables, temporary candidates, one-time tool observations, and pending decisions. These should decay after the episode or thread ends.

Some information should not be remembered at all: denied assumptions, casual one-time remarks, unconfirmed guesses, and low-value details.

### Criteria for Memory Writes

Memory writes should be governed by four criteria.

Stability asks whether the information is likely to hold across future turns and tasks. Reusability asks whether it will reduce future interaction cost or improve success. Confirmability asks whether it was explicitly confirmed by the user, reliably returned by a tool, or supported by repeated evidence. Risk asks whether retaining it incorrectly would cause harm.

A memory write is a decision action, not merely storage. Training samples should show why something is written, to which layer, with what confidence, and under what expiration policy.

```json
{
  "memory_event": "upsert",
  "memory_id": "pref_format_table_first",
  "key": "preference_format",
  "value": {"default": "table first, then summary", "language": "en"},
  "evidence": {
    "source": "user_confirmed",
    "episode_id": "ep_0009",
    "occurrence_count": 3
  },
  "confidence": 0.85,
  "decay": {
    "policy": "time_decay",
    "half_life_days": 60,
    "override_on_conflict": true
  }
}
```

### Relationship Between Memory Writes and Task Closure

Memory writing belongs inside the task loop. The agent should first understand what role the information played in the current task, then decide whether it has cross-task value.

If memory writes happen too early, the system may turn temporary statements into durable facts. If they happen too late or not at all, users must repeatedly provide stable preferences.

### Memory Layers, Long-Term Preferences, and Data Decay

A practical memory hierarchy includes instantaneous context, working memory, session summary, and long-term preference.

Instantaneous context supports local generation. Working memory maintains current-task variables and progress. Session summary compresses an episode for later recovery. Long-term preference stores cross-episode information that influences future behavior.

Long-term preference should be operational, not a vague user profile. Useful examples include default language, document structure, terminology choices, approval order, file-naming habits, and recurring tool workflows.

Decay is mandatory. Time decay lowers confidence over time. Conflict decay lowers old memory when new evidence arrives. Usage decay demotes memory that is never recalled or verified. Datasets should include memory overwrite, downgrade, and archive examples, not only successful writes.

### Migration Rules Between Memory Layers

Layers need migration rules. A piece of information may move from visible context to working memory, from working memory to session summary, and from repeated session evidence to long-term preference.

For example, "give me the Chinese version first for this report" is probably working memory. Repeated preference across tasks, such as "default to Chinese first and preserve English terms," may become long-term preference.

Migration depends on reuse value and evidence strength, not on surface wording.

### Long-Term Preferences Should Converge Gradually, Not Be Written Once

A one-time statement is often context, not a durable preference. Users may say "make it brief" because they are in a hurry. A true long-term preference appears across tasks and survives conflict resolution.

Long-term preference records should maintain evidence count, last verification time, applicable scope, confidence, and override rules. This turns memory formation into gradual convergence rather than a single irreversible write.

### Memory Summaries, Conflict Resolution, and Timeliness Management

Summaries are the middle layer that converts noisy dialogue into reusable state. A good summary captures goal, progress, confirmed constraints, open questions, and likely recovery point. It is not a literary conversation digest.

Conflict resolution makes memory trustworthy. Latest explicit user correction should override older preference. High-confidence tool observation should override model guess. Confirmed information should override unconfirmed information.

Timeliness turns memory from a static database into a dynamic state system. Each memory should record source, write time, last verification time, scope, confidence, and invalidation condition.

### Granularity Control for Memory Summaries

A summary that is too detailed becomes a compressed transcript. A summary that is too abstract cannot support recovery. The right granularity is determined by recovery cost.

For task agents, a useful summary should answer: what is the current goal, what is complete, what constraints are confirmed, what is blocked, and where should the system resume next time?

### Priority System for Conflict Resolution

A practical priority order is: latest explicit user instruction, high-confidence tool observation, confirmed information, task-local constraint, long-term preference, model inference.

If the long-term preference says "answer in Chinese" but the current episode says "use English this time," the current episode wins. If an old state says a document is available but a fresh tool result says it was deleted, the tool result wins.

![Figure 20-2: Memory layers and update flow for task-oriented agents](../../images/part6/图20_2_zh.png)

*Figure 20-2: Memory layers and update flow for task-oriented agents*

### Recall Hit Rate, Wrong Recall, and Forgetting Strategy

Memory quality is not only whether something is recalled. It is whether the right thing is recalled at the right time with the right weight.

Wrong recall can be worse than no recall because the model acts confidently on false context. Examples include applying a previous project's naming convention to the current project, using an expired constraint, or treating a one-time request as a permanent preference.

Forgetting is part of memory engineering. A good agent should forget low-value, low-confidence, or expired information, and recover by asking or retrieving when needed.

### Recall Should Consider Usability, Not Only Similarity

Semantic similarity alone is insufficient. The best memory is the one still valid and useful for the current state transition.

Recall should consider active thread, expiry, source confidence, task-progress match, and whether a newer version exists. Otherwise a semantically similar memory may be actionably wrong.

## 20.4 Replay Testing and Failure Review

### Multi-Turn Replay Evaluation, State-Drift Evaluation, and Memory-Dependency Evaluation

Multi-turn agents should be evaluated through replay, not only static benchmark answers. Replay evaluation runs a full interaction in order and checks whether the model takes state-consistent actions at key nodes.

State-drift evaluation tests whether the model gradually leaves the original goal or role during long interaction. It asks whether the agent stayed on track, not only whether it eventually finished.

Memory-dependency evaluation creates cases that require historical memory even when the current window does not repeat it. This separates long-context reading from true memory use.

```python
THREAD_FIELDS = {
    "mail": {"recipient", "subject", "body"},
    "resume": {"resume_file", "target_role", "bullet_style"},
}


def detect_thread_contamination(events):
    findings = []
    for event in events:
        thread_id = event["thread_id"]
        state_keys = set(event.get("after_state", {}))
        for other_thread, exclusive_fields in THREAD_FIELDS.items():
            if other_thread == thread_id:
                continue
            leaked = state_keys & exclusive_fields
            if leaked:
                findings.append({
                    "event_id": event.get("id"),
                    "thread_id": thread_id,
                    "leaked_fields": sorted(leaked),
                })
    return findings
```

### The Basic Unit of Replay Evaluation Should Be State Transition, Not Reply

If each turn is scored only as text, replay evaluation collapses back into single-turn evaluation. Agent replay should score whether the before-state was read correctly, the action was legal, and the after-state was updated correctly.

A response may sound natural while resuming the wrong thread or treating expired variables as valid. That should fail.

### Why Long-Horizon Drift Is More Dangerous Than Local Error

Local errors are often visible and repairable. Long-horizon drift accumulates gradually. The system may begin with a small deviation, write the wrong state, then plan on top of it. By the time users notice, the error is embedded in several intermediate states.

Replay tests should actively search for trajectories that remain linguistically smooth while drifting semantically.

### Error Attribution for "Remembering the Wrong Thing" and "Forgetting the Important Thing"

"Remembering the wrong thing" usually points to a write-policy, conflict-resolution, or recall-ranking problem. The system may store an unconfirmed guess, turn a one-time remark into durable preference, or fail to overwrite old evidence.

"Forgetting the important thing" may come from recall failure, excessive decay, poor summary, or thread-switching that lost key slots.

Failure review should ask whether the problem happened at write, update, recall, decay, or recovery.

### Failure Review Should Move From Outcome Attribution to Process Attribution

Outcome attribution asks why the final answer failed. Process attribution asks where the first wrong state entered the trajectory.

For example, a wrong email after recovery might trace back to an old thread not being suspended, a missing attachment validation, or a long-term preference overriding a task-specific instruction.

Only process attribution tells the team which data, labels, or tests to add.

### Red-Line Tests Before Multi-Turn Interaction Goes Online

Multi-turn agents need stricter pre-launch red-line tests than single-turn systems.

Critical red lines include thread contamination, wrong memory persistence, recovery failure, and feedback failure. These break user trust because they make the system feel uncontrollable.

Red-line tests should be extracted from real failure logs and become part of the data flywheel.

### Typical High-Risk Scenarios Red-Line Tests Should Cover

High-risk scenarios include rapid task switching, strong user correction, cross-session recovery, time-sensitive tasks with stale state, permission boundary changes, and memory conflicts.

These are not edge cases in real use. They are common pressure points for task-oriented agents.

### Table: Multi-Turn Failure Modes and Detection Actions

| Failure mode | Typical symptom | Likely root cause | Detection action |
|---|---|---|---|
| State drift | Long conversation leaves original goal | Missing state fields or weak trajectory supervision | Replay long tasks and compare goal consistency |
| Thread contamination | Task A parameters enter task B | Unclear thread segmentation | Construct parallel-thread tests |
| Wrong write | Guess or one-time fact enters long-term memory | Low write threshold | Audit memory-write events |
| Wrong recall | Expired or irrelevant memory is used | Ranking ignores time and confidence | Add expired distractors to evaluation |
| Critical forgetting | Recovery loses key slots | Poor summary or missing recovery state | Test interruption and recovery |
| Feedback failure | User correction does not update state | Feedback treated as text only | Mark correction turns and inspect after-state |
| Over-persistence | Short-term fields survive task end | Missing TTL or decay | Check task-close cleanup |
| Over-forgetting | Stable preference is not reused | Weak long-term extraction | Cross-session preference tests |
| Wrong recovery | Agent remembers task but resumes wrong step | Missing progress field | Evaluate stage localization |

*Table 20-1: Multi-turn failure modes and detection actions*

## 20.5 Cases and Continuity

### Multi-Turn Data Cases for AI Assistants, Customer-Service Agents, and Office Agents

AI assistants emphasize long-term preferences and lightweight task recovery. They need continuity across writing, research, study, and revision tasks.

Customer-service agents emphasize episode completeness, state-machine transitions, and auditable feedback. User correction must update ticket state, not merely trigger polite language.

Office agents are the most complex because they involve tools, documents, schedules, emails, and multiple threads. They expose whether the system can keep task containers separate and recover after interruption.

### Data Focus in AI Assistant Scenarios

AI assistants should focus on gradual preference formation and project-background summarization. Users often continue long-term themes without restating background.

Datasets should include "continue last task," "use the previous structure," and "revise based on last feedback" cases, with clear labels for durable preference versus current-episode instruction.

### Data Focus in Customer-Service Agent Scenarios

Customer-service data should emphasize state-machine completeness and failure recovery. Typical stages include intake, identity verification, information collection, solution confirmation, escalation, and closure.

Every confirmation, denial, correction, and escalation should map to an internal state update.

### Data Focus in Office Agent Scenarios

Office-agent data should emphasize parallel threads, cross-tool observations, cross-session recovery, and long-term preferences that affect execution path.

The dataset should deliberately mix document editing, calendar work, email drafting, and spreadsheet updates so thread isolation can be learned and tested.

### Coupling With RAG, Feedback Loops, and Privacy Governance

Memory is naturally coupled with RAG, feedback loops, and privacy governance.

Memory is best for user preferences, task continuity, and interaction history. RAG is best for external facts, documents, and time-sensitive knowledge. Feedback loops convert user corrections, tool failures, and logs into new training and test data. Privacy governance decides what may be stored, for how long, and under which usage boundary.

### Boundary Modeling Between Multi-Turn Memory and RAG

A stable rule is: external, fast-changing, source-sensitive facts should be retrieved; user-specific, task-continuity, and interaction-cost-reducing information may be remembered.

If a model learns to remember the world, it will store stale facts. If it relies on retrieval for stable preferences, it will repeatedly ask the user for the same thing.

### Feedback Loops Should Change State, Not Only Answers

In single-turn systems, feedback often rewrites an answer. In multi-turn agents, feedback should update state. "Not this file" should update the target file. "Use the old version" should change the version field. "The approver is Wang" should update the workflow.

Feedback samples should therefore include before-state, correction, after-state, and next action.

### Privacy Governance as a Reverse Constraint on Memory Design

Privacy governance should shape memory design from the beginning. An item may be useful but unsuitable for long-term storage. A memory may be technically recallable but not allowed in every context.

Field-level permissions, purpose limitation, expiry, deletion, and auditability must be part of memory schema design.

### Table: Session Fields and Memory Fields

| Field category | Example field | Meaning | Recommended layer | Decay |
|---|---|---|---|---|
| Session | `session_id` | Continuous interaction container | Session | No |
| Session | `episode_id` | Task process identifier | Episode | No |
| Session | `task_thread_id` | Parallel task-thread identifier | Thread | No |
| Session | `active_state` | Current state such as planning, acting, waiting | Working memory | Yes |
| Session | `pending_slots` | Key missing or unconfirmed information | Working memory | Yes |
| Session | `last_action_result` | Summary of recent action or tool result | Working memory | Yes |
| Session | `interruption_flag` | Whether the task is suspended | Working memory | Yes |
| Session | `progress_stage` | Current task phase and completion | Working memory or summary | Yes |
| Memory | `preference_language` | Stable language preference | Long-term preference | Slow decay |
| Memory | `preference_format` | Output structure, file format, terminology habit | Long-term preference | Slow decay |
| Memory | `stable_constraints` | Durable rules or approval order | Long-term preference | Slow decay |
| Memory | `project_background` | Ongoing project context | Summary or long-term preference | Update decay |
| Memory | `rejected_memory` | Explicitly denied or invalidated information | Conflict layer | Fast decay or archive |
| Memory | `memory_confidence` | Confidence and verification status | Metadata | Yes |
| Memory | `last_verified_at` | Last verification or use time | Metadata | No |
| Memory | `ttl_or_decay_policy` | Lifecycle rule | Metadata | No |

*Table 20-2: Session fields and memory fields*

## Chapter Summary

The core of multi-turn agent data engineering is not making conversations longer. It is organizing interaction into state trajectories that can be learned, recovered, and evaluated. What determines long-horizon agent quality is not only language generation, but state representation, memory layering, feedback loops, and decay rules.

This chapter covered four major design problems: how to segment session, episode, and task thread; how to represent dialogue state, task progress, and long-term preference; how to define memory write, update, recall, and forgetting; and how to use replay evaluation, failure attribution, and red-line tests to turn online failures into data assets.

Long-term memory is not an extended context window. State transition is not a side effect of natural-language continuity. Only when multi-turn interaction is treated as a process driven by state, memory, and feedback can teams build reliable long-horizon task agents.

## References

Asai A, Wu Z, Wang Y, Sil A, Hajishirzi H (2024) Self-RAG: Learning to Retrieve, Generate, and Critique Through Self-Reflection.

Budzianowski P, Wen T-H, Tseng B-H, Casanueva I, Ultes S, Ramadan O, Gasic M (2018) MultiWOZ: A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling. In: EMNLP, pp 5016-5026.

Carlini N, Tramer F, Wallace E, Jagielski M, Herbert-Voss A, Lee K, Roberts A, Brown T, Song D, Erlingsson U, Oprea A, Raffel C (2021) Extracting Training Data from Large Language Models. In: USENIX Security Symposium.

Lewis P, Perez E, Piktus A, Petroni F, Karpukhin V, Goyal N, Kuttler H, Lewis M, Yih W-T, Rocktaschel T, Riedel S, Kiela D (2020) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In: NeurIPS.

Liu N F, Lin K, Hewitt J, Paranjape A, Bevilacqua M, Petroni F, Liang P (2024) Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Association for Computational Linguistics 12:157-173.

Packer C, Wooders S, Lin K, Fang V, Patil S, Stoica I, Gonzalez J E (2023) MemGPT: Towards LLMs as Operating Systems.

Park J S, O'Brien J C, Cai C J, Morris M R, Liang P, Bernstein M S (2023) Generative Agents: Interactive Simulacra of Human Behavior. In: UIST.

Schick T, Dwivedi-Yu J, Dessi R, et al. (2023) Toolformer: Language Models Can Teach Themselves to Use Tools. In: NeurIPS.

Shinn N, Cassano F, Gopinath A, Narasimhan K, Yao S (2023) Reflexion: Language Agents with Verbal Reinforcement Learning.

Wang W, Dong L, Cheng H, et al. (2023) Augmenting Language Models with Long-Term Memory. In: NeurIPS.

Williams J D, Raux A, Ramachandran D, Black A (2013) The Dialog State Tracking Challenge. In: SIGDIAL, pp 404-413.

Yao S, Zhao J, Yu D, Du N, Shafran I, Narasimhan K, Cao Y (2023) ReAct: Synergizing Reasoning and Acting in Language Models. In: ICLR.

Young S, Gasic M, Thomson B, Williams J D (2013) POMDP-Based Statistical Spoken Dialog Systems: A Review. Proceedings of the IEEE 101(5):1160-1179.

Zhong W, Guo L, Gao Q, Ye H, Wang Y (2024) MemoryBank and Long-Term Memory Mechanisms for LLM Agents.
