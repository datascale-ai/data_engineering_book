**Part Introduction**

As large models evolve from "able to answer" toward "able to reason, call tools, and engage in multi-turn collaboration," the focus of data engineering is shifting as well. In the past, much data work centered on outcomes, concerned mainly with whether the model produced the correct answer. But in reasoning models and Agent systems, outcome supervision alone is no longer sufficient to support real-world tasks. Whether a model can think step by step, use tools correctly, and maintain consistent state and leverage memory across multi-turn interactions is becoming the new set of key questions.

For this reason, this part focuses on the new class of data problems around models that "can think, use tools, and remember," and discusses how data engineering should be upgraded from outcome supervision to process supervision and trajectory supervision. Data here is no longer just single-turn question–answer pairs; it gradually extends to more complex structures such as reasoning processes, tool-call chains, interaction state, and memory evolution. The criteria for data quality likewise shift: it is no longer only whether the answer is correct, but further whether the process is sound, whether execution succeeds, and whether state remains stable.

Around this shift, this part will unfold along three directions—reasoning data, tool-use data, and agent memory and multi-turn interaction data—systematically discussing the construction, organization, evaluation, and governance of data for Agents, and laying the groundwork for the subsequent rollout of Projects P06 and P07. The core question it seeks to answer is: when the model is no longer just "generating answers" but "completing tasks," how should data engineering be upgraded?

# Chapter 18 Chain-of-Thought and Reasoning Data Engineering

As large models move from "able to answer" toward "able to reason," the priorities of data engineering shift noticeably. For ordinary SFT data, teams tend to focus on whether the input is clear, the output correct, and the style stable. But for reasoning tasks such as math, logic, and code, having only "question–answer" pairs is generally not enough. The key to such tasks is not just what the model finally says, but whether it arrives at that answer along a verifiable, reproducible, and generalizable reasoning path. Many instabilities observed after deployment look on the surface like "occasional wrong answers," but at their root they stem from training data that rewards only final outcomes without constraining intermediate processes—causing the model to apply inconsistent or even erroneous internal strategies across different problem types, difficulty levels, and contexts.

Reasoning data engineering is therefore not a simple extension of ordinary SFT. While retaining the foundational generation methods of SFT and synthetic data, it further introduces a system of reasoning trajectory representation, step-level verification, error classification, difficulty stratification, and process supervision. It requires teams to do more than "produce more samples"; they must build a complete pipeline that spans problem generation, trajectory construction, automatic validation, process labeling, and curriculum organization. Only then will the model learn not a scattered set of answer mappings but a more stable repertoire of reasoning behaviors that hold up across arithmetic, program repair, symbolic derivation, and logical judgment.

This chapter is aimed at teams building math, logic, and code reasoning datasets. It systematically discusses why looking only at final answers conceals reasoning defects, how to represent reasoning trajectories, how to perform step-level verification and error classification, how to design difficulty stratification and sample organization schemes, and how to operationalize a sustainable, scalable reasoning data pipeline in real-world engineering.

## 18.1 Why Looking Only at Final Answers Conceals Reasoning Defects

### A correct answer does not equal correct reasoning

In reasoning tasks, the final answer is merely the external landing point of the entire solution process, not sufficient evidence of the process itself. A model that gives a correct answer may indeed have completed the full chain of condition recognition, rule invocation, state update, and conclusion verification—or it may have simply memorized a template, latched onto surface patterns, or stumbled onto the correct result after a series of wrong steps. For ordinary generation tasks, this phenomenon of "imprecise process but acceptable result" may not be a serious problem. But for highly constrained reasoning tasks such as math, logic, and code, this is precisely one of the most important risk sources to watch for.

In a math problem, the model may, in some step, use an illegal transposition, cancellation, or theorem, yet still arrive at the correct numerical value because the problem's structure is simple. In a logical task, the model may not have truly expanded the premises and matched the rules, but instead directly applied some familiar concluding form. In code repair, the model may not have understood the true cause of the bug, but happened to modify a local spot that makes the current tests pass. On the surface, all such samples can be labeled "successful." But from the standpoint of capability learning, what the model has learned is not a stable solving mechanism but some high-risk shortcut.

If training data retains only "question–answer" pairs without explicitly constraining the intermediate process, the reward signal during training is severely compressed. The model does not need to demonstrate how it reached the result, nor does it pay any cost for skipped steps, pseudo-explanations, or unsupported inferences along the way. It only needs to learn to output an endpoint that will be judged "correct." The resulting capability is often highly sensitive to problem type, phrasing, length, and context. Any slight variation in problem form, or any need for a longer reasoning chain, can rapidly destabilize the model.

More importantly, the decoupling of correct answers from correct reasoning directly misleads a team's assessment of model capability. If a team interprets "improved accuracy" simply as "improved reasoning," it may persistently overestimate the system's true level across the training, evaluation, and deployment stages. A model that excels only at hitting the surface of the answer may look strong enough in offline evaluation, yet struggle to maintain reliable performance in real-world use.

### Why correct results can still be low-quality samples

In data engineering practice, many samples enter the training set not because their process has been proven reliable, but only because their final result passed the acceptance condition. This works fine for simple tasks, but in reasoning tasks it creates a kind of hidden data pollution: low-quality processes are repeatedly learned as high-quality positives merely because their results happened to be correct.

The most dangerous aspect of this pollution is that it does not surface as explicit errors. The sample will not announce on its face, "I am a bad sample." Instead, it slips into the data pipeline wearing the coat of "correct answer." For the model, the signal conveyed is: even if the intermediate process is sloppy, as long as the final result clears the bar, this behavior can be rewarded. Over time, the model increasingly tilts toward opportunistic paths rather than learning reusable, transferable reasoning structures.

In reasoning data engineering, then, a correct result should be regarded as the lowest bar of acceptability—not as sufficient evidence of quality. Truly high-value samples should not only have correct results but also have interpretable intermediate steps, verifiable key transformations, and locatable local errors. Only such samples are worth repeatedly reinforcing during training.

### Blind spots of outcome supervision in complex tasks

Outcome supervision is well suited to tasks with clear objectives where paths are unimportant or intermediate states are hard to define. In classification tasks, we care whether the label is correct; in simple extraction tasks, whether the fields match; in shallow QA, often only whether the answer is consistent with the facts. But once a task involves multi-step dependencies, multi-constraint coupling, or long-chain solving, outcome supervision exhibits clear blind spots.

First, outcome supervision cannot tell us at which layer the error occurred. A single final error may stem from misunderstanding the problem, omitting a condition, a local computational mistake, misuse of a rule, state inconsistency, or simply a format deviation in the final output. If we retain only whole-problem correctness, all these qualitatively different problems get compressed into the same failure label. Second, outcome supervision cannot distinguish between "almost right but a small last-step slip" and "all wrong but coincidentally correct." The former usually has strong training value because the model's main path is already near correct; the latter is a high-risk sample because it may package an error mode as a positive. Third, outcome supervision cannot provide fine-grained feedback for data iteration. At most, the team learns that the overall success rate on a class of problems is low, without knowing which type of step, rule, or variable operation is the persistent culprit.

This blind spot is especially typical in math tasks. Two models may both fail the same problem—one because basic arithmetic is unstable, the other because of a critical conditional misjudgment. Without analyzing the process, the team cannot decide whether to add more basic arithmetic samples or to emphasize explicit display of rule premises. The same holds for logic tasks. A wrong conclusion may stem from missing premises, misjudged quantifier scope, exceeding rule scope, or failed branch handling. In code tasks, looking only at whether a patch passes tests likewise cannot reveal whether the fix actually addresses the root cause, introduces new side effects, or merely overfits the test cases.

The problem with outcome supervision is therefore not just "not seeing the intermediate process," but rather that it compresses inherently heterogeneous errors into a single label, robbing training, analysis, and correction of direction. The team cannot tell what to fix; it can only blindly add more data or tweak training parameters—an approach that often treats symptoms but not root causes.

### Why outcome supervision misleads data analysis

In practice, data analysis often relies on macro metrics—accuracy, pass rate, whole-problem success rate, average score. These are useful, but if a task is fundamentally process-dependent, such metrics only reflect final performance and cannot reveal internal mechanisms. Teams can easily fall into the illusion that as long as overall metrics are rising, the system is steadily improving.

The catch is that rising metrics may stem from very different causes. Sometimes the model truly learned more stable reasoning paths; other times it merely memorized more high-frequency patterns or developed stronger template-matching ability on a particular benchmark. Without a process-level view, the team cannot tell these cases apart. Data analysis can look encouraging while the model has not actually gained sufficient generalization.

Worse, outcome supervision lets many genuinely important problems "hide" on the dashboard. For example, a class of problems may have a decent answer-correctness rate, but a substantial fraction of those samples contains logical leaps; or a code-repair model's pass rate keeps rising while the proportion of patches introducing side effects rises in lockstep. Without step-level labels and process verification, these phenomena are virtually impossible to surface through ordinary outcome reports. The team mistakenly believes the system is evolving healthily—until deployment amplifies these latent issues.

### How reasoning errors manifest as instability in deployment

What a deployment environment fears most is not "consistently bad" but "seemingly capable, actually erratic." Many post-deployment issues are not that the model lacks capability altogether, but that it has not formed a stable process mechanism. On some inputs it lands on a suitable path and performs well; on slightly different inputs, it falls back to a coarse template and performance collapses. This alternation between good and bad inspires deeper distrust than uniformly poor performance.

Such instability has several common manifestations. First, slight rewordings of the input cause the model's intermediate reasoning to drift. The essence of the problem is unchanged, but the model switches to an unreliable path because of changes in surface vocabulary, narrative order, or context format. Second, once the reasoning chain lengthens, front-to-back state consistency breaks down. Variables defined earlier are silently overwritten later; constraints established earlier are forgotten in intermediate steps; local conclusions begin contradicting each other. Third, in code tasks, the model may produce seemingly professional cause analysis, but the eventual implementation does not match the repair logic the analysis claims. Fourth, when wrong, the model often remains highly fluent and confident, making process defects harder for evaluators and end users to detect.

From an engineering perspective, this kind of instability is especially dangerous because it cannot be eliminated simply by adding more static homogeneous samples. If new data is still in "question–answer" form, the training set merely keeps reinforcing the same signal: process does not matter, only the endpoint matters. The model may keep improving hit rates on certain benchmarks, but its internal reasoning structure remains fragile. Once exposed to real environments, this fragility manifests as random fluctuation, distributional sensitivity, and hard-to-reproduce failures.

### Why process defects are amplified in long-chain tasks

In short-chain tasks, an erroneous process can sometimes be masked by the problem's own simplicity. With few steps, few states, and few constraints, even when the model is somewhat imprecise mid-way, it may still luck back onto the correct track at the end. But in long-chain tasks, process defects almost certainly accumulate and amplify. The reason is simple: each step is a precondition for the next, and once local state goes wrong, all subsequent judgments are built on erroneous foundations.

In a lengthy math problem, an early misdefinition of a variable renders all subsequent derivation meaningless. In multi-hop logic, a single missed premise causes the entire argument chain to collapse. In complex code repair, misjudging the root cause sends the entire patch design in the wrong direction. The longer the chain, the more important process stability becomes, and the more apparent the limitations of outcome supervision: when errors accumulate to late stages, the team sees only "the final answer was wrong" without seeing which layer collapsed first.

This is why reasoning data engineering must emphasize process. Not because process is intrinsically "fancier," but because in long-chain tasks, process is the principal carrier of correctness. Without process constraints, a model's performance on complex tasks is unlikely to stabilize.

### From "answer-oriented" to "process-oriented"

The core shift for a reasoning data team is from "answer-oriented" to "process-oriented." This is not to say the final answer is unimportant; rather, in math, logic, and code, the answer is the result of the process and cannot substitute for the process itself. If data engineering is built solely around answers, what the model ultimately learns will look more like an output-hit strategy. Only by incorporating process into data structure, validation mechanisms, and quality evaluation does the model have a chance to gradually develop stable solving habits.

Being process-oriented does not mean preserving endless intermediate text. It means preserving the key intermediate information that actually determines correctness: which conditions were identified, which rules were invoked, which states changed, which local conclusions supported subsequent steps. These are what reasoning data should truly focus on. Only when such elements enter the training pipeline can the team answer: How did the model get it right? Why did it get it wrong? Which samples are worth keeping, and which must be cleaned?

## 18.2 Representations of Reasoning Trajectories

### Differences among CoT, scratchpad, program-of-thought, and tree-of-thought

The first question for reasoning data engineering is not "how to generate more processes" but "in what form should the process exist?" Different representations are not merely cosmetic differences in format—they determine sample readability, verification difficulty, storage cost, annotation burden, and the structure of training objectives. A poorly chosen representation makes everything downstream less efficient: samples are hard to unify, verification is hard to automate, training converges unstably, and quality analysis loses traction.

CoT (chain-of-thought) emphasizes step-by-step explanation in natural language. It best matches the human intuition of "explaining a process," making it well suited for textbook-style math problems, explanatory logic problems, and scenarios that need to display reasoning intent. CoT is highly readable, friendly to human review, and convenient for explicitly stating key judgments in the task. But its drawbacks are equally clear: natural language has too much freedom, easily mixing in unverifiable descriptions; sample length and style vary widely, increasing the difficulty of quality control.

Scratchpad is closer to a draft-paper-style intermediate record. It usually does not unfold all thoughts into full natural language but only keeps intermediate variables, key computations, local judgments, and necessary markers. This form is highly effective for arithmetic, multi-step symbolic operations, and short-chain logic, since it can pack denser useful information into shorter text. Compared with CoT, scratchpad is often more amenable to automation and easier to align with intermediate states. The risk is that if designed too sparsely, it may lose enough semantic anchors, making human inspection difficult and even hindering the model's learning of cross-step dependencies.

Program-of-Thought goes one step further, expressing intermediate reasoning as executable program snippets, pseudocode, sequences of expressions, or other formal intermediate representations. For math, symbolic transformation, and program analysis tasks, this form is particularly valuable because many intermediate steps can be verified directly by an executor or a rule system. As a result, much work that previously relied on human judgment can be handed off to structured tools. Its biggest advantage is strong verifiability, but the cost is also higher: the representation schema must be sufficiently standardized, and the task itself must permit some degree of formal expression.

Tree-of-Thought no longer views the reasoning process as a single linear chain but explicitly records multiple candidate branches, the choices among them, backtracking, and comparisons. For complex planning, search-based reasoning, and tasks that require weighing multiple candidate solutions, this kind of representation is highly valuable, since real solving processes often involve trial, abandonment, and reselection. But construction and use costs are the highest. It must record not only "what was finally done" but also "what else might have been done," along with some evaluation of those branches. Forcibly using a tree structure when the task does not require multi-branch exploration only adds noise and cost.

### Difference between linear and branched trajectories

In linear reasoning tasks, the process is centered on sequential dependency: each step determines the next, and each step builds on its predecessor. Here, a linear trajectory is enough to capture most of the key information. Most math computation problems, routine code bug fixes, and standard chain-style logical derivations fit linear structures well. Linear trajectories are clear, compact, and easy to verify, and they slot naturally into ordinary SFT or process supervision training.

But not all reasoning is naturally linear. When a task involves comparison of multiple plans, local trial and error, search-style backtracking, or planning branches, a single linear trajectory flattens out much of the important information. The model may finally produce a correct path, but if the data never explicitly records candidate branches and the reasons for their elimination, the model cannot learn "why not the other path." In some complex tasks, this branch-selection ability is itself a key part of capability.

Whether to adopt a linear trajectory should therefore not be decided by annotation convenience alone, but by the task's solving structure. If a task is essentially a single main path, do not manufacture branches artificially; if a task depends on comparison and backtracking, do not compress the final path into a history-less straight line.

### Representation schemas for math, code, and logic tasks

For math tasks, a good schema usually includes the problem statement, known conditions, the goal quantity, a sequence of steps, intermediate variables, and the final conclusion. Sometimes it should also record the action type of each step—substitution, expansion, cancellation, elimination, differentiation, integration boundary handling, and so on. The benefit is that the team knows not only "what characters the model wrote" but also "what operation it performed." When downstream needs rule checking or error classification, action labels dramatically reduce analysis difficulty.

**Code example: an actionable "reasoning trajectory sample schema" (math scenario)**

The example below shows a structured trajectory with "action labels + intermediate expressions." Downstream, one can apply equivalence / executable checks to `expr`, whitelist checks to `action`, and consistency checks to `vars`.

```json
{
  "id": "math_trace_00031",
  "problem": "Given x + 3 = 10, find x.",
  "given": ["x + 3 = 10"],
  "target": "x",
  "steps": [
    {"i": 1, "action": "transpose", "expr": "x = 10 - 3", "vars": ["x"]},
    {"i": 2, "action": "compute", "expr": "x = 7", "vars": ["x"]}
  ],
  "final_answer": "7",
  "meta": {"difficulty": "basic", "verifier": "arith_v1"}
}
```

For code tasks, the schema is better organized around "problem localization—cause analysis—repair plan—code change—verification result." Unlike math, intermediate process in code tasks is not abstract reasoning chain but a solution process tied directly to program state. A high-quality code sample should retain not only before/after code snippets but also the test that triggered the failure, failure logs, localization rationale, candidate repair strategies, and final verification results. Otherwise the model learns only local patch mapping rather than a full debugging logic.

For logic tasks, the schema should emphasize premises, rule invocations, local conclusions, and conflict checks. Logical errors are often not surface-language issues but rather incomplete derivation bases, mistaken applicability conditions of rules, or cross-step leaps. Without making explicit "from which premises which step is derived," many seemingly fluent chains lack true logical legitimacy. For more complex logic tasks, fields for branch discussion, counterexample testing, and changes in the hypothesis set can be added, making the trajectory more amenable to downstream verification.

### Representation differences between explicit and implicit state

A frequently overlooked issue in trajectory design is whether intermediate state should be written out explicitly or left implicitly embedded in natural language. The problem with many low-quality samples is precisely that key state changes are not expressed clearly but hidden between sentences. The model appears to say much, but the conditional shifts, variable updates, and rule scopes that actually drive subsequent steps are not written down.

Explicit state representation has the advantage of turning critical dependencies from "implicit understanding" into "visible objects." For example, the current variable value, current equation form, and current sub-goal in a math problem; the current fault location, the object currently being modified, and the current test status in a code task; the current set of premises, the current set of conclusions, and the current conflict status in a logic task. Once these states are made explicit, downstream verification and correction become much easier.

Implicit state is more text-economical, but it places greater demands on both models and annotators and is more prone to ambiguity. A sample that depends too much on default contextual inference may be understood differently by different readers and parsed differently by different verifiers. For reasoning data that must be produced and quality-controlled at scale, making key state explicit is generally the safer engineering choice.

### Cost trade-offs between short and detailed trajectories

In practice, many teams swing from one extreme to another: initially retaining only final answers, and after realizing process matters, pursuing ever longer and more detailed reasoning text. In fact, trajectories should be not as long as possible but as effective as possible. What matters is not word count but how much critical, verifiable, reusable information is packed in.

Trajectories that are too short omit key steps and prevent the model from learning a stable process. The model knows how to jump from problem to answer, but not how to organize variables in between, invoke rules, or progressively shrink the problem space. Trajectories that are too long introduce a different problem: large amounts of rhetorical language, repetitive phrasings, and unverifiable explanations enter the sample, shifting the training focus from key state changes to surface language. The model may learn to "sound like it is thinking earnestly" without actually learning more reliable reasoning behavior.

A more sensible principle is to retain key transformations, key decisions, key invocations, and key state updates, while compressing redundant prose that does not constrain the result. Basic arithmetic and short-chain logic generally only need short trajectories; complex proofs, program repair, and multi-branch planning need finer-grained records. Trajectory length should not be specified uniformly but should follow task characteristics and verification needs.

### Trajectory density matters more than trajectory length

Rather than asking "how long should a sample be," ask "how much useful information is carried by each segment of text." This is the question of trajectory density. A high-density trajectory, even if not long, clearly conveys intermediate state, key derivations, and local decisions; a low-density trajectory, no matter how lengthy, may carry only rhetoric without constraint.

High-density trajectories typically share several features. First, key steps are never skipped. Second, intermediate state changes are clear. Third, redundant rhetoric is scarce. Fourth, downstream verifiers can directly apply checks to a large portion of the content. Such samples are more valuable for training because what the model is exposed to is a cleaner process signal.

Low-density trajectories often manifest as: extensive natural-language elaboration but quick passes over critical operations; "abundant explanation" on the surface but insufficient intermediate evidence; large style variations across samples that defy unified QA. Such samples may look like "human explanations" but are not necessarily a quality resource for data engineering. The goal is not for data to read like humans, but for data to be usable.

### The choice of representation should serve verification

A reasoning trajectory is not for showing off "what the model thought" but for turning the process into a data object that can be processed, filtered, and trained on. Therefore, the choice of representation must be co-designed with downstream verification, not left as an afterthought after freely generating large amounts of process text. If the process itself is unparseable, unstructurable, and unverifiable, then no matter how long the trajectory, it is just an expensive pile of text.

This means design should already consider which parts can be rule-verified, which can be execution-verified, which need a judge model, and which must be retained for human spot checks. For example, if math steps adopt explicit action labels and intermediate expressions, rule programs can more easily check legality; if code samples explicitly separate "cause analysis" from "code change," it is easier to judge whether the analysis matches the implementation; if logic samples record premise references, cross-step leaps are easier to identify.

From an engineering standpoint, the degree of structure in a reasoning trajectory effectively determines the ceiling of automated verification. The more heterogeneous, free-form, and reliant on implicit understanding a trajectory is, the higher the cost of verification; the cleaner, more field-based, and more explicit it is, the easier downstream QA can run stably. A mature data system never treats representation as a mere matter of writing style; it views it as core infrastructure for the entire reasoning data pipeline.

### Table: Types of Reasoning Samples and Their Applicable Tasks

| Sample type | Primary representation | Applicable tasks | Strengths | Limitations |
|---|---|---|---|---|
| Answer-only | Question + final answer | Simple QA, classification, low-depth reasoning tasks | Low cost, high throughput | Cannot expose process defects |
| CoT | Step-by-step natural-language reasoning | Math problems, common-sense logic, explanatory reasoning | Highly readable, easy for human review | Hard to fully automate verification |
| Scratchpad | Draft-style intermediate variables and key steps | Arithmetic, multi-step symbolic processing, short-chain logic | Compact and efficient, easy to align with key states | Slightly less interpretable |
| Program-of-Thought | Pseudocode, program snippets, executable intermediate expressions | Math computation, program reasoning, structured solving | Executable verification, high robustness | Higher construction cost |
| Tree-of-Thought | Multiple candidate branches and selection process | Search-style planning, complex logic, multi-path decisions | Captures exploration and backtracking | Complex annotation, high training cost |
| Correction-type | Erroneous trajectory + corrected trajectory | Math correction, code repair, logic debugging | Helps learn correction ability | Difficult to construct and annotate errors |
| Self-reflection | Initial answer + self-check + revised result | Complex QA, reasoning enhancement, code review | Helps improve stability | Easily introduces templated reflection noise |



![Figure 18-1: Reasoning data construction and verification flowchart](../../images/part6/图18_1.png)

*Figure 18-1: Reasoning data construction and verification flowchart*

## 18.3 Automatic Verification and Error Classification

### Why reasoning data must not be generated without verification

For ordinary SFT data, a sample tends to have at least preliminary training value as long as the question is clear, the answer acceptable, and the style broadly consistent. Reasoning data is different. The core value of reasoning data comes not just from "having an answer" but from "whether the intermediate process is credible." If a sample has a long reasoning chain but has not been verified, what it brings may not be capability gain but process pollution. The model absorbs the erroneous intermediate steps, pseudo-explanations, and unsupported leaps, ultimately learning a behavior that "talks about process" more skillfully but produces more noise.

Reasoning data engineering thus must not treat verification as an afterthought to generation; it should treat verification as a basic threshold for samples entering the training set. A reasoning data pipeline without verification can rely on high throughput to obtain many samples early on, but as scale grows, noise accumulates faster. Because reasoning samples are typically longer, more free-form, and have more covert local errors, relying solely on final-answer filtering is far from sufficient. The more complex the process, the more essential it is to establish explicit checks before samples enter training.

Even more importantly, verification serves not only to "remove erroneous samples." It also plays an equally important role: converting initially chaotic process text into structured quality signals. If a sample is marked as failed, the team needs to know not just "that it failed" but also whether it was an arithmetic error, a logical leap, exceeding rule scope, a pseudo-explanation, or a local state inconsistency. Without this layer, subsequent data revision, sample regeneration, curriculum stratification, and process supervision all lack sufficiently fine-grained grounding.

### Why automatic verification is a precondition for scaling reasoning data

Once reasoning data scales to a certain volume, relying entirely on humans to read and judge becomes nearly infeasible. Not because humans are unimportant, but because the verification difficulty of reasoning samples far exceeds that of plain answer-checking. For an answer-only sample, humans usually just compare whether the final output is correct. For a reasoning sample, they must check whether each intermediate step is valid, whether state remains consistent, and whether local explanations actually support the next conclusion. This workload balloons with step count.

Automatic verification is therefore not about "eliminating humans" but about deploying human resources where they matter most. Rule systems can handle format, action legality, and local consistency checks; executors can handle runnability checks for computations, programs, and symbolic steps; test systems can verify code behavior; judge models can take over some semantic process judgments. Through such mechanisms, teams can use automation to filter out most obvious problems first, then have humans review boundary cases, high-value hard examples, and systematic errors. This division of labor is the basis for reasoning data to be truly scalable and continuously iterable.

Without automatic verification, teams tend to get stuck in a dilemma: either limit throughput to control noise—keeping sample volume low—or pursue scale at the cost of quality, letting erroneous processes flood the training set. The former keeps reasoning data construction at small-workshop scale; the latter makes the model increasingly skilled at imitating low-quality processes. Automatic verification establishes a sustainable engineering path between these two extremes.

### Rule verification, execution verification, unit tests, and judge models

Automatic verification of reasoning data cannot be achieved by a single mechanism. Different tasks, steps, and representations require layered combinations of verification means. Generally, rule verification, execution verification, unit tests, and judge models constitute the four most common verification capabilities, each covering a different class of errors.

Rule verification is best suited to portions with stable form, explicit constraints, and direct program matching. Examples include the format legality of math steps, whether variables have been defined, whether action labels conform to a whitelist, whether rule invocations in logic problems fall within the allowed set, whether fields in code samples are complete, and whether a patch reaches the correct file. Such verification is cheap, fast, and highly controllable, making it ideal as a first-layer large-scale screening mechanism. It cannot resolve every issue, but it can quickly remove many low-level errors and structurally chaotic samples.

Execution verification suits intermediate processes that can be "run." Examples include expression computation in math, symbolic steps, intermediate programs in program-of-thought, and local function execution in code samples. Whenever a step can be formally executed, verification reliability is usually much higher than natural-language judgment. The significance of execution verification lies in turning "looks reasonable" into "actually runs." For many reasoning tasks, this is the key step from surface plausibility to genuine truth.

**Code example: execution verification of "arithmetic/expression steps" (safe simplified version)**

Without introducing extra dependencies, one can start with a "controlled-expression" evaluator that allows only numbers and `+ - * / ( )`, verifying whether the right-hand side of each equation can be computed and whether it matches the given target.

```python
import ast
import operator as op


OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv
}


def safe_eval(expr: str) -> float:
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Num):  # py<3.8
            return node.n
        if isinstance(node, ast.Constant):  # py>=3.8
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("illegal constant")
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = _eval(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        raise ValueError("illegal expression")

    tree = ast.parse(expr, mode="eval")
    return float(_eval(tree))


def verify_step(step_expr: str) -> bool:
    # Demo only: validate forms like "x = 10 - 3"
    if "=" not in step_expr:
        return False
    left, right = [s.strip() for s in step_expr.split("=", 1)]
    _ = left  # further variable-name checks could be added here
    safe_eval(right)  # passing this layer means it computes successfully
    return True


if __name__ == "__main__":
    print(verify_step("x = 10 - 3"))  # True
    print(verify_step("x = __import__('os').system('rm -rf /')"))  # False
```

Unit tests mainly serve code repair, program synthesis, and structured tool-call tasks. They check not only that the final program runs but also that the repair truly satisfies the expected behavior. For code tasks, looking only at the generated text is generally insufficient; the real quality standard is whether program behavior is correct, whether boundary conditions are covered, and whether new side effects have been introduced. Unit tests play the role of "behavioral ground truth" here, much closer to real usage standards than textual similarity or surface explanation quality.

Judge models fill the gap left by rule and execution verification on semantic judgments. For example, whether an explanation truly supports its conclusion, whether a logical step constitutes a leap, whether a code-fix rationale matches the patch's actual behavior, whether a piece of self-reflection truly identifies the root cause. Such questions cannot be fully formalized, yet cannot be ignored. The value of judge models is in providing a kind of approximate semantic review at scale. They are not absolutely reliable on their own and should be combined with rule and execution verification rather than used as the sole final standard.

### Why verification systems usually need multiple stages

In practice, teams often hope for a single "universal verifier"—as if a sufficiently strong module could unify all reasoning quality issues. But the complexity of reasoning data makes this nearly impossible. A single sample may simultaneously contain format issues, execution errors, semantic leaps, and local redundancy, with different problems suited to different tools. An effective verification system is usually not a single judge but a multi-stage cascade.

Generally, the first stage should be a cheap, high-recall screening to clear the most obvious low-quality samples. Rule verification and basic format checking suit this role. The second stage should move into stronger constraints of truthfulness verification—execution verification, test runs, expression evaluation—responsible for confirming whether key steps actually hold. The third stage enters costlier semantic judgment: judge models or human spot checks, for samples that look formally fine but whose process quality is still in question. This tiering has two benefits: it reserves expensive judgment for a small number of borderline samples, and it lets different verification signals cross-correct rather than operate in isolation.

Multi-stage verification also yields finer-grained failure reasons. Whether a sample dies at the format layer, execution layer, or semantic layer leads to entirely different corrective actions. If a team has only a unified "pass/fail" verdict, it knows the sample is unfit but does not know what to do next.

### Classifying arithmetic errors, logical leaps, pseudo-explanations, and hallucinated steps

Errors in reasoning data are not all the same thing. If a team labels every failed sample simply as "wrong," the value of process supervision is greatly diminished. Different errors correspond to different root causes and different remediation strategies. A mature reasoning data system should classify errors into several operationally meaningful types rather than stopping at a vague overall assessment.

Arithmetic errors are the most intuitive class. They typically manifest as basic computation failures, sign errors, faulty substitutions, miscopied formulas, missed boundary conditions, and so on. They are easy to locate and usually easy to verify automatically. Often dismissed as "low-level errors," they actually have enormous impact in long-chain reasoning, since a small computational slip can derail the entire downstream chain.

Logical leaps are more covert. They are not miscomputations but jumps to the next conclusion without showing necessary intermediate justifications. The model appears to be reasoning efficiently but may have skipped the most critical proof or judgment. The danger is that the model develops a hazardous habit: as long as the final conclusion sounds reasonable, intermediate steps can be omitted. This pattern may be hard to expose on short samples but collapses easily on complex tasks.

Pseudo-explanations are a kind of process noise especially common in the era of large models. Their surface signature is "saying a lot," but these explanations bear no causal relation to actual reasoning progress. In math samples, pseudo-explanations may show as long natural-language preambles that fail to justify why a formula holds; in code samples, as analyses that sound professional but bear no correspondence to the actual patch; in logic tasks, as rephrasings of the problem statement that add no verifiable intermediate information.

Hallucinated steps are graver still: the model introduces conditions, theorems, APIs, variables, or environmental assumptions that do not exist in the problem. Such errors may not even surface immediately if one looks only at the final result, but they are highly damaging to training because they teach the model that "fabricating intermediate bases is permissible." Once this preference is reinforced, many subsequent reasoning outputs will exhibit uncontrolled expansion.

### Exceeding rule scope, breaking consistency, and state drift

Beyond the typical error classes above, reasoning data also contains some more structural errors worth labeling separately. The first is exceeding rule scope: the model uses operations, rules, or external knowledge not allowed for the current task. For instance, a logic problem that requires strict derivation from given premises, yet the model inserts common-sense supplements; a math problem restricted to elementary methods, yet the model invokes higher-order results directly; a program repair task limited to a specific file, yet the model modifies other locations to bypass the constraint. The danger of exceeding scope is that it is often misjudged as "clever," when in fact it violates the task definition.

The second is consistency breakage. The model defines a variable, premise, or constraint in one step but quietly changes its meaning later, or forgets a state that was already established. Long-chain tasks are especially prone to this. It is neither as conspicuous as hallucination nor as easily caught as arithmetic errors, but it severely undermines reasoning stability. A process that contradicts itself across steps cannot be reliable overall, even if each local step seems plausible.

The third is state drift. State drift means the model gradually deviates from the original task space during long-chain generation. It may start out solving the problem and end up explaining something else; start fixing a class-A bug and end up writing a class-B optimization; start reasoning from one set of premises and unconsciously switch to another semantic environment. State drift is one of the most troublesome deployment failures because it is rarely a single point of error but rather a systemic deviation produced by an entire chain gradually drifting.

### Step-level labels and process quality scoring

If verification results stop at "pass" or "fail," their value is still limited. What truly supports process supervision and subsequent training are step-level labels—pinning quality information to specific intermediate steps. A reasoning sample should not be merely "whole-problem correct" or "whole-problem wrong"; ideally each step is marked as correct, suspicious, wrong, redundant, leaping, unverifiable, or already corrected later. Only when errors are localized to steps can subsequent training, filtering, and correction be properly targeted.

The first significance of step-level labels is that samples are no longer a single black box. The team can know which local positions are most error-prone, which actions fail most often, and in which task stages leaps or hallucinations tend to occur. Second, step-level labels enable more flexible training strategies. Not every "sample with errors" needs to be discarded wholesale. If a long sample has a reliable majority of process with only a local misstep, it may still have great value. Teams can mask only the erroneous step, construct a local correction task, or transform the sample into "error identification—correction" training data.

Process quality scoring sits above step-level labels and provides a more synthetic evaluation of the entire trajectory. A sample's process quality typically depends not only on final correctness but on local correctness, consistency, completeness, verifiability, redundancy, and alignment between explanations and behavior. In math, local legality and chain completeness are especially important; in code, behavior verification and consistency between repair rationale and implementation matter more; in logic, premise coverage and rule applicability tend to be central. With quality scoring, teams can add a "priority" dimension beyond "keep or discard," letting training absorb more of the genuinely stable, clear, reusable high-quality processes.

### Why process scoring should not depend solely on final correctness

In many pipelines, whole-problem correctness naturally becomes the most prominent scoring indicator—simple, intuitive, easy to align with benchmarks. But if process scoring still relies primarily on final results, so-called "process supervision" degenerates into a variant of outcome supervision. True process scoring should as much as possible derive from the intermediate structure itself, rather than referencing only the final answer.

For instance, a sample that ultimately gets the wrong answer but has nine-tenths of its steps correct, with a clearly correctable last-step error, still holds value for training local correction and process robustness. Conversely, a sample with the correct final answer but filled with leaps, pseudo-explanations, and latent hallucinations may pass on outcome but does not deserve a high score. In other words, process scoring must be willing to demote "right answer, bad process" samples and to distinguish "wrong answer, mostly sound process" samples from ordinary failures. Only then will the scoring system not, in turn, reinforce the old outcome-oriented habits.




![Figure 18-2: Process supervision labeling diagram](../../images/part6/图18_2.png)

*Figure 18-2: Process supervision labeling diagram*

### Table: Error Types and Corrective Actions

| Error type | Typical manifestation | Common cause | Recommended corrective action |
|---|---|---|---|
| Arithmetic error | Wrong computation, miscopied signs, faulty substitution | Unstable basic computation, long trajectory causing local slips | Recompute with an executor, replace the wrong step and replay the downstream chain |
| Logical leap | Directly writes conclusion without necessary premises | Over-compressed training samples, model prefers shortcuts | Insert missing steps, explicitly include rules and justifications |
| Pseudo-explanation | Lots of explanation, but no causal link to conclusion or modification | Over-pursuit of "explanation-like" style | Remove vague description, keep only statements that constrain the next step |
| Hallucinated step | Introduces conditions, theorems, APIs, or variables not in the problem | Over-association by the model, insufficient constraints | Cross-check against problem and environment, force regeneration of that step |
| Exceeding rule scope | Uses disallowed reasoning rules or illegal transformations | Unclear schema, missing verifier | Add rule whitelist checks, roll back to nearest legal state |
| Pseudo code-fix | Patch passes some tests but does not fix the root cause | Insufficient test coverage, optimizing only surface pass rate | Add tests, perform diff analysis, add failure-cause alignment verification |
| Redundant step | Adds descriptions that do not affect the result but add noise | Loose generation template, length preference biased | Compress steps, retain only key state-transition information |
| Inconsistent step | Variable names, constraints, or conclusions conflict across steps | Long-chain context drift | Cross-step consistency checking, rewrite the conflicting interval |

## 18.4 Difficulty Curriculum and Sample Organization

### Why reasoning data should not be mixed randomly

In many ordinary SFT tasks, randomly shuffling samples and training uniformly often works well enough, because internal structural differences between tasks are not too large and the model mainly learns input–output mappings. Reasoning tasks are different. Reasoning ability tends to be hierarchical: basic computation, variable management, rule invocation, local consistency, long-chain memory, branch comparison, error backtracking—these abilities do not mature simultaneously but typically need to be built up step by step, from simple to complex, from short to long.

If a team mixes all difficulties, types, and lengths of reasoning samples from the outset, the model tends to be pulled by conflicting signals early in training. Before it has mastered stable short-chain reasoning, it is forced to handle highly complex long-chain tasks; before it has learned basic rule invocation, it faces multi-branch selection and complex correction. This not only lowers training efficiency but can also instill bad habits: facing complex problems by skipping intermediate steps, covering uncertainty with vague explanations, or relying on high-frequency templates to evade real structured solving.

Reasoning data organization therefore must consider not just coverage but learning order. Curriculum learning is not the pursuit of a rigid "low-to-high" textbook ordering, but recognition that reasoning ability needs to be shaped in stages. The model should first establish stable local behavior, then progressively carry more complex global structures; first learn short-chain consistency, then long-chain dependencies and multi-stage decisions; first identify clear errors, then gradually encounter ambiguous boundaries and high-interference scenarios. Without such organization, however large the reasoning dataset, it may not reliably translate into reasoning ability.

### Difficulty bucketing, curriculum learning, and staged feeding

The first step of difficulty bucketing is to clarify what "difficulty" actually consists of. For reasoning tasks, difficulty is not just superficial length or whether the final answer looks complex. More meaningful difficulty dimensions include: the number of steps required, the span of cross-step dependencies, whether there are branch decisions, whether backtracking is needed, whether external tool execution is involved, whether rule invocations are confusable, and whether local errors cascade quickly. Only by incorporating these dimensions does difficulty bucketing avoid degenerating into simple length sorting.

The core of curriculum learning is not mechanically sorting data from low to high difficulty but controlling which structural complexities the model encounters at which stage. Early in training, samples should emphasize clear rules, short steps, and clean local states, letting the model first learn to execute basic actions stably. The mid stage gradually introduces long chains, branches, error identification, and local correction, so the model extends its process control on existing foundations. Late stages then introduce highly complex, multi-path, multi-error, and cross-task composite samples, pushing the model toward stronger integrated reasoning ability.

Staged feeding does not mean later stages abandon earlier-stage data. On the contrary, a mature practice is to retain a proportion of low-difficulty samples to prevent the model from forgetting basic rules while learning complex structures. This is especially important in reasoning tasks. Many high-level errors stem not from insufficient high-level ability but from basic steps destabilizing in complex environments. If later stages contain no basic samples, the model may become more "articulate" on complex tasks but regress in basic computation and consistency maintenance.

### How difficulty should be defined

If difficulty is defined too coarsely for reasoning tasks, curriculum design easily becomes distorted. A common pitfall is to bucket only by length, word count, or surface complexity. But a long passage does not mean a truly complex reasoning structure; a short problem may require critical rule selection and conditional switching. Difficulty is better defined at the structural level than at the surface level.

An operationally meaningful difficulty taxonomy should consider at least several dimensions. The first is step depth—how many key state transitions are needed from problem to conclusion. The second is dependency span—whether early information must be preserved long-distance into the later stages. The third is branch complexity—whether multiple possible paths must be compared. The fourth is verification complexity—whether intermediate steps are easy to check by rules or executors. The fifth is correction complexity—how easy it is to locate and repair an intermediate error.

In math tasks, basic arithmetic and single-step algebraic transformations are clearly low difficulty; problems involving multi-stage variable substitution, case analysis, and long formula expansion are higher. In logic, direct derivation from a single rule is low difficulty; tasks with many premises, near-similar rules, distractor elimination, or proof by contradiction are more complex. In code tasks, local syntax fixes and simple function modifications are basic; repairs requiring cross-file understanding, contextual dependency, and hidden side-effect handling rise to higher tiers. Only by decomposing difficulty into such finer structural factors can curriculum learning truly serve capability growth.

### Curriculum learning is more than "easy before hard"

On the surface, curriculum learning may seem to mean simply "give simple samples first, then complex ones," but truly effective curriculum design goes much further. Reasoning ability is not a single monotonically rising curve but the joint development of multiple sub-capabilities. A model may already handle relatively long chains yet still err frequently on basic variable consistency; it may handle simple correction yet fail to stably choose paths in multi-branch tasks. Curriculum learning is thus not just "overall difficulty rises" but the alternating reinforcement of different sub-capabilities.

This implies that curriculum arrangement can be multi-dimensional rather than linear. One stage might emphasize short chains with high rule density; the next, medium-length chains with richer branching; one stage focuses on root-cause localization in code repair, the next on verification and regression analysis after fixing. In short, the essence of curriculum learning is not laying out a single ordering but purposefully organizing training samples according to the model's weakest current sub-capabilities.

From an engineering perspective, this also brings a practical benefit: the team can more clearly observe training gains. If the curriculum simply raises all sample difficulty uniformly, it becomes hard to analyze which capability improved and which is still weak. With explicit emphasis at each stage, data feedback and error distributions become more interpretable.

### Positives, negatives, correction samples, and self-reflection samples

A reasoning dataset containing only "standard correct solutions" may look clean but is incomplete. Real reasoning ability includes not only advancing along correct paths but also identifying, understanding, correcting, and avoiding errors. If the training set contains only perfect answers, the model has a hard time learning to handle failure modes, let alone building self-correction in complex scenarios.

Positive samples remain the foundation. They define what counts as a qualified process and provide correct rule invocation, state updates, and conclusion formation. Without positives, the model has no notion of what the target behavior should look like. But positives alone are far from enough. Negative samples show the model which paths not to walk, which steps look fluent but are wrong, which conclusions sound reasonable but lack support. They are indispensable references for training process discriminators, judge models, and reward models.

Correction samples go further. They show not only the error but also how to go from error back to correctness. For math correction, logic debugging, and code repair, such samples are especially valuable, since many real tasks are not solved from scratch but corrected from existing errors. A good correction sample retains the erroneous steps, identifies the error type, explains why it is wrong, and gives a legal corrected trajectory. Repeated exposure to such data helps the model learn local backtracking and recovery rather than collapsing entirely upon any error.

Self-reflection samples sit at another level. They show how the model or a teacher system examines its initial process, discovers latent problems, and revises its output. Unlike correction samples, self-reflection emphasizes the internal mechanism of "discovering errors," not merely providing a corrected answer. Such samples help build more stable reasoning agents and improve long-chain robustness, but they are also the most easily abused. If reflection templates are too rigid—producing pro forma "upon rechecking I find an error" without truly locating the process problem—they only add new template noise.

### Why negative and correction samples cannot be constructed casually

Once teams recognize the importance of negative and correction samples, they often adopt a simple approach: degrade a correct process slightly, or let a model freely generate erroneous trajectories, and label them as negatives. This yields high throughput but carries large risk. Low-quality error samples do not necessarily help the model learn "what is wrong"; they may expose it to large numbers of meaningless or unnatural error modes.

Valuable negatives should satisfy several conditions. First, the error type should be clear and not a jumble of unrelated mistakes. Second, the error should be task-realistic, ideally corresponding to errors that appear frequently in deployment or actual generation, not artificially manufactured oddities. Third, the error and the correct version should align so the model can understand "exactly which step differs and why." Correction samples especially require this. If the erroneous trajectory and the corrected trajectory bear no correspondence—just a wrong version and a right version—the model only learns a "wrong output–correct output" surface mapping, not process-level correction ability.

Negative and correction samples must therefore be verified and filtered as well. Not every error is worth keeping. Those worth keeping have clear error boundaries, explicit correction logic, and support process learning.

### Organizing real-world and synthetic problems

Real-world and synthetic problems are not an either/or choice in reasoning data engineering; they serve different roles. Real-world problems define the realism of the problem space, the naturalness of error distributions, and the credibility of task boundaries. Synthetic problems supplement coverage, increase long-tail sample counts, control difficulty distribution, and target capability gaps. A mature data system organizes the two as a continuous chain rather than placing them in opposition.

Generally, real-world problems make better seed corpora. Teams can collect high-quality samples from textbooks, competition problems, real code-repo issues, OJ problems, logic test suites, and curated cases to build a base dataset with clear structure and adequate verification. These real samples can enter training directly and also serve as reference templates for subsequent synthesis. Based on them, teams can perform numerical perturbations, condition swaps, rule switching, goal transformations, error injection, solution-style transfer, and other extensions to gradually generate large numbers of synthetic problems that remain in-distribution with real tasks.

But synthesis in reasoning scenarios cannot just be "tweak the problem." Truly high-value synthetic data must inherit the structural constraints of real problems and additionally possess verifiability and stratifiability. That is, what comes out is not just "more problems" but "more problems that fit a verification loop, can be clearly labeled by difficulty, and can support process supervision." If synthesis emphasizes only surface diversity without process quality, then no matter how large the scale, it is only manufacturing more unstable samples.

### Retain basic generation methods from SFT and synthetic data

Although reasoning data engineering emphasizes process supervision, it does not start from scratch. The foundational generation methods from SFT and synthetic data engineering remain the bedrock of reasoning data construction. High-quality seed sample filtering, template-based expansion, condition substitution, role constraints, hard-example injection, multi-model collaborative generation, rule-based filtering, human spot checks—these methods remain necessary in reasoning scenarios. The question is not whether they still apply but how they need to be upgraded.

The upgrade has three main aspects. First, the generation goal upgrades from "usable answer" to "verifiable process." Second, the quality standard upgrades from "whole-problem broadly correct" to "key steps checkable, error types attributable." Third, sample organization upgrades from "unified mixed ingestion" to "structured arrangement by difficulty, error type, and process form." In other words, reasoning data engineering does not negate prior SFT and synthesis methods; it pushes them to a stricter process level.

### Practical thinking on curriculum organization and sample ratios

After difficulty stratification, sample ratios must still be addressed in practice. How many basic positives, how many high-difficulty samples, how many correction samples, and how many reflection samples to invest at each stage directly shapes the model's behavioral preferences. If too many high-difficulty samples are introduced too soon, the model may start learning complex surface forms before basic rules stabilize; if negatives and correction samples are too plentiful, the model may over-focus on error contexts, blurring the clarity of standard paths.

A more solid approach is: early in training, focus on high-quality positives and low-to-medium difficulty samples to establish the basic process pattern; mid-stage, gradually raise the share of long-chain, branch, and correction samples so the model learns to handle complexity on top of stable paths; late-stage, fill in samples for weak categories based on error analysis—if logical leaps are frequent in a class, add logic samples with explicit premise citations; if pseudo-explanations are common in a code-repair class, add strongly constrained "reason–patch alignment" samples.

There is no fixed golden ratio across all tasks. The key is that sample ratios should serve the capability most needed at the current stage, not be split evenly for the sake of surface balance. Reasoning data engineering is not about displaying samples; it is about using data to control the learning path.

### Why difficulty stratification helps subsequent evaluation and iteration

Difficulty stratification helps not only training but also evaluation and data iteration. Without a difficulty structure in training and validation sets, teams see only a single overall score and cannot easily tell at which tier the model fails. With samples bucketed by step count, rule complexity, branching, correction depth, and the like, evaluation can more clearly indicate whether the model is stuck at basic short chains, long-chain consistency, branch selection, or error correction.

This is crucial for data recycling. Data engineering improvement should not be blind quantity expansion but targeted repair. If the model loses on mid-to-high difficulty logic mainly due to exceeding rule scope, add adversarial samples with explicit rule boundaries; if code repair frequently shows "correct explanation, misaligned patch," targeted additions of reason–implementation alignment data are warranted. The clearer the difficulty stratification, the more precise the iteration actions, and the closer the data system gets to a true closed loop.

### From data pile-ups to data curricula

The biggest difference between reasoning data engineering and ordinary data pile-ups is the pursuit of capability-shaping paths through organized data, not merely larger sample counts. A reasoning dataset without curriculum, stratification, or error organization—however large—may amount to randomly throwing processes of mixed quality into the trainer. What the model finally learns then tends to be a mixed surface style, not a stable reasoning structure.

A data system that has gone through difficulty bucketing, sample-type curation, error-mode control, and iterative feedback can, even at moderate scale, more reliably accumulate dependable capability. In such a system, no class of sample stands alone: positives define the standard path, negatives delimit error boundaries, correction samples shape recovery ability, self-reflection samples strengthen self-checking, and difficulty stratification governs when and in what proportion each enters training. Only when these elements work together does reasoning data truly progress from "data pile-up" to "data curriculum."

## 18.5 Engineering Cases and Connections

### A math reasoning dataset construction case

Imagine a math reasoning dataset project covering middle school through introductory college. Early in the project, the team can collect real problems covering algebra, geometry, functions, probability, and other subdomains, and define a basic schema for each class—problem statement, known conditions, goal, standard solution, step sequence, intermediate expressions, and answer. A teacher model or rule program then generates initial trajectories, after which expression evaluators, symbolic computation tools, and rule checkers verify whether each step is legal.

In this flow, the team should not stop at "whole-problem accuracy." More importantly, failed samples should be opened up: is the issue frequent basic arithmetic errors, frequent loss of variable substitutions and conditional constraints, or a tendency to write pretty but empty explanations? Only by mapping these errors into statistically meaningful types can the data team improve generation templates, add targeted samples, and adjust verification rules in turn. Finally, the dataset should be ingested along two dimensions—difficulty and error type—serving basic SFT, process supervision training, and correction ability training respectively.

### Why math reasoning datasets are an ideal scenario for process supervision

A key reason math tasks are particularly suited to reasoning data engineering is that they naturally have strong stepwise structure. Unlike open-ended writing or general chit-chat, math problem solving usually includes relatively explicit intermediate state changes: how known conditions are organized, how variables are defined, how expressions are transformed, how rules are invoked, how conclusions are drawn. Process here is not auxiliary commentary—it is part of the problem itself. As a result, errors in math tasks rarely stay abstract; they manifest concretely as a step being illegal, a condition being omitted, a substitution being invalid, or a theorem's prerequisites being unmet.

These structural features make math reasoning data particularly suitable for step-level labels and automatic verification. Many intermediate steps can be formally reviewed via expression evaluators, symbolic computation tools, and rule checkers. The team does not have to rely solely on human intuition; it can offload some critical judgments to reproducible verification modules. For data engineering, this means math reasoning data can be not only generated but also systematically QA'd, stratified, and iterated.

Furthermore, math tasks help teams establish basic methodology for process supervision. What counts as a step, a leap, an illegal transformation, or a local error is easier to delimit in math than in general natural-language tasks. A team that first builds this process-oriented data pipeline in math and then transfers it to more complex logic and code tasks usually accumulates stable engineering experience more readily.

### Organizing math problem sources

A usable math reasoning dataset is not just random problem collection followed by uniform solution generation. The problem source itself needs organization. Generally, real problems should form the initial core seed pool, since they determine the dataset's task boundaries, language style, and error distribution. Textbook exercises, basic competition problems, standardized exam problems, online problem banks, and curated cases can all serve as seeds. But these sources should not simply be concatenated; they should be reorganized by knowledge domain, solving structure, and difficulty tier.

For example, in algebra one should distinguish equation solving, identity transformations, function evaluation, and recurrence; in geometry, shape property judgment, angle relations, auxiliary line construction, and proof chains; in probability and statistics, event decomposition, formula substitution, conditional probability, and combinatorial logic. The point of such partitioning is not directory tidiness but later precision in defining schemas, verification rules, and difficulty buckets.

Beyond real problems, teams can build synthetic extensions from these seeds. But math synthesis should not be mere number replacement. A higher-quality approach preserves the original reasoning skeleton and applies controlled extensions to condition combinations, target variables, distractors, edge cases, and solution paths. The same class of equation problem can be perturbed in coefficients to control the number and form of solutions; the same class of geometry problem can be conditioned added or removed to control proof-chain length; the same class of probability problem can have its event structure adjusted to introduce more complex conditional branches. Only then does synthetic data truly serve difficulty stratification and capability extension, rather than churning out surface-different but kernel-identical duplicates.

### How to operationalize the math sample schema

Schema design in math reasoning data determines how smoothly downstream verification and training proceed. A too-loose schema preserves only coarse "problem–solution–answer" information and cannot support process supervision; a too-complex one inflates annotation cost and lowers throughput. Schemas in math tasks should be organized around key intermediate states, not around surface text.

A reasonably mature math sample should at minimum contain fields for problem, known conditions, goal, solution steps, intermediate expressions, and final conclusion. To strengthen verifiability, teams can add action labels and justification labels—marking each step as substitution, expansion, transposition, elimination, differentiation, integration, case analysis, or conclusion merging; and supplying "by which rule does this hold" justifications for key steps (equation properties, function monotonicity, geometric theorems, probability formulas). The verifier then checks not just textual matching but whether actions are legal, justifications match, and state is continuous.

For higher-level math datasets, schemas can also include difficulty labels, knowledge-point labels, error-hotspot labels, and solution-type labels. The same problem might be tagged "multi-step algebraic transformation," "case analysis," "boundary-condition sensitive," "prone to sign errors." Such metadata does not directly participate in solving but is critical for curriculum learning, sample ratios, and error analysis. It helps the team understand the dataset's internal structure rather than treating the whole problem bank as an undifferentiated pool.

### The closed loop of generation and verification for math reasoning trajectories

Initial trajectories in math data engineering can come from multiple sources. A teacher model can supply more natural, language-style processes; a rule program can supply more strongly constrained formal steps; human experts can supply high-precision references for critical samples. In real engineering, the most effective approach is usually a combination: large-scale candidate generation by a teacher model, local checks by rule programs or executors, with human review reserved for high-value or high-difficulty samples.

The key to this closed loop is not "how closely a generated solution resembles the standard answer" but whether intermediate steps can be checked layer by layer. For algebra problems, expression evaluators can check whether a transformation preserves equivalence; for function problems, symbolic computation can check correctness of derivatives, integrals, and extremum judgments; for geometry problems, even though pure natural-language proofs are harder to automate fully, local relationships, theorem references, and condition coverage can still be semi-automatically checked. This converts many processes that "seem plausible" into processes where "key steps actually hold."

After verification, failed samples should not simply be discarded. They are themselves important sources for process supervision. Some failed samples are suitable for cleaning and regeneration; others for conversion into correction samples; still others for training judge models or error classifiers. The point of math data engineering is not to preserve only spotless standard solutions but, through verification, to reorganize processes of different quality levels and error types so each finds its most suitable training use.

### Error decomposition and feedback loops in math data

In math reasoning datasets, whole-problem failure is just a surface phenomenon. What truly matters is breaking failure into operationally meaningful error categories. Some samples fail because of unstable basic arithmetic; some because of illegal formula transformations; some because of omitted conditions; some because the model never truly derives but writes a long, plausible-looking explanation. Without this decomposition, the team can only vaguely say "performance on this class is poor" without knowing what data to add, which templates to revise, or which verification rules to tighten.

Math reasoning projects therefore need an error feedback loop. Errors do not end at verification but flow back into data design. If a class of problems frequently loses variable substitutions, the schema may need a more explicit intermediate-variable field; if a class often exhibits logical leaps, the process template may be over-compressed; if a class consistently generates long, empty explanations, the quality-scoring system may need stronger penalties for redundancy and pseudo-explanation. Through such feedback, data engineering converges over time rather than spinning in a mechanical loop of "collect more problems, generate more processes."

Going further, error statistics can directly guide curriculum organization. If the model mainly fails on basic computation in low-difficulty algebra, strengthen basic-step samples first; if it mainly fails on long-chain consistency and case analysis at mid-to-high difficulty, add relevant process data. Only when errors flow back in a structured way does a math reasoning dataset gain real iterative vitality.

### Ingestion and use partitioning of the math dataset

Math reasoning data is not well served by a single uniform ingestion form. A more sensible approach is to split data into subsets according to sample quality, process completeness, and intended use. Some high-quality positives go into basic SFT to shape standard solving style and basic process format; samples with adequate step labels go into process supervision training to reinforce local correctness and intermediate state consistency; samples containing explicit errors and corrections are better suited for training correction, self-checking, or process-discrimination abilities.

This partitioning matters because not all data fits the same learning method. A sample with minor errors but overall sound process is risky as a direct SFT positive but very valuable as an "error identification—correction" sample. Conversely, a concise standard solution with few process fields may not support fine-grained process supervision but still suits basic behavior shaping. Mature data teams learn to assign uses based on sample characteristics, not just make binary "good keep, bad delete" decisions.

### A code repair and program synthesis reasoning data case

The engineering flow for code tasks resembles math's, but verification relies more heavily on the program execution environment. A high-quality code repair dataset should not retain only "buggy code–fixed code" pairs; it should include the failing test that triggered the defect, error logs, the localization process, the rationale for changes, the patch content, and post-fix test results. The model then learns the full closed loop of problem solving, not just static text mapping.

Program synthesis tasks likewise need process representations. For a sample mapping a natural-language requirement to a code implementation, teams can retain requirements analysis, intermediate design, key function sketches, boundary-condition statements, and the final implementation, then verify via unit tests, static checks, and runtime alignment with the spec. If all intermediate state is omitted and only the requirement and code are retained, the model easily learns surface patterns but cannot stably plan program structure on new tasks.

Code data is also especially suited to constructing correction samples. Program environments naturally support a "fail–locate–fix–reverify" closed loop, which more readily yields high-value process data than pure natural-language scenarios. Teams can build a continuously expanding reasoning-and-repair data pipeline from existing repositories, OJ problems, test suites, and failure logs.

### Why code tasks are better suited to process supervision than ordinary text tasks

A natural advantage of code tasks is that they have not only textual processes but also program-behavior ground truth. For pure natural-language reasoning, whether steps are legal often still relies on rule-based or semantic judgment; in code tasks, many intermediate behaviors can be verified directly through compilation, execution, testing, and static analysis. This means code reasoning processes can be bound to real execution results rather than remaining at plausibly worded explanations.

This makes code repair and program synthesis ideal scenarios for process supervision. Whether a model truly understood a bug should not be judged from its explanatory text but from whether the patch passes tests, preserves original functionality, and introduces no new exceptions. Whether a model truly implemented a requirement should not be judged by surface similarity to some template but by whether program behavior conforms to the spec. Because code tasks possess this executability, data engineering teams can build "process–verification–feedback" closed loops earlier than in general natural-language scenarios.

But this also means the bar for code reasoning data is higher. A sample with eloquent text but unsound behavior is not high-quality data—it is more dangerous than ordinary noise, as it reinforces a particularly bad pattern: using explanation to mask behavioral defects. The more code reasoning samples emphasize process, the more execution verification must be a core constraint rather than an add-on check.

### Sample structure of code repair data

A code repair sample directly usable for reasoning training cannot consist only of input code and post-fix code. It should at least also include failure triggers, error messages, affected locations, rationale for localization, repair ideas, the specific patch, and verification results. This structure is not for "more information's sake" but reflects that code repair is essentially a multi-stage solving process. Retaining only before/after code diffs teaches the model only local text replacement patterns, not real debugging behavior.

For example, error logs tell the model how failure surfaces; tests tell the model the external behavioral constraints; the localization process tells the model why a particular code region was targeted; the repair rationale tells the model the causal logic behind the patch; verification results tell the model whether the modification truly resolved the issue. These elements form the "process evidence chain" of code repair. Missing any of them pushes the model toward surface-patch learning rather than a full diagnosis-and-repair loop.

In more complex scenarios, code repair samples can also include candidate repair plans and their elimination reasons. Especially when multiple repair paths exist, just keeping the final patch is insufficient; the model should also see "why not these other plans." This is especially helpful for training stronger repair decision-making.

### Why intermediate design in program synthesis must be preserved

Program synthesis tasks look like direct "requirement-to-code" mapping, but truly high-quality implementations are rarely written in one shot. They usually pass through requirements decomposition, interface design, core data structure decisions, key function sketches, boundary-condition considerations, and exception-handling planning. If these stages are omitted entirely, leaving only the requirement and final program, the model is biased toward surface code-template matching rather than program-structure planning.

For simple problems, this defect may not be obvious because short code tasks can be done in one hop. But once requirements are even moderately complex—multi-function coordination, state management, boundary-condition handling, performance constraints, or test-driven development—samples without intermediate design leave the model lacking stable planning ability. Local code may look polished, but overall structure is scattered, or early design conflicts with later implementation.

Program synthesis data should therefore preserve intermediate design traces. Pull out requirement analysis and function responsibility splits, write out key data structures and interface sketches explicitly, state boundary conditions and exception branches before implementation. Models trained this way do not just "write code"; they are more likely to learn to "plan first, then implement, then verify"—a capability far more important for complex program tasks than local code fluency.

### Verification loops and use of failures in code data

The most valuable aspect of code tasks is that failures themselves are highly structured and usable. Compilation failures, test failures, runtime errors, static-check warnings, and unmet performance constraints are not mere "negative results"—they are process signals with clear localization value. Unlike pure natural-language scenarios, failures in code tasks naturally come with context: error line numbers, exception stacks, test assertion info, type-mismatch hints. Retained, these are themselves extremely valuable components of reasoning data.

Code data engineering therefore should not view failures as things to be filtered out but as important sources for high-value process samples. A single failure can yield at least three kinds of data: the failure sample itself as training data for error identification; the failure-to-fix trajectory as a correction sample; multi-round failures and stepwise corrections forming more complex self-reflection and iterative repair samples. Such data is hard to obtain in volume from ordinary text tasks but flows naturally from code environments.

In this sense, code data accumulation does not depend on "value only from successful fixes"; every "fail–locate–modify–verify" cycle produces process data. As long as the team structurally preserves each link in the chain, a continuously expanding reasoning-and-repair data pipeline can be built.

### Two sources of code data: real engineering and problem banks

Sources of code reasoning data divide into roughly two classes. One is real engineering environments—repo issues, historical bug-fix records, pull requests, CI failure logs, regression reports. This data is highly realistic, with natural errors and rich context, well suited to training repair and analysis abilities close to real workflows. But its issues are equally clear: contexts may be very long, noise is heavy, environmental dependencies are complex, and cleaning costs are high.

The other comes from problem banks and controlled environments—OJ problems, teaching cases, hand-crafted bug sets, restricted programming tasks. These are easier to control in terms of task boundaries, test environments, and difficulty tiers, and are well suited for program synthesis and basic code repair training. The downside is limited real-world complexity, with error distributions possibly diverging from real engineering scenes.

Mature teams usually do not rely on only one source. Problem-bank data serves well as the main source for basic ability training and process-format unification; real engineering data complements it for higher-order robustness, complex-context handling, and real-world repair behavior. Combined, the code reasoning data system remains controllable without divorcing from real development.

### Constructing code correction samples

Code tasks are especially suited to building correction samples because behavioral changes are tracked relatively objectively. A high-quality correction sample is not a static "buggy code–correct code" pair but should include the failure trigger, root-cause identification, repair action, and post-fix verification. Where possible, intermediate failed versions should also be retained, showing that fixing is rarely one-shot but often approaches correctness over several attempts.

For example, a sample can first show the original failing test and logs, then give the initial localization, then a first repair attempt and its new failure, then the corrected official patch and passing tests. Such data is especially valuable because it tells the model not only "what the correct answer is" but also "how errors are exposed, how they are identified, why a correction had to be adjusted." This sense of process is exactly what many code-generation systems lack most in real environments.

### Connections with previous SFT and synthetic data chapters

Viewed across the book, reasoning data engineering does not replace SFT and synthetic data engineering; it extends them deeper. Earlier SFT chapters addressed "how the model should answer"; the synthetic data chapters addressed "how to mass-produce usable samples"; this chapter addresses "when a task is fundamentally process-dependent, how to turn the reasoning process itself into a designable, verifiable, trainable data object."

A mature team's methodological progression is usually continuous: first use SFT methods to define task interfaces and output specifications; then use synthetic data methods to expand coverage and fill long tails; finally, for reasoning tasks, add trajectory representation, step verification, difficulty curriculum, and correction loops. Only by linking these three layers does a team truly possess a data engineering chain oriented toward reasoning ability, rather than a collection of disjointed data production tricks.

### This chapter's relation to previous chapters is progression, not replacement

In the book's structure, reasoning data engineering is easily misread as "a fancier new chapter," as if its appearance renders prior SFT and synthetic data methods obsolete. That is not the case. Reasoning data engineering stands precisely because earlier parts have solved foundational problems: how to define input/output interfaces, how to unify style and format, how to generate initial samples stably, how to establish basic quality control. Without these prerequisites, the reasoning process itself would be hard to organize as a stable data object.

The relation between this chapter and prior content is therefore progression, not replacement. SFT provides the base of behavioral boundaries and output forms; synthetic data provides coverage expansion and large-scale production; reasoning data engineering builds on these to answer "how to make process itself a training object." If the three are siloed, teams often fall into local optimization—doing only surface format unification without process quality control; only mass generation without verification and curriculum; only reasoning process while neglecting interface and distribution design.

### A continuous chain from SFT to synthesis to reasoning process supervision

A mature team's method chain does not leap suddenly from ordinary data to complex process data; it deepens layer by layer. First, SFT methods clarify task interfaces, answering style, output format, and basic behavioral boundaries, giving the model "answer-as-required" capability. Next, synthetic data and seed-extension methods raise coverage, fill long tails, and increase data throughput, ensuring the model is not stuck on a tiny set of templated cases. At the reasoning stage, trajectory representation, automatic verification, error classification, difficulty curriculum, and correction samples are introduced to shift the model from "can answer" to "can answer stably."

The significance of this chain is connecting otherwise scattered data engineering techniques into a capability-growth system. SFT is no longer just format tuning; synthetic data is no longer just scale expansion; reasoning process supervision is no longer just adding a bit of intermediate text. Together they constitute the complete data path oriented toward reasoning ability. What the team gains is no longer a pile of isolated data assets but a reasoning data engineering system that continuously produces, verifies, corrects, and upgrades.

### Why a team gains real reasoning-building capability only by linking all three layers

If a team does only SFT without synthesis and process supervision, the model typically learns only surface behavior on limited samples; if only synthesis without process verification, the model absorbs more samples but possibly just larger-scale noise; if only process supervision without prior task interfaces and sample generation, the data system lacks unified standards and is hard to scale. The real question is never "which method matters most" but whether these methods can be organized as continuous engineering.

Only when SFT defines boundaries, synthetic data expands coverage, and reasoning process supervision raises intermediate reliability does a team truly have the conditions for building reasoning ability. Data engineering then is no longer just supplying raw material for training; it actively shapes the model's solving habits, error-handling tendencies, and capability growth trajectory. This is the core message of this chapter: reasoning ability does not grow naturally out of a single model trick. It needs a complete, mutually supportive system of data engineering mechanisms.

## Chapter Summary

The core of reasoning data engineering is not getting the model to "say a bit more about its thinking process," but rather bringing the intermediate process into a designable, verifiable, stratifiable, trainable engineering closed loop. Looking only at the final answer conceals reasoning defects, because a correct result is not the same as a correct process; outcome-only supervision overlooks key error sources in complex tasks, since the model may stumble onto the right answer via unstable or even incorrect paths. To address this, teams need first to design appropriate reasoning trajectory representations—letting CoT, scratchpad, program-of-thought, and tree-of-thought each serve their suitable tasks; next to build step-level automatic verification systems combining rule verification, execution verification, unit tests, and judge models; further to classify errors into arithmetic errors, logical leaps, pseudo-explanations, hallucinated steps, and so on, and based on these to construct step-level labels and process quality scores; and finally, through difficulty bucketing, curriculum learning, positive/negative organization, correction samples, and self-reflection sample design, to elevate reasoning data from scattered samples into a sustainably evolving data system.

For math, logic, code, and similar tasks, truly high-value data is not "data with correct answers" but "data with reliable processes, attributable errors, organized difficulty, and closed-loop verification." This is the fundamental distinction between reasoning data engineering and ordinary SFT data engineering.

## References

<!-- TBD: papers, blogs, tools, and official documentation cited in this chapter. See publishing/citations_progress.md for the completion strategy. -->
