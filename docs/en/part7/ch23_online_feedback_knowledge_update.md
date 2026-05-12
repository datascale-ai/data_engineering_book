# Chapter 23: Online Feedback Loops and Knowledge Updates

When a RAG system goes live, the real data engineering work has not ended — it has entered a new phase. Before launch, the system mostly faces a controlled test set, a curated knowledge base, and relatively standardized questions. After launch, it starts facing real users, real business processes, and a continuously changing knowledge environment. Users ask questions in non-standard phrasings, enter conversations with omitted conditions, and pose new questions while policies, products, and processes keep evolving. At this point, whether the system is reliable no longer depends solely on how completely the initial knowledge base was built; it depends on whether the system can continuously discover problems, fix them, and update itself from real usage.

This chapter therefore focuses on the data loop that comes after a system goes live. It discusses how to turn online signals — logs, clicks, ratings, corrections, tickets, and human handoffs — into governable data assets. The concern is not whether a single answer is correct, but whether the system can establish stable feedback routing, error attribution, knowledge updates, version rollback, and an operations cadence, so that every failure becomes input to subsequent improvement.

Around this goal, the chapter starts with the necessity of an online feedback flywheel, and then discusses event collection and feedback routing, knowledge updates and version governance, metric dashboards and operational cadence, and online incident reviews and SOPs. The core question it answers is: as an LLM application moves from launch to long-term operation, how should data engineering support the system's continuous improvement.

------

## 23.1 Why "Going Live" Is Only the Starting Point of the Data Flywheel

### 23.1.1 From Offline Validation to Real Usage: Launch Is Not the Endpoint

In many LLM application projects, going live is often treated as a phase-end milestone. Once the system is built, tested, connected to the knowledge base, deployed to production, and stable on a small set of sample questions, the team often considers the project to be in its wrap-up stage. However, for RAG systems, knowledge assistants, enterprise Q&A systems, and any LLM application facing real users, going live is not the end of data work — it is the point at which the data flywheel truly starts to spin.

The reason is that the problem distribution seen during the offline phase is typically only a small fraction of the system's real usage. When development teams build evaluation sets, they tend to choose questions that are relatively clear, controllable, and easy to annotate — for example, "What is the reimbursement standard under a given policy?", "How do I enable a particular product feature?", or "Which materials are required for a given process?" These questions have clear answers and map easily to specific document fragments in the knowledge base. But after launch, user questions tend to be more complex, more vague, and closer to real business context. Users do not ask questions following document titles, nor do they proactively supply complete conditions. They might ask: "In my situation, can I claim this?", "Does that process from last time still apply?", or "Why did this metric suddenly change?" Such questions often contain omissions, references, context dependencies, implicit constraints, and cross-document composition. Only in real interaction does the system expose problem types that offline evaluation can hardly cover, which is why data engineering for LLM applications generally cannot stop at building a knowledge base once or completing a single evaluation set. Production is not a static exam; it is a continuously evolving business environment in which user questions, knowledge content, business rules, permission scopes, product versions, and organizational processes all change over time. If a system lacks post-launch feedback collection, issue triage, knowledge updating, and version governance, its capabilities will not only fail to improve — they may gradually degrade as knowledge becomes stale, user needs shift, and errors accumulate.

The core question of Chapter 23 is therefore how to turn post-launch real usage into a data asset, and how to feed that data back into the knowledge base, retrieval system, generation strategy, and evaluation system continuously. Compared with the RAG data pipelines discussed in Chapter 21 and the multimodal visual retrieval discussed in Chapter 22, this chapter focuses more on the closed-loop mechanisms that operate after the system is running: how the system perceives failure, attributes problems, updates knowledge, verifies whether updates are effective, and incorporates these actions into a stable operational rhythm.

From this perspective, going live with an LLM application is not delivery completion but entry into a new phase of the data lifecycle. The key in this phase is no longer one-off construction, but continuous collection, continuous diagnosis, continuous backfilling, and continuous governance. Only with such a closed-loop capability can RAG and multimodal knowledge applications move from "demoable" to "operable."

---

### 23.1.2 The Real Problem Distribution Is Only Exposed After Deployment

Offline evaluation sets usually have clear boundaries. They come from existing documents, questions curated by business experts, typical samples constructed by the project team, or sets of questions automatically generated by models. Such data is very important for pre-launch validation, but it cannot fully represent how real users ask questions. Real user questions typically exhibit three salient characteristics: non-standard phrasing, incomplete conditions, and shifting goals.

First, user phrasing is not always consistent with terminology in the knowledge base. If the system relies solely on standard terms from documents to retrieve, it may fail to recall the correct knowledge. Online feedback data can expose these differences in real expressions, helping the system add synonyms, business aliases, and query rewriting rules.

Second, user questions often lack key conditions. For instance, an employee asks "Can this expense be reimbursed?" without stating the expense type, location, employee identity, project ownership, or approval status. If the system answers directly, it may ignore applicability conditions; if it recognizes missing conditions, it should follow up or give a conditional answer. Such issues are hard to cover well in offline evaluations because evaluation samples are usually curated into relatively complete questions, while real users often supply only fragmentary information.

Third, user goals may shift across multiple turns of interaction. A user may start by asking about a policy clause, then move to "help me decide if my case applies," and further to "give me a paragraph I can send to finance." This means the system needs to move from single-turn Q&A toward task-style interaction, and the multi-turn trajectories recorded in online logs are an important data source for understanding real task flows.

This gap between the real problem distribution and the offline evaluation distribution can be called "post-launch distribution drift." It does not necessarily mean model capability has degraded; rather, it indicates the system has entered a more realistic and complex data environment. Many RAG systems perform unstably after launch — not because the offline evaluation was fabricated, but because the offline evaluation cannot cover the long-tail distribution of production questions.

*Table 23-1: Differences Between Offline Evaluation Questions and Real Online Questions*

| Dimension              | Offline Evaluation Questions                | Real Online Questions                                       | Impact on the System                                              |
| ---------------------- | ------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------- |
| Phrasing               | Standard terms, well-structured             | Colloquial, abbreviations, many aliases                     | Affects query understanding and retrieval recall                  |
| Condition completeness | Usually contains necessary conditions       | Often omits identity, time, version, and scenario           | Requires follow-up questions, condition recognition, risk warning |
| Question boundary      | Mostly single questions                     | Multi-intent, multi-constraint, cross-document composition  | Requires decomposition and multi-route retrieval                  |
| Context dependency     | Usually no context or fixed context         | Constantly changing across multi-turn interactions          | Requires state management and conversation memory                 |
| How errors surface     | Through human judgment or metric computation | Through follow-ups, downvotes, corrections, abandonment    | Requires online feedback collection and attribution               |

As Figure 23-1 contrasts, in real systems, this gap directly shapes data engineering priorities. Before launch, teams focus more on knowledge base coverage, parsing quality, index structure, and basic evaluation; after launch, teams must also focus on user phrasing, failure samples, feedback annotation, knowledge expiration, permission mis-recall, citation trustworthiness, and multi-turn task continuity. In other words, data engineering before launch emphasizes "getting the system built," while data engineering after launch emphasizes "making the system continuously better."

![Figure 23-1: From Offline Evaluation to Real Online Problem Distribution](../../images/part7/图23_1.png)

*Figure 23-1: From Offline Evaluation to Real Online Problem Distribution*

The real online problem distribution also has a strong temporal dimension. Certain problems may suddenly surge due to a new policy release, a new product launch, an organizational reshuffle, or an external event. For example, after a reimbursement policy update, employees may cluster around questions about old-versus-new rule differences; after a new feature launches, customers may cluster around usage and limitations; after an earnings release, analysts may cluster around why certain metrics changed. Without online question monitoring, it is hard to detect such shifts in time, let alone drive synchronized updates of the knowledge base and retrieval strategy. Therefore, production-grade LLM applications must treat online questions as a continuously generated data asset. User questions are not noise — they are the most authentic demand signals the system receives; user failures are not isolated incidents but entry points for system improvement. Only by establishing a collection and analysis mechanism oriented toward the online distribution can a team truly understand the audience, scenarios, and boundaries the system serves.

---

### 23.1.3 Why Online Feedback Is More Valuable Than Offline Samples

Online feedback matters because it simultaneously offers three properties: authenticity, timeliness, and behavioral signal.

Authenticity means online feedback comes from real users performing real tasks. Offline evaluation samples are often human-curated with clear question boundaries and answers that are relatively easy to annotate; online feedback, in contrast, contains real user expressions, real context, and real business goals. It not only tells the team "can the system answer the standard question," but also "does the system help the user complete the task."

Timeliness means online feedback reflects the latest changes in knowledge and demand. Enterprise knowledge bases, product documentation, process policies, and business rules are constantly updated. Even a high-quality offline evaluation set can quickly go stale. Online feedback can expose new problems, new terms, new demands, and new conflicts at the first moment. For example, after a new product version launches, if users frequently ask "does the old API still work?", it suggests the documentation may need a migration note; if users repeatedly press on the applicability scope of a policy, the original clause may be unclear or the retrieval may not have covered all the conditions.

Behavioral signal means users express attitudes not only through explicit feedback but also through behaviors that indirectly reveal system quality. Explicit feedback includes upvotes, downvotes, corrections, written comments, and human flags; implicit feedback includes whether the user continues asking, rewrites the question, clicks on cited sources, copies the answer, hands off to a human, or abandons the session. Compared with a single correct/incorrect label, these behavioral signals are much closer to the real user experience.

For example, a user who does not downvote but rewrites the same question three times in a row suggests the previous answers may not have met their need; a user who clicks on a cited source and stays for a long time suggests the answer triggered a need to verify; a user who immediately hands off to a human after seeing the answer suggests the system may not be covering high-risk or complex judgment scenarios. When such signals are collected systematically, they can be transformed into high-value training samples, evaluation samples, and knowledge update tasks.

*Table 23-2: Types and Value of Online Feedback Signals*

| Feedback Type     | Typical Source                              | Problem It Reflects                                    | Optimization Actions It Can Drive                    |
| ----------------- | ------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------- |
| Explicit positive | Upvote, acceptance, "this helped"           | Answer likely met the need                             | Distill high-quality Q&A samples, strengthen templates |
| Explicit negative | Downvote, correction, error report          | Answer wrong, incomplete, or untrustworthy             | Add to failure bank, trigger error attribution       |
| Follow-up         | User keeps asking, adds conditions          | Previous answer insufficient or condition unclear      | Improve follow-up strategy and context assembly      |
| Rewrite           | User rephrases and re-asks                  | Query understanding or recall failed                   | Add synonym expressions and query-rewrite samples    |
| Citation click    | User views source document                  | User needs to verify evidence                          | Improve citation readability and evidence locating   |
| Human handoff     | Ticket, human takeover                      | System cannot complete high-risk judgment              | Build human-review and backfill flows                |
| Session abandonment | Drop-off, no continuation                 | UX failure without explicit feedback                   | Use as weak negative samples for analysis            |

Online feedback also helps teams build a prioritization mindset. In the offline phase, teams may chase overall metric improvements — Recall@k (number of relevant results in the top k), Answer Accuracy, Citation Accuracy, and so on. After launch, what truly needs prioritization is often not all errors, but high-frequency errors, high-risk errors, and high-impact errors. High-frequency errors mean many users encounter the same issue, so fixes pay off quickly; high-risk errors involve legal, financial, medical, compliance, or permission-sensitive scenarios that must be controlled first; high-impact errors may appear in critical business processes such as approvals, reimbursements, contract review, customer service, and incident handling. Online feedback supplies error frequency, user impact scope, and business consequences, helping the team direct scarce resources toward the most valuable fixes.

It is important to note that online feedback is not equivalent to directly trainable data. User feedback often contains noise, emotion, misclicks, and missing context. A downvote may be because the answer is genuinely wrong, but it could also be because the answer is too long, the tone is off, no direct conclusion is given, or the user's input was incomplete. Therefore, online feedback must be cleaned, attributed, annotated, and routed before it can become a usable data asset. This is precisely the data-engineering core of the online feedback loop: not simply collecting logs, but turning user behavior into structured problems, turning structured problems into fix tasks, and turning fix tasks into new knowledge, indexes, evaluations, and model improvements.

---

### 23.1.4 Why Systems Get Worse Over Time Without a Feedback Pipeline

An LLM application without a feedback pipeline may perform well at launch but gradually degrade after running for some time. "Degradation" here does not necessarily mean model parameters get worse; it means the mismatch between the system and the real business keeps widening.

The first type of degradation comes from knowledge expiration. Enterprise policies, product features, API docs, organizational processes, pricing rules, and compliance requirements all change. If the knowledge base is not updated in time, the system keeps answering based on stale knowledge. More dangerously, stale knowledge often still looks reasonable, so users have trouble detecting it. For instance, the system may answer reimbursement standards based on an old policy or explain feature limits based on an old product manual. Such errors are insidious and especially require online feedback and version governance to catch.

The second type of degradation comes from shifts in user phrasing. As users become familiar with the system, the way they ask changes. Early users may try standard questions, while later users pose more complex, more specific, and more task-oriented questions. A system that only optimizes for the pre-launch evaluation set will increasingly fail to cover real needs. Users will then repeatedly follow up, rewrite their questions, or switch to human channels, and both usage and trust will decline.

The third type of degradation comes from error accumulation. RAG and multimodal systems usually consist of multiple stages: collection, parsing, chunking, indexing, retrieval, reranking, context assembly, generation, and citation. A small error at any stage can amplify downstream. Without a failure-sample pipeline and error-attribution mechanism, such problems persist and recur across more user questions.

The fourth type of degradation comes from broken organizational ownership. After launch, the system may enter an unmaintained state: the algorithm team thinks delivery is done, the business team sees problems as model-capability issues, the platform team only cares about service availability, and content maintainers only upload documents. Without a unified feedback loop, user issues bounce among teams but never get clearly fixed. Over time, the system appears live but its quality keeps declining.

*Table 23-3: Typical Degradation Paths Without an Online Feedback Pipeline*

| Source of Degradation | Concrete Symptom                                  | User-Side Perception                  | Root Cause                                                |
| --------------------- | ------------------------------------------------- | ------------------------------------- | --------------------------------------------------------- |
| Knowledge expiration  | Old policies, APIs, and processes still retrieved | Plausible but actually wrong answers   | Lack of knowledge versioning and expiration governance    |
| Phrasing mismatch     | Colloquial queries miss formal documents          | "Off-topic" answers                   | Lack of online query analysis and synonym backfill        |
| Repeated errors       | Same errors keep recurring                        | "The system never learns"             | Lack of failure bank and fix loop                         |
| Citation failure      | Answers cite removed or wrong locations           | Users can't verify answers            | No citation anchor validation or index update             |
| Broken ownership      | Issues can't be routed to a team or module        | Long fix cycles                       | Lack of feedback routing and operational mechanisms       |
| Evaluation distortion | High offline metrics but low online satisfaction  | "Tests well but unusable in practice" | No online sample backflow into the offline evaluation set |

Such degradation is not a problem of a single model or module, but the result of a system that has not entered a continuous operational state. An LLM application without a feedback pipeline is like a production system that gets one health check at launch and is never monitored, reviewed, fixed, or version-updated afterward. As the business environment evolves, the system naturally drifts further from real demand.

The online feedback loop, therefore, is not a "nice-to-have" operations feature — it is a basic capability for production-grade LLM applications. It determines whether the system can learn from usage, turn failures into improvements, and remain stable amid changes in knowledge and demand.

---

### 23.1.5 The Basic Structure of the Data Flywheel: From Feedback to Improvement

The online feedback loop can be understood as a data flywheel. "Data flywheel" here does not mean the system magically gets better unsupervised; it means the system can continuously generate new data through real usage, and after this data is filtered, annotated, attributed, and governed, it feeds back to improve the knowledge base, indexes, retrieval, generation, and evaluation systems, thereby improving the next round of user experience.

As shown in Figure 23-2, a complete data flywheel usually includes six stages: event collection, feedback routing, error attribution, fix action, regression evaluation, and post-deployment validation. Event collection is the entry point. The system needs to record the user query, conversation context, retrieval results, context-assembly content, generated answer, cited sources, user feedback, and subsequent behavior. Without sufficiently complete logs, downstream attribution is nearly impossible. For example, if only the final answer is logged but recalled results are not, one cannot tell whether the error came from retrieval or generation; if cited sources are not logged, one cannot tell whether the answer was based on the correct evidence. Feedback routing turns raw feedback into problem types. Negative feedback from users may correspond to knowledge gaps, retrieval failures, citation errors, generated hallucinations, permission issues, or product UX problems. Different problems need to enter different queues — they cannot all be dumped on the model or content team. Error attribution localizes failure phenomena to specific stages of the system pipeline. An incorrect answer may be because the document was never ingested, or it was ingested but parsing failed, or parsing was correct but chunking broke the semantics, or retrieval recalled the wrong version, or the model failed to follow the evidence. The more accurate the attribution, the more effective the fix. Fix actions include knowledge supplementation, document re-parsing, metadata correction, index rebuild, query-rewrite augmentation, rerank sample augmentation, prompt adjustments, refusal-policy updates, and evaluation-set expansion. A fix is not merely "edit the answer," but mapping the error to reusable data and system improvements. Regression evaluation verifies whether a fix is effective. Every knowledge update, index update, or strategy change may bring new issues. The system therefore needs to run regression tests using the golden set, the online failure set, and targeted challenge sets to ensure fixing the current issue does not break existing capabilities. Post-deployment validation observes the online effect after the fix — for example, whether negative feedback on similar questions drops, citation click-through improves, the human handoff rate decreases, or the number of follow-up turns decreases. These metrics then feed into the next round of feedback analysis.

![Figure 23-2: The Online Feedback Data Flywheel for LLM Applications](../../images/part7/图23_2.png)

*Figure 23-2: The Online Feedback Data Flywheel for LLM Applications*

The key to the data flywheel is not "the more automated, the better," but "every category of feedback has a stable destination." For low-risk, high-frequency issues, automated rules and model assistance can handle routing and backfill; for high-risk issues, human review, expert annotation, and pre-deployment approval are required. Mature systems usually adopt a hybrid mode of "automation + human governance," using automation for scale and humans for high-risk decisions and quality calibration.

The data flywheel must also be combined with version management. Every backfill should record its source, triggering issue, fix action, scope of impact, and evaluation results. Otherwise, the team cannot tell which batch of data, which index change, or which policy adjustment produced a given improvement. Without versions, there is no review; without review, there is no stable iteration.

---

### 23.1.6 Section Summary

This section discussed why going live with an LLM application is not the endpoint of data work, but the starting point of the online feedback loop and data flywheel. Offline evaluation helps validate the system before launch, but real user problems are only fully exposed in production. Post-launch problem distributions tend to be more colloquial, long-tailed, task-oriented, and context-dependent, so the system must perceive real demand through online feedback continuously.

Online feedback is highly valuable because it simultaneously offers authenticity, timeliness, and behavioral signals. Upvotes, downvotes, follow-ups, rewrites, citation clicks, human handoffs, and session drop-offs are all important clues to system quality. With structured processing, these signals can enter the failure bank, knowledge update queue, evaluation expansion, and broader optimization pipeline.

A system without a feedback pipeline faces knowledge expiration, phrasing mismatches, repeated errors, citation failures, and broken ownership. It may perform well in the early days but gradually exhibit "tests well, used poorly." Production-grade LLM applications must therefore organize feedback collection, routing, attribution, fixes, regression evaluation, and validation into a stable data flywheel.

The next section will further discuss the event collection and feedback routing mechanisms in the online feedback loop, focusing on how feedback sources such as logs, clicks, ratings, corrections, tickets, and human handoffs feed into a unified data processing pipeline, and how to distinguish among knowledge gaps, retrieval defects, generation defects, and policy defects.

---

## 23.2 Event Collection and Feedback Routing

### 23.2.1 Why the Feedback Loop Is First an Event Engineering Problem

The online feedback loop does not begin with a "user downvote"; it begins with whether the system can record one interaction event completely. In an LLM application, a seemingly simple Q&A often involves many critical stages behind the scenes: user input, context state, retrieval request, recalled results, reranked results, context assembly, model generation, cited sources, user behavior, and subsequent feedback. If this information is not systematically logged, the team can hardly tell where a failure actually occurred, nor can it turn online issues into reusable data assets. This section therefore starts from a basic point: the foundation of the online feedback loop is not the annotation platform or the training script — it is the event collection system. Only when the system can decompose a user interaction into a traceable, auditable, replayable chain of events can the downstream feedback routing, error attribution, knowledge updating, and regression evaluation rest on a reliable foundation.

In traditional web or app systems, event collection focuses more on product-behavior metrics such as clicks, views, dwell time, and conversion. In LLM applications, event collection must also cover model-side and data-side information. In a RAG Q&A system, for instance, simply logging "what the user asked" and "what the model answered" is not enough. The system must also log which documents were recalled at that moment, the score of each document, which fragments ended up in the context, which sources the model cited, and whether the user later followed up or handed off to a human. Without this information, failure review becomes very difficult. If a user downvotes an answer but the system only kept the final reply, the team cannot tell whether the issue was that the knowledge base had nothing relevant, retrieval failed to recall the right fragment, rerank misordered them, context assembly omitted a key condition, or the model produced a wrong conclusion on top of correct evidence. In the end, all issues get lumped into a vague "the model answers poorly," and fix actions lose focus. Worse, incomplete event collection can lead a system to repeatedly fix the wrong stage. For example, the real cause of a problem may be a stale document version, but because the system did not log the version of the recalled document, the team may mistake it for a prompt issue and keep tuning the prompt; another real cause may be permission filtering hiding the correct document, but if permission-judgment results are not logged, the team may attribute it to insufficient vector-retrieval recall. Over time, such misattribution wastes massive engineering resources and masks the real data problems.

Production-grade LLM applications, therefore, must design event collection as the first layer of infrastructure in the data loop. It must serve both online monitoring and offline review; record both user behavior and model inputs/outputs; and cover both successful and failed samples. Only in this way can the system precipitate real usage into data that can be analyzed, annotated, and backfilled.

---

### 23.2.2 Logs, Clicks, Ratings, Corrections, Tickets, and Human Handoffs

Online feedback signals usually come from multiple channels. Their trustworthiness, granularity, and intended use differ, so they cannot simply be lumped together. A mature feedback system typically collects logs, clicks, ratings, corrections, tickets, and human handoffs at the same time, and unifies them into a single event schema.

Logs are the most fundamental source of feedback. They record the complete pipeline during system execution, including the user query, session id, retrieval results, context fragments, model output, cited sources, response latency, error codes, and policy hits. Logs do not in themselves express user satisfaction, but they provide the evidence necessary for downstream error attribution. Without logs, the team can only see failed outcomes; with logs, the team can see the failure process.

Click behavior reflects how users interact with results. In a RAG system, whether the user clicks on cited sources, expands further evidence, copies the answer, or jumps to the original document can serve as indirect signals about answer trustworthiness and usability. For example, a user who frequently clicks citations but still keeps asking may indicate the citations are present but the explanation is insufficient; a user who accepts the answer without clicking any citation may indicate the answer is clear enough, or that the user is insensitive to sources. Such signals require contextual judgment to interpret.

Ratings are the most common form of explicit feedback, including upvotes, downvotes, star ratings, and "did this solve your problem?" buttons. Rating signals are simple, direct, and easy to count, but they are also susceptible to subjective bias. For instance, a user may downvote because the answer was too long, or because the system refused to answer a high-risk question — that does not necessarily mean the answer was factually wrong. Ratings are therefore well suited as an entry point for problem discovery and should not be used directly as training labels.

Corrections are more valuable than ratings because they usually contain the user's specific indication of what is wrong. A user may flag "wrong citation," "this policy has expired," "the answer ignored probationary employees," or "the value read from the chart is incorrect." Such feedback can directly enter the error-attribution pipeline and help build high-quality backfill samples. That said, corrections also need review, because user corrections are not always accurate — especially in specialized domains, where business experts or content owners should confirm them.

Tickets are an important source of complex issues. In enterprise settings, many users will not provide complete feedback in the chat interface, but instead submit issues through customer support systems, internal ticketing systems, ops platforms, or business workflows. Tickets typically contain richer context, screenshots, attachments, and the outcome of human handling — making them well suited for building high-value failure samples. However, ticket data tends to be structurally heterogeneous and requires redaction, denoising, field mapping, and task-type tagging.

Human handoff is a critical feedback signal in high-risk or complex tasks. When the system cannot answer, the user requests a human, or policy mandates a handoff, those handoff events should all be logged. Human handoff not only indicates the system's current capability boundary but also provides a trace of how experts handle the case. If the system also records how the human eventually solved the problem, it can learn which knowledge is missing, which follow-up conditions are necessary, and which answering strategies need adjustment.

*Table 23-4: Online Feedback Sources and Their Data Value*

| Feedback Source | Typical Content                                                       | Strengths                                       | Limitations                              | Suitable Downstream Actions                       |
| --------------- | --------------------------------------------------------------------- | ----------------------------------------------- | ---------------------------------------- | ------------------------------------------------- |
| System logs     | Query, recalled fragments, context, answer, citations, latency        | Complete pipeline, supports postmortems         | Does not directly express satisfaction   | Error attribution, performance diagnostics, regression evaluation |
| Click behavior  | Citation clicks, evidence expansion, copy, jumping to source          | Reflects real usage behavior                    | Multiple plausible semantic explanations | Evidence usability analysis, citation optimization |
| Ratings         | Upvote, downvote, star rating, "resolved?"                            | Simple, direct, easy to aggregate               | Subjective noise                         | Failure-sample triage, satisfaction monitoring    |
| Corrections     | Flagged errors, supplied correct answers, expired-content reports     | High information density, high value            | Requires review                          | Knowledge updates, sample backfill                |
| Tickets         | Issue description, screenshots, handling process, final conclusion    | Rich context, suitable for complex cases        | Heterogeneous structure, costly to process | Specialized dataset construction, SOP improvement |
| Human handoff   | Reason for handoff, expert trace, final reply                         | Reveals system capability boundaries            | Depends on standardized human processes   | High-risk policy optimization, expert sample bank |

These feedback sources together form the raw data layer of the online feedback loop. In practice, teams should not rely on a single signal but build multi-signal fusion. A downvote sample accompanied by citation clicks, repeated follow-ups, and human handoff deserves more priority than a generic downvote; a session with no explicit downvote but three consecutive rewrites should also be flagged as a potential failure sample.

To enable such fusion, the system needs to unify events across three levels: session, question, and answer. The session level records the user's full trajectory across an interaction; the question level records each user input along with its retrieval and generation process; the answer level records the model output, cited evidence, user feedback, and subsequent behavior. Only with such hierarchical event structure does feedback routing move beyond coarse statistics into an actionable engineering loop.

---

### 23.2.3 The Difference Between Explicit and Implicit Feedback

Online feedback can be divided into explicit feedback and implicit feedback. Explicit feedback is evaluation or correction proactively given by the user; implicit feedback is the satisfaction, confusion, or failure signal indirectly revealed by user behavior. The two play different roles in the data loop and must be used distinctly.

The strength of explicit feedback is semantic clarity. When users downvote or submit a correction, they usually mean the answer did not meet expectations. For the data-operations team, explicit negative feedback is the most direct entry point for failure samples. Explicit positive feedback can be used to distill high-quality Q&A pairs and analyze which answer formats, evidence organization, and explanatory styles are best received. However, explicit feedback suffers from coverage gaps. Most users do not actively provide feedback, especially when the answer is mediocre but not severely wrong — they may simply leave or rephrase. Explicit feedback can also carry emotional or misclick noise. For instance, a user may downvote because the system did not produce the conclusion they wanted, even though the system's refusal was the correct safety policy; a user may downvote because the answer was too conservative, even though from a compliance standpoint a conservative answer is preferable.

The strength of implicit feedback is broad coverage. Nearly every user behavior can serve as an implicit signal: dwell time, number of follow-ups, query rewrites, citation clicks, copying the answer, switching pages, human handoff, and session drop-off. These signals do not require active user evaluation and are therefore closer to real usage. But their interpretation is harder. Continuing to ask may mean the answer was incomplete or that the user wants to dig deeper; clicking citations may mean the answer is trustworthy or that it is being doubted; leaving the session may mean the question is resolved or that the system was unhelpful. Implicit feedback usually cannot serve as a standalone label and must be analyzed alongside explicit feedback, log data, and business context.

*Table 23-5: Comparison of Explicit and Implicit Feedback*

| Dimension      | Explicit Feedback                                              | Implicit Feedback                                                       |
| -------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Typical forms  | Upvote, downvote, rating, correction, comment                  | Follow-up, rewrite, citation click, copy, human handoff, drop-off       |
| Strengths      | Direct semantics, easy to interpret                            | Broad coverage, close to real behavior                                  |
| Limitations    | Low coverage, subjective noise                                 | Ambiguous meaning, requires contextual interpretation                   |
| Data value     | Good entry point for failure samples and human review queues   | Good for behavior analysis, weak labels, and trend monitoring           |
| Usage approach | Should pass through review before entering annotation/backfill | Must be combined with logs and other signals for judgment               |
| Risks          | User misjudgment, emotional or malicious feedback              | Misinterpreting behavior; treating normal exploration as failure        |

In practice, explicit feedback can be treated as a "strong signal" and implicit feedback as a "weak signal." Strong signals are suitable as direct entries to human review and the failure bank; weak signals are suitable for surfacing anomalous trends and shortlisting candidate problems. For example, a rising downvote rate on a category of questions is a strong failure signal; rising average follow-up turns, falling citation click-through, and rising handoff rate — even without many downvotes — also indicate possible quality regression.

To use implicit feedback more reliably, the system can design composite rules. For instance, a session can be flagged as "likely unresolved" if all of the following hold: the user rephrases a similar question within 30 seconds after receiving an answer; two consecutive turns recall different documents but no answer is accepted; the session ultimately hands off to a human or files a ticket. Such composite signals are more reliable than any single behavior and are better suited as candidates for routing and sampling.

---

### 23.2.4 Event Collection Schema: From Raw Logs to Usable Samples

To turn online feedback into a data asset, a unified event schema is required. The role of the event schema is to organize raw interaction logs into a data structure that can be searched, analyzed, annotated, and replayed. Without a unified schema, feedback data scatters across logging systems, instrumentation platforms, support systems, databases, and model services, making a closed loop impossible.

A complete event schema should include at least six categories of information: user and session info, query info, retrieval info, generation info, feedback info, and governance info. User and session info identifies the context of an interaction. It need not include the user's real identity, but should record anonymized user ID, session ID, tenant, permission scope, language, device, and timestamp. For enterprise systems, it should also record the user's organization, role, or permission group, in order to detect over-recall across permissions. Query info describes the user input itself, including the raw query, normalized query, query-rewrite result, identified intent, entities, constraints, and contextual references. For instance, when a user asks "Can this still be claimed?", the system should record what "this" refers to in the previous turn, "claim" maps to reimbursement intent, and which conditions are missing (expense type, location, etc.). Retrieval info is one of the most important parts of event collection in a RAG system. The system must record the index version queried, recalled document IDs, chunk IDs, scores, rerank scores, the fragments that finally entered the context, fragments that were filtered out, and the reasons for filtering. Without this, retrieval failure and generation failure are hard to tell apart. Generation info includes model version, prompt version, context length, final answer, cited sources, refusal policy, generation latency, and safety-policy hits. For multimodal systems, it should also record image regions, bboxes, OCR text, table structures, and visual evidence IDs. Feedback info includes both explicit and implicit feedback: explicit signals such as upvotes, downvotes, ratings, user corrections, and free-text comments; implicit signals such as follow-ups, rewrites, citation clicks, copy-answer events, human handoffs, ticket submissions, and session drop-offs. Governance info covers PII redaction status, log retention period, whether the sample is allowed to be used for training, and whether it is in the human-review queue.

*Table 23-6: Core Fields of the Online Feedback Event Schema*

| Field Category   | Core Fields                                                                       | Purpose                                                                |
| ---------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Session info     | session_id, turn_id, timestamp, tenant, user_role                                 | Reconstruct interaction context and permission scope                   |
| Query info       | raw_query, normalized_query, intent, entities, missing_slots                      | Analyze user intent, entities, and missing conditions                  |
| Retrieval info   | index_version, retrieved_docs, chunk_ids, scores, rerank_scores                   | Judge correctness of recall and ranking                                |
| Context info     | selected_context, citation_anchors, context_length                                | Reproduce the evidence actually seen by the model                      |
| Generation info  | model_version, prompt_version, answer, refusal_flag, latency                      | Judge generation policy and model behavior                             |
| Feedback info    | rating, correction_text, clicks, follow_up, handoff_flag                          | Identify explicit and implicit feedback                                |
| Governance info  | pii_status, training_allowed, review_status, retention_policy                     | Control compliance, review, and data-usage boundaries                  |

At the implementation level, the event schema should serve not only log storage but also downstream sample construction. That is, when designing fields, the team should anticipate how these data will later flow into evaluation sets, the failure bank, the knowledge update queue, and model training. For example, `index_version` and `prompt_version` look like engineering fields, but they determine whether version-level attribution is possible later; `citation_anchors` looks like a display field, but it determines whether the answer can be verified against the correct evidence; `training_allowed` looks like a compliance field, but it determines whether the data can flow into training or fine-tuning.

![Figure 23-3: Online Feedback Event Collection and Routing Pipeline](../../images/part7/图23_3.png)

*Figure 23-3: Online Feedback Event Collection and Routing Pipeline*

---

### 23.2.5 Feedback Routing: Knowledge Gap, Retrieval Defect, Generation Defect, and Policy Defect

After collection, the most critical work is feedback routing — mapping user feedback to different categories of system problems and pushing them into corresponding fix queues. Without routing, all feedback piles up as vague "user dissatisfaction" and yields no real improvement.

In RAG and LLM applications, common problems can be divided into four categories: knowledge gap, retrieval defect, generation defect, and policy defect. A knowledge gap means the knowledge base does not contain enough information to answer the user's question. This can happen because relevant documents have not been ingested, documents are outdated, new business areas have not been supplemented, the knowledge granularity is too coarse, or the information exists only in human experience and has not been documented. The typical symptom is that the system cannot retrieve relevant evidence, or the answer can only speak in generalities. Fix actions usually include adding documents, updating FAQs, adding structured fields, introducing expert knowledge, or establishing a knowledge-update workflow. A retrieval defect means the correct answer exists in the knowledge base, but the system did not recall it or did not rank it at the top. This is very common in RAG systems. It can stem from mismatch between query phrasing and document terminology, chunking that breaks semantics, embeddings that fail on domain terms, missing keyword indexes, wrong metadata filters, or failed rerank ranking. Fix actions include adding synonyms, optimizing query rewrite, adjusting the chunking strategy, supplementing metadata, improving hybrid retrieval, or adding rerank training samples. A generation defect means the correct evidence was retrieved, but the model went wrong while generating the answer — ignoring evidence, misreading it, overgeneralizing, omitting conditions, citing incompletely, or producing the wrong style. Fix actions usually include adjusting prompts, adding format constraints, introducing answer templates, strengthening citation requirements, adding refusal rules, or adding the failure samples to a generation evaluation set. A policy defect means the system has design issues in high-risk, permission, refusal, follow-up, or handoff policies. For instance, the user's question lacks key conditions but the system does not follow up; the user's request involves high-risk judgment but the system answers directly; a user lacks permission to view certain documents but the system still recalls them; certain questions should hand off to a human but the system keeps generating. Such issues are usually not model-capability shortcomings but undefined product or safety policies.

The key to feedback routing is establishing executable judgment rules. A team cannot simply tag feedback as "error"; it must further ask: does the correct knowledge exist? Was it recalled? Did it enter the context? Did permission or safety policies fire? Together, these questions form the minimal diagnostic chain for feedback routing. For one downvote sample, for example, analysis can proceed as follows: if no correct content exists in the knowledge base → knowledge gap; if it exists but was not recalled → retrieval defect; if recalled but did not enter the final context → ranking or context-assembly issue; if context is correct but the answer is wrong → generation defect; if the question itself lacks conditions but the system did not follow up → policy defect; if the answer is correct yet the user remains unhappy → likely a product-experience issue.

---

### 23.2.6 Feedback Queues and Operational Ownership

Feedback routing is not only a technical classification problem; it is also an ownership problem. Each class of problem should map to an explicit processing queue, owning team, SLA, and acceptance criteria. Otherwise, even if the system can identify the problem, it cannot drive fixes.

Knowledge gaps are usually owned by content owners, business experts, or knowledge-base operators, who decide whether to add documents, update clauses, add FAQs, or restructure knowledge. Retrieval defects are usually owned by the data engineering and retrieval engineering teams, focusing on chunks, indexes, metadata, query rewrite, and rerank. Generation defects are usually owned by the model-application team and involve prompts, context formatting, citation constraints, and generation policies. Policy defects require joint decision-making among product, business, compliance, and security teams, because they touch on whether the system should answer, how it should answer, and when it should hand off. Product-experience issues are usually owned by the product and frontend teams.

To improve processing efficiency, feedback queues should have priorities. Priority can be determined jointly by impact scope, risk level, frequency, and fix cost. High-frequency low-risk problems can be batched; low-frequency high-risk problems require immediate human review; low-frequency low-risk problems can enter a periodic optimization pool; high-frequency high-risk problems should trigger a dedicated postmortem and version-level fix. In a mature online feedback system, every feedback sample should have an explicit status, such as: pending routing, pending review, pending fix, fixed, pending regression, deployed, closed. Only this way can the team track whether an online failure sample really completed the loop, rather than sitting in the issue list.

---

### 23.2.7 Section Summary

This section discussed event collection and feedback routing in the online feedback loop. For an LLM application, the feedback loop is first an event-engineering problem. The system must record user input, retrieval results, context assembly, model output, cited sources, user behavior, and subsequent feedback in full, in order to support downstream error attribution and data backfill.

Online feedback sources include logs, clicks, ratings, corrections, tickets, and human handoffs. Different signals have different data value and noise characteristics, so they must be organized through a unified event schema. Explicit feedback is semantically direct and well suited as a failure-sample entry point; implicit feedback has broader coverage and is well suited for trend monitoring and candidate sample selection.

After collection, the system must route problems into knowledge gaps, retrieval defects, generation defects, policy defects, and product-experience issues. Different problems map to different owning teams and fix actions. Only when each class of feedback enters an explicit queue and passes through review, fix, regression, and deployment validation does online feedback truly become a data flywheel driving continuous improvement.

The next section will further discuss knowledge updates, rollback, and version governance, focusing on how new-knowledge ingestion, old-knowledge expiration, conflicting-content governance, canary releases, and rapid rollback support long-term stable operation of production LLM applications.

---

## 23.3 Knowledge Updates, Rollback, and Version Governance

### 23.3.1 Why Knowledge Updates Must Be Engineered as a Managed Process

The ultimate goal of the online feedback loop is not to collect user problems or compute satisfaction; it is to enable the system to continuously fix itself based on real usage. The most central category of fix action is the knowledge update. For RAG systems, enterprise knowledge assistants, customer-service bots, compliance Q&A, and multimodal document retrieval systems, the knowledge base is not a static asset — it is a dynamic asset that changes with the business, policies, products, processes, and external environment.

In practice, the complexity of knowledge updates comes from three sources. First, knowledge itself has a shelf life. Corporate policies are adjusted, product features iterate, API fields are deprecated, pricing rules change, organizations restructure, and laws and regulations get updated. For such content, old knowledge does not simply become "less valuable" — it may become wrong information. If a RAG system keeps recalling stale knowledge, it will produce answers that look well-grounded but are actually outdated. Such errors are more dangerous than having no answer, because users tend to trust answers that carry citations. Second, knowledge can conflict. A single system may contain formal policies, historical policies, FAQs, meeting minutes, customer-service scripts, product manuals, and ad-hoc notices. These sources differ in timestamp, authority, and applicable scope. Without knowledge-priority and conflict-governance rules, the system may simultaneously recall contradictory evidence within a single answer. For example, a formal policy may require three-level approval but an old FAQ still says two-level approval; product docs may say a feature is enterprise-only but customer-service scripts simplify it as "available to all users." Such conflicts cannot be resolved by the model alone; they must be handled in advance by knowledge governance. Third, knowledge updates affect system behavior. Adding documents, modifying metadata, rebuilding indexes, or adjusting chunking strategies can change retrieval results. One update may fix one class of problems while breaking another that was working. Knowledge updates must therefore be versioned, tested, canaried, and monitored — just like code releases — rather than directly overwriting the production knowledge base.

Production-grade RAG systems must upgrade knowledge updates from a "document maintenance action" into an "engineering release process." Every knowledge change should have an explicit source, change description, owner, scope of impact, evaluation result, release time, and rollback plan. Only then can the system remain stable amid continuous updates rather than drifting out of control under frequent changes.

---

### 23.3.2 New-Knowledge Ingestion: From Document Onboarding to Usable Knowledge Units

New-knowledge ingestion is the most common scenario in knowledge updates. It may come from new policy releases, new product documentation, new FAQ additions, new earnings reports, new contract templates, or knowledge gaps revealed by online feedback. Compared with initial knowledge base construction, post-launch ingestion places greater emphasis on incrementality, controllability, and verifiability.

A complete new-knowledge ingestion process usually has five steps: source verification, content parsing, structuring, index update, and post-deployment validation.

Source verification is the first step. The system must record where the knowledge comes from, whether it is licensed for use, whether it is the official version, and whether business or legal teams have approved it. For internal enterprise systems, the authority of document sources matters enormously. Formal policies, approved product manuals, and FAQs confirmed by business owners should have higher priority; meeting minutes, casual chat records, and personal notes should be weighted lower or treated only as auxiliary references. Content parsing converts the document into a machine-processable structure. For PDFs, Word docs, web pages, tables, images, and scans, parsing quality directly determines downstream retrieval effectiveness. Post-launch ingestion must not skip parsing-quality checks, especially for documents involving tables, charts, headers/footers, footnotes, version numbers, and section hierarchies. If parsing loses applicability conditions, units of measure, table structure, or document version, even successful indexing may yield wrong answers. Structuring converts parsed content into knowledge units and adds metadata such as source document, section path, publish time, effective time, expiration time, applicable subjects, permission scope, document version, knowledge type, and authority level. For production-grade RAG systems, metadata is not auxiliary — it is the basis for retrieval, filtering, ranking, and conflict governance. Index update writes knowledge units into the retrieval system. Depending on the architecture, this may involve vector indexes, keyword indexes, structured indexes, graph databases, table indexes, and multimodal indexes. Incremental updates must pay special attention to index consistency: has the new knowledge been written to all index types, do old indexes need deletion or down-weighting, are parent/child indexes synchronized, and do multimodal chart regions stay aligned with their textual descriptions. Post-deployment validation is the final step before production. The system should run a regression test using a relevant question set to verify that the new knowledge is recalled, cited, and generated correctly. If the new knowledge was added to fix an online failure case, the original failing query should be replayed to confirm the issue is resolved.

*Table 23-7: Key Checks in the New-Knowledge Ingestion Flow*

| Stage              | Key Question                                | Focus                                                         | Common Risks                                          |
| ------------------ | ------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------- |
| Source verification | Is the document trustworthy and usable?     | Source, permissions, owner, approval status                   | Informal documents entering production knowledge base |
| Content parsing    | Is the document read correctly?             | Sections, tables, charts, footnotes, version numbers          | Garbled tables, lost conditions, OCR errors           |
| Structuring        | Are usable knowledge units formed?          | Chunks, metadata, scope, citation anchors                     | Missing version, effective time, or permission info   |
| Index update       | Are correct indexes written?                | Vector, keyword, structured fields, multimodal indexes        | Index inconsistency, old content not retired          |
| Post-deploy validation | Is the issue actually fixed?            | Failure-query replay, citation check, regression evaluation   | Fixes one issue but introduces new errors             |

New-knowledge ingestion should further distinguish "additive updates" from "replacement updates." Additive updates add new content to existing knowledge — new FAQs, new cases, new chart explanations; replacement updates mean old knowledge no longer applies — policy version upgrades, deprecated API fields, pricing adjustments. The risks differ: additive updates mainly concern recall coverage and ranking; replacement updates must focus on retiring old knowledge and governing conflicts.

---

### 23.3.3 Old-Knowledge Expiration and Conflicting-Content Governance

The most easily overlooked issue in knowledge updates is not the failure to add new knowledge — it is the failure to retire old knowledge. In many systems, teams keep adding new documents but rarely actively clean up old ones. The result is a knowledge base that grows ever larger, retrieval that gets noisier, and a model that may see conflicting old-versus-new evidence at generation time.

Old-knowledge expiration usually takes three forms: time expiration, version expiration, and scope expiration. Time expiration means a piece of knowledge is valid only within a specific window — temporary notices, phased policies, promotional rules, quarterly reports, project plans, all of which have clear time boundaries. Time-expired knowledge does not necessarily need to be deleted but must be correctly down-weighted or filtered during retrieval and generation. Version expiration means a new version replaces an old one. After product manual v3.0 ships, some feature descriptions in v2.0 may no longer apply; after a 2025 policy release, certain processes from the 2023 policy may be obsolete. Without document-version and effective-status fields, old versions are hard to keep out of recall. Scope expiration means the knowledge is still valid, but does not apply to the current user or scenario. For instance, a reimbursement policy may apply only to full-time employees and not to interns; a feature may apply only to the enterprise edition; a procedure may apply only to domestic business, not overseas branches. Scope expiration is not about staleness but about applicability conditions that have not been properly recognized.

Conflicting-content governance must operate at the knowledge level, the index level, and the generation level. At the knowledge level, each piece of knowledge should have an authority level and an applicable scope. Formal policies outrank FAQs; the newest version outranks older ones; structured fields outrank informal descriptions; content confirmed by business owners outranks user comments or meeting minutes. At the index level, retrieval should recognize time, version, permissions, and applicable subjects. Expired content can be deleted, archived, down-weighted, or made visible only in historical queries. Content that should be kept but not recalled by default should be controlled via metadata filters. At the generation level, the model should be asked to recognize evidence conflicts. When multiple versions or contradictory fragments appear in the context, the model should not naively concatenate answers but pick trustworthy evidence based on version, date, and authority, and where necessary surface a "the knowledge base currently has conflicts; human confirmation required." That said, this capability cannot rely solely on the model — upstream data governance remains key.

*Table 23-8: Strategies for Old-Knowledge Expiration and Conflict Governance*

| Problem Type      | Typical Symptom                                       | Data-Governance Strategy                                  | System-Side Handling                                   |
| ----------------- | ----------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------ |
| Time expiration   | Temporary notices, quarterly rules still recalled      | Add effective-time and expiration-time fields              | Filter or down-weight by query time                    |
| Version expiration | Old and new product manuals hit simultaneously        | Maintain version numbers and current-version flags         | Default to recalling latest; archive old versions      |
| Scope expiration  | Policies cited for users they don't apply to          | Tag applicable subjects, regions, roles, permissions       | Metadata filters and permission filters                |
| Source conflict   | FAQ contradicts formal policy                          | Establish authority levels and source priority             | Boost authoritative content in rerank                  |
| Content conflict  | Same metric, process, or rule has contradictions       | Set up conflict detection and human review queue           | Conflict warning or refusal                            |
| Citation failure  | Answer cites removed or relocated documents            | Maintain citation anchors and document mappings            | Citation validation and index rebuild                  |

The difficulty of conflict governance is that it spans teams. Content teams update documents, business teams confirm rules, platform teams own indexing, algorithm teams own retrieval/ranking, and product teams own display and warnings. Without a clear process, conflicting content easily lingers. The knowledge base should therefore set up periodic audits — focused reviews of high-recall documents, expired documents, low-trust sources, and content with frequent user feedback.

---

### 23.3.4 Hot Updates, Scheduled Updates, and Audited Updates

Knowledge updates do not all use the same release method. Depending on risk level, time sensitivity, and business impact, updates fall into three modes: hot updates, scheduled updates, and audited updates.

Hot updates suit low-risk, time-sensitive content — FAQ additions, typo fixes, minor wording updates, low-risk product hints. Their advantage is fast response, suitable for quickly fixing online issues; the risk is that, without automated validation, wrong content can enter production just as quickly. Hot updates must therefore include at least basic checks: document-format validation, parsing-success validation, index-write validation, and small-scale query replay.

Scheduled updates suit periodically changing knowledge — help center synced daily, product docs updated weekly, monthly operations reports, quarterly earnings, or policy bundles. Their advantage is steady cadence, suitable for batch validation and resource planning. They are typically executed by schedulers and produce an update report afterward: new document count, deletions, parsing failures, indexed updates, and regression results.

Audited updates suit high-risk content — compliance policies, financial rules, medical guidelines, legal clauses, permission policies, contract templates, and critical business processes. These cannot be auto-deployed; they must go through owner sign-off, expert review, regression evaluation, approval trail, and canary release. For such knowledge, update speed is not the only goal — correctness, traceability, and rollback are more important.

![Figure 23-4: Knowledge Update, Canary Release, and Rollback Governance Flow](../../images/part7/图23_4.png)

*Figure 23-4: Knowledge Update, Canary Release, and Rollback Governance Flow*

In practice, the three update modes usually coexist. The system can allow low-risk content to go through hot updates, medium-risk content through scheduled updates, and high-risk content through audited updates. This requires the knowledge update platform to support risk grading: each update request is automatically or semi-automatically graded based on knowledge type, source, scope of impact, and user population, and routed to the corresponding flow. A wording tweak in a customer-service FAQ can go through hot updates; a batch of product-doc version syncs can go through scheduled updates; anything touching contract liability, financial approvals, or medical advice must go through audited updates. Such tiering balances efficiency and safety — preventing every update from being slowed by heavy approvals, while also preventing high-risk content from sneaking into production unchecked.

---

### 23.3.5 Version Freezes, Canary Releases, and Rapid Rollback

The final risk control on knowledge updates is version governance. Without it, three key questions cannot be answered: what version is currently in production? Which update introduced this wrong answer? If an update is bad, can we quickly revert to the previous stable version?

Concretely, version governance comprises version freezes, canary releases, and rapid rollback. A version freeze fixes the knowledge base, indexes, prompts, model configuration, and evaluation set state at a point in time to form a reproducible release version. For a production RAG system, a "version" should record not just a document folder but the full dependency: source document versions, parser version, chunking strategy, embedding model, index-build parameters, rerank model, prompt template, and permission rules. Only with this can the team reproduce issues for postmortem. Canary release is a key way to reduce knowledge-update risk. Updates should not be released to all users at once. Important updates can roll out first to internal testers, a small fraction of real users, or specific tenants, while monitoring retrieval hit rate, answer accuracy, citation click-through, negative-feedback rate, and human handoff rate. If canary metrics are stable, expand gradually. For multi-tenant systems, canaries can also be staged by department, region, business line, or user role. Rapid rollback restores the system to the last stable version when an update misfires. Rollback capability requires retaining the old index, the old knowledge bundle, and old configurations. If every update overwrites the production index without snapshots, rollback becomes very hard. A more mature approach is "blue/green indexes" or "dual-index release": build the new index, switch traffic only after validation; if something goes wrong, switch back immediately.

Rollback applies not only to content but also to indexing strategies and generation strategies. An update may not touch any document but adjust chunk granularity or rerank weights, degrading online recall; a prompt update may improve completeness but hurt citation fidelity. All such changes should be brought under version management and rollback. A reliable version governance system should record at least: version number, release time, change content, change source, owner, scope of impact, evaluation results, canary scope, monitored metrics, rollback point, and approval record. These are not only for incident handling but also for long-term review and collaboration.

---

### 23.3.6 Section Summary

This section discussed knowledge updates, rollback, and version governance in the online feedback loop. For production LLM applications, the knowledge base is not a one-off static asset but a dynamic system requiring continuous updating, validation, and governance. New-knowledge ingestion must go through source verification, content parsing, structuring, index update, and post-deployment validation; old-knowledge expiration and conflict governance rely on metadata for time, version, scope, authority, and permissions. Update methods should be tiered by risk: hot updates for low risk, scheduled updates for medium risk, audited updates for high risk. Meanwhile, the system must establish version freezes, canary releases, and rapid rollback to ensure every change is traceable, verifiable, and reversible. In the online feedback loop, knowledge updates are not isolated. They connect to user failure feedback, error attribution, fix tasks, regression evaluation, and online monitoring. Only when knowledge updates are folded into an engineered governance process can a RAG system remain reliable, controllable, and sustainably evolvable in a changing business environment.

The next section will further discuss metric dashboards and operational cadence, focusing on how indicators such as online success rate, human handoff rate, correction rate, and knowledge-hit rate feed into weekly operations meetings, focused postmortems, and major-version release cadence, thereby grounding the data loop in the team's daily work.

---

## 23.4 Metric Dashboards and Operational Cadence

### 23.4.1 Why the Online Feedback Loop Needs Dashboards

If the online feedback loop only fixes individual failure samples, it easily falls into "firefighting operations." Users downvote a question, the team fixes it; the business flags a knowledge gap, the team adds a document; an answer cites the wrong source, the team manually tweaks an index entry. In early stages this can be quick and responsive, but as user base, knowledge base, and business scenarios grow, single-point fixes become ineffective. The team must upgrade from "handling problems" to "operating a system," and dashboards are the foundational tool that enables this shift.

A dashboard does more than show system status; it turns online feedback into a management language that is observable, comparable, attributable, and actionable. For a production RAG or LLM application, "is the system healthy?" cannot be judged only by API availability or latency, nor by user traffic growth. What matters more is: does the system actually answer user questions, is it based on correct knowledge, does it reduce human handling cost, does it stay stable after knowledge updates, and does it apply the right policy in high-risk scenarios.

The dashboard in the online feedback loop should therefore cover four kinds of metrics: quality metrics, behavior metrics, operations metrics, and risk metrics. Quality metrics focus on whether answers are correct, citations reliable, retrieval on-target; behavior metrics focus on whether users accept, follow up, or hand off; operations metrics focus on issue-handling efficiency, knowledge-update cadence, and fix-loop progress; risk metrics focus on incorrect answers, high-risk wrong answers, permission breaches, and stale-knowledge recall. Only with all four can the team tell whether the system is actually getting better or just looking better on a single metric. For example, if answer accuracy rises while human handoff rate also rises, it may mean the system handles low-risk questions better but cannot handle complex ones; if recall is high but citation click-through is low, answers may look sourced but users do not trust the citation display; if user satisfaction rises but knowledge updates lag, short-term experience may be good but there is latent stale-knowledge risk. The dashboard makes these tensions visible, preventing the team from being misled by a single metric.

---

### 23.4.2 Online Success Rate, Human Handoff Rate, Correction Rate, and Knowledge-Hit Rate

As shown in Table 23-9, the most commonly used core metrics in the online feedback loop include the online success rate, human handoff rate, correction rate, and knowledge-hit rate. They measure system performance from four angles: user outcomes, human cost, error exposure, and knowledge coverage.

The online success rate measures whether user problems are effectively solved. It can be estimated jointly from explicit and implicit feedback — clicking "resolved," upvoting, copying the answer, not following up, or ending the session after the answer can all serve as success signals. But success is not a simple binary: different scenarios define it differently. In customer service, success may mean no human handoff; in an enterprise knowledge base, it may mean clicking on the correct citation and proceeding to the next action; in compliance, it may mean not giving a direct conclusion but properly flagging risk and routing to human review.

The human handoff rate measures the fraction of tasks the system cannot complete on its own. It reflects both capability boundaries and policy design. In high-risk domains, some handoff is necessary; but if even ordinary questions are handed off frequently, the knowledge base, retrieval, or generation strategy is lacking. The handoff rate should be analyzed by question type, not chased to be as low as possible. In legal, medical, or financial-approval scenarios, a low handoff rate combined with a high error rate is a danger signal.

The correction rate measures how often users or human reviewers find errors — including user-initiated corrections, expert-review findings, and error tags from ticket flows. A rising correction rate does not necessarily mean the system has gotten worse; it may mean the feedback channel is easier to use, users are more willing to report, or the system covers more complex scenarios. It must be analyzed together with issue volume, difficulty, release version, and knowledge-update batches.

The knowledge-hit rate measures whether user questions can be matched to effective content in the knowledge base. It tracks knowledge-level coverage, not just retrieval recall. If the knowledge base has no answer to a question, no retrieval algorithm can produce a reliable response. A low hit rate usually signals knowledge gaps, un-ingested documents, expired documents, missing metadata, or questions outside the system's boundary.

*Table 23-9: Core Metrics of the Online Feedback Loop*

| Metric              | Meaning                                                                 | Typical Computation                                                                       | Problems It Helps Surface                                          |
| ------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Online success rate | Whether user problems are effectively solved                            | Resolved sessions / total sessions, or composite from explicit & implicit feedback        | Unusable answers, poor experience, incomplete task completion      |
| Human handoff rate  | Whether the system needs human intervention                             | Handoff sessions / total sessions                                                         | Capability boundary, overly strict policies, retrieval failures    |
| Correction rate     | Fraction with user or reviewer-flagged errors                           | Correction samples / answered samples                                                     | Factual errors, citation errors, expired knowledge                 |
| Knowledge-hit rate  | Whether the knowledge base has effective evidence for the question      | Questions with valid evidence / total questions                                           | Knowledge gaps, missing ingestion, insufficient metadata           |
| Citation accuracy   | Whether answer citations support the conclusion                         | Correctly cited answers / answers with citations                                          | Citation misalignment, weak evidence, context-assembly errors      |
| Follow-up rate      | Whether users need to keep asking                                       | Sessions with follow-ups / total sessions                                                 | Incomplete answers, unclear conditions, unclear phrasing           |

These metrics complement one another. Online success rate is the outcome metric, human handoff rate is the cost/boundary metric, correction rate is the error-exposure metric, and knowledge-hit rate is the asset metric. Operations cannot stare at any single one; they must observe combinations.

For instance, if both online success rate and knowledge-hit rate drop, prioritize knowledge gaps or expirations; if hit rate is normal but answer accuracy drops, suspect generation or context assembly; if accuracy is normal but follow-up rate rises, the answer may be unclear or lack action steps; if handoff rate spikes, decide whether business complexity has risen or the policy is misfiring.

---

### 23.4.3 Weekly Operations, Focused Postmortems, and Major-Release Cadence

Dashboards only deliver value when they enter a stable operational cadence. Otherwise, they only show data passively without driving improvements. For production LLM applications, we recommend three tiers of cadence: weekly operations meetings, focused postmortems, and major-release reviews.

Weekly operations meetings focus on day-to-day operating state — core metric trends, online failure samples, knowledge-update progress, and the pending-issue queue. The goal is not to solve every technical detail but to make sure the team knows the system's top issues, what is in progress, and what risks need escalation. The meeting should include product, data engineering, algorithm, platform, and business content owners, so that feedback does not stay siloed.

Focused postmortems handle clusters of issues. For example, when finance-report Q&A errors spike for a week, or when a product release triggers a flood of feedback about inconsistent feature descriptions, a focused postmortem is needed. It should walk the specific pipeline: how the user asked, what was recalled, how the answer was generated, whether the citation was correct, why the user was dissatisfied, and whether the root cause is knowledge gap, retrieval defect, generation defect, or policy defect. The conclusion must translate into concrete fix actions, not vague "to-be-optimized."

Major-release reviews cover wide-scope knowledge updates, indexing-strategy changes, model upgrades, or prompt-system overhauls. They cannot rely on dev self-testing alone; they require regression evaluation, canary release, online monitoring, and a rollback plan. Especially for updates touching high-frequency knowledge, critical processes, or high-risk answering policies, the scope of impact and acceptance criteria must be explicit before release.

*Table 23-10: Operational Cadence of the Online Feedback Loop*

| Mechanism            | Frequency       | Main Inputs                                                          | Main Outputs                                          | Participants                                                          |
| -------------------- | --------------- | -------------------------------------------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------- |
| Weekly operations    | Weekly          | Dashboards, top failure categories, knowledge-update queue           | Prioritization, ownership assignment, risk escalation | Product, data engineering, algorithm, business content, platform       |
| Focused postmortem   | On trigger      | A class of high-frequency or high-risk failure samples               | Root-cause analysis, fix plan, regression samples     | Owners, business experts, algorithm and platform                       |
| Major-release review | Before release  | Change description, regression results, canary results, rollback plan | Release decision, canary scope, rollback point        | Project lead, platform, business, compliance, quality lead             |
| Monthly quality review | Monthly       | Trend metrics, cost metrics, feedback summary                        | Periodic quality report, resource planning            | PM, product, data operations, technical lead                          |

The crucial element of cadence is closed-loop record-keeping. Each meeting should produce explicit issues, owners, due dates, acceptance metrics, and follow-up channels. For example, "improve retrieval" is not a valid task; "for reimbursement-policy questions, add 30 synonym phrasings, raising Recall@5 on the failure subset from 62% to above 85%" is. Only when tasks are metricized, sample-anchored, and versioned can meetings push real improvement.

---

### 23.4.4 Ownership Boundaries and Collaboration in the Feedback Loop

The online feedback loop usually involves multiple teams, so ownership boundaries must be clear. A single user downvote may touch knowledge-base content, retrieval strategy, model generation, product UX, and permission control. Without clear division of labor, the issue bounces among teams and ultimately no one owns it.

A workable approach is to bind feedback to root-cause categories and corresponding teams. Knowledge gaps to content/business owners; retrieval defects to data engineering and retrieval teams; generation defects to the model-application team; policy defects jointly to product, business, and compliance; platform stability to the infrastructure team; product-experience issues to product and frontend. Issues that cannot be classified in one shot go into a joint-postmortem queue led by data operations.

The loop also needs a status-flow mechanism. From entry to closure, a feedback sample typically traverses: pending routing, pending confirmation, pending fix, pending regression, pending deploy, closed. Each status needs explicit entry and exit conditions. "Pending fix" requires a root-cause tag and fix plan; "pending regression" requires bound regression samples; "closed" requires verification or a metric change as evidence.

Collaboration also needs prioritization rules. Not every issue can be processed at once; teams must order them by impact, risk, frequency, and fix cost. High-risk issues take priority even when low-frequency; high-frequency low-risk issues fit batch optimization; low-frequency low-risk issues can accumulate in a long-term pool; high-frequency high-risk issues should trigger incident-level response.

Finally, the feedback loop needs knowledge precipitation. Every postmortem should enter a case library, with background, failure symptom, root cause, fix action, deployment outcome, and lessons learned. Over time, these cases become the team's operations knowledge base — helping new members understand common system issues and informing evaluation-set design and pre-deployment checklists.

---

### 23.4.5 Section Summary

This section discussed dashboards and operational cadence in the online feedback loop. For production-grade LLM applications, dashboards are not just data displays — they are the infrastructure that turns online feedback into operational decisions. Online success rate, human handoff rate, correction rate, knowledge-hit rate, citation accuracy, and follow-up rate together form a quality-observation framework that helps the team spot knowledge gaps, retrieval defects, generation defects, policy defects, and UX issues.

Metrics must enter a stable operational cadence to drive continuous improvement. Weekly operations meetings handle daily issues and prioritization, focused postmortems address high-frequency or high-risk clusters, and major-release reviews control risk in knowledge, index, model, and policy changes. Meanwhile, the feedback loop must clarify ownership, establish issue status flow and prioritization rules, and prevent online feedback from circulating among teams without ever closing.

The maturity of the online feedback loop ultimately shows in whether the team can continuously convert real user problems into data assets, fix tasks, evaluation samples, and knowledge updates. The next section will, through an online knowledge-update SOP and incident review, discuss how to ground these mechanisms in concrete production processes.

---

## 23.5 Case Reviews and SOPs

### 23.5.1 Why an Online Knowledge-Update SOP Is Necessary

The previous sections discussed the online feedback loop, event collection, feedback routing, knowledge updates, version governance, and metrics operations. This section grounds these mechanisms in actual production processes, discussing how to use SOPs (Standard Operating Procedures) to manage online knowledge updates and incident reviews.

For a production RAG system, a knowledge update is not simply uploading a document or having an engineer manually rebuild an index. A seemingly small knowledge change can affect retrieval, citation, and risk judgment for many user questions. Without a standard procedure, teams easily run into the following: new knowledge is deployed without validation, letting wrong content into production; old knowledge is not retired in time, causing old/new conflicts; after an index update, no regression test is run, so previously correct questions begin to fail; when issues arise, the source of change is not recorded, making it impossible to identify which update introduced the regression. The core goal of the online knowledge-update SOP is to turn "knowledge change" into engineering actions that are approvable, executable, verifiable, and reversible. It must make explicit who is responsible at each stage, what the inputs and outputs are, what the passing criteria are, and how exceptions are handled.

A complete online knowledge-update SOP usually contains at least seven stages: change proposal, source verification, parsing and structuring, conflict detection, index update, regression evaluation, and canary release with monitoring. Low-risk knowledge can use a compressed flow; high-risk knowledge must add human approval, compliance review, and rollback planning.

*Table 23-11: Online Knowledge-Update SOP*

| Stage                | Input                                              | Key Actions                                                 | Output                  | Passing Criteria                                  |
| -------------------- | -------------------------------------------------- | ----------------------------------------------------------- | ----------------------- | -------------------------------------------------- |
| Change proposal      | New document, change notes, feedback samples       | Fill in source, scope of impact, owner                      | Knowledge change ticket | Clear rationale and explicit owner                 |
| Source verification  | Document source, permissions, approval record      | Verify trustworthiness, legality, scope                     | Source verification     | Trusted source, compliant permissions              |
| Parsing & structuring | Raw document, parser config                       | Extract sections, tables, charts, metadata                  | Structured knowledge units | Complete content, complete metadata             |
| Conflict detection   | New and old knowledge units                        | Check version, time, and scope conflicts                    | Conflict report         | High-risk conflicts resolved                       |
| Index update         | Knowledge units, index config                      | Update vector, keyword, structured indexes                  | New index version       | Index built successfully, traceable                |
| Regression evaluation | Golden set, failure-sample set, focused set      | Test recall, citations, answer correctness                  | Evaluation report       | Metrics meet threshold, no severe regressions      |
| Canary release       | New index version, canary policy                   | Small-traffic deploy and feedback monitoring                | Canary result           | No noticeable quality degradation                  |
| Full release/rollback | Canary results, monitoring metrics                | Expand traffic or revert to previous version                | Release record          | Successful release or safe rollback                |

In practice, an SOP should not be overly elaborate, or the team will bypass it; nor should it be too coarse, or it will fail to control risk. A workable approach is to execute different flows by risk tier. Low-risk FAQ updates can automate parsing, indexing, and small-scale regression; updates touching finance, contracts, healthcare, or compliance must go through expert review and canary validation.

---

### 23.5.2 Case: Stale Knowledge Causes a Wrong Answer

We illustrate with an enterprise internal-policy Q&A system, showing how, without a knowledge-update loop, online issues happen and how to fix them via SOP.

A company launched an internal knowledge assistant to answer employee questions on reimbursement, leave, procurement, and approval processes. Initially it performed well on most standard questions. Two months after launch, the finance team released a new travel-reimbursement policy, adjusting lodging standards, transportation allowances, and approval authority. However, the old policy remained in the knowledge base, and although the new policy was uploaded to the document system, structuring and index updates were never completed. A few days later, employees began asking: "What is the maximum hotel reimbursement for tier-one cities on business trips?" The system recalled the lodging standard from the old policy and produced an answer with citations. Because the answer included source links, users took it as trustworthy and submitted reimbursement claims under the old standards. The finance review caught the discrepancy and escalated the issue to system operations. The postmortem found this was not a random model hallucination but a textbook case of stale knowledge. The retrieved document did exist; it simply was no longer the current version. The new policy had been released but not entered the production index. The old policy lacked expiration and version-status fields, so retrieval could not down-weight or filter it.

The case revealed three data-engineering problems: first, the knowledge base lacked effective-time and expiration-time management; second, the new-knowledge release did not trigger an index update; third, the system did not run regression evaluation on high-frequency policy questions. Fix actions should address all three rather than just edit one answer.

*Table 23-12: Error Attribution and Fix Actions for the Stale-Knowledge Case*

| Symptom                            | Root-Cause Category          | Specific Cause                                | Fix Action                                                                       |
| ---------------------------------- | ---------------------------- | --------------------------------------------- | -------------------------------------------------------------------------------- |
| System cites old policy            | Knowledge version governance | Old document has no expiration status         | Add version, effective time, and expiration time to policy documents             |
| New policy not recalled            | Index-update defect          | New document not in the production index      | Trigger new-policy parsing, structuring, and index rebuild                       |
| Users cannot judge version         | Citation-display defect      | Answer omits policy version and effective date | Show version number and effective date in the citation                          |
| Same issue not detected earlier    | Regression-evaluation defect | Lack of policy-specific regression set        | Build a travel-policy regression sample set                                      |
| No verification after fix          | Operations-loop defect       | No canary monitoring or acceptance metrics    | Canary release with monitoring on negative-feedback rate for reimbursement Q&A   |

The case shows that online incidents are rarely single-point failures; they reflect inadequate knowledge-lifecycle management. The right fix is not to delete one wrong answer but to systematize "policy version governance" — including metadata completion, index updates, citation display, evaluation-set expansion, and dashboard refresh.

---

### 23.5.3 Automated Backflow of High-Value Feedback Samples

Not every piece of user feedback in the online feedback loop deserves human postmortem. A production system may generate huge volumes of logs, ratings, clicks, and follow-up behaviors daily — manual processing would be prohibitively expensive. The system therefore needs to identify high-value feedback automatically and prioritize routing to review and backfill.

High-value feedback samples usually share traits: they involve high-frequency questions, come from high-risk scenarios, carry strong negative feedback, contain user correction text, trigger human handoff, relate to recent knowledge updates, or cluster as similar failures within a short window. The system can compute a priority score for each feedback sample to decide whether it enters human review, the knowledge-update queue, or the evaluation set.

Below is a simplified example of a priority-scoring rule for feedback samples. It is not full production code; it illustrates how to combine explicit feedback, implicit behavior, risk level, and knowledge-update status into an interpretable filter rule.

```python id="5uhrnq"
from dataclasses import dataclass

@dataclass
class FeedbackEvent:
    query: str
    explicit_negative: bool = False      # Downvote, error report, or explicit "not helpful"
    has_correction: bool = False         # Contains user-provided correction text
    followup_count: int = 0              # Number of follow-up turns
    reformulated: bool = False           # User rephrased and resubmitted the question
    human_handoff: bool = False          # Whether the session was handed off to a human
    risk_level: str = "low"              # low / medium / high
    related_to_recent_update: bool = False
    frequency_7d: int = 1                # Occurrences of similar questions in the last 7 days


def score_feedback(event: FeedbackEvent) -> int:
    score = 0

    if event.explicit_negative:
        score += 3
    if event.has_correction:
        score += 4
    if event.followup_count >= 2:
        score += 2
    if event.reformulated:
        score += 2
    if event.human_handoff:
        score += 4
    if event.risk_level == "medium":
        score += 2
    elif event.risk_level == "high":
        score += 5
    if event.related_to_recent_update:
        score += 3
    if event.frequency_7d >= 10:
        score += 3
    elif event.frequency_7d >= 3:
        score += 1

    return score


def route_feedback(event: FeedbackEvent) -> str:
    score = score_feedback(event)

    if event.risk_level == "high" and (event.explicit_negative or event.human_handoff):
        return "expert_review_queue"

    if score >= 10:
        return "priority_failure_queue"
    elif score >= 6:
        return "sampling_review_queue"
    else:
        return "monitoring_pool"


event = FeedbackEvent(
    query="Under the new travel policy, can tier-one-city hotel costs still be reimbursed under the old standard?",
    explicit_negative=True,
    has_correction=True,
    followup_count=2,
    human_handoff=True,
    risk_level="high",
    related_to_recent_update=True,
    frequency_7d=12,
)

print(score_feedback(event))
print(route_feedback(event))

```

This snippet embodies an important principle: a feedback sample's value is not determined by downvote alone; it should be judged by multiple signals together. A low-risk downvoted question may not need immediate handling; a high-risk question, even if it occurred only once, may need expert review; a question that recurs at high frequency within a short window likely indicates a knowledge gap or systemic retrieval failure.

In production systems, such rules are usually combined with model classifiers. The system can first use rules to surface obvious high-value samples, then use a classification model to attribute root cause — knowledge gap, retrieval defect, generation defect, policy defect, or UX issue. Rules give interpretability; models give extensibility; the combination fits complex online environments better.

------

### 23.5.4 An Example Online Knowledge-Update SOP

Combining the case and the automated backflow above yields a fairly complete online knowledge-update SOP. Here is a simplified version suitable for an enterprise RAG system.

Step 1: From user feedback, tickets, and logs, the system surfaces failure samples and routes them to the corresponding queues by priority. High-risk issues go straight to expert review; ordinary issues enter sampling review or periodic processing.

Step 2: Operators or the system perform initial root-cause attribution — knowledge gap, knowledge expiration, retrieval defect, generation defect, or policy defect. If undetermined, the full event chain is retained and routed to joint postmortem.

Step 3: For knowledge-update-class issues, create a knowledge change ticket. The ticket should include source, user query, wrong answer, wrong citation, correct knowledge source, owner, risk level, and target deploy time.

Step 4: The knowledge owner adds or revises documents and verifies sources. For high-risk content, business experts or compliance owners must confirm. After confirmation, the document enters parsing and structuring.

Step 5: The system performs conflict detection and index update. If old/new conflicts are found, confirm the authoritative source and applicable scope first rather than letting both versions enter the production index simultaneously.

Step 6: Run regression evaluation. The evaluation set should include at least the original failure sample, similar-question samples, golden-set samples, and high-risk boundary samples. Only when key metrics meet thresholds does the change proceed to canary release.

Step 7: Canary deploy and monitor. If negative-feedback rate, human handoff rate, or citation-error rate spikes, roll back immediately; if canary is stable, expand traffic and finally roll out fully.

Step 8: Close the feedback ticket and precipitate the postmortem record. The record should state root cause, fix actions, affected version, deployment outcome, and preventative measures. These records become important inputs to future operations meetings and the case library.

The value of this SOP is that it turns one online failure into a complete data loop. A failure is no longer just a UX issue; it becomes input for knowledge updates, evaluation enhancement, and system governance. As this flow runs continuously, the system not only fixes individual issues but steadily improves robustness to similar issues.

------

### 23.5.5 Section Summary

This section used the stale-knowledge case to discuss how the online feedback loop is grounded in a concrete SOP. In production RAG systems, many errors are not random model hallucinations but the combined failure of knowledge versioning, index updating, citation display, and regression evaluation. Incident review must therefore not stop at "the answer was wrong" but localize the specific stage in the knowledge lifecycle.

To improve feedback processing efficiency, the system must automatically identify high-value feedback and rank by signals such as explicit negative feedback, user corrections, follow-up behavior, human handoff, risk level, recent updates, and frequency. Combining rules and models, large volumes of online feedback can be turned into manageable review queues, fix tasks, and evaluation samples.

The core of the online knowledge-update SOP is to organize failure-sample collection, root-cause analysis, knowledge change, structuring, index update, regression evaluation, canary release, and postmortem precipitation into a stable process. Only when these actions form an institutionalized closed loop can an LLM application move from "launched and usable" to "reliable over the long term."
