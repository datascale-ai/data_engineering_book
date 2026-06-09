# Chapter 23: Online Feedback Loops and Knowledge Updates

## Abstract

The deployment of a Retrieval-Augmented Generation (RAG) system marks the beginning of the data flywheel, not its end. Once live, the system faces real users who phrase queries non-standardly and omit necessary conditions, as well as a knowledge environment that shifts continuously as policies, products, and processes evolve. Reliability no longer depends on whether the initial knowledge base was complete; it depends on whether the system can continuously discover, fix, and update itself through real usage. This chapter focuses on the data feedback loop that operates after deployment, discussing how online signals—logs, clicks, ratings, corrections, tickets, and human handoffs—can be transformed into governable data assets. The chapter first argues for the necessity of the online feedback flywheel, then discusses how event collection and feedback routing extract usable failure samples and improvement signals from noisy data. It goes on to explain how knowledge updates, rollbacks, and version governance enable changes to be written safely without degrading existing capabilities, then establishes a metrics dashboard and operational cadence to sustain long-term operation. Finally, through post-incident reviews and Standard Operating Procedures (SOPs), the chapter shows how every failure can become an input for subsequent improvement.


When a RAG system completes deployment, the real data engineering work does not end—it enters a new phase. Before deployment, the system primarily faces controllable test sets, curated knowledge bases, and relatively standard questions; after deployment, it begins facing real users, real business processes, and a continuously changing knowledge environment. Users ask questions using non-standard expressions, enter conversations with omitted conditions, and pose new questions as policies, products, and processes are continuously updated. At this point, whether the system is reliable no longer depends solely on whether the initial knowledge base was built completely, but on whether it can continuously discover problems, fix them, and update itself through real usage.

This chapter therefore focuses on the data feedback loop after system deployment, discussing how online signals—logs, clicks, ratings, corrections, tickets, and human handoffs—can be transformed into governable data assets. The concern is not whether a single answer is correct, but whether the system can establish stable feedback routing, error attribution, knowledge updates, version rollbacks, and an operational cadence that makes every failure an input for subsequent improvement.

Toward this goal, the chapter begins with the necessity of the online feedback flywheel, then covers event collection and feedback routing, knowledge updates and version governance, metrics dashboards and operational cadence, and post-incident reviews with SOPs. The core question it answers is: as a large-model application transitions from successful deployment to long-term operation, how should data engineering support the system's continuous improvement?

## Keywords

Online feedback; knowledge updates; RAG operations; log governance; version rollback; data flywheel

## Learning Objectives

- Explain why system deployment is only the starting point of the data flywheel, not the end of data work.
- Design event collection and feedback routing mechanisms to extract usable failure samples from explicit and implicit signals.
- Build knowledge update, rollback, and version governance processes that safely write changes without degrading existing capabilities.
- Design online metrics dashboards and operational cadences to support long-term operations.
- Transform every failure into an input for subsequent improvement through post-incident reviews and Standard Operating Procedures.

---

## 23.1 Why "Deployment Complete" Is Only the Starting Point of the Data Flywheel

### 23.1.1 From Offline Validation to Real Usage: Deployment Is Not the Finish Line

In many large-model application projects, deployment is treated as a milestone endpoint. After the system completes development, passes testing, connects to the knowledge base, is deployed to the production environment, and performs stably on a small set of example questions, the team often concludes that the project has entered its wrap-up phase. However, for RAG systems, knowledge assistants, enterprise Q&A systems, and large-model applications serving real users, deployment is not the end of data work—it is the starting point at which the data flywheel truly begins to turn.

This phenomenon arises because the problem distribution observed during the offline phase typically represents only a small fraction of the system's real usage scenarios. When development teams build evaluation sets, they tend to select questions that are relatively clear, controllable, and easy to annotate—for example, "What is the reimbursement standard for a given policy?", "How do I enable a specific product feature?", or "What materials need to be submitted for a given process?" These questions have clear answers and can easily be mapped to specific document fragments in the knowledge base. After deployment, however, users ask questions that are more complex, more ambiguous, and closer to real business contexts. Users do not phrase questions according to document titles, nor do they proactively provide complete conditions. They might ask: "Can I get reimbursed for my situation?", "Does that process from last time still apply?", or "Why did this metric suddenly change?"—questions that often contain omissions, anaphora, context dependence, implicit constraints, and cross-document combinations. This phenomenon—whereby real data environments are only entered after deployment—is consistent with observations that machine learning systems accumulate technical debt through data dependencies, changes in the external world, and hidden feedback loops (Sculley et al. 2015; Amershi et al. 2019). A system only exposes problem types that offline evaluation cannot cover through real interactions; this is why data engineering for large-model applications often cannot stop at building a knowledge base once or completing a single evaluation set. The production environment is not a static examination room but a continuously evolving business arena, where user questions, knowledge content, business rules, permission scopes, product versions, and organizational processes all change over time. If the system lacks mechanisms for post-deployment feedback collection, problem routing, knowledge updates, and version governance, system capabilities will not only fail to improve continuously—they may actually degrade as knowledge expires, user needs shift, and errors accumulate.

Chapter 23 therefore addresses the core question of how to transform real post-deployment usage into data assets, and how to feed those data continuously back into the knowledge base, retrieval system, generation strategy, and evaluation framework. Compared to Chapter 21's discussion of RAG data pipelines and Chapter 22's discussion of multimodal visual retrieval, this chapter focuses more on the closed-loop mechanisms after the system is running: how the system perceives failure, how it attributes problems, how it updates knowledge, how it verifies that updates are effective, and how these actions can be integrated into a stable operational cadence.

From this perspective, deploying a large-model application is not delivering a finished product but entering a new phase of the data lifecycle. The key in this phase is no longer one-time construction but continuous collection, continuous diagnosis, continuous backfilling, and continuous governance. Only when a system possesses this closed-loop capability can RAG and multimodal knowledge applications move from being demonstrable to being operationally sustainable.

---

### 23.1.2 The Real Problem Distribution Only Surfaces After Deployment

Offline evaluation sets typically have clearly defined boundaries. They come from existing documents, business-expert-curated questions, typical examples constructed by project teams, or model-generated question sets (Gama et al. 2014; Koh et al. 2021). These data are important for pre-deployment validation, but they cannot fully represent how real users ask questions. Real user questions have three distinctive characteristics: non-standard expression, incomplete conditions, and unstable objectives.

First, users' phrasing does not always align with the terminology in the knowledge base. If the system relies solely on standard document terminology for retrieval, it may fail to recall the correct knowledge. Online feedback data can expose these real expression discrepancies and help the system supplement synonyms, business aliases, and query rewriting rules.

Second, user questions often lack key conditions. For example, an employee might ask "Can this expense be reimbursed?" without specifying the expense type, location, employee status, project assignment, or approval status. If the system answers directly, it may ignore applicable conditions; if it can recognize missing conditions, it should ask follow-up questions or provide conditional answers. These question types are difficult to cover adequately in offline evaluation, because evaluation samples are typically organized as fairly complete questions, whereas real users often provide only fragmentary information.

Third, user objectives may shift across multi-turn interactions. A user might initially ask about a policy clause, then move to "Help me determine whether my situation qualifies," and then further request "Give me a statement I can send to finance." This means the system must move from single-turn Q&A toward task-oriented interaction, and multi-turn trajectories recorded in online logs are an important data source for understanding real task flows.

The divergence between this real problem distribution and the offline evaluation distribution can be called "post-deployment distribution shift." It does not necessarily mean that model capabilities have degraded; rather, it indicates that the system has entered a more real and more complex data environment. Many RAG systems are unstable after deployment not because offline evaluation was fraudulent, but because offline evaluation cannot cover the long-tail distribution of production problems.

*Table 23-1: Differences Between Offline Evaluation Questions and Online Real Questions*

| Dimension | Offline Evaluation Questions | Online Real Questions | Impact on the System |
| --- | --- | --- | --- |
| Expression | Standard terminology, complete structure | Colloquial, abbreviations, aliases common | Affects query understanding and retrieval recall |
| Condition completeness | Usually contains necessary conditions | Often omits identity, time, version, scenario | Requires follow-up questions, condition identification, and risk warnings |
| Question boundary | Mostly single questions | Multi-intent, multi-constraint, cross-document combinations | Requires question decomposition and multi-path retrieval |
| Context dependence | Usually no context or fixed context | Continuously changing across multi-turn interactions | Requires state management and session memory |
| Error exposure mode | Through manual judgment or metric computation | Through follow-up questions, downvotes, corrections, abandonment | Requires online feedback collection and attribution |

As shown in the comparison in Figure 23-1, in real systems this divergence directly influences data engineering priorities. Before deployment, teams focus more on knowledge base coverage, parsing quality, index structure, and baseline evaluation; after deployment, teams must also attend to user expression, failure samples, feedback annotation, knowledge expiration, permission mis-recall, citation credibility, and multi-turn task continuity. In other words, pre-deployment data engineering emphasizes "building the system," while post-deployment data engineering emphasizes "making the system continuously better."



![Figure 23-1: From Offline Evaluation to the Online Real Problem Distribution](../../images/part7/图23_1zh.png)

*Figure 23-1: From Offline Evaluation to the Online Real Problem Distribution*



The real problem distribution after deployment also has a clear temporal dimension. Certain questions will surge suddenly in response to new policy releases, new product launches, organizational changes, or external events. For example, after a reimbursement policy is updated, employees will concentrate their questions on the differences between old and new rules; after a new feature launches, customers will concentrate their questions on usage and limitations; after a financial report is published, analysts will concentrate their questions on the reasons behind certain metric changes. Without online question monitoring, it is very difficult to detect these changes in time, let alone drive simultaneous updates to the knowledge base and retrieval strategy. Therefore, production-grade large-model applications must treat online questions as a continuously generated data asset. User questions are not noise—they are the most authentic demand signals the system receives; user failures are not isolated incidents but entry points for system optimization. Only by establishing collection and analysis mechanisms oriented toward the online distribution can a team truly understand the users, scenarios, and boundaries the system serves.

---

### 23.1.3 Why Online Feedback Is More Valuable Than Offline Samples

Online feedback is important because it simultaneously possesses three characteristics: authenticity, timeliness, and behavioral signals.

Authenticity means that online feedback comes from real users' actual behavior in real tasks. Offline evaluation samples are often manually curated, with clear question boundaries and answers that are relatively easy to annotate; however, these signals often carry position bias and interpretive ambiguity, and cannot be used directly as strong supervision labels without correction (Joachims 2002; Joachims et al. 2017). Online feedback, in contrast, contains users' real expressions, real context, and real business objectives. It tells the team not only "whether the system can answer standard questions" but also "whether the system can help users complete tasks."

Timeliness means that online feedback can reflect the latest changes in knowledge and demand. Enterprise knowledge bases, product documentation, process policies, and business rules are continuously updated. Even a high-quality offline evaluation set may become outdated quickly. Online feedback, by contrast, can immediately expose new problems, new terminology, new requirements, and new conflicts. For example, if users frequently ask "Can the old interface still be used?" after a new product version launches, this suggests that the documentation may need migration notes; if users frequently ask follow-up questions about the scope of applicability of a policy, this suggests that the original text may be insufficiently clear or that the retrieval results do not fully cover the conditions.

Behavioral signals mean that users express their attitudes not only through explicit feedback but also expose system quality indirectly through their behavior. Explicit feedback includes upvotes, downvotes, corrections, text comments, and manual annotations; implicit feedback includes whether users continue to ask follow-up questions, whether they rephrase their question, whether they click on cited sources, whether they copy the answer, whether they transfer to a human agent, and whether they abandon the session. Compared to a single correct/incorrect label, these behavioral signals are closer to real experience.

For example, a user who does not downvote but reformulates the same question three times in a row suggests that the system's preceding answers may not have met the need; a user who clicks on a cited source and spends a relatively long time there suggests that the answer may have triggered a verification need; a user who immediately transfers to a human agent after receiving an answer suggests that the system may have failed to cover high-risk or complex judgment scenarios. If these signals are collected systematically, they can be transformed into high-value training samples, evaluation samples, and knowledge update tasks.



*Table 23-2: Types and Value of Online Feedback Signals*

| Feedback Type | Typical Source | Problem Reflected | Optimization Actions That Can Be Driven |
| --- | --- | --- | --- |
| Explicit positive feedback | Upvotes, adoption, confirmation of helpfulness | Answer likely meets the need | Retain high-quality Q&A samples, reinforce templates |
| Explicit negative feedback | Downvotes, corrections, error reports | Answer is incorrect, incomplete, or untrustworthy | Enter failure sample repository, trigger error attribution |
| Follow-up behavior | User continues asking, supplements conditions | Previous answer was insufficient or conditions were unclarified | Optimize follow-up strategy and context assembly |
| Reformulation behavior | User rephrases and asks again | Query understanding or retrieval failure | Add synonym expressions and query rewrite samples |
| Citation clicks | User views source document | User needs to verify evidence | Optimize citation readability and evidence localization |
| Human handoff | Ticket, human takeover | System cannot complete high-risk judgment | Establish human review and backfill process |
| Session abandonment | Interruption, no further continuation | User experience failure with no explicit feedback | Include as weak negative sample for analysis |



The value of online feedback also lies in its ability to help teams establish problem-priority awareness. During the offline phase, teams may pursue overall metric improvement—for example, Recall@k (the number of successfully recalled relevant results among the top k results), Answer Accuracy, Citation Accuracy, and so on. After deployment, however, what truly needs to be prioritized is not every error but high-frequency errors, high-risk errors, and high-impact errors. High-frequency errors affect a large number of users with similar problems, so fixing them yields high returns; high-risk errors involve sensitive scenarios such as legal, financial, medical, compliance, and permissions, and must be prioritized for control; high-impact errors may appear in critical business processes such as approval, reimbursement, contract review, customer service, and operations incident handling. Online feedback can provide error frequency, user impact scope, and business consequences, thereby helping teams direct limited resources to the most valuable repair directions.

It should be noted that online feedback does not equal data that can be used directly for training. User feedback often contains noise, emotion, misoperation, and missing context. For example, a user's downvote may be because the answer is genuinely wrong, but it may also be because the answer is too long, the tone is inappropriate, it does not give a direct conclusion, or the user's own input was incomplete. Therefore, online feedback must undergo cleaning, attribution, annotation, and routing before it can be transformed into usable data assets. This is precisely the data engineering core of the online feedback loop: not simply collecting logs, but transforming user behavior into structured problems, transforming structured problems into repair tasks, and transforming repair tasks into new knowledge, indices, evaluations, and model improvements.

---

### 23.1.4 Why a System Without a Feedback Loop Degrades Over Time

A large-model application without a feedback loop, even if it performs well at deployment, may progressively degrade after a period of operation. This "degradation" does not necessarily manifest as worse model parameters; rather, it manifests as a growing mismatch between the system and real business needs.

The first type of degradation comes from knowledge expiration. Enterprise policies, product features, interface documentation, organizational processes, pricing rules, and compliance requirements all change. If the knowledge base is not updated in time, the system will continue to answer based on outdated knowledge. More dangerously, outdated knowledge often still looks reasonable, making it difficult for users to detect the error promptly. For example, the system may answer reimbursement standards based on an old policy version, or explain feature limitations based on outdated product documentation. This type of error has a concealed nature and requires online feedback and version governance to detect (Sculley et al. 2015; Breck et al. 2017).

The second type of degradation comes from shifts in user expression. As users become more familiar with the system, their questioning patterns change. Early users may try standard questions; later, they will pose more complex, more specific, and more task-oriented questions. If the system continues to optimize only for the pre-deployment evaluation set, it will increasingly fail to cover real needs. Users will then frequently ask follow-up questions, rephrase their questions, or switch to human support channels, and both usage rates and trust levels will decline.

The third type of degradation comes from accumulated errors. RAG and multimodal systems typically consist of multiple stages: collection, parsing, chunking, indexing, retrieval, re-ranking, context assembly, generation, and citation. A small error in any stage can be amplified in subsequent stages. Without a mechanism for failure sample feedback and error attribution, these problems will persist for a long time and recur across more user questions.

The fourth type of degradation comes from organizational responsibility fragmentation. After deployment, the system may enter a state of unmaintained operation: the algorithm team considers the project delivered, the business team treats the problems as model capability issues, the platform team only cares about service availability, and content maintainers are only responsible for uploading documents. Without a unified feedback loop, user problems flow among multiple teams but cannot be resolved into clear remediation actions. Over time, the system remains nominally online while its actual quality continues to decline.



*Table 23-3: Typical Degradation Paths in the Absence of an Online Feedback Mechanism*

| Degradation Source | Specific Manifestation | User-Perceived Effect | Root Cause |
| --- | --- | --- | --- |
| Knowledge expiration | Old policies, old interfaces, old processes retrieved | Answers appear reasonable but are actually wrong | No knowledge version updates or expiration governance |
| Expression mismatch | Colloquial user questions cannot match formal documents | System answers off-topic | No online query analysis or synonym expression backfill |
| Repeated errors | Same type of error recurs repeatedly | Users feel system "never learns" | No failure sample repository or repair feedback loop |
| Citation invalidation | Answers cite outdated documents or wrong locations | Users cannot verify answers | No citation anchor validation or index updates |
| Responsibility fragmentation | Problem cannot be attributed to a team or module | Long repair cycles | No feedback routing or operational mechanisms |
| Evaluation distortion | Offline metrics high but online satisfaction low | System "tests well, works poorly" | Offline evaluation set has no online sample feedback |



This degradation is not a problem with any single model or module; it is the result of the system never entering a sustained operational state. A large-model application without a feedback loop is like a production system that receives a health check only once before going live: metrics are acceptable at deployment, but afterward there is no monitoring, no review, no repair, and no version updates. As the business environment changes, the system naturally drifts further from real needs.

Online feedback loops are therefore not an "icing on the cake" operational function but a fundamental capability of production-grade large-model applications. They determine whether the system can learn from user interactions, whether failures can be turned into improvements, and whether stability can be maintained through knowledge and demand changes.

---

### 23.1.5 The Basic Structure of the Data Flywheel: From Feedback to Improvement

The online feedback loop can be understood as a data flywheel. A data flywheel does not mean the system automatically improves without supervision; rather, it means the system can continuously generate new data through real usage, and that these data—after filtering, annotation, attribution, and governance—feed back into the knowledge base, index, retrieval, generation, and evaluation frameworks to improve the next round of user experience.

As shown in Figure 23-2, a complete data flywheel typically comprises six stages: event collection, feedback routing, error attribution, remediation actions, regression evaluation, and online validation. Event collection is the entry point of the flywheel. The system must record user queries, session context, retrieval results, context assembly content, generated answers, citation sources, user feedback, and subsequent behavior. Without sufficiently complete logs, subsequent attribution is nearly impossible. For example, if only the final answer is recorded but not the retrieval results, it is impossible to determine whether the error originated in retrieval or generation; if citation sources are not recorded, it is impossible to determine whether the answer was based on correct evidence (Es et al. 2024). Feedback routing transforms raw feedback into problem types. A user's negative feedback may correspond to missing knowledge, retrieval failure, citation error, generation hallucination, permission issues, or product experience problems. Different problems need to enter different processing queues and cannot all be handed to the model team or content team. Error attribution localizes failures to specific stages in the system pipeline. For example, an incorrect answer may be because the document was never ingested, or because it was ingested but parsing failed, or because parsing was correct but chunk splitting destroyed the semantics, or because retrieval recalled the wrong version, or because the model did not follow the evidence. The more accurate the attribution, the more effective the remediation action. Remediation actions include knowledge supplementation, document re-parsing, metadata correction, index rebuilding, query rewrite augmentation, re-ranking sample supplementation, prompt adjustment, refusal policy updates, and evaluation set expansion. Remediation is not simply "correcting an answer" but mapping errors to reusable data and system improvements. Regression evaluation verifies whether remediation is effective. Every knowledge update, index update, or strategy adjustment may introduce new problems. Therefore, the system needs to run regression tests using golden sets, online failure sample sets, and specialized challenge sets, ensuring that fixing current problems does not degrade existing capabilities (Yu et al. 2024). Online validation observes the post-fix online effect. For example, whether negative feedback on similar questions decreases, whether citation click rates improve, whether human handoff rates decline, and whether the number of follow-up turns decreases. These metrics in turn enter the next round of feedback analysis.



![Figure 23-2: The Online Feedback Data Flywheel for Large-Model Applications](../../images/part7/图23_2zh.png)

*Figure 23-2: The Online Feedback Data Flywheel for Large-Model Applications*



The key to the data flywheel is not "the higher the degree of automation, the better" but "every type of feedback finds a stable destination." For low-risk, high-frequency problems, automated rules and model-assisted processing can complete routing and backfilling; for high-risk problems, human review, expert annotation, and deployment approval are required. Mature systems typically adopt a "automation + human governance" hybrid model, using automation for scale processing and human judgment for high-risk decisions and quality calibration.

In addition, the data flywheel must be combined with version management. Every backfill should record its source, the triggering problem, the remediation action, the scope of impact, and the evaluation results. Otherwise, teams cannot determine whether a given online performance improvement was driven by a specific batch of data, a specific index adjustment, or a specific strategy change. Without versions, there is no retrospective; without retrospectives, there is no stable iteration.

---

### 23.1.6 Section Summary

This section discussed why deploying a large-model application is not the end of data work but the starting point of the online feedback loop and data flywheel. Offline evaluation helps the system complete pre-deployment validation, but real user questions are fully exposed only in the production environment. The post-deployment problem distribution is typically more colloquial, long-tailed, task-oriented, and context-dependent, so the system must continuously perceive real needs through online feedback.

Online feedback is highly valuable because it simultaneously provides authenticity, timeliness, and behavioral signals. Users' upvotes, downvotes, follow-up questions, reformulations, citation clicks, human handoffs, and session interruptions are all important clues about system quality. After structured processing, these signals can enter the failure sample repository, knowledge update queue, evaluation set expansion, and system optimization pipeline.

Systems without a feedback loop face knowledge expiration, expression mismatch, repeated errors, citation invalidation, and organizational responsibility fragmentation (Lewis et al. 2020; Mallen et al. 2023). They may perform well in the early post-deployment period but, as the business and user needs change, gradually exhibit the phenomenon of "testing well, working poorly." Production-grade large-model applications must therefore organize feedback collection, problem routing, error attribution, remediation actions, regression evaluation, and online validation into a stable data flywheel.

The next section will further discuss the event collection and feedback routing mechanisms in the online feedback loop, focusing on how logs, clicks, ratings, corrections, tickets, and human handoffs enter a unified data processing pipeline, and how to distinguish among missing knowledge, retrieval defects, generation defects, and strategy defects.

---

## 23.2 Event Collection and Feedback Routing

### 23.2.1 Why the Feedback Loop Is First an Event Engineering Problem

The online feedback loop does not start with "a user downvotes"—it starts with whether the system can completely record every interaction event. In large-model applications, a seemingly simple question-and-answer exchange often involves multiple critical stages behind the scenes: user input, context state, retrieval request, recalled results, re-ranking results, context assembly, model generation, citation sources, user behavior, and subsequent feedback. If this information is not systematically recorded, teams will find it very difficult to determine where a failure occurred and will be unable to transform online problems into reusable data assets. This section therefore begins with a fundamental observation: the foundation of the online feedback loop is not an annotation platform or a training script, but an event collection infrastructure. Only when the system can decompose a user interaction into a traceable, auditable, and retrospectively reviewable event chain does subsequent feedback routing, error attribution, knowledge updating, and regression evaluation have a reliable basis.

In traditional Web or mobile application systems, event collection focuses more on product behavior metrics such as clicks, page views, dwell time, and conversion rates. In large-model applications, event collection must additionally cover model-side and data-side information. For example, in a RAG Q&A system, recording only "what the user asked" and "what the model answered" is insufficient. The system must also record which documents were recalled at the time, what the score of each document was, which fragments were ultimately included in the context, which sources the model cited, and whether the user subsequently asked follow-up questions or transferred to a human agent. Without this information, error retrospectives become very difficult. For instance, a user downvotes an answer, but the system has only saved the final response. At this point, the team cannot determine whether the problem arose because the knowledge base lacked relevant content, because retrieval failed to recall the correct fragment, because re-ranking ordered the results incorrectly, because context assembly omitted a key condition, or because the model generated a wrong conclusion from available evidence. Ultimately, all problems will be attributed vaguely to "poor model answers," causing remediation efforts to lose focus. More seriously, if event collection is incomplete, the system may repeatedly fix the wrong stage. For example, a real problem may be that the document version has expired, but because the system did not record the version number of the recalled document, the team may mistakenly attribute it to a prompt problem and keep adjusting the prompt; another problem may be that the correct document is invisible due to permission filtering, but if the log does not record the permission judgment result, the team may mistakenly attribute it to insufficient vector retrieval recall. In the long run, this kind of misattribution wastes large amounts of engineering resources and obscures genuine data problems (Breck et al. 2017).

Production-grade large-model applications therefore need to design event collection as the first layer of infrastructure for the data feedback loop. It must serve both online monitoring and offline retrospectives; it must record both user behavior and model inputs and outputs; and it must cover both successful samples and failure samples. Only then can the system deposit real usage processes as data that is analyzable, annotatable, and backfillable.

---

### 23.2.2 Logs, Clicks, Ratings, Corrections, Tickets, and Human Handoffs

Online feedback signals typically come from multiple channels. Different signals have different levels of credibility, granularity, and usage patterns, so they cannot simply be mixed together for processing. A mature feedback system typically collects multiple event types simultaneously—logs, clicks, ratings, corrections, tickets, and human handoffs—and organizes them under a unified event schema.

Logs are the most fundamental feedback source. They record the complete chain of system operation, including user queries, session IDs, retrieval results, context fragments, model output, citation sources, response latency, error codes, and policy hit status. Logs may not directly express user satisfaction, but they provide the necessary evidence for subsequent error attribution. Without logs, a team can only see failure results; with logs, the team can see the failure process. Research on click logs in information retrieval demonstrates that user behavior can reflect ranking and result quality, but raw clicks are influenced by display position, snippet quality, and task intent, and typically require bias correction and contextual modeling before being used for learning-to-rank or sample backfilling (Joachims 2002; Chapelle and Zhang 2009; Joachims et al. 2017).

Click behavior reflects how users interact with the results provided by the system. In RAG systems, whether users click on cited sources, expand additional evidence, copy answers, or navigate to original documents can all serve as indirect signals of answer credibility and usability. For example, a user who frequently clicks citations but then continues to ask follow-up questions may indicate that citations exist but explanations are insufficient; a user who adopts an answer without clicking citations at all may indicate that the answer is sufficiently clear, or that the user is insensitive to source attribution. These signals require integrated judgment in context.

Ratings are the most common form of explicit feedback, including upvotes, downvotes, star ratings, and "was this resolved?" confirmation buttons. Rating signals are simple, direct, and easy to aggregate, but they are also susceptible to subjective user factors. For example, a user might downvote because the answer is too long, or because the system refused to answer a high-risk question, neither of which necessarily means the answer is factually wrong. Ratings are therefore suitable as an entry point for problem discovery, but should not be used directly as training labels.

Correction feedback is more valuable than ratings because it typically contains the specific errors pointed out by users. For example, a user may mark "wrong citation," "this policy has expired," "the answer omits the situation for probationary employees," or "the value in the chart was misread." This type of feedback can directly enter the error attribution process and help teams build higher-quality backfill samples. Correction feedback also requires review, however, because user-provided corrections are not always accurate—especially in professional domains, where confirmation by domain experts or content owners is required.

Tickets are an important source of complex problem cases. In enterprise settings, many users do not provide complete feedback directly in the chat interface; instead, they submit problems through customer service systems, internal ticketing systems, operations platforms, or business process systems. Tickets typically include more detailed background, screenshots, attachments, and human handling results, making them suitable for building high-value failure samples. However, ticket data often has complex structure and requires desensitization, denoising, field mapping, and task type annotation.

Human handoffs are key feedback signals in high-risk or complex tasks. When the system cannot answer, when users request human handling, or when policy mandates a transfer to a human agent, these handoff events should all be recorded. Human handoffs not only indicate the current capability boundary of the system but also provide the expert handling trajectory. If it is possible to record how the human ultimately resolved the problem, the system can learn from that about which knowledge was missing, which follow-up conditions were necessary, and which response strategies need adjustment.



*Table 23-4: Online Feedback Sources and Data Value*

| Feedback Source | Typical Content | Advantages | Limitations | Suitable Downstream Actions |
| --- | --- | --- | --- | --- |
| System logs | Query, recalled fragments, context, answer, citations, latency | Complete pipeline, usable for retrospectives | Does not directly express user satisfaction | Error attribution, performance diagnosis, regression evaluation |
| Click behavior | Citation clicks, evidence expansion, answer copying, document navigation | Reflects real usage behavior | Semantic interpretation is not unique | Evidence usability analysis, citation optimization |
| Rating feedback | Upvotes, downvotes, star ratings, resolution confirmation | Simple, direct, easy to aggregate | High subjective noise | Failure sample filtering, satisfaction monitoring |
| User corrections | Error identification, correct answer provided, expired content reported | High information density, high value | Requires review and confirmation | Knowledge updates, sample backfilling |
| Ticket records | Problem description, screenshots, handling process, final conclusion | Complete background, suitable for complex cases | Inconsistent structure, high processing cost | Specialized dataset construction, SOP optimization |
| Human handoffs | Transfer reason, expert handling trajectory, final response | Exposes system capability boundary | Depends on human process standards | High-risk strategy optimization, expert sample retention |



These feedback sources together constitute the raw data layer of the online feedback loop. In engineering practice, teams should not rely on only one type of feedback signal but should build a multi-signal fusion mechanism. For example, a downvoted sample that is also accompanied by citation clicks, repeated follow-up questions, and a human handoff is more worthy of priority analysis than an ordinary downvote; a session with no explicit downvote but where the user reformulated the same question three times in a row should also be identified as a potential failure sample.

To achieve this fusion, the system needs to organize events from different sources at three levels: session, question, and answer. The session level records the complete trajectory of a user during a continuous interaction; the question level records each user input along with its retrieval and generation process; the answer level records model output, cited evidence, user feedback, and subsequent behavior. Only by establishing this hierarchical event structure can feedback routing move beyond coarse-grained statistics and enter an actionable engineering feedback loop.

---

### 23.2.3 The Distinction Between Explicit and Implicit Feedback

Online feedback can be divided into explicit feedback and implicit feedback. Explicit feedback is an evaluation or correction actively expressed by the user; implicit feedback is a satisfaction, confusion, or failure signal indirectly reflected in user behavior. The two play different roles in the data feedback loop and must be used distinctly.

The advantage of explicit feedback is semantic clarity. When a user downvotes or submits a correction, it typically means the system's response did not meet expectations. For data operations teams, explicit negative feedback is the most direct entry point for failure samples. Explicit positive feedback can be used to retain high-quality Q&A pairs and to analyze which response formats, evidence organization styles, and explanatory approaches are more readily accepted by users. However, explicit feedback has a coverage problem. Most users do not provide proactive feedback, especially when the system answer is mediocre but not seriously wrong—users may simply leave or rephrase their question. In addition, explicit feedback may carry emotional and accidental noise. For example, a user might downvote because the system did not produce their desired conclusion, but the system's refusal may be the correct safety strategy; a user might downvote because the answer is too conservative, but from a compliance perspective, a conservative answer may be more appropriate.

The advantage of implicit feedback is wide coverage. Almost all user behaviors can become implicit feedback signals, including dwell time, number of follow-up questions, question reformulation, citation clicks, answer copying, page navigation, human handoffs, and session abandonment. These signals do not require users to actively evaluate, so they are closer to the real usage process. However, implicit feedback is more difficult to interpret. A user continuing to ask follow-up questions may be because the answer was incomplete, or because the user wants to explore further; a user clicking on citations may be because the answer is trustworthy, or because the answer is suspicious; a user leaving the session may be because the problem has been resolved, or because the system provided no help. Therefore, implicit feedback generally cannot be used as a label alone but must be analyzed in combination with explicit feedback, log information, and business context.

*Table 23-5: Comparison of Explicit and Implicit Feedback*

| Dimension | Explicit Feedback | Implicit Feedback |
| --- | --- | --- |
| Typical form | Upvotes, downvotes, ratings, corrections, comments | Follow-up questions, reformulation, citation clicks, copying, human handoff, abandonment |
| Advantages | Semantically direct, easy to interpret | Wide coverage, close to real behavior |
| Limitations | Low coverage rate, significant subjective noise | Non-unique meaning, requires contextual interpretation |
| Data value | Suitable as failure sample entry point and human review queue | Suitable for behavioral analysis, weak labels, and trend monitoring |
| Usage approach | Enters annotation or backfill pipeline after review | Requires combined judgment with logs and other signals |
| Risk | User misjudgment, emotional feedback, malicious feedback | Misinterpreting behavior, mistaking normal exploration for failure |

In engineering practice, explicit feedback can be viewed as "strong signals" and implicit feedback as "weak signals." Strong signals are suitable for directly entering human review and the failure sample repository; weak signals are suitable for discovering anomalous trends and filtering candidate problems (Hu, Koren and Volinsky 2008; Joachims et al. 2017). For example, if the downvote rate for a category of questions rises, that is a strong failure signal; if the average follow-up turn count for a category of questions rises, citation click rates fall, and human handoff rates rise, even without many downvotes, this indicates that the system may have undergone quality degradation.

To use implicit feedback more reliably, the system can design combination rules. For example, a session can be marked as "likely unresolved" when the following conditions are met: the user reformulates a similar question within 30 seconds of receiving an answer; retrieval hits two different documents across two consecutive turns but neither answer is adopted; and the session ends with a human handoff or ticket submission. Such combined signals are more reliable than any single behavior and are more suitable as candidate samples for subsequent routing and spot-checking.

---

### 23.2.4 The Event Collection Schema: From Raw Logs to Usable Samples

To transform online feedback into data assets, a unified event schema must be designed. The purpose of an event schema is to organize raw interaction logs into a data structure that is searchable, analyzable, annotatable, and replayable. Without a unified schema, feedback data will be scattered across log systems, instrumentation systems, customer service systems, databases, and model services, making it difficult to form a feedback loop.

A complete event schema should include at least six categories of information: user and session information, query information, retrieval information, generation information, feedback information, and governance information. User and session information identifies the context to which an interaction belongs. It does not necessarily need to store the user's real identity, but should record anonymous user ID, session ID, tenant, permission scope, language, terminal, and timestamp. For enterprise systems, it is also necessary to record the organizational unit, role, or permission group to which the user belongs, in order to determine whether certain answers involve unauthorized recall. Query information describes the user input itself, including the raw query, normalized query, query rewriting result, recognized intent, entities, constraint conditions, and context references. For example, when a user asks "Can I still expense this?", the system needs to record that "this" references the specific expense item from the previous turn, "expense" corresponds to a reimbursement intent, and missing conditions include expense type and location. Retrieval information is one of the most important parts of event collection in a RAG system. The system needs to record the index version being queried, recalled document IDs, chunk IDs, scores, re-ranking scores, fragments ultimately included in the context, filtered-out fragments, and reasons for filtering. Without this information, retrieval failures and generation failures are very difficult to distinguish. Generation information includes model version, prompt version, context length, final answer, citation sources, refusal strategy, generation latency, and safety policy hit status (Breck et al. 2017; Kreuzberger, Kühl and Hirschl 2023). For multimodal systems, it is also necessary to record image regions, bounding boxes, OCR text, table structures, and visual evidence IDs. Feedback information includes both explicit and implicit feedback. Explicit feedback includes upvotes, downvotes, ratings, user corrections, and text comments; implicit feedback includes follow-up questions, reformulations, citation clicks, answer copying, human handoffs, ticket submissions, and session abandonment. Governance information includes data desensitization status, log retention period, whether use for training is permitted, and whether the record enters the human review queue.

*Table 23-6: Core Fields of the Online Feedback Event Schema*

| Field Category | Core Fields | Description |
| --- | --- | --- |
| Session information | session_id, turn_id, timestamp, tenant, user_role | Used to reconstruct interaction context and permission scope |
| Query information | raw_query, normalized_query, intent, entities, missing_slots | Used to analyze user intent, entities, and missing conditions |
| Retrieval information | index_version, retrieved_docs, chunk_ids, scores, rerank_scores | Used to determine whether recall and ranking are correct |
| Context information | selected_context, citation_anchors, context_length | Used to retrospectively review the evidence actually seen by the model |
| Generation information | model_version, prompt_version, answer, refusal_flag, latency | Used to assess generation strategy and model behavior |
| Feedback information | rating, correction_text, clicks, follow_up, handoff_flag | Used to identify explicit and implicit feedback |
| Governance information | pii_status, training_allowed, review_status, retention_policy | Used to control compliance, review, and data use boundaries |

At the implementation level, the event schema should not only serve log storage but also downstream sample construction. That is, when designing fields, teams need to think ahead about how these data will eventually enter evaluation sets, failure sample repositories, knowledge update queues, and model training pipelines. For example, `index_version` and `prompt_version` may appear to be purely engineering fields, but they determine whether version attribution is possible after the fact; `citation_anchors` may appear to be only a display field, but it determines whether one can verify that the answer was based on correct evidence; `training_allowed` may appear to be only a compliance field, but it determines whether the data can enter subsequent training or fine-tuning pipelines.

![Figure 23-3: Online Feedback Event Collection and Routing Pipeline](../../images/part7/图23_3zh.png)

*Figure 23-3: Online Feedback Event Collection and Routing Pipeline*



---

### 23.2.5 Problem Routing: Missing Knowledge, Retrieval Defects, Generation Defects, and Strategy Defects

After feedback collection, the most critical task is problem routing. Problem routing means mapping user feedback to different system problem types and directing them to corresponding remediation queues. Without a routing mechanism, all feedback accumulates as vague "user dissatisfaction" and cannot drive effective improvement.

In RAG and large-model applications, common problems can be divided into four categories: missing knowledge, retrieval defects, generation defects, and strategy defects. Missing knowledge refers to a situation where the knowledge base lacks sufficient information to answer the user's question. This may be because relevant documents were never ingested, documents have expired, new business scenarios have not yet been supplemented, knowledge granularity is too coarse, or information exists in human expertise but has not been documented. The typical manifestation of missing knowledge is that the system cannot retrieve relevant evidence, or can only give a vague answer. Remediation actions typically include supplementing documents, updating FAQs, adding structured fields, introducing expert knowledge, or establishing a knowledge update process. Retrieval defects refer to situations where the correct answer exists in the knowledge base but the system did not recall it or did not rank it highly. This type of problem is very common in RAG systems. It may arise from mismatches between query expressions and document terminology, chunk splitting that destroys semantics, embeddings unable to cover specialized terminology, missing keyword indices, incorrect metadata filtering, or failed re-ranking. Remediation actions include adding synonyms, optimizing query rewriting, adjusting chunking strategy, supplementing metadata, improving hybrid retrieval, or adding re-ranking training samples. Generation defects refer to situations where correct evidence was retrieved, but the model generated a problematic answer. For example, the model ignores evidence, misreads evidence, over-infers, omits conditions, provides incomplete citations, or produces output in a style that does not meet requirements. Remediation actions for this type of problem typically include adjusting prompts, adding format constraints, introducing answer templates, strengthening citation requirements, adding refusal rules, or including failure samples in the generation evaluation set. Strategy defects refer to design problems in the system's policies around high-risk scenarios, permissions, refusals, follow-up questioning, and human handoffs. For example, a user's question lacks key conditions but the system does not ask for clarification; a user's request involves a high-risk judgment but the system directly provides a conclusion; certain users do not have permission to access certain documents but the system still recalls the relevant content; certain questions should be transferred to a human agent but the system continues to generate answers. These problems are typically not model capability failures but poorly designed product and safety policies.

The key to problem routing is establishing actionable judgment rules. Teams cannot simply label feedback with an "error" tag but must further determine: does correct knowledge exist? Was it recalled? Did it enter the context? Did permission or safety policies trigger? These judgments together constitute the minimum diagnostic chain of feedback routing. For example, for a user downvote sample, the following sequence can be followed: if the knowledge base lacks the correct content, classify as missing knowledge; if the knowledge base has the correct content but it was not recalled, classify as a retrieval defect; if it was recalled but did not enter the final context, classify as a ranking or context assembly problem; if the context is correct but the answer is wrong, classify as a generation defect; if the question lacks conditions but the system did not ask for clarification, classify as a strategy defect; if the answer is correct but the user is still unsatisfied, it may be a product experience problem.

---

### 23.2.6 Feedback Routing Queues and Operational Responsibility

Feedback routing is not only a technical classification problem but also an operational responsibility problem. Each category of problem should correspond to a clearly defined processing queue, responsible team, handling deadline, and acceptance criteria. Otherwise, even if the system can identify problems, it cannot drive remediation.

Missing knowledge is typically handled by content owners, domain experts, or knowledge base operations staff. They need to determine whether to supplement documents, update terms, add FAQs, or revise knowledge structures. Retrieval defects are typically handled by data engineering and retrieval engineering teams, focusing on chunks, indices, metadata, query rewriting, and re-ranking. Generation defects are typically handled by the model application team, involving prompts, context formatting, citation constraints, and generation strategies. Strategy defects require joint decision-making by product, business, compliance, and security teams, because they involve whether and how the system should answer and when to transfer to a human agent. Product experience problems are typically handled by product and frontend teams. At the organizational process level, making sample status, responsibility assignment, and acceptance criteria explicit also aligns with MLOps requirements for end-to-end responsibility chains: the quality of a model application depends not only on training or inference but also on the collaborative boundaries among data, deployment, monitoring, and operations teams (Amershi et al. 2019; Kreuzberger, Kühl and Hirschl 2023).

To improve processing efficiency, feedback queues should have priority levels. Priority can be determined jointly by impact scope, risk level, frequency of occurrence, and remediation cost. For example, high-frequency low-risk problems can be fixed in batches; low-frequency high-risk problems require immediate human review; low-frequency low-risk problems can enter a periodic optimization pool; high-frequency high-risk problems should trigger a special retrospective and version fix. In a mature online feedback system, every feedback sample should have a clearly defined status—for example: pending routing, pending review, pending fix, fixed, pending regression, deployed, and closed. This allows teams to track whether an online failure sample has truly entered the feedback loop rather than remaining on a problem list.

---

### 23.2.7 Section Summary

This section discussed the event collection and feedback routing mechanisms in the online feedback loop. For large-model applications, the feedback loop is first an event engineering problem. The system must completely record user inputs, retrieval results, context assembly, model outputs, citation sources, user behavior, and subsequent feedback in order to support subsequent error attribution and data backfilling.

Online feedback sources include logs, clicks, ratings, corrections, tickets, and human handoffs. Different feedback signals have different data values and noise characteristics and therefore need to be organized through a unified event schema. Explicit feedback is semantically more direct and is suitable as a failure sample entry point; implicit feedback has wider coverage and is suitable for trend monitoring and candidate sample filtering.

After feedback collection, the system needs to route problems into categories of missing knowledge, retrieval defects, generation defects, strategy defects, and product experience problems. Different problems correspond to different responsible teams and remediation actions. Only when every type of feedback can enter a clearly defined queue and undergo review, remediation, regression, and online validation can online feedback truly be transformed into a continuously improving data flywheel.

The next section will further discuss knowledge updates, rollbacks, and version governance, focusing on how new knowledge injection, outdated knowledge invalidation, conflict content governance, gray releases, and rapid rollback support the long-term stable operation of production-grade large-model applications.

---

## 23.3 Knowledge Updates, Rollbacks, and Version Governance

### 23.3.1 Why Knowledge Updates Must Be Managed as an Engineering Process

The ultimate goal of the online feedback loop is not simply to collect user problems or measure satisfaction, but to enable the system to continuously repair itself based on real usage. The most central type of remediation action is knowledge updates. For RAG systems, enterprise knowledge assistants, customer service bots, compliance Q&A systems, and multimodal document retrieval systems, the knowledge base is not a static asset but a dynamic asset that changes continuously with business, policy, product, process, and external environment.

In real applications, the complexity of knowledge updates comes primarily from three aspects. First, knowledge itself has a temporal dimension. Enterprise policies adjust, product features iterate, interface fields are deprecated, pricing rules change, organizational structures reorganize, and laws and regulations may also be updated. For this type of content, outdated knowledge is not simply "declining in value"—it may become incorrect information. If a RAG system continues to recall outdated knowledge, it will generate answers that appear to have evidence but are actually expired. This type of error is more dangerous than providing no answer, because users often trust an answer more when it comes with citations. Second, knowledge may conflict internally. A single system may simultaneously contain formal policies, historical policies, FAQs, meeting minutes, customer service scripts, product manuals, and temporary notices. These sources differ in their time, authority, and scope of applicability. If the system has no knowledge priority rules or conflict governance mechanisms, it may recall contradictory evidence within the same answer. For example, the formal policy specifies that a process requires three levels of approval, but an old FAQ still says two levels; the product documentation says a feature is only available in the enterprise version, but the customer service script, simplified for ease of communication, says "all users can use it." These conflicts cannot be resolved solely by model judgment but must be handled proactively by knowledge governance mechanisms. Third, knowledge updates affect system behavior. Adding a batch of documents, modifying metadata, rebuilding the index, or adjusting the chunking strategy can all change retrieval results. An update may fix one type of problem but break another type that was previously handled correctly. Knowledge updates must therefore be versioned, tested, gradually deployed, and monitored—like code releases—and must not be applied directly to the production knowledge base without controls.

From the perspective of dynamic knowledge systems, the advantage of RAG is that knowledge can be updated through external corpora, but this advantage holds only when the knowledge source, index, and generation strategy are governed in concert; otherwise, non-parametric memory likewise introduces the risk of expiration and conflict (Lewis et al. 2020; Mallen et al. 2023). Production-grade RAG systems must therefore upgrade knowledge updates from "document maintenance actions" to "engineering release processes." Every knowledge change should have a clear source, change description, responsible person, scope of impact, evaluation results, release time, and rollback strategy. Only in this way can the system remain stable through continuous updates rather than gradually losing control through frequent changes.

---

### 23.3.2 New Knowledge Injection: From Document Ingestion to Usable Knowledge Units

New knowledge injection is the most common scenario in knowledge updates. It may come from newly issued policies, newly released product documentation, new FAQ supplements, new financial report ingestion, new contract template imports, or knowledge gaps exposed by online feedback. Compared to initial knowledge base construction, new knowledge injection after deployment emphasizes incrementality, controllability, and verifiability.

A complete new knowledge injection process typically includes five steps: source validation, content parsing, structured processing, index updates, and online validation.

Source validation is the first step in knowledge injection. The system needs to record where the new knowledge comes from, whether usage rights exist, whether it is an official version, and whether it has been confirmed by business or legal teams. For enterprise internal systems, the authority of the document source is very important. Formal policies, approved product manuals, and FAQs confirmed by business owners should have higher priority; meeting minutes, temporary chat records, and personally compiled documents should be assigned lower weight and used only as supplementary references. Content parsing converts documents into machine-processable structures. For PDFs, Word documents, web pages, tables, images, and scanned files, parsing quality will directly determine subsequent retrieval effectiveness. Knowledge injection after deployment should not skip parsing quality checks, especially for complex documents involving tables, charts, headers and footers, footnotes, version numbers, and chapter hierarchies. If the parsing stage loses applicable conditions, numerical units, table structures, or document versions, even a successfully indexed result may produce incorrect answers. Structured processing converts parsed content into knowledge units, requiring supplementation of metadata such as source document, chapter path, publication date, effective date, expiration date, applicable subjects, permission scope, document version, knowledge type, and authority level. For production-grade RAG systems, metadata is not supplementary information but an important basis for retrieval, filtering, ranking, and conflict governance. Index updates write knowledge units into the retrieval system. Depending on the system architecture, this may involve vector indices, keyword indices, structured indices, graph databases, table indices, and multimodal indices. When performing incremental updates, particular attention should be paid to index consistency: whether new knowledge has completed all index type writes, whether old indices need to be deleted or downweighted, whether parent-child indices are updated in sync, and whether multimodal chart regions are consistent with textual descriptions. Online validation is the final step before new knowledge enters production. The system should use relevant question sets for regression testing, verifying that new knowledge can be correctly recalled, correctly cited, and correctly generated. If new knowledge was added to fix a specific online failure problem, the original failure query should be re-run to confirm whether the problem has been resolved.

*Table 23-7: Key Checkpoints in the New Knowledge Injection Process*

| Stage | Key Question | Inspection Focus | Common Risks |
| --- | --- | --- | --- |
| Source validation | Is the document trustworthy and usable? | Source, permissions, responsible person, approval status | Non-official documents entering the production knowledge base |
| Content parsing | Was the document read correctly? | Chapters, tables, charts, footnotes, version numbers | Table corruption, condition loss, OCR errors |
| Structured processing | Were retrievable knowledge units formed? | Chunks, metadata, scope of applicability, citation anchors | Missing version, effective date, or permission information |
| Index updates | Was the correct index written? | Vector, keyword, structured fields, multimodal indices | Index inconsistency, old content not taken offline |
| Online validation | Was the problem truly resolved? | Failure query replay, citation verification, regression evaluation | Local problem fixed but new errors introduced |

New knowledge injection should also distinguish between "supplementary updates" and "replacement updates." Supplementary updates add new content on top of the existing knowledge—for example, adding new FAQs, new cases, or new chart descriptions. Replacement updates mean that old knowledge is no longer applicable—for example, a policy version upgrade, a deprecated interface field, or a changed pricing rule. The two carry different risks. Supplementary updates primarily concern recall coverage and ranking; replacement updates must additionally address outdated knowledge invalidation and conflict governance.

---

### 23.3.3 Outdated Knowledge Invalidation and Conflict Content Governance

The most easily overlooked problem in knowledge updates is not that new knowledge was not added, but that old knowledge was not removed. In many systems, teams continuously add new documents to the knowledge base while rarely actively cleaning up old ones. The result is a continually growing knowledge base with increasingly chaotic retrieval results, and the model may simultaneously see new and old conflicting evidence during generation.

Outdated knowledge typically takes three forms: temporal invalidation, version invalidation, and scope invalidation. Temporal invalidation refers to knowledge that is only valid within a specific time period. For example, temporary notices, phased policies, promotional rules, quarterly reports, and project schedules all have clear time boundaries. Temporally invalidated knowledge does not necessarily need to be deleted, but must be correctly downweighted or filtered during the retrieval and generation stages. Version invalidation refers to a new version superseding an old version. For example, after Product Manual v3.0 is released, certain feature descriptions in v2.0 may no longer apply; after the 2025 edition of a policy is released, processes in the 2023 edition may be deprecated. If the system has no document version and effective status fields, it is very difficult to prevent old versions from being recalled. Scope invalidation refers to knowledge that remains valid but does not apply to the current user or scenario. For example, a reimbursement policy applies only to full-time employees, not interns; a product feature applies only to the enterprise version, not the individual version; an operation process applies only to domestic operations, not overseas branches. Scope invalidation is not knowledge expiration but rather applicable conditions not being correctly identified.

Knowledge conflicts can also be regarded as a data quality problem: when the same entity, rule, or metric appears inconsistently across different sources, the system must resolve the discrepancy through version, authority, and scope of applicability rather than handing conflicting evidence directly to the generation model to adjudicate on the fly (Breck et al. 2017; Gao et al. 2023). Conflict content governance requires joint handling at the knowledge, index, and generation levels. At the knowledge level, an authority level and scope of applicability need to be set for each piece of knowledge. Formal policies take precedence over FAQs; the latest version takes precedence over old versions; structured fields take precedence over informal descriptions; content confirmed by business owners takes precedence over user comments or meeting minutes. At the index level, the retrieval system needs to be able to identify time, version, permissions, and applicable subjects. For content that has already expired, options include deletion, archiving, downweighting, or making it visible only in historical query scenarios. For content that still needs to be retained but should not be recalled by default, its use scope should be controlled through metadata filters. At the generation level, the model needs to be instructed to recognize evidence conflicts. When the context contains multiple versions or mutually contradictory fragments, the model should not simply concatenate the answer but should select credible evidence based on version, date, and authority level—and if necessary, prompt the user that "the current knowledge base contains conflicts requiring human confirmation." However, this capability cannot rely entirely on the model's own judgment; upstream data governance remains key.

*Table 23-8: Strategies for Outdated Knowledge Invalidation and Conflict Governance*

| Problem Type | Typical Manifestation | Data Governance Strategy | System-Side Handling |
| --- | --- | --- | --- |
| Temporal invalidation | Temporary notices and quarterly rules still being recalled | Add effective date and expiration date fields | Filter or downweight based on query time |
| Version invalidation | New and old product manuals both matched | Establish version number and current version flag | Recall latest version by default, archive old versions |
| Scope invalidation | Policy inapplicable to user being cited | Tag applicable subjects, regions, roles, permissions | Metadata filter and permission filtering |
| Source conflict | FAQ inconsistent with formal policy | Establish authority level and source priority | Boost weight of authoritative content during re-ranking |
| Content conflict | Same metric, process, or rule contradicts itself | Establish conflict detection and human review queue | Prompt user or refuse to answer in conflict scenarios |
| Citation invalidation | Answer cites deleted or migrated document | Maintain citation anchors and document mappings | Citation validation and index rebuilding |

The difficulty of conflict content governance is that it often spans multiple teams. The content team is responsible for document updates, the business team for rule confirmation, the platform team for index mechanisms, the algorithm team for retrieval ranking, and the product team for display and notifications. Without a clear process, conflicting content can easily remain in the system for a long time. The knowledge base should therefore establish a regular inspection mechanism to conduct targeted reviews of high-frequency recalled documents, expired documents, low-trust sources, and content with significant user feedback.

---

### 23.3.4 Hot Updates, Scheduled Updates, and Audited Updates

Knowledge updates do not come in only one release pattern. Based on risk level, timeliness requirements, and scope of business impact, update modes can be divided into hot updates, scheduled updates, and audited updates.

Hot updates are appropriate for low-risk, high-urgency content, such as FAQ additions, typo corrections, minor description updates, and low-risk product tips. The advantage of hot updates is fast response, enabling quick fixes of online problems; the risk is that if automated validation is lacking, erroneous content can quickly enter production. Hot updates must therefore include at least basic checks, such as document format validation, parsing success validation, index write validation, and small-scale query replay.

Scheduled updates are appropriate for periodically changing knowledge, such as help center content synced daily, product documentation updated weekly, operational reports released monthly, quarterly financial reports, or policy packages. The advantage of scheduled updates is a stable cadence, which is convenient for batch validation and resource planning. They are typically suited to automatic execution via scheduling systems, generating an update report upon completion that includes the number of added documents, deleted documents, parsing failures, index updates, and regression evaluation results.

Audited updates are appropriate for high-risk content, such as compliance policies, financial rules, medical guidelines, legal terms, permission policies, contract templates, and critical business processes. Audited updates cannot be automatically deployed directly but require responsible party confirmation, expert review, regression evaluation, approval documentation, and gray release. For this type of knowledge, update speed is not the only goal—correctness, traceability, and rollback capability are more important.

![Figure 23-4: Knowledge Update, Gray Release, and Rollback Governance Process](../../images/part7/图23_4zh.png)

*Figure 23-4: Knowledge Update, Gray Release, and Rollback Governance Process*



In practice, all three update modes often coexist. The system can allow low-risk content to follow the hot update path, medium-risk content to follow the scheduled update path, and high-risk content to follow the audited update path. This requires the knowledge update platform to have risk classification capabilities: every update request should be automatically or semi-automatically graded based on knowledge type, source, scope of impact, and user group, and then enter the corresponding process. For example, wording optimization of a customer service FAQ can follow the hot update path; version synchronization of a batch of product documents can follow the scheduled update path; content involving contractual liability, financial approval, or medical advice must follow the audited update path. This layered mechanism can strike a balance between efficiency and safety, avoiding both the slowdown of all updates due to inefficient review and the entry of high-risk content into production without inspection.

---

### 23.3.5 Version Freezing, Gray Releases, and Rapid Rollback

The ultimate risk control for knowledge updates depends on version governance. Without version governance, three key questions cannot be answered: what version is the current production knowledge base? Which update introduced a particular erroneous answer? If an update goes wrong, can the system quickly revert to the last stable version?

Specifically, the main methods of version governance can be summarized as three types: version freezing, gray releases, and rapid rollback. Version freezing means fixing the state of the knowledge base, index, prompt, model configuration, and evaluation set at a given point in time to form a reproducible release version. For production-grade RAG systems, a version should not only record document folders but should record the complete dependency chain, including original document versions, parser version, chunking strategy, embedding model, index construction parameters, re-ranking model, prompt templates, and permission rules. Only then can a team accurately perform post-incident retrospectives when problems occur. Gray releases are an important way to reduce knowledge update risk. Knowledge updates should not always be released to all users at once. For important updates, the system can first make them available to internal test users, a small fraction of real users, or specific tenants, observing metrics such as retrieval hit rates, answer accuracy rates, citation click rates, negative feedback rates, and human handoff rates. If gray release performance is stable, the rollout can be gradually expanded. For multi-tenant enterprise systems, gray releases can also be performed by department, region, business line, or user role. Rapid rollback means restoring the system to the last stable version when an update goes wrong. Rollback capability requires the system to retain old version indices, old version knowledge packages, and old version configurations. If every update directly overwrites the production index without retaining snapshots, rollback becomes very difficult. A more mature approach is to use "blue-green indexing" or "dual-index release": first build the new index, then switch traffic after validation passes; if a problem occurs, immediately switch back to the old index. The value of version freezing and dual-index release lies in ensuring reproducibility. MLOps research typically emphasizes joint versioning of data, code, model, configuration, and runtime environment; for RAG, the knowledge package, chunking strategy, embedding model, and index construction parameters must also be included within the version boundary (Amershi et al. 2019; Kreuzberger, Kühl and Hirschl 2023).

Rollback applies not only to knowledge content but also to index strategies and generation strategies. For example, an update may not have modified documents but may have adjusted chunk granularity or re-ranking weights, causing online recall quality to decline; a prompt update may have improved answer completeness while reducing citation faithfulness. These changes should all be included in version management and the rollback scope. A reliable version governance system should record at minimum: version number, release time, change content, change source, responsible person, scope of impact, evaluation results, gray release scope, monitoring metrics, rollback point, and approval records. This information is used not only for incident handling but also for long-term retrospectives and team collaboration.

---

### 23.3.6 Section Summary

This section discussed knowledge updates, rollbacks, and version governance in the online feedback loop. For production-grade large-model applications, the knowledge base is not a static asset built once and for all but a dynamic system requiring continuous updating, continuous validation, and continuous governance. New knowledge injection must go through source validation, content parsing, structured processing, index updates, and online validation; outdated knowledge invalidation and conflict content governance must rely on metadata for time, version, scope, authority, and permissions. Knowledge update modes should be tiered by risk level: low-risk content can be hot-updated, medium-risk content is suitable for scheduled updates, and high-risk content must follow audited updates. At the same time, the system must establish version freezing, gray release, and rapid rollback mechanisms to ensure that every knowledge change is traceable, verifiable, and reversible. In the online feedback loop, knowledge updates are not isolated actions. They connect user failure feedback, error attribution, remediation tasks, regression evaluation, and online monitoring. Only when knowledge updates are incorporated into an engineering governance process can RAG systems remain reliable, controllable, and sustainably evolving in an ever-changing business environment.

The next section will further discuss metrics dashboards and operational cadences, focusing on how online success rate, human handoff rate, correction rate, knowledge hit rate, and other metrics can enter weekly operations meetings, specialized retrospectives, and major version upgrade cycles, thereby embedding the data feedback loop into teams' daily work.

---

## 23.4 Metrics Dashboards and Operational Cadence

### 23.4.1 Why the Online Feedback Loop Requires a Metrics Dashboard

If the online feedback loop remains only at the level of fixing individual failure samples, it easily falls into "reactive remediation operations." A user downvotes a problem; the team fixes that problem. A business reports a knowledge gap; the team supplements a document. One answer has a citation error; the team manually adjusts an index entry. This approach can respond quickly in the early post-deployment period, but as user scale grows, the knowledge base expands, and business scenarios multiply, point-by-point fixes gradually become ineffective. Teams need to upgrade from "handling problems" to "operating the system," and metrics dashboards are the fundamental tool for achieving this transformation.

The purpose of a metrics dashboard is not merely to display system runtime status but to transform online feedback into a management language that is observable, comparable, attributable, and decision-actionable. For production-grade RAG or large-model applications, whether the system is healthy cannot be judged solely by whether interfaces are available and latency is normal, nor only by whether user traffic is growing. More importantly: does the system actually answer user questions? Is it based on correct knowledge? Is it reducing the cost of human processing? Does it remain stable after knowledge updates? Does it adopt the correct strategy in high-risk scenarios?

Metrics dashboards in the online feedback loop should therefore cover four categories of metrics simultaneously: quality metrics, behavioral metrics, operational metrics, and risk metrics. Quality metrics concern whether answers are correct, whether citations are reliable, and whether retrieval hits. Behavioral metrics concern whether users adopt answers, whether they ask follow-up questions, and whether they transfer to human agents. Operational metrics concern problem processing efficiency, knowledge update cycles, and remediation loop progress. Risk metrics concern incorrect answers, high-risk erroneous responses, permission overreach, and outdated knowledge recall. Only when these metrics are presented together can a team judge whether the system is genuinely improving or only appearing to improve on specific metrics. For example, if a system's answer accuracy rises but human handoff rates also rise, this may indicate that the system performs better on low-risk questions but cannot handle complex ones; if a system's retrieval recall rate is high but citation click rates are very low, this may indicate that answers appear to have sources but users do not trust the citation display; if a system's user satisfaction rises but knowledge updates are severely lagging, this may mean short-term experience is fine but there is a future outdated-knowledge risk. The value of dashboards lies in making these tensions visible, preventing teams from being misled by single metrics.

---

### 23.4.2 Online Success Rate, Human Handoff Rate, Correction Rate, and Knowledge Hit Rate

As shown in Table 23-9, in the online feedback loop, the most commonly used set of core metrics includes online success rate, human handoff rate, correction rate, and knowledge hit rate. These respectively measure system performance from four angles: user outcomes, human costs, error exposure, and knowledge coverage. Metrics design typically avoids relying on single averages alone. Online experiment and search system evaluation literature typically emphasizes simultaneously observing success rates, user behavior, error costs, and long-tail impact, because the rise of a single metric may mask degradation in high-risk sub-scenarios (Kohavi, Tang and Xu 2020; Joachims et al. 2017).

Online success rate measures whether user questions are effectively resolved by the system. It can be estimated jointly from explicit and implicit feedback. For example, a user clicking "resolved," upvoting, copying an answer, not asking further follow-up questions, or ending the session after an answer can all serve as success signals. However, online success rate is not a simple binary metric, because "success" is defined differently across scenarios. For customer service Q&A, success may mean the user does not transfer to a human agent; for an enterprise knowledge base, success may mean the user clicked the correct citation and completed a subsequent operation; for compliance Q&A, success may not be directly providing a conclusion but correctly flagging the risk and directing to human review.

Human handoff rate measures the proportion of tasks the system cannot complete independently. It can reflect both the system's capability boundary and whether the strategy design is reasonable. In high-risk domains, an appropriate level of human handoffs is necessary; but if ordinary questions also frequently trigger handoffs, this indicates deficiencies in the knowledge base, retrieval, or generation strategy. Human handoff rate should be analyzed in combination with question type and should not simply be pursued as a number to minimize. In legal, medical, financial approval, and similar scenarios, a low handoff rate accompanied by a high error rate is itself a danger signal.

Correction rate measures the frequency with which users or human reviewers discover errors. It includes user-initiated corrections, errors found during expert review, and error annotations flowing back from tickets. A rising correction rate does not necessarily mean the system has gotten worse—it may also indicate that feedback entry points are easier to use, that users are more willing to provide feedback, or that the system is covering more complex scenarios. Therefore, correction rate needs to be analyzed jointly with question volume, question difficulty, the current online version, and knowledge update batches.

Knowledge hit rate measures whether user questions can be matched to valid content in the knowledge base. It focuses on knowledge-level coverage rather than just the recall performance of the retrieval algorithm. If a question has no answer in the knowledge base at all, even if the retrieval system performs normally, a reliable answer cannot be generated. A low knowledge hit rate typically indicates knowledge gaps, documents not ingested, expired documents, missing metadata, or user questions that exceed the system's scope.

*Table 23-9: Core Metrics for the Online Feedback Loop*

| Metric | Primary Meaning | Typical Computation | Primary Problems It Reveals |
| --- | --- | --- | --- |
| Online success rate | Whether user questions are effectively resolved | Resolved sessions / total sessions, or combined estimate from explicit and implicit feedback | Unusable answers, poor experience, incomplete tasks |
| Human handoff rate | Whether the system requires human intervention | Sessions transferred to human / total sessions | System capability boundary, overly strict strategy, or retrieval failure |
| Correction rate | Rate at which users or reviewers discover errors | Corrected samples / answered samples | Factual errors, citation errors, outdated knowledge |
| Knowledge hit rate | Whether valid evidence for the question exists in the knowledge base | Questions with valid evidence hits / total questions | Missing knowledge, documents not ingested, insufficient metadata |
| Citation accuracy | Whether answer citations support the conclusion | Correctly cited answers / total answers with citations | Citation misalignment, insufficient evidence, context assembly errors |
| Follow-up rate | Whether users need to continue asking follow-up questions | Sessions with follow-up questions / total sessions | Incomplete answer, unclarified conditions, unclear expression |

These metrics have complementary relationships. Online success rate is a results metric; human handoff rate is a cost-and-boundary metric; correction rate is an error exposure metric; knowledge hit rate is a knowledge asset metric. During system operations, teams should not focus on only one metric but should observe how they change in combination.

For example, if online success rate declines while knowledge hit rate also declines, prioritize investigating missing or expired knowledge; if knowledge hit rate is normal but answer accuracy declines, the problem may be in generation or context assembly; if answer accuracy is normal but follow-up rates rise, the problem may be that answers are not direct enough or lack operational steps; if human handoff rate suddenly rises, further analysis is needed to determine whether business complexity has increased or whether system policy has incorrectly triggered.

---

### 23.4.3 Weekly Operations Meetings, Specialized Retrospectives, and Major Version Upgrade Cycles

A metrics dashboard only drives genuine impact when it enters a stable operational cadence. Otherwise, the dashboard only passively displays data and cannot drive system improvement. For production-grade large-model applications, a three-tier operational cadence is recommended: weekly operations meetings, specialized retrospectives, and major version upgrade reviews.

Weekly operations meetings focus on routine system running status, primarily reviewing core metric changes, online failure samples, knowledge update progress, and pending problem queues. Their goal is not to deeply resolve every technical detail but to ensure that the team knows the system's most important current problems, which problems are being addressed, and which risks need escalation. Weekly operations meetings should typically include product, data engineering, algorithm, platform, and business content owners jointly, avoiding feedback being confined to a single team.

Specialized retrospectives address a category of problems that has been exposed in concentration. For example, if financial report Q&A errors have increased significantly in a given week, or if users have provided large volumes of feedback about inconsistent feature descriptions after a product version launch, a specialized retrospective is needed. Specialized retrospectives should focus on specific problem chains: how the user asked the question, what the system recalled, how the answer was generated, whether the citation was correct, why the user was unsatisfied, and whether the root cause belongs to missing knowledge, a retrieval defect, a generation defect, or a strategy defect. Retrospective conclusions must be translated into clear remediation actions rather than remaining at "future optimization."

Major version upgrade reviews address larger-scope knowledge base updates, index strategy adjustments, model version upgrades, or prompt system changes. Major version upgrades cannot rely only on development team self-testing but must go through regression evaluation, gray releases, online monitoring, and rollback plans. Especially for updates involving high-frequency knowledge, critical business processes, or high-risk response strategies, impact scope and acceptance criteria must be made explicit before deployment.

*Table 23-10: Operational Cadence for the Online Feedback Loop*

| Operational Mechanism | Frequency | Primary Inputs | Primary Outputs | Participating Roles |
| --- | --- | --- | --- | --- |
| Weekly operations meeting | Weekly | Metrics dashboard, top failure sample categories, knowledge update queue | Priority ordering, responsibility assignment, risk escalation | Product, data engineering, algorithm, business content, platform |
| Specialized retrospective | Problem-triggered | Certain high-frequency or high-risk failure samples | Root cause analysis, remediation plan, regression samples | Relevant owners, domain experts, algorithm and platform |
| Major version upgrade review | Pre-version release | Change description, regression evaluation, gray release results, rollback plan | Release decision, gray release scope, rollback point | Project lead, platform, business, compliance, quality owner |
| Monthly quality retrospective | Monthly | Trend metrics, cost metrics, user feedback summary | Periodic quality report, resource planning | Project management, product, data operations, technical lead |

The key to operational cadence is forming closed-loop records. Every meeting should clarify the problem, responsible person, deadline, acceptance metrics, and follow-up tracking approach. For example, "improve retrieval effectiveness" is not a qualified task; "supplement 30 synonym expressions for reimbursement-policy-type questions and raise Recall@5 for related failure samples from 62% to above 85%" is an executable task. Only when tasks are quantified by metrics, samples, and versions can operations meetings drive genuine improvement.

---

### 23.4.4 Responsibility Boundaries and Collaboration Mechanisms in the Feedback Loop

The online feedback loop typically involves multiple teams, so responsibility boundaries must be clearly defined. A user downvote may simultaneously implicate knowledge base content, retrieval strategy, model generation, product interaction, and permission control. Without a clear division of labor, this type of problem can easily circulate among teams and ultimately be owned by no one.

One effective approach is to bind feedback problems to responsible teams by root cause type. Missing knowledge is the responsibility of content or business owners; retrieval defects are the responsibility of data engineering and retrieval teams; generation defects are the responsibility of the model application team; strategy defects are jointly handled by product, business, and compliance; platform stability problems are the responsibility of the infrastructure team; product experience problems are the responsibility of the product and frontend teams. For problems where root cause cannot be determined in a single pass, they enter a joint retrospective queue led by the data operations lead.

The feedback loop also needs a problem state transition mechanism. From entering the queue to final closure, a feedback sample typically goes through states such as "pending routing, pending confirmation, pending fix, pending regression, pending deployment, and closed." Each state should have clearly defined entry conditions and exit conditions. For example, the "pending fix" state must include a root cause label and a remediation plan; the "pending regression" state must be bound to regression samples; the "closed" state must be supported by validation results or changes in online metrics as evidence.

Collaboration mechanisms should also address priority rules. Not all problems can be handled simultaneously, and teams need to rank them by impact scope, risk level, frequency of occurrence, and remediation cost. High-risk problems should be prioritized even if low-frequency; high-frequency low-risk problems are suitable for batch optimization; low-frequency low-risk problems can enter a long-term accumulation pool; high-frequency high-risk problems should trigger a dedicated incident-level response.

In addition, the feedback loop needs knowledge retention. Every problem retrospective should enter a case repository, including problem background, failure manifestation, root cause, remediation actions, deployment results, and lessons learned. After long-term accumulation, these cases will become the team's operational knowledge base, helping new members understand common system problems and guiding evaluation set design and deployment checklists in return.

---

### 23.4.5 Section Summary

This section discussed metrics dashboards and operational cadences in the online feedback loop. For production-grade large-model applications, dashboards are not simple data display tools but the infrastructure for transforming online feedback into operational decisions. Online success rate, human handoff rate, correction rate, knowledge hit rate, citation accuracy, and follow-up rate together constitute the system quality observation framework, helping teams identify missing knowledge, retrieval defects, generation defects, strategy defects, and experience problems.

Metrics must enter a stable operational cadence to drive continuous system improvement. Weekly operations meetings address daily issues and priority ordering; specialized retrospectives analyze high-frequency or high-risk failures; major version upgrade reviews control the risks of knowledge, index, model, and strategy changes. At the same time, the feedback loop must define clear responsibility boundaries, establish problem state transition mechanisms and priority rules, and prevent online feedback from circulating among teams without being resolved.

The maturity of the online feedback loop is ultimately reflected in whether the team can continuously transform real user questions into data assets, remediation tasks, evaluation samples, and knowledge updates. The next section will further discuss online knowledge update SOPs and post-incident case reviews, examining how these mechanisms are implemented in concrete production processes.

---

## 23.5 Case Reviews and SOPs

### 23.5.1 Why Online Knowledge Update SOPs Are Necessary

In the preceding sections, we discussed the online feedback loop, event collection, feedback routing, knowledge updates, version governance, and operational metrics. This section further grounds these mechanisms in actual production processes, discussing how to manage online knowledge updates and post-incident reviews through SOPs (Standard Operating Procedures).

For production-grade RAG systems, knowledge updates are not simply a matter of uploading documents or having an engineer manually rebuild an index. A seemingly minor knowledge change may affect the recall results, answer citations, and risk judgments for a large number of user questions. Without a standard process, teams can easily encounter the following types of problems: new knowledge goes live without validation, causing erroneous content to enter production; old knowledge is not taken offline in time, causing new and old version conflicts; index updates occur without regression testing, causing previously correct questions to start failing; a problem occurs but change sources were not recorded, making it impossible to determine which update introduced the error. The core goal of an online knowledge update SOP is therefore to transform "knowledge changes" into engineering actions that can be approved, executed, verified, and rolled back. It needs to specify who is responsible at each stage, what the inputs are, what the outputs are, what the acceptance criteria are, and how to handle anomalies.

A complete online knowledge update SOP typically includes at least seven stages: change submission, source validation, parsing and structuring, conflict detection, index update, regression evaluation, and gray release with monitoring. For low-risk knowledge, the process can be condensed; for high-risk knowledge, human approval, compliance checks, and rollback plans must be added.

*Table 23-11: Online Knowledge Update SOP*

| Stage | Input | Key Actions | Output | Acceptance Criteria |
| --- | --- | --- | --- | --- |
| Change submission | New documents, change description, feedback samples | Fill in change source, impact scope, and responsible person | Knowledge change record | Change rationale is clear, responsible person is identified |
| Source validation | Document source, permission information, approval records | Validate credibility, legality, and applicability | Source validation result | Source is trustworthy, permissions are compliant |
| Parsing and structuring | Original document, parsing configuration | Extract chapters, tables, charts, metadata | Structured knowledge units | Content is complete, metadata is comprehensive |
| Conflict detection | New and old knowledge units | Check version conflicts, temporal conflicts, scope conflicts | Conflict report | High-risk conflicts have been resolved |
| Index update | Knowledge units, index configuration | Update vector index, keyword index, structured index | New index version | Index construction succeeded, traceable |
| Regression evaluation | Golden set, failure sample set, specialized set | Test recall, citation, answer correctness | Evaluation report | Metrics reach threshold, no critical regressions |
| Gray release | New index version, gray release strategy | Small-traffic deployment with feedback monitoring | Gray release results | No obvious quality degradation |
| Full release / rollback | Gray release results, monitoring metrics | Expand traffic or revert to old version | Release record | Deployment succeeded or safe rollback completed |

In practice, the SOP should not be designed to be overly cumbersome, or teams will bypass the process; nor should it be too coarse, or risk cannot be controlled. A workable approach is to execute different processes by risk tier. For example, low-risk FAQ updates can automatically complete parsing, indexing, and small-scale regression; when content involving finance, contracts, medical care, or compliance is involved, expert review and gray validation must be required.

---

### 23.5.2 Case Study: Knowledge Expiration Leading to Incorrect Answers

The following uses an enterprise internal policy Q&A system as an example to show how online problems arise when there is no knowledge update feedback loop, and how to fix them through the SOP.

An enterprise deployed an internal knowledge assistant to answer employee questions about reimbursement, leave, procurement, and approval processes. The system performed well initially, able to answer most standard questions. Two months after deployment, the finance department published a new version of the business travel reimbursement policy, adjusting accommodation standards, transportation allowances, and approval authority. However, the old policy version remained in the knowledge base; although the new policy was uploaded to the document system, it had not completed structured parsing and index updates. A few days later, employees began asking: "If I travel to a tier-1 city, what is the maximum accommodation reimbursement?" The system recalled the accommodation standards from the old policy and generated an answer with citations. Because the answer included source links, users believed it to be trustworthy and submitted reimbursement claims according to the old standards. Finance reviewers found the standards inconsistent and subsequently reported the problem to the system operations team. The retrospective revealed that the error was not a simple model hallucination but a classic knowledge expiration problem. The document the system recalled genuinely existed, but it was no longer the currently valid version; the new policy had already been released but had not entered the production index; the old policy had no expiration date or version status field, so the retrieval system could not determine that it should be downweighted or filtered.

This case exposed three data engineering problems: first, the knowledge base lacked management of effective and expiration dates; second, new policy publication did not trigger an index update; and third, the system did not run regression evaluations for high-frequency policy questions. Remediation actions should also focus on these three problems rather than simply correcting a single answer.

*Table 23-12: Error Attribution and Remediation Actions for the Knowledge Expiration Case*

| Problem Manifestation | Root Cause Classification | Specific Cause | Remediation Action |
| --- | --- | --- | --- |
| System cited old policy version | Knowledge version governance defect | Old document had no expiration status | Add version number, effective date, and expiration date to policy documents |
| New policy not recalled | Index update defect | New document did not enter production index | Trigger parsing, structuring, and index rebuild for new policy |
| User unable to judge version | Citation display defect | Answer did not display policy version and effective date | Display version number and effective date in citations |
| Similar problems not detected in advance | Regression evaluation defect | Lack of specialized evaluation set for policy questions | Build a business travel policy regression sample set |
| No validation after fix | Operational loop defect | No gray release monitoring or acceptance metrics | Gray deploy and monitor negative feedback rate for reimbursement questions |

This case illustrates that online incidents are often not the result of a single module failing, but the result of poor knowledge lifecycle management. The correct remediation approach is not merely to delete the old answer, but to systematize "policy version governance," including metadata completion, index updates, citation display, evaluation set expansion, and monitoring dashboard updates. It also demonstrates that RAG citations do not automatically equal trustworthiness: if citations point to expired documents or incorrect versions, the system actually amplifies the risk of outdated knowledge in a more authoritative form, so citation accuracy and knowledge timeliness must be evaluated jointly (Mallen et al. 2023; Es et al. 2024).

---

### 23.5.3 Automated Backfilling of High-Value Feedback Samples

In the online feedback loop, not every piece of user feedback is worth entering into human review. A production system may generate large volumes of logs, ratings, clicks, and follow-up behaviors every day; if all of them are handled manually, the cost will be very high. The system therefore needs to automatically identify high-value feedback samples and prioritize them for entry into the review and backfill pipeline.

High-value feedback samples typically have the following characteristics: they involve high-frequency questions, come from high-risk scenarios, are accompanied by strong negative feedback, contain user correction text, triggered a human handoff, are related to recent knowledge updates, or represent the same type of failure appearing in concentration over a short period. The system can compute a priority score for each feedback sample to determine whether it should enter human review, the knowledge update queue, or evaluation set expansion. Filtering this type of sample is analogous to the sample selection problem in active learning and online learning: the system should prioritize samples with high uncertainty, high risk, high frequency, or those that can represent an error cluster, rather than processing all logs uniformly (Settles 2009; Kohavi, Tang and Xu 2020).

The following is a simplified example of feedback sample priority scoring. It is not complete production code but demonstrates how to combine explicit feedback, implicit behavior, risk level, and knowledge update status into an interpretable filtering rule.

```python
from dataclasses import dataclass

@dataclass
class FeedbackEvent:
    query: str
    explicit_negative: bool = False      # Whether the user downvoted, reported an error, or explicitly indicated unhelpfulness
    has_correction: bool = False         # Whether the event includes user correction text
    followup_count: int = 0              # Number of subsequent follow-up turns
    reformulated: bool = False           # Whether the question was reformulated and resubmitted
    human_handoff: bool = False          # Whether the session was transferred to a human agent
    risk_level: str = "low"              # low / medium / high
    related_to_recent_update: bool = False
    frequency_7d: int = 1                # Number of occurrences of this question type in the past 7 days


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
    query="Under the new business travel policy, can tier-1 city accommodation still be reimbursed at the old standard?",
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

This code embodies an important principle: the value of a feedback sample is not determined solely by whether it was downvoted, but should be judged by multiple signals together. A low-risk question, even if downvoted, may not need immediate attention; a high-risk question, even if it appears only once, may require expert review; if a question appears at high frequency over a short period, it may represent a knowledge gap or systemic retrieval failure.

In production systems, this type of rule is typically further combined with model classifiers. For example, the system can first use rules to filter out clearly high-value samples, then use a classification model to determine their root cause type—such as missing knowledge, retrieval defect, generation defect, strategy defect, or product experience problem. Rules provide interpretability; models provide scalability; the combination is more suitable for complex online environments.

---

### 23.5.4 Example Online Knowledge Update SOP

Combining the preceding case and automated backfill mechanism, a relatively complete online knowledge update SOP can be formed. The following presents a simplified version suitable for enterprise RAG systems.

Step one: the system discovers failure samples from user feedback, tickets, and logs, and based on priority rules, they enter corresponding queues. High-risk problems should immediately enter the expert review queue; ordinary problems can enter the sampled retrospective or periodic processing pool.

Step two: operations staff or the system automatically performs a preliminary root cause determination. The problem is categorized as missing knowledge, knowledge expiration, retrieval defect, generation defect, or strategy defect. If it cannot be determined, the complete event chain is retained and enters a joint retrospective.

Step three: if the problem belongs to the knowledge update category, a knowledge change record is created. The change record should include the problem source, user query, erroneous answer, erroneous citation, correct knowledge source, responsible person, risk level, and expected deployment time.

Step four: the knowledge owner supplements or revises the document and performs source validation. For high-risk content, confirmation by domain experts or compliance owners is required. After confirmation, the document enters the parsing and structuring pipeline.

Step five: the system completes conflict detection and index updates. If conflicts between new and old knowledge are detected, the authoritative source and scope of applicability should be confirmed first, rather than allowing both pieces of content to enter the production index simultaneously.

Step six: run regression evaluation. The evaluation set should include at minimum the original failure samples, similar question samples, golden set samples, and high-risk boundary samples. Only after key metrics reach thresholds can the update proceed to gray release.

Step seven: gray deploy and monitor metrics. If negative feedback rates, human handoff rates, or citation error rates rise abnormally, immediately roll back; if gray release performance is stable, expand traffic and finally perform a full release.

Step eight: close the feedback record and archive the retrospective. The retrospective record should document the root cause, remediation actions, affected version, deployment results, and preventive measures for the future. These records will become important inputs for subsequent operations meetings and the case repository.

The value of this SOP is that it transforms every online failure into a complete data feedback loop. Failures are no longer only user experience problems but become inputs for knowledge updates, evaluation enhancement, and system governance. As this process operates continuously, the system can not only fix individual problems but also continuously improve robustness against similar problems.

---

### 23.5.5 Section Summary

This section used an online knowledge expiration case to discuss how the online feedback loop is implemented as a concrete SOP. In production-grade RAG systems, many errors are not random model hallucinations but the combined result of failures in knowledge version management, index updates, citation display, and regression evaluation. Post-incident reviews must therefore go beyond "the answer was wrong" and locate the specific stage in the knowledge lifecycle where the failure occurred.

To improve feedback processing efficiency, the system needs to automatically identify high-value feedback samples and rank them by priority based on signals including explicit negative feedback, user corrections, follow-up behavior, human handoffs, risk level, recent updates, and question frequency. By combining rules and models, the system can transform large volumes of online feedback into manageable review queues, remediation tasks, and evaluation samples.

The core of the online knowledge update SOP is organizing failure sample collection, root cause analysis, knowledge changes, structured processing, index updates, regression evaluation, gray releases, and retrospective documentation into a stable process. Only when these actions form a institutionalized feedback loop can large-model applications move from "deployable" to "reliably operational over the long term."

---

## Chapter Summary

This chapter treats system deployment as the starting point of the data flywheel and demonstrates that long-term reliability depends on the system's ability to continuously discover and fix problems through real usage. Toward this goal, the chapter presents a design for event collection and feedback routing that transforms online signals—logs, clicks, ratings, corrections, and human handoffs—into failure samples and improvement leads that are governable and attributable; it explains how knowledge updates, rollbacks, and version governance enable changes to be written safely while preserving existing capabilities, making every knowledge base modification traceable and reversible.

At the operational level, the chapter establishes metrics dashboards and operational cadences that converge dispersed feedback actions into a stable cycle of discovery, attribution, updating, and validation; it further uses post-incident reviews and SOPs to deposit handling experience as reusable operational procedures. These methods together support the continuous improvement of large-model applications as they transition from deployment to long-term operation. At this point, the core stages of application-level data engineering have been fully presented.

## References

Amershi S, Begel A, Bird C, DeLine R, Gall H, Kamar E, Nagappan N, Nushi B, Zimmermann T (2019) Software Engineering for Machine Learning: A Case Study. In: Proceedings of the 41st International Conference on Software Engineering: Software Engineering in Practice, pp 291–300.

Breck E, Cai S, Nielsen E, Salib M, Sculley D (2017) The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction. In: Proceedings of the IEEE International Conference on Big Data, pp 1123–1132.

Chapelle O, Zhang Y (2009) A Dynamic Bayesian Network Click Model for Web Search Ranking. In: Proceedings of the 18th International Conference on World Wide Web, pp 1–10.

Es S, James J, Espinosa-Anke L, Schockaert S (2024) RAGAS: Automated Evaluation of Retrieval Augmented Generation. In: Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, pp 150–158.

Gama J, Žliobaitė I, Bifet A, Pechenizkiy M, Bouchachia A (2014) A Survey on Concept Drift Adaptation. ACM Computing Surveys 46(4):1–37.

Gao Y, Xiong Y, Gao X, Jia K, Pan J, Bi Y, Dai Y, Sun J, Wang M, Wang H (2023) Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv preprint arXiv:2312.10997.

Hu Y, Koren Y, Volinsky C (2008) Collaborative Filtering for Implicit Feedback Datasets. In: Proceedings of the 2008 IEEE International Conference on Data Mining, pp 263–272.

Huyen C (2022) Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications. O'Reilly Media.

Joachims T (2002) Optimizing Search Engines Using Clickthrough Data. In: Proceedings of the Eighth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp 133–142.

Joachims T, Swaminathan A, Schnabel T (2017) Unbiased Learning-to-Rank with Biased Feedback. In: Proceedings of the Tenth ACM International Conference on Web Search and Data Mining, pp 781–789.

Koh P W, Sagawa S, Marklund H, Xie S M, Zhang M, Balsubramani A, Hu W, Yasunaga M, Phillips R L, Gao I, Lee T, David E, Stavness I, Guo W, Earnshaw B A, Haque I S, Beery S, Leskovec J, Kundaje A, Pierson E, Levine S, Finn C, Liang P (2021) WILDS: A Benchmark of in-the-Wild Distribution Shifts. In: Proceedings of the 38th International Conference on Machine Learning, pp 5637–5664.

Kohavi R, Tang D, Xu Y (2020) Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing. Cambridge University Press.

Kreuzberger D, Kühl N, Hirschl S (2023) Machine Learning Operations (MLOps): Overview, Definition, and Architecture. IEEE Access 11:31866–31879.

Lewis P, Perez E, Piktus A, Petroni F, Karpukhin V, Goyal N, Küttler H, Lewis M, Yih W-t, Rocktäschel T, Riedel S, Kiela D (2020) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In: Advances in Neural Information Processing Systems 33, pp 9459–9474.

Mallen A, Asai A, Zhong V, Das R, Khashabi D, Hajishirzi H (2023) When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories. In: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, pp 9802–9822.

Sculley D, Holt G, Golovin D, Davydov E, Phillips T, Ebner D, Chaudhary V, Young M, Crespo J-F, Dennison D (2015) Hidden Technical Debt in Machine Learning Systems. In: Advances in Neural Information Processing Systems 28, pp 2503–2511.

Settles B (2009) Active Learning Literature Survey. University of Wisconsin–Madison Computer Sciences Technical Report 1648.

Yu H, Gan A, Zhang K, Tong S, Liu Q, Liu Z (2024) Evaluation of Retrieval-Augmented Generation: A Survey. arXiv preprint arXiv:2405.07437.
