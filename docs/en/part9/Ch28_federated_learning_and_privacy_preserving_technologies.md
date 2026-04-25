---

# Chapter 28: Federated Learning and Privacy Protection Technology

---

## Summary of this chapter

In the previous chapter, we discussed the data compliance framework and governance, clarified the engineering importance of "compliance shift left" and Privacy by Design, and established a system baseline based on RoPA, DPIA, and classification and grading. However, in the face of highly sensitive data (C3 level) or cross-organizational data islands, relying solely on institutional "data availability and invisibility" or traditional access control and desensitization often cannot completely block the physical risk of data leakage. As machine learning systems, especially large model systems, continue to penetrate into the core business processes, data exposure no longer only occurs in traditional links such as database export, report display, or manual query, but begins to appear in feature construction, parameter training, joint modeling, and inference calls.

This chapter further advances the perspective from "system and process governance" to "technical governance in the training, collaboration and inference stages", and systematically introduces Federated Learning (Federated Learning, FL) and Privacy Enhancing Technologies (PETs). We not only discuss "what" these technologies are, but also focus on analyzing why they must enter the early stage of architecture design, and how to combine, verify and govern them in real engineering systems. In other words, this chapter is not concerned with a "technical glossary" at the conceptual level, but with a practical issue: how organizations should redesign collaborative training systems when data cannot flow freely.

This chapter first explains why post-event encryption and post-desensitization are often insufficient in machine learning, especially in large model scenarios, and further reveals the structural contradiction between data availability and privacy protection. Subsequently, we will build a complete technical panorama and compare the protection objects, applicable stages, system costs and combination boundaries of federated learning, differential privacy (DP), secure multi-party computation (MPC), trusted execution environment (TEE) and homomorphic encryption (HE). On this basis, this chapter will further expand on the key issues of project implementation: how to evaluate accuracy loss, communication overhead and delay increase, how to design the components and control flow of the federated training system, how to understand the applicable prerequisites of horizontal federation, vertical federation and federated fine-tuning, and how to verify the real performance of the privacy system through member inference, gradient inversion, model poisoning and backdoor testing.

Finally, this chapter will combine two cross-subject collaboration cases of medical care and finance, and connect them with the P09 privacy protection pipeline, Ch27 compliance governance framework, and Ch22 multi-modal visual retrieval capabilities to show how privacy protection technology can form a closed loop of "governance-training-verification-auditing" in complex AI systems. The focus of P09 is to form a front-end governance link through classification, permissions, desensitization, isolation, auditing, pre-inspection and accident review, and this chapter discusses how to continue to ensure the privacy boundaries between the training stage and the model stage when facing cross-subject modeling after the front-end governance is completed.

---

## 28.0 Learning Objectives

After studying this chapter, readers should be able to:

* Understand why privacy protection cannot just stay at the institutional governance and field desensitization layer, but must enter the machine learning system and model training architecture in advance.
* Master the basic principles, protection objects, applicable stages and main costs of the five core privacy protection technologies (FL, DP, MPC, TEE, HE).
* Distinguish the three implementation modes of horizontal federation, vertical federation and federated fine-tuning, and understand their prerequisite differences in data distribution and collaboration relationships.
* Understand the main attack surfaces in federated learning, including membership inference, gradient inversion, model poisoning and backdoor attacks.
* Establish privacy technology selection capabilities and be able to make engineering trade-offs between accuracy, latency, bandwidth, explainability and compliance pressure.
* Master the verification ideas of privacy enhancement systems, and understand why "introducing privacy technology" does not mean "really passing privacy verification."
* Understand how the technology in this chapter works with the P09 privacy-preserving data pipeline, Ch27 compliance governance system, and Ch22 multi-modal retrieval architecture.

---

## Scene introduction

Two top tertiary hospitals (Hospital A and Hospital B) and a top AI research institution (Institution C) plan to jointly train a large multi-modal auxiliary diagnosis model for rare diseases. Hospital A has a large amount of high-value clinical text medical records, and Hospital B has corresponding medical imaging and gene sequencing data. All three parties believe that this cooperation has obvious social value: the number of cases seen by a single hospital is limited, and rare disease samples are naturally scarce. If a joint sample space cannot be formed across institutions, it will be difficult to improve the generalization ability of the model.

However, the project was stopped by both the legal and security departments during the DPIA (Data Protection Impact Assessment) stage in the early stages of project establishment. The reason is very clear. First, these data are medical and health C3 highly sensitive data and are absolutely not allowed to leave the hospital intranet. Second, even if hospital A and hospital B remove direct identifiers such as patient names and ID numbers locally, if research institution C jointly encodes multi-modal features, it may still encounter member inference or feature inversion attacks through the alignment relationship between visual entities and text entities, and reversely restore the patient's identity. Third, in cross-agency collaboration, no party is willing to bear the institutional responsibilities and technical risks brought about by "centralized aggregation of raw data."

Faced with the double attack of "data islands" and "strong compliance supervision", the traditional paradigm of "summarizing data into a data lake for centralized training" has completely failed. The team must find an architectural solution that can achieve joint optimization of models without sharing raw data. This is the realistic background for the emergence of federated learning and privacy-enhancing technologies: data no longer flows freely, and collaboration must be reorganized through protocols, algorithms, and system boundaries.

![Figure 28-1: Diagram of privacy and compliance conflicts in cross-institutional medical data collaboration](../../images/part9/图28_1_跨机构医疗数据协作中的隐私与合规冲突示意图.png)
*Figure 28-1: Schematic diagram of privacy and compliance conflicts in cross-institutional medical data collaboration*

---

## 28.1 Why privacy protection must be front-loaded into architectural design

As we highlighted in Ch27, compliance costs can rise rapidly as the project life cycle progresses, so it is imperative to persist with “compliance shift left.” In terms of underlying technology implementation, privacy protection also follows this iron law. For AI systems, the later you consider privacy, the easier it is to leave irreversible structural flaws in data access, feature engineering, training processes, and inference interfaces. Many teams only focus on model effects and training throughput in the early stages of the project, and regard privacy issues as a secondary matter that "will be fixed after the model is run through." As a result, it is often discovered that the system architecture itself cannot meet regulatory requirements after entering the acceptance, launch, or compliance review stages.

This problem can be partially alleviated in traditional business systems by adding field permissions, adding logs, adding approval, or adding desensitization. However, in machine learning systems, once the risk enters the training process, it is difficult to completely eliminate it through post-processing means. The reason is that the model itself compresses the statistical rules, long-tail sample characteristics and even individual extreme information in the training data into the parameter space. In other words, data exposure is no longer just a question of who exported a table, but a question of whether the model has learned things that should not be remembered.

### 28.1.1 Why post-encryption and post-desensitization are often not enough

Traditional security thinking emphasizes database encryption, transmission encryption and front-end display desensitization. These measures are certainly important in traditional business systems, but are not sufficient in machine learning, especially large model scenarios.

The first is the memory effect of the model. Deep models, especially models with large parameter sizes, tend to memorize long-tail samples in the training set. If some training samples themselves are highly sensitive, have low recurrence frequency, and are unique in form, the model may retain enough information in the parameters, allowing an attacker to indirectly recover the private fragments in the training samples through prompts, API probing, or targeted sampling of the output.

The second is the **reversible problem of high-dimensional features**. Many teams think that "removing names, mobile phone numbers, and ID numbers" means security, but this only removes direct identifiers. For image embeddings, text vectors, multi-modal alignment features, gradient information, or intermediate representations, it is still possible for an attacker to reconstruct part of the original content through inversion methods. In other words, surface field desensitization does not automatically equal model layer privacy security.

The third is that the training stage is more dangerous than the display stage. In many organizations, security concerns still remain on "whether the front-end page is coded" and "whether the mobile phone number is removed from the exported table." However, the real high risks are often the data loading, feature caching, sample splicing, log printing, intermediate result storage and model update synchronization process during the training phase. The model has been leaked before it goes online, and it does not necessarily have to wait until the front-end display step before problems occur.

Therefore, privacy protection cannot be just a "surface shielding" after data enters the system, but must become part of the training and collaboration mechanism itself.

### 28.1.2 Structural contradiction between data availability and privacy protection

Privacy protection and data availability are not simple antagonistic relationships, but a long-standing structural tension. If the system pursues extreme privacy, the most direct way is to completely cut off the data connection, extremely restrict the query, or inject a large amount of noise into the training and output, but this will quickly damage the model's recognition ability, regression performance, and business availability. On the contrary, if the system pursues extreme availability and allows full plaintext sharing and unconstrained joint modeling across entities, the model effect may be better, but compliance risks and liability exposure will be simultaneously amplified.

The task of architectural design is not to eliminate this contradiction, but to find the "Minimum Necessary Exposure Surface" under specific business scenarios. The so-called minimum available exposure surface is to compress the visible data range, exchangeable information types, saveable intermediate results and exposeable model capabilities to the minimum on the premise of meeting the business goals. This goal determines that privacy enhancement technology is never the icing on the cake of "it's better to have it, but it's okay without it", but a prerequisite for the existence of many cross-subject AI systems.

![Figure 28-2: Schematic diagram of the structural contradiction between data availability and privacy protection](../../images/part9/图28_2_数据可用性与隐私保护的结构性矛盾示意图.png)
*Figure 28-2: Schematic diagram of the structural contradiction between data availability and privacy protection*

### 28.1.3 The concept of “privacy budget” in engineering systems

Differential Privacy (DP) provides an engineering quantifiable expression for this trade-off, namely the privacy budget. Privacy budget is usually expressed as $\epsilon$. It is not a budget in a financial sense, but an upper bound measurement of the system's allowed leakage risk. Generally speaking, the smaller $\epsilon$ is, the stronger the protection is, but the loss of accuracy and usability is usually greater; the larger $\epsilon$ is, the smaller the noise is and the model is more useful, but the risk of leakage will also be significantly increased.

In engineering systems, privacy budget is not an abstract mathematical constant, but a governance variable that must be incorporated into training rounds, number of queries, experiment frequency, and external service quotas. If a team only writes epsilon in a paper but has no mechanism to record the budget consumed for each training, each evaluation, and each external release, then the system does not truly manage privacy risks in practice. In other words, the budget management capability itself is part of the privacy system.

### 28.1.4 Paradigm shift from “data security” to “training security”

In traditional data governance, the security focus is usually "who accessed the database", "which fields were exported" and "which reports were shared". But in AI systems, the focus of risk begins to shift. What organizations need to care about is not only whether the data is seen, but how the data is represented, how the gradients are uploaded, how the parameters are aggregated, how the model is debugged, whether the output may leak training samples, and whether an attacker can deduce the original individual information by interacting with the model.

This means that the security goal shifts from "protecting the database" to "protecting the training process, collaboration boundaries and model behavior". All subsequent technical routes in this chapter, including FL, DP, MPC, TEE and HE, actually serve this paradigm shift: when the data itself cannot be moved freely, the system must re-establish security on the training protocol, aggregation logic and output boundaries.

![Figure 28-3: Governance focus migration diagram from data security to training security] (../../images/part9/图28_3_从数据安全到训练安全的治理重心迁移图.png)
*Figure 28-3: Governance focus migration diagram from data security to training security*

---

## 28.2 Panorama of technical roadmap

To solve the problem of privacy computing, the industry has currently evolved into five major technical schools. They are not mutually exclusive, but have different emphasis on protection objects, calculation loss and applicable scenarios. The key to understanding these technologies is not to memorize definitions, but to see clearly what they protect, what they sacrifice, and which stage of the system they are suitable for placement.

![Figure 28-4: Panoramic matrix diagram of privacy enhancement technology](../../images/part9/图28_4_隐私增强技术全景矩阵图.png)
*Figure 28-4: Panoramic matrix diagram of privacy enhancement technology*

### 28.2.1 Analysis and comparison of core technologies

| Technical schools | Core principles | Protection objects | Applicable stages | Implementation costs and main bottlenecks |
| :--------------- | :-------------------------- | :------------- | :------------- | :-------------------------- |
| **Federated Learning (FL)** | The data does not move and the model does not move; each node trains locally and only interacts with gradients or parameters.      | The original training data does not go out of the domain directly.   | Model training and fine-tuning | High communication overhead; risk of gradient leakage; sensitive to node heterogeneity.   |
| **Differential Privacy (DP)** | Inject noise into the data, gradients, or output to mask individual contributions.         | Protect whether the individual is participating in training or inquiry. | Training, statistical release, federated aggregation | Accuracy is impaired; budget management is complex; parameter adjustment is difficult.         |
| **Secure Multi-Party Computation (MPC)** | Through protocols such as secret sharing, multiple parties can complete joint calculations without exposing inputs. | Input data and intermediate results.     | Joint statistics, joint intersection, and joint risk control | There are many communication rounds and large delays; it is not suitable for large-scale in-depth training.     |
| **Homomorphic Encryption (HE)** | Perform operations directly on the ciphertext, and obtain the same result as the plaintext after decryption.        | Data content during calculation.    | Secure reasoning, secure aggregation | Extremely high computing power consumption; limited supported operators.            |
| **Trusted Execution Environment (TEE)** | Execute code and process data in a trusted enclave of hardware.             | Runtime memory data and key logic.  | Secure aggregation, sensitive reasoning, key management | Rely on specific hardware; there are side channel risks; the root of trust is external to the hardware manufacturer. |

These five types of technologies can be roughly divided into two groups: one group emphasizes "**data does not go out of the domain or is not seen in plain text**", such as FL, MPC, and HE; the other group emphasizes "**even if observations occur, reduce individual identifiableness or runtime visibility**", such as DP and TEE. Such a division helps to understand why they are often not substitution relationships, but combination relationships.

### 28.2.2 Technology combination route and scenario selection principles

In real projects, a single technology often cannot solve all problems, so a combined route is more common.

The most typical is **FL + DP**. Federated learning solves the problem of "the original data does not leave the domain", and differential privacy solves the problem of "uploading updates or model output may still expose individual information." This combination is very suitable for scenarios where multiple parties jointly train but the risk of member inference must be controlled. The second typical combination is **FL + HE**. After the client is trained locally, it uploads the encrypted updates to the center for aggregation. The center cannot see the plaintext gradient, thereby reducing the risk of observation on the center side. The third category is **TEE + FL**, which puts the aggregation logic into a trusted execution environment to reduce the possibility of the cloud side or host system snooping on intermediate training results. For problems such as financial list intersection and joint anti-fraud, a common combination is **MPC + PSI**. The focus is not on training large models, but on safely completing multi-party set operations and joint statistics.

Three questions should be answered first when selecting. First, who is the object of protection, whether it is the original data, individual identity, runtime memory, or the calculation result itself. Second, who is the attacker? Is it a central platform, a cooperative organization, an external attacker, or a black-box adversary with model query capabilities. Third, what is the most unbearable price for the system, whether it is reduced accuracy, increased latency, bandwidth consumption, or increased implementation complexity. Only by clarifying these three issues can the technical route have practical significance.

---

### 28.2.3 Federated Learning (FL) Deep Digging

Federated learning is not just as simple as “keeping data locally”, it is essentially a mechanism to reorganize training control. In traditional centralized training, data is pulled into a unified environment, and the trainer masters all samples; in federated training, the training logic must accept the fact that the data domains of the participants are separated from each other, the network status is different, the resource capabilities are different, and the trust assumptions are also different. If a model wants to learn, it can only synchronize parameters or gradients across domains, rather than synchronizing raw data across domains.

#### (1) Basic training closed loop

A typical federated training process usually includes the following steps. First, the central coordinator issues the initial model or the global model of the current round. Subsequently, each participating node trains on local data for several steps to generate local parameter updates or gradient updates. Each node then sends these updates to the aggregator. The aggregator performs average, weighted average or more complex robust aggregation strategies to obtain a new global model, and then sends it to each node to enter the next round of iteration.

Such a round-based training process looks similar to distributed training, but the core difference between the two is that distributed training assumes that the training nodes are controlled by the same organization, and the data can usually be viewed as a whole segmentation; federated learning naturally assumes that the participants are independent of each other, have limited trust, and have strong business differences in local data. Therefore, federated training is not simply "slower distributed training", but a set of collaboration paradigms with business boundaries and governance constraints.

![Figure 28-5: Basic training closed loop of federated learning](../../images/part9/图28_5_联邦学习基本训练闭环.png)
*Figure 28-5: Basic training closed loop of federated learning*

#### (2) FedAvg and local update mechanism

The most classic algorithm in federated learning is FedAvg. The intuition is very simple: after each client trains locally for several steps, the model parameters are updated and uploaded, and the central end averages them according to the sample size or weight, and then forms a new global model. FedAvg is widely used because it has a simple structure, is easy to implement, and has good convergence performance in many medium-complexity scenarios.

However, FedAvg also exposes a typical trade-off of the federated system: if the number of local training steps is too few, there will be many communication rounds, and the cost of bandwidth and synchronization is high; if the number of local training steps is too many, each client will move further in the direction of its own local data distribution, resulting in a more serious drift of the aggregated model, which is often referred to as client drift. In engineering, it is usually necessary to make joint adjustments between the local epoch number, client participation ratio, learning rate and aggregation frequency.

#### (3) Non-IID data problem

In a federated environment, data on each node is usually not independently and identically distributed. Different institutions may have completely different user structures, device sources, regional attributes, annotation standards, and data scales. This will lead to slow convergence of the global model, individual large nodes dominating the training direction, a few nodes not receiving benefits for a long time, and even a situation where the global model is acceptable on average, but is completely unusable for key participants.

The non-IID problem determines that federated training is not as simple as "putting data in another place", but collaborative training with strong business heterogeneity. For scenarios such as medical care, finance, and government affairs, this heterogeneity is often more difficult to deal with than the algorithm itself, because it comes from actual differences in the institution itself, rather than a technical variable that can be easily smoothed out by code.

#### (4) The real boundary of federated learning

The core commitment of federated learning is that “the original data does not leave the domain in principle”, but this does not mean “absolute privacy and security”. If uploaded gradients, parameter updates, training logs, or intermediate metrics can be analyzed, it is still possible for an attacker to infer local data characteristics. Therefore, federated learning is often the "pedestal" for privacy systems rather than the complete answer. As long as systems still need to exchange some kind of analyzable information across domains, differential privacy, secure aggregation, robust aggregation, and auditing strategies must continue to be considered.

---

### 28.2.4 Digging into Differential Privacy (DP)

The core goal of differential privacy is not to make data "completely invisible", but to ensure that whether a single sample participates in the system will not significantly affect the output results. In other words, even if an attacker has a large amount of background knowledge, it is difficult to determine whether an individual appears in the training data based on model output, statistical results, or certain visible signals during the training process.

#### (1) Protection objects of DP

DP focuses on protecting the “indistinguishability of individual contributions”. It does not promise that business results must be kept confidential, nor does it promise that the system will not be leaked at all, but provides a probabilistic sense of protection: in the presence or absence of an individual sample, the difference in system output distribution is limited to a controllable range. This kind of protection is especially suitable for dealing with membership inference risks, because what attackers most often do is to determine whether a person is in the training set.

#### (2) Local DP and Central DP

From the perspective of deployment location, differential privacy can be roughly divided into Local DP and Central DP. Local DP requires each user or client to add noise themselves before the data leaves local, so the trust assumption is the weakest, but since noise enters the system earlier, the accuracy loss is usually more noticeable. Central DP adds noise uniformly on the center side, and the model quality is usually better, but the premise is that you must trust that the center can correctly perform budget management and noise injection.

In a federal scenario, both approaches are possible. If the organization lacks sufficient trust in the center, it will prefer Local DP; if the organization can accept that the center assumes stronger governance responsibilities, it may adopt Central DP in exchange for better model effects.

#### (3) DP-SGD ideas in training

In model training, the most common idea of ​​differential privacy is DP-SGD. The core steps usually include: first clipping the single-sample gradient to limit the maximum impact of each sample on the overall update; then injecting noise of a specific distribution into the aggregated gradient; and finally accumulating and recording the consumption of the privacy budget during the training process. This process embodies the engineering essence of DP: instead of "adding some noise to the gradient casually", it first limits the individual influence boundary, and then injects noise on this controllable boundary, thereby making the risk measurement interpretable.

![Figure 28-6: Schematic diagram of DP-SGD training process](../../images/part9/图28_6_DPSGD训练流程示意图.png)
*Figure 28-6: Schematic diagram of DP-SGD training process*

#### (4) Why is DP difficult to adjust?

The biggest difficulty in differential privacy is not "whether noise can be added", but "how to find an acceptable range between availability, budget and stability." A theoretically beautiful parameter combination may directly destroy the model in real business. On the other hand, if the epsilon is set too high in order to maintain the indicator, the actual protection significance may be lost. Therefore, DP projects often get stuck on a very practical problem: it is not that the theory is not valid, but that the business cannot accept such loss of accuracy.

---

### 28.2.5 Deep Digging into Secure Multi-Party Computation (MPC)

Secure multi-party computing is more suitable for scenarios with a relatively clear structure, clear calculation rules, and a limited number of participants, such as joint intersection, joint statistics, joint risk control scoring, etc. Its core value is that all parties can collaboratively complete certain calculations without exposing their original inputs, and only obtain visible results.

#### (1) Intuitive understanding of secret sharing

The basic intuition of MPC can be understood as "split the original data into multiple incomprehensible fragments, and then let these fragments participate in joint operations." The fragments in the hands of any single participant are not enough to recover the original data. Only through protocol interaction can the target output be finally obtained. This mechanism is suitable for scenarios where the computing logic is relatively fixed and all parties involved want to strictly control the input exposure range.

#### (2) What is MPC suitable for and what is not suitable for?

MPC is particularly suitable for list intersection (PSI), blacklist comparison, joint statistics and partial rule-based joint modeling, because the function structure of these tasks is relatively clear, and many parties are more concerned about "whether the results can be calculated safely" rather than "whether a huge deep model can be efficiently trained." However, it is not suitable for deep neural network training with a large number of parameters, nor is it suitable for complex online scenarios with high frequency, low latency, and many participating nodes. The reason is not that it can't be done, but that communication rounds and computational overhead tend to quickly bring down project availability.

---

### 28.2.6 Homomorphic Encryption (HE) Deep Digging

The biggest attraction of homomorphic encryption is that "the ciphertext is computable". In other words, even if the data exists in encrypted form, the system can still perform certain forms of operations on it without decrypting it, and finally obtain results consistent with plaintext operations after decryption. This feature makes HE theoretically very privacy-preserving because the computing nodes do not need to see the plaintext.

#### (1) Why is HE so strong?

HE is especially suitable for scenarios where "data must be handed over to an incompletely trusted environment for processing, but the plain text cannot be exposed", such as cross-cloud inference, aggregation statistics in an encrypted state, or some secure inference tasks. For architectures where the central platform is untrustworthy but must undertake computing tasks, HE provides a very attractive protection idea: transforming "the computing party cannot be trusted" into "the computing party cannot see the plain text at all".

#### (2) Why is HE so important?

The problem with HE is that engineering is extremely expensive. Ciphertext operations are usually much slower than plaintext operations, and nonlinear operators, complex control flows, and large-scale matrix operations in the model will significantly increase the difficulty of implementation. Therefore, HE is often suitable for use in some specific links, such as secure aggregation, limited inference or key sub-module encryption calculations, but is not suitable as a full replacement for the entire complex deep training pipeline.

---

### 28.2.7 Trusted Execution Environment (TEE) Deep Digging

The logic of TEE is "trust the hardware enclave, not the host environment." It uses a protected execution space provided by the CPU or hardware platform to allow sensitive code and data to run in an isolated environment. Even if the host system, administrator or cloud vendor operation and maintenance has higher permissions, the plain text data and runtime status cannot be easily viewed.

#### (1) Value of TEE

TEE is particularly suitable for protecting critical nodes in the system, such as aggregation services in federated training, key management services, or certain highly sensitive inference tasks. When organizations cannot fully trust cloud vendor operations, container operating environments, or host operating systems, TEE can provide an additional layer of protection for "the most critical part of the logic that must be executed centrally."

#### (2) Risks and limitations of TEE

TEE is not a silver bullet. It relies on a hardware root of trust, which itself is subject to supply chain, implementation flaws, and side-channel attacks. At the same time, TEE's resource limitations, debugging complexity, and compatibility costs with existing platforms will also significantly increase the difficulty of implementation. Therefore, it is more common in engineering to use TEEs as critical node hardeners rather than shoehorning the entire system into trusted enclaves.

---

## 28.3 Selection and system cost evaluation

The introduction of privacy enhancement technology essentially uses system performance, engineering complexity and organizational collaboration costs in exchange for stronger compliance and lower leakage risks. Architects must make these costs explicit, rather than packaging privacy solutions as "zero-cost upgrades." In real projects, many privacy technologies fail not because of wrong concepts, but because no one tells the business side in advance: how much computing power, how much delay, how much bandwidth, how much debugging costs, and how much organization and coordination costs you will have to pay for it.

### 28.3.1 Accuracy loss, delay increase and communication overhead

The first is **accuracy loss**. It mainly comes from factors such as differential privacy noise, MPC/HE's approximation of complex operators, non-IID data distribution in a federated environment, and limited training rounds. For the business side, the most intuitive question is usually not "What is the privacy budget?" but "How much will the indicator drop?" This shows that for privacy technology to be truly implemented, utility changes must be explained in a language that businesses can understand.

The second is the increase in latency. MPC and HE will significantly increase computing latency, while federated learning is often affected by node waiting, network synchronization, slow client tailing, and retry mechanisms. For online reasoning systems, if the system response requirement is milliseconds, then many heavy-duty privacy computing solutions are almost impossible to achieve.

The third is **communication overhead**. In large model or high-dimensional model scenarios, the scale of federated parameter or gradient transmission is extremely large. If encryption expansion is added, bandwidth often becomes the first bottleneck. Many teams think the problem lies in the algorithm, but in fact the problem first dies on the network.

### 28.3.2 When to choose institutional governance and when to choose technical governance

Not all projects need to “take the big step in privacy computing”. Decisions should be made in conjunction with the Ch27 classification and grading framework. For L1 (C1) low-sensitivity data, institutional governance is preferred, and requirements are usually met through RBAC, routine masking, audit logs, and approval flows. For sensitive data in L2 (C2), you can first use isolation environment, partial field restrictions, usage constraints and lightweight federated analysis. Only when the project enters L3 (C3) highly sensitive data, cross-subject collaboration, cross-border scenarios or external joint training, it is necessary to upgrade "data not leaving the domain" from an institutional requirement to a system physical boundary, and then introduce technical governance methods such as FL, MPC, HE or TEE.

This type of judgment is consistent with the governance boundaries of P09. The goal of P09 is to form a front-end governance closed loop through classification, permissions, desensitization, isolation, auditing and incident handling, so that sensitive data can be controlled before entering the training or analysis system. Because of this, the privacy enhancement technology discussed in Ch28 does not replace P09, but continues to advance to the training stage after P09.

### 28.3.3 Communication optimization strategy

Communication optimization in federated learning is the key to its implementation. A theoretically feasible solution, if each round has to upload a very large update vector and wait for all nodes to complete training and resynchronize, the system will soon lose availability due to network and time costs.

One of the common strategies is gradient compression. For example, only upload the top-k important gradients, use low-bit quantization, or sparsely process parameter updates. The core of these methods is not to pursue mathematical perfection, but to significantly reduce the transmission volume within an acceptable loss of accuracy. The second type of strategy is to reduce synchronization frequency. By increasing the number of local training steps and reducing the number of global synchronization rounds, more local computing power can be exchanged for less network overhead. However, this approach will also bring about client drift, which requires a more robust aggregation strategy. The third type of strategy is **Asynchronous Federation**. It allows different nodes to participate in training at different paces, which helps alleviate the problem of slow node tailing, but the consistency and convergence analysis of the global model will be more complicated.

![Figure 28-7: Communication cost breakdown diagram in federated training](../../images/part9/图28_7_联邦训练中的通信成本分解图.png)
*Figure 28-7: Communication cost breakdown diagram in federated training*

### 28.3.4 Accuracy Optimization Strategy

After introducing DP or other privacy constraints, the engineering team usually needs to proactively perform accuracy compensation. A simple but often effective idea is to first select a stronger pre-trained basic model, and then do lighter downstream optimization under privacy constraints. Because the stronger the basic model, the better it can maintain acceptable performance under limited budget and limited data.

Another important strategy is to handle sample and client differences in a more fine-grained manner, such as group training by organization, group or task type, and then aggregation at a higher level. For DP scenarios, you can also make systematic parameter adjustments on the clipping threshold, noise intensity, number of training rounds, and optimizer. It should be emphasized that parameter tuning in privacy scenarios is more complicated than that of ordinary models, because you not only have to look at the loss curve and verification set indicators, but also the budget consumption and attack verification results.

### 28.3.5 System complexity and operation and maintenance costs

Privacy technology adds not just computing power costs, but also key management difficulty, node failure recovery complexity, debugging invisibility, log compliance requirements, and collaboration costs across legal, security, algorithm, and platform teams. Many privacy projects fail not because the thesis proposal is wrong, but because the operation and maintenance phase simply does not know how to troubleshoot problems. For example, if the aggregator is put into TEE, the intermediate state is difficult to directly observe; if HE is used, it is difficult for developers to troubleshoot abnormalities like debugging ordinary numerical programs; if DP is used, when the model deteriorates, you cannot directly judge whether the budget is too small, the clipping is too strong, or the samples themselves are too few.

Therefore, when evaluating privacy-enhancing technologies, it’s not just about GPUs and CPUs, it’s also about organizational maintainability. The solution that can really be implemented is often not the most "advanced" solution, but the one with the best balance between safety, accuracy, cost and operation and maintenance.

---

## 28.4 Implementation model, verification and online governance

### 28.4.1 Horizontal FL

Horizontal federation is suitable for situations where the feature spaces of all participants are basically the same but the user groups are different. For example, two regional banks have similar customer characteristics fields, but serve different regional populations; another example is that hospitals in different cities have similar structured case fields, but their patient samples do not overlap. In this type of scenario, the "columns" of all parties are basically the same, but the "rows" are different. Therefore, it is more suitable to combine the collaborative capabilities at the sample level to alleviate the problem of insufficient sample size of a single institution.

The biggest advantage of horizontal federation is that its structure is intuitive and easy to understand and deploy. But it does not automatically mean that the training effect must be good, because there may still be significant deviations in the user structure, label ratio and behavior patterns of different institutions.

### 28.4.2 Vertical FL

Vertical federation is suitable for situations where the participants have similar user groups but master different feature dimensions. For example, a bank knows the user's financial characteristics, and an e-commerce company knows the consumption preference characteristics. The two hope to establish a joint risk control model without directly exchanging original fields. The key to the problem at this time is not the number of samples, but the complementarity of feature dimensions among different parties.

The engineering difficulty of vertical federation is usually higher than that of horizontal federation, because it involves not only model collaboration, but also issues such as sample alignment, identity matching, and feature linkage. If strong privacy constraints are added, the system complexity will further increase.

![Figure 28-8: Schematic diagram comparing horizontal federation and vertical federation](../../images/part9/图28_8_横向联邦与纵向联邦对比示意图.png)
*Figure 28-8: Schematic diagram comparing horizontal federation and vertical federation*

### 28.4.3 Federated Fine-Tuning

For large model scenarios, more and more systems adopt federated fine-tuning instead of federated full-parameter training. The reason is very practical: full parameter training communication is too heavy, the cost is too high, and the privacy exposure is also greater. Federated fine-tuning usually combines PEFT methods such as LoRA, Adapter, and Prefix Tuning to exchange only smaller-scale adaptation layer parameters between institutions, rather than all the weights of the entire base model.

This approach has two direct benefits. First, communication costs are significantly reduced, making it easier to implement in a multi-agency environment. Second, local institutions can retain more control over the underlying model and private corpus while sharing fewer parameter updates. Therefore, federated fine-tuning is likely to become one of the mainstream forms of cross-agency large model collaboration in the future.

### 28.4.4 Secure Aggregation

For federated training to truly hold, one question must be answered: Can the aggregator see individual updates from each client? If the answer is "yes", then although federated learning prevents the original data from going out of the domain, it may still expose the risk of gradient leakage to the central end. The goal of secure aggregation is to allow the central end to only see the aggregated results, but not the original updates of individual clients.

This is important because many people confuse federated learning with secure aggregation. In fact, federated learning is just a way to organize training, and security aggregation is a very critical but not automatically owned security module. A federated system without security aggregation often still retains a clear central observation plane.

### 28.4.5 Online governance and grayscale strategy

Privacy-enhancing systems must also do grayscale, rollback, and abort. Many teams imagine privacy technology as the ability to "take effect silently in the background", but in fact it will have a significant impact on model performance, system latency and participant collaboration, so it is more necessary to be cautious when launching it online. In practice, before a federal or privacy-enhancing system is launched, small-scale institutional gray scale, budget thresholds, parameter update frequency limits, data domain change approval, abnormal training suspension conditions, and partner exit mechanisms should usually be set.

Such governance requirements may seem cumbersome, but their essence is to explicitly incorporate “privacy risks” into system change management. For industries with strong supervision, the real danger is not the technology itself, but that after the technology goes online, no one knows when to stop, who has the authority to make changes, and how to trace back any problems.

---

## 28.5 Attack and Defense in Federated Learning

A mature privacy chapter cannot only talk about technical solutions, but also must talk about the attack surface. Otherwise, it is easy for readers to mistakenly think that "data does not leave the domain" is safe enough. In fact, the attack surface in a federated environment is no less than in centralized training, but the attack paths have changed. The attacker does not necessarily need to have access to the original database, but can also carry out the attack by observing gradients, participating in training, manipulating updates, or analyzing model output.

### 28.5.1 Membership Inference Attack

The goal of the member inference attack is to determine whether a certain sample appears in the training set. In medical and financial scenarios, this judgment itself may already constitute a major privacy leak. For example, if an attacker can determine whether a patient appears in a rare disease training set, even if the attacker does not know all the details of the patient's medical records, it is enough to cause serious ethical and legal problems.

Federated learning does not automatically eliminate this risk. Because the global model may still exhibit higher confidence, more stable prediction patterns, or special output behavior for training samples. An attacker can exploit these differences to infer whether a certain sample participated in training.

### 28.5.2 Gradient Inversion Attack (Gradient Inversion)

Gradient inversion attacks illustrate another core risk of federated learning: even if the original data does not go out of domain, the uploaded gradient or parameter update itself may contain enough information to allow the attacker to recover the original training sample or its approximate characteristics. For image tasks, the attacker may reconstruct the sample outline; for text tasks, the attacker may recover keywords, sentence patterns, and even some sensitive fragments.

The reason why this type of attack is dangerous is that it directly breaks through the first layer of illusions that many organizations have about federated learning: the fact that the data is not uploaded does not mean that the information is not uploaded. As long as the uploaded content retains an analyzable structure, the attack surface remains.

![Figure 28-9: Schematic diagram of gradient inversion attack](../../images/part9/图28_9_梯度反演攻击示意图.png)
*Figure 28-9: Schematic diagram of gradient inversion attack*

### 28.5.3 Model Poisoning and Backdoor Attacks

In a federated environment, an attacker does not necessarily have to steal data, but can also pollute the global model by uploading manipulated model updates via a malicious client. If the attack goal is to degrade overall performance, this is model poisoning; if the attack goal is to cause the model to output incorrect results under specific trigger conditions, this is often called a backdoor attack.

This type of attack deserves special attention because federated learning itself encourages multiple parties to participate in training, and multi-party participation also means that it is difficult for the central platform to completely determine whether each client is behaving normally. For medical diagnosis, financial approval and public service systems, even if the attacker does not steal any original data, as long as the attacker can systematically distort model decision-making, it is still a major security incident.

### 28.5.4 Defense Mechanism

The defense of the federal system can be deployed at three levels. The first is **privacy defense**, such as differential privacy, gradient clipping and secure aggregation, which focus on reducing the analyzability of single samples and single client updates. The second is **robust aggregation**, such as Median, Trimmed Mean, Krum and other robust aggregation strategies, which focus on reducing the impact of malicious clients on the global model. The third is **Anomaly Detection**, which monitors and intercepts client update amplitude, update direction, loss changes, and distribution anomalies.

It should be pointed out that these three types of defense are usually not substitutional relationships. Differential privacy is more focused on preventing inference and reconstruction, robust aggregation is more focused on anti-poisoning, and anomaly detection is more focused on finding problems during runtime. A truly credible federal system must often combine these three types of capabilities.

### 28.5.5 Why offensive and defensive verification must be included in the online standards

If the system has not undergone member inference testing, inversion testing and poisoning robustness testing, then the fact that "the system uses privacy technology" does not itself mean that the risk has been controlled. The real implementation of compliance is not to configure technical terms, but to verify whether the attack surface is really reduced. In other words, privacy protection cannot be proven only by design documentation, but must also be proven by offensive and defensive experiments and pre-launch verification results.

This is also an area that many organizations tend to overlook in privacy engineering: they spend a lot of energy introducing what technologies are used, but rarely answer "How do you prove that these technologies are really effective under your data, your model, and your attack hypothesis?" The online governance emphasized in this chapter is precisely to set this verification as the threshold for going online, rather than to remedy it afterwards.

---

## 28.6 Federation system architecture design

At this point, the chapter must move from "Technical Concepts" to "Systems Engineering." Because whether the technical route can be implemented ultimately depends on how the system splits components, how to organize the control flow, how to record versions and budgets, and how to handle failures and collaborate on changes. In other words, federated learning is not an algorithmic function, but a distributed collaboration system.

### 28.6.1 Core components of the federated system

A complete federal training system usually includes five types of core components. The first category is **Coordinator/Orchestor**, which is responsible for training round scheduling, participant registration, task orchestration and model version control. The second category is **Client Runtime**, which is deployed locally in each organization and is responsible for data loading, local training, policy execution and local log management. The third category is **Aggregator**, which is responsible for summarizing uploaded updates and generating global models. The fourth category is **Privacy Engine**, which is responsible for performing gradient clipping, noise injection, budget recording, secure aggregation, or key-related operations. The fifth category is **Audit & Governance Layer**, which is responsible for logs, approvals, audit trails and abnormal alarms.

A mature federated system often does not pile these capabilities into one service, but clearly separates training orchestration, privacy control and audit governance. The reason for this is that privacy policies and model policies are not always maintained by the same team. If the boundaries are unclear, problems such as "the model is updated, but the privacy parameters are not synchronized" and "the budget is exhausted, but the training is still running" can easily occur later.

![Figure 28-10: Overall architecture diagram of the federal system](../../images/part9/图28_10_联邦系统整体架构图.png)
*Figure 28-10: Overall architecture diagram of the federated system*

### 28.6.2 Data flow and control flow

In the design of federated systems, a very easily overlooked but extremely important issue is the distinction between **data flow** and **control flow**. Data flow refers to how raw data, local samples, intermediate features, gradients, or parameter updates flow through the system. Control flow refers to who will issue training tasks, who will decide participation rounds, who will approve policy changes, who will record budget consumption, and who has the authority to terminate training.

Many architectural problems do not stem from the algorithm, but from the failure to clearly distinguish between the two types of flows, resulting in the confusion of permission design and risk responsibility. For example, although a central service does not directly contact the original data, if it can force the distribution of any training tasks, modify any privacy parameters and bypass audits, then it still has excessive power over the control flow. This is also a high-risk structure for strong regulatory scenarios.

### 28.6.3 Model version, budget version and audit version

In traditional model platforms, many teams only record the model version number and online time. But for federation and privacy-enhancing systems, this is far from enough. The system should also explicitly record: which model version uses what privacy budget; which institutions are introduced in which round of training; which pruning threshold, noise strategy or safe aggregation scheme is used in which aggregation; which alarms are related to this model version.

This means that version control in the federated system is not just about "saving a weight file", but must be able to trace the entire training governance process. Only in this way, when there is a problem with the model, a partner agency raises a question, or the audit department asks, can the organization explain how the model was formed, within which boundaries it was trained, and why it was allowed to go online.

### 28.6.4 Failures, Retries, and Party Exits

Federated systems naturally face problems such as instability of participants, network interruptions, inconsistent node performance, and training dropouts. Therefore, the system must design a fault-tolerance strategy after the client goes offline, whether to continue aggregation when some participants fail, a training round rollback mechanism, and processing rules when the partner organization temporarily or permanently withdraws.

This is very realistic. Centralized training assumes that training nodes are maintained by the same team, while clients in a federated environment may belong to different institutions, different networks, and different security policies. The disconnection of a certain node is not necessarily a technical failure, but may also be caused by changes in organizational policies, suspension of approvals, or risk control triggers. Therefore, fault tolerance of federated systems is not only a distributed system issue, but also an organizational collaboration issue.

---

## 28.7 Connection with P09 privacy protection pipeline

From a system perspective, the focus of P09 is not on federal training itself, but on forming a privacy protection data pipeline through classification, permissions, desensitization, isolation, auditing, pre-inspection and accident review. The goal is not to "do a desensitization," but to establish a processing system that can explain the boundaries of responsibility so that sensitive data can be safely controlled before entering the training or analysis system.

### 28.7.1 P09 solves the governance problem "before entering training"

In P09, the system first generates the compliance scope, classification policy, access policy and privacy technology options, then performs classification, desensitization, isolation, alarm and audit on the original records, and completes the governance closed loop through preflight and postmortem. In other words, P09 first answers: Which data can enter the subsequent system, which must be isolated, which fields must be removed, which accesses should be audited, and which exceptions must be reviewed.

Although this type of work is not equivalent to federated learning, it is the basic governance that must be completed before federated learning. Because if pre-classification and boundary control are not done well, any subsequent federated training may run on the wrong data boundaries.

### 28.7.2 Federated learning solves the problem of "cross-subject joint training"

Even if P09 has preprocessed the data cleanly enough, it does not mean that the data is suitable for centralized aggregation. Federated learning and PETs solve another problem: how to continue modeling when the original data cannot be shared centrally. In other words, P09 ensures the quality of governance before data enters the model system, while Ch28 discusses how cross-subject collaborative training can continue to maintain privacy boundaries after the data enters their respective local domains.

### 28.7.3 How to form a closed loop between the two

Looking at the system of the entire book, this closed loop can be written clearly. Ch27 is responsible for the compliance framework and grading standards at the institutional level; P09 is responsible for the classification, desensitization, permissions, isolation, auditing and pre-inspection of data before entering the model system; Ch28 is responsible for privacy protection in the training and collaboration stages; Ch22 is responsible for multi-modal retrieval and application capability undertaking. In this way, the entire system is no longer an isolated chapter, but a complete link from "data governance" to "model governance" to "application governance".

![Figure 28-11: Closed-loop diagram of compliance governance, privacy pipeline, federated training and application capabilities](../../images/part9/图28_11_合规治理隐私流水线联邦训练与应用能力闭环图.png)
*Figure 28-11: Closed-loop diagram of compliance governance, privacy pipeline, federated training and application capabilities*

### 28.7.4 Why is this connection important?

The problem with many projects is that front-end management is done by one team, and the training system is done by another team. There is no unified strategy between the two. The result is that although a lot of classification and isolation have been done in the past, the training system still reintroduces risks through caching, debugging, parameter synchronization or output logs. This chapter should emphasize that P09 and the federal system are not two isolated modules, but should share classification levels, audit strategies, risk thresholds, and exception handling logic. Only in this way, privacy protection is not about "a certain process looks safe", but a consistent boundary throughout the entire life cycle.

---

## 28.8 In-depth exploration of industry cases

### 28.8.1 Medical scenario: cross-hospital multi-modal rare disease model

Return to the medical case at the beginning of this chapter. This case best illustrates why a single technology is often not enough. First of all, the data is highly sensitive to C3 and is naturally not allowed to flow freely; secondly, text, images, and genes are cross-modal high-risk combinations, and desensitization alone is not enough to eliminate the risk of re-identification after alignment; thirdly, the participants are cross-institutional and do not fully trust each other, and no party is willing to assume the responsibility of building a centralized lake.

In this case, the more reasonable route is often not "centralized training + desensitization patching", but a combined solution: front-end governance takes over the classification, access control and direct PII removal of P09; federated fine-tuning is used in the training phase to avoid centralized sharing of original data across agencies; differential privacy is superimposed before uploading updates to reduce the risk of member inference; the central aggregation node can cooperate with secure aggregation or TEE for additional protection; member inference and inversion testing must be performed before going online.

The key to this case is not whether it is technically possible to centralize the data, but that the governance boundaries simply do not allow it. Because of this, the value of privacy enhancement technology is not just “making the system more secure”, but “making possible collaborative training that would otherwise be impossible.”

### 28.8.2 Financial Scenario: Joint Anti-Fraud and Blacklist Matching

Financial scenarios are different from medical care. The goal is often not multi-modal generation or complex representation learning, but joint risk control, blacklist intersection, anomaly identification and rule enhancement. Both institutions may have a certain size of suspicious account information, but neither institution can directly exchange complete user lists because this involves both customer privacy and business boundaries and compliance responsibilities.

In this case, if the task is to do joint intersection or intersection statistics, MPC/PSI is usually given priority; if the task is to jointly train a risk control model, FL can be considered; if the system is still worried about the privacy leakage of the query results, more stringent result auditing and query restrictions can be added. Compared with medical scenarios, finance places more emphasis on rule accuracy, low false positives, and interpretability. Therefore, its technical route is often more biased towards "safe computing + audit traceability" rather than simply pursuing the highest model performance.

![Figure 28-12: Comparison chart of privacy technology roadmaps in medical and financial scenarios](../../images/part9/图28_12_医疗与金融场景的隐私技术路线对比图.png)
*Figure 28-12: Comparison of privacy technology roadmaps in medical and financial scenarios*

---

## 28.9 From training to inference: a full-link perspective on privacy protection

Many readers tend to understand privacy issues as "training issues," but in real systems, training, deployment, querying, and accountability may all constitute leakage surfaces. Risks in the training phase include data access, gradient upload, aggregation, parameter storage, and debugging logs; risks in the inference phase include leaks induced by prompt words, exposure to output overfitting, accumulation of query frequencies to form inferences, and leakage of training traces in API return content; and in the audit and accountability phase, organizations must also answer within what boundaries the model was trained, which budgets have been consumed, and which updates introduced problems.

In other words, privacy protection is not a temporary layer added during the training phase, but a continuous management object throughout the model life cycle. Only when an organization understands privacy as a "full-link attribute" rather than "an additional capability of a certain module" can it truly form a stable, reusable, and auditable system design.

---

## 28.10 Practical Guide: When to use and when not to use

First, don’t over-engineer for low-sensitivity, single-subject, low-risk data. If the data itself is not sensitive and is used entirely in a controlled environment within a single organization, directly introducing federated learning or MPC will only increase complexity and may not bring real benefits. Second, don’t treat federated learning as “naturally safe.” Federated learning solves the problem of data location and does not automatically solve gradient leakage, member inference, model poisoning and backdoor risks. Third, technology selection should first ask about collaboration boundaries and then algorithm preferences. The real questions that should be answered first are: whether the original data can go out of the domain, whether the central end is trustworthy, whether the participants trust each other, how high the online delay requirement is, whether it is necessary to prevent member inference, and whether the audit is required to be explainable.

A practical experience is: when you find that a project neither involves highly sensitive data nor cross-subject collaboration, but still attempts to build the most complex privacy computing stack, it often means that the architecture has begun to be over-designed. On the other hand, when you are faced with medical, financial, cross-border or government affairs collaboration, but you are still trying to rely only on "field desensitization + access control" to get through, it means that the architectural design seriously underestimates the actual risks.

---

## 28.11 Summary of this chapter

Starting from "Why privacy protection must be front-loaded in architecture design", this chapter systematically introduces five types of privacy enhancement technologies, FL, DP, MPC, HE and TEE, and analyzes their differences in protection objects, computing costs and applicable scenarios. We emphasize that federated learning is not a simple slogan of "data does not leave the domain", but a complete system design regarding training control rights, parameter flow methods and collaboration boundaries. Differential privacy provides a quantifiable individual protection framework. Multi-party secure computing is suitable for joint statistics and intersection scenarios. Homomorphic encryption provides theoretically strong protection capabilities for ciphertext calculations. The trusted execution environment establishes runtime trust boundaries for key nodes at the hardware level.

More importantly, this chapter advances privacy issues from "introduction to technical terms" to the level of "engineering system governance". We discussed issues such as non-IID data, communication overhead, accuracy loss, privacy budget, attack and defense, system architecture, version tracking, and online governance, demonstrating that the real difficulty of privacy enhancement technology is not whether the algorithm can be found, but how to embed it into the complete model life cycle.

Finally, combined with the P09 privacy protection pipeline and medical and financial cases, we can see that a mature AI privacy governance system is not a single breakthrough in a certain technology, but a combination of institutional governance, data governance, training governance, verification governance and audit governance. A truly implementable system must answer four questions at the same time: how to enter data, how to train models, how to verify risks, and how to trace responsibilities.

---


