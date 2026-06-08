# Chapter 30: Internal Data Markets and Sharing Governance

## Chapter Guide

The previous chapters discussed data catalogs, metadata governance, data productization, data contracts, and data valuation. Together they answer several foundational questions: can data be found, understood, depended on, and measured for value? Findability, accessibility, interoperability, and reusability are the conditions that allow data assets to enter an organization's collaboration network (Wilkinson et al. 2016). Inside an enterprise, however, data creates value only after one more barrier is crossed: can the right people reuse the right data, within the right boundaries, in the right way?

Many organizations do not lack data. Their data is trapped behind team, system, and permission boundaries. One team cleans customer master data while another repeats the same work. One project standardizes field definitions while another reparses raw logs. A high-quality metric definition works inside one business domain but cannot be discovered or requested by other departments.

This state is expensive. Duplicate collection consumes business-system capacity and external data budgets. Duplicate cleaning creates similar but inconsistent datasets. Unclear permissions push access requests through email and chat. Opaque quality makes consumers discover unreliability only after integrating the data. Worse, these costs are scattered across projects and rarely accounted for at the organizational level. Each team sees a small local shortcut; the organization absorbs duplicate construction, audit risk, and fractured definitions.

An internal data market addresses this "data exists but does not flow" problem. It is not merely a prettier catalog or an approval form. It connects data assets, data products, authorization processes, usage audit, value feedback, and sharing incentives into a continuing governance system. Research on data marketplaces emphasizes not only listings, but also matching supply and demand, transaction rules, quality signals, and trust mechanisms (Schomm, Stahl and Vossen 2013).

The goal is not unconditional openness. Quite the opposite: an internal data market should make sharing more bounded, evidence-based, and accountable. It should define which data can be self-served, which requires approval, which can be used only for limited purposes, and which requires anonymization or isolated computation. Data governance research treats decision rights, accountability, and control mechanisms as central to governance design (Khatri and Brown 2010; Abraham, Schneider and vom Brocke 2019).

This chapter addresses four questions. First, why enterprises need internal data markets, and how markets reduce the cost of duplicate collection, duplicate cleaning, unclear permissions, and opaque quality. Second, how to design roles and processes for providers, consumers, approvers, platform teams, and security/compliance teams. Third, how to govern sharing, authorization, and audit, including purpose limits, access logs, expiration, and violation handling. Fourth, how to move from sharing to a data product ecosystem. The chapter also covers operating metrics, incentives, an end-to-end customer-risk data product case, anti-patterns, and an acceptance checklist.

## 30.1 Why Enterprises Need Internal Data Markets

### 30.1.1 The Cost of Data Silos Is More Than Inconvenience

Data silos are often described as a collaboration problem: Team B cannot access data held by Team A. That is true, but incomplete. The deeper issue is that silos continuously produce duplicate cost, quality divergence, risk blind spots, and delayed decisions.

When data cannot be shared reliably, each team tends to collect a copy it can control. This may look faster in the short term. Over time, the organization accumulates many similar sources with different definitions and quality levels. Those copies enter reports, models, risk rules, customer profiles, and operational systems. As dependencies grow, they become hard to reconcile. When metrics disagree, teams spend time explaining why "my customer count" differs from "your customer count." This explanation cost is one of the most common and persistent wastes in enterprise data governance.

Silos also destroy economies of scale in data quality. If one shared asset serves many teams, one quality fix benefits many downstream uses. If every team creates a copy, each team separately handles missing values, duplicates, anomalies, and definition changes. One team fills nulls as "unknown," another drops them, and a third interprets them as "not applicable." Downstream answers diverge even when everyone began with the same raw data.

An internal data market reorganizes these scattered efforts. It lets high-quality assets be discovered, requested, authorized, reused, and reviewed under controlled conditions. Treating information as a measurable and operable asset is a core theme of information asset management and data value management (Moody and Walsh 1999; Laney 2017).

### 30.1.2 Duplicate Collection: Invisible Budget Consumption

Duplicate collection is one of the first problems an internal data market should reduce. In large enterprises, the same data category is often collected independently by multiple teams: customer profiles by marketing, service, risk, and finance; supplier data by procurement, contract, finance, and risk systems; device status by production, operations, and quality systems.

Each team can usually justify the local copy: its system needs fresher data, its fields fit the business better, its permission process is easier to control, or its project schedule cannot wait for enterprise governance. The problem is that if the organization does not record total duplicate cost, duplication becomes the default.

Duplicate collection includes interface cost, external purchase cost, processing cost, and reconciliation cost. The market changes the default. Before launching a new collection effort, consumers should search the market for reusable data products. If an existing product meets the need, they request authorization. If it partially meets the need, they request product enhancement. Some scenarios still need independent pipelines, such as ultra-low-latency risk decisions, isolated sensitive computation, or legally required data separation. But duplication should require a reason; it should not be the path of least resistance.

### 30.1.3 Duplicate Cleaning: Similar Data, Inconsistent Answers

Duplicate cleaning is even more hidden. Multiple teams may use the same raw data, yet produce different assets because their cleaning rules differ. For example, order amount may involve refunds, discounts, taxes, and fees. Marketing may exclude refunded orders, finance may count refunds as negative amounts, and risk may use transaction initiation amount. Their views of customer spending power will differ. Phone-number deduplication may also vary by phone, ID number, or phone plus name, producing different customer counts.

The market should not force one universal cleaning rule for every purpose. Instead, it should productize, document, and make choices explicit. A shared data product must explain its business definition, cleaning rules, suitable scenarios, and unsuitable scenarios. If the same raw source legitimately supports several definitions, the market may list several derived products, but their lineage and documentation must show the relationship. Consumers then choose knowingly rather than receiving a result that only appears standard.

### 30.1.4 Unclear Permissions: Sharing Becomes Risk

Many organizations still handle data access through email, chat, offline confirmation, and temporary permissions. Applicants do not know whom to ask. Data owners do not know whether they can approve. Security teams do not know the purpose. Platform teams open access but do not own the business judgment.

This creates two opposite failures. Legitimate use is delayed, and inappropriate access may be granted through informal communication or stale permissions. The root cause is the absence of institutionalized decision rights for data sharing.

An internal data market must split authorization responsibility clearly. Providers confirm content, quality, and definitions. Business approvers judge whether the purpose is reasonable. Security and compliance teams judge sensitivity, anonymization, and use limits. Platform teams execute permissions, record logs, and reclaim access at expiration. Once these roles are formalized, sharing becomes auditable, reviewable, and optimizable rather than dependent on personal relationships.

The point is not to make approval heavy. It is to make approval evidence-based. Clear low-risk requests can use fast paths. Sensitive data, vague purposes, external sharing, or model training should trigger stricter review.

### 30.1.5 Opaque Quality: A Market Cannot Build Trust

Every market needs trust. If consumers cannot see quality status, they will not connect a data product to core production workflows. A catalog with many products can still become a rich-looking but unused directory.

Quality opacity usually appears in four ways: consumers do not know whether data is fresh, whether fields are stable, what the missing/duplicate/anomaly rates are, or who will fix quality problems. Each shared data product should therefore carry quality information: update time, quality grade, key-field completeness, recent incidents, SLA, owner, and feedback channel. High-value products should also show quality trends.

Quality transparency changes consumer behavior. A stable, well-documented, responsive product will be reused. A low-quality product without ownership will not circulate even if it is listed. The market must govern quality, responsibility, and feedback, not only listings.

### 30.1.6 Core Value Proposition of an Internal Data Market

An internal data market has four basic value propositions:

1. **Make data discoverable.** Consumers can search existing assets and products before creating new pipelines.
2. **Make sharing bounded.** Use is authorized, purpose-constrained, and audited rather than uncontrolled.
3. **Make quality judgeable.** Consumers can see definitions, quality status, SLA, and historical issues before using data.
4. **Make reuse sustainable.** High-value shared data becomes maintained data products with feedback and stable supply.

This is what distinguishes a market from an ordinary catalog. A catalog answers what data exists. A market further answers who can use it, how it can be used, how well it is used, and whether it deserves continued supply.

![Internal data market architecture](../../images/part9/ch30_fig01.png)

*Figure 30-1: Internal data market architecture*

### 30.1.7 Section Summary

This section explained why enterprises need internal data markets. Duplicate collection, duplicate cleaning, unclear permissions, and opaque quality create continuing cost, risk, and definition fragmentation. A market turns data from scattered resources into governable internal assets that can flow efficiently within clear boundaries.

## 30.2 Roles and Processes in a Data Market

### 30.2.1 A Market Is a Collaboration Model, Not Just a System

When building an internal data market, teams often start from system functions: search, request button, approval page, permission provisioning, access logs, and ratings. These functions matter, but they are only the surface. Data management bodies of knowledge place these capabilities across governance, metadata, quality, security, and architecture rather than in one isolated system (DAMA International 2017).

The real market is a set of cross-role collaboration relationships. If role boundaries are unclear, even a complete system will fail: providers do not know what to maintain, consumers do not know how to express needs, approvers do not know approval standards, platform teams only open access, and compliance teams can only punish after the fact.

This chapter divides market roles into five categories: data providers, data consumers, approvers, platform teams, and security/compliance teams. These are governance responsibilities, not fixed organization names. One team may hold several roles; one role may be shared by several teams. Every sharing action must still have accountable parties.

### 30.2.2 Data Providers: From Owning Data to Supplying Products

Providers produce and maintain shared data. In a market, they must take product responsibility rather than merely "owning" a table. Data mesh's data-as-a-product idea similarly emphasizes that domain teams should own discoverability, understandability, trustworthiness, and usability (Dehghani 2022).

Providers define asset boundaries: a table, a table group, a metric package, a tag package, an API, or a product containing documents, examples, quality reports, and a contract. They explain business definitions, maintain quality, configure monitoring and incident response, and manage change. Field additions, deprecations, definition changes, source replacements, and refresh-cadence changes must be communicated through contracts or change notices. Providers do not finish their work by "putting data out"; they operate a dependable product.

### 30.2.3 Data Consumers: From Asking for Data to Declaring Purpose

Consumers use shared data. They should not merely ask for "table X." They should state purpose, scenario, access method, data scope, usage duration, and downstream impact.

Purpose declaration is central. The same data used for reporting, model training, customer outreach, risk decisions, or external disclosure has different risk. Without purpose, approvers and compliance teams cannot judge whether access is reasonable. Consumers should also follow least privilege: request the minimum fields, time range, and access needed. Regional sales analysis may need city-level aggregates, not customer-level details. Feature development may need anonymized behavior features, not raw identifiers.

Consumers should also provide feedback after use: whether the data met the need, whether fields were clear, whether quality was stable, and whether the request process worked.

### 30.2.4 Approvers: Institutionalizing Business Reasonableness

Approvers judge whether a sharing request is reasonable for the business. They may come from the data's business domain, a governance committee, or the data owner organization. They should not simply approve or reject; they should decide based on data classification, purpose, consumer identity, historical authorization, and risk requirements.

Approvers should ask whether the purpose is real and clear, whether requested data matches the purpose, whether lower-sensitivity alternatives exist, whether access duration is reasonable, whether data will be reshared, and whether model training or automated decision making is involved.

For low-sensitivity routine uses, approvers can define auto-approval or fast-approval rules. High-sensitivity, cross-domain, external, or personal-rights-affecting uses require stricter approval.

### 30.2.5 Platform Teams: Turning Process Into Executable Capability

Platform teams implement the market process as system capability: catalog, search, listing, request forms, approval flows, permission provisioning, access control, log collection, expiration, quality display, and feedback. Modern data platforms connect ingestion, storage, processing, service, governance, and observability across the data lifecycle (Reis and Housley 2022).

Platform teams also standardize interfaces. Products may appear as tables, APIs, files, metrics, features, vector indexes, or knowledge bases, but the market should provide a unified discovery, request, and audit experience. The platform should not replace business approval or compliance judgment. It makes the process executable, traceable, and automatable.

If a platform can list data but cannot write authorization into permission systems, the market remains a display layer. If it can open access but cannot record purpose and expiration, audit fails. The market must integrate identity management, permission systems, catalogs, lineage, quality monitoring, ticketing, and audit platforms.

### 30.2.6 Security and Compliance Teams: Making Sharing Boundaries Explicit

Security and compliance teams define sharing boundaries based on sensitivity, regulations, internal policies, and scenario risk. Policies include data classification, anonymization requirements, access control, purpose limits, retention, audit, cross-border or external sharing restrictions, and violation handling.

Compliance should not appear only as a final veto. Better practice is to embed rules into the market platform: some high-sensitive fields are never shared in plaintext, some data can be used only in isolated analysis environments, some requests require a project ID and business owner, and some authorizations expire within 90 days. Systematized rules improve efficiency because routine cases can be handled automatically and high-risk cases receive human attention.

### 30.2.7 Standard Market Process

A standard process has seven stages:

1. **Data product listing.** The provider registers the product with definitions, fields, quality indicators, sensitivity level, SLA, owner, examples, and request conditions.
2. **Consumer discovery.** The consumer searches by keyword, domain, tag, metric, scenario, and lineage.
3. **Request submission.** The consumer selects scope, access method, purpose, duration, project, and high-risk use flags such as model training or external disclosure.
4. **Policy evaluation.** The platform routes the request to auto-approval, fast approval, or multi-party review based on data level, purpose, consumer identity, and authorization history.
5. **Approval decision.** Business approvers and compliance teams decide, possibly requiring reduced scope, anonymized versions, or more explanation.
6. **Permission provisioning.** The platform writes the approval into permission systems and records scope, expiration, purpose limits, and responsible parties.
7. **Usage audit and feedback.** The platform records logs, frequency, abnormal behavior, expiration, and feedback, then feeds results into product operations.

![Authorization approval workflow](../../images/part9/ch30_fig02.png)

*Figure 30-2: Authorization approval workflow*

### 30.2.8 Role Responsibility Matrix

A responsibility matrix makes the process executable and auditable. Governance organization design must fit business complexity, centralization, and decision rights; no single model fits all enterprises (Weber, Otto and Osterle 2009; Otto 2011; Alhassan, Sammon and Daly 2016).

*Table 30-1: Internal data market role responsibility matrix*

| Governance action | Data provider | Data consumer | Approver | Platform team | Security/compliance team |
| --- | --- | --- | --- | --- | --- |
| Product listing | Define definitions, fields, quality, and owner | Provide demand feedback | Confirm business ownership | Provide listing tools and templates | Confirm sensitivity rules |
| Data discovery | Maintain searchable descriptions and tags | Search and compare products | Monitor key-asset coverage | Provide search, recommendation, and catalog capability | Provide compliance tags |
| Data request | Answer product questions | State purpose, scope, and duration | Judge business reasonableness | Provide request forms and workflow | Judge risk and restrictions |
| Authorization approval | Confirm content and supply capability | Add purpose details | Make business decision | Orchestrate approval flow | Make compliance decision |
| Permission provisioning | Confirm authorized scope | Use within scope | Confirm approval result | Write to permission systems | Define controls |
| Usage audit | Watch abnormal-use feedback | Accept logging and checks | Review authorization reasonableness | Collect access logs | Monitor violation risk |
| Quality maintenance | Fix issues and notify changes | Report quality problems | Coordinate cross-domain impact | Display quality metrics | Assess quality risk |
| Expiration and reclaim | Confirm renewal eligibility | Renew or stop use | Approve renewal | Reclaim permissions automatically | Review long-term access |

The matrix can be adjusted, but each action needs an accountable owner, not only participants.

### 30.2.9 Section Summary

This section decomposed the market into five roles and seven process stages. Providers productize data, consumers declare purpose and least-privilege scope, approvers judge business reasonableness, platform teams automate process, and security/compliance teams define boundaries. With these roles connected, the market becomes an executable sharing governance mechanism rather than a static directory.

## 30.3 Sharing, Authorization, and Audit

### 30.3.1 Core Principles of Sharing Governance

Sharing governance must balance efficiency and safety. If every request passes through heavy approval, consumers will bypass the market. If everything is self-service, sensitive data loses boundaries and the organization fails audit and privacy obligations. Security and privacy control systems emphasize risk-based access, audit, purpose constraints, and accountability rather than uniform control strength (NIST 2020a; NIST 2020b).

Four principles form the baseline:

1. **Least privilege.** Consumers receive only the minimum data scope and permission needed for the declared purpose.
2. **Purpose binding.** Authorization is tied to a project, purpose, and time window, not a permanent opening for a person.
3. **Traceability.** Who accessed what data, when, and for what purpose must be auditable.
4. **Dynamic reclaim.** Permissions expire or change when projects end, purposes change, people move, or risks change.

### 30.3.2 Data Requests

A good request form should not only ask "which table." It should help consumers express the need clearly. Required information should include applicant, team, project ID, business owner, data product, fields, time range, access method, duration, purpose, downstream system, model-training flag, customer-outreach flag, and external-sharing flag.

For sensitive data, the form should ask why anonymized, aggregated, or lower-sensitivity alternatives cannot be used. Templates should help consumers describe purposes correctly. Purpose categories such as business analysis, risk modeling, customer operations, product experimentation, audit, regulatory reporting, and R&D testing can drive different approval routes.

### 30.3.3 Authorization Models

Traditional data permissions focus on tables, databases, directories, or system accounts. That works for simple analytics, but not for an internal data market. Role-based access control provides a foundation, while attribute-based access control extends decisions with subject, resource, environment, and purpose attributes (Ferraiolo and Kuhn 1992; Sandhu et al. 1996; Hu et al. 2014).

The market should move from table permissions to purpose permissions. A purpose permission combines data object, field scope, row scope, access method, use scenario, validity period, environment, and reshare limits. For the same customer dataset, a finance audit project may access customer IDs, transaction amounts, and contract status; a marketing analysis project may access only anonymized IDs and aggregated tags; a model training project may be forced into an isolated environment with no plaintext export.

Approval results should therefore produce structured authorization policies, not merely "allow table A." Permission systems, query engines, API gateways, anonymization systems, and audit systems should execute the same policy.

### 30.3.4 Purpose Limits

Data authorization is often mistaken for ownership transfer. Once a team receives data, it may continue using it in other projects, models, reports, or even share it further. This invalidates the original approval. The approved object is a specific purpose, not unlimited use.

Purpose limits may prohibit external sharing, customer outreach, automated decisions, model training, plaintext export, joining with certain data classes, or use outside a specified environment. These limits should appear in the request page, approval record, permission policy, and product documentation.

For high-risk data, technical controls can enforce limits: sensitive data is queried only inside a secure sandbox, exports are aggregate-only, model training jobs must register experiment IDs, and API calls must carry project tokens. Purpose limits make sharing sustainable because providers and compliance teams can trust that data will not spread without boundaries.

### 30.3.5 Access Logs

Access logs are the foundation of sharing governance. Without logs, the organization cannot know whether authorization was used as intended or trace impact after an incident. Auditability and traceability are also core capabilities in security-control baselines (NIST 2020a).

Logs should record actor, time, data product, version, field scope, query conditions, access method, purpose identifier, authorization ID, returned row count, and anomaly flags. Highly sensitive data may also require source device, network context, export behavior, and downstream processing lineage.

Logs are not only for blame. They support market operations. High access frequency may identify core assets. Many requests but little actual use may indicate request friction, poor documentation, or quality issues. Frequent access near expiration suggests renewal or migration reminders. Abnormal downloads, off-hours access, expanded query scope, or use inconsistent with purpose should trigger review.

### 30.3.6 Expiration and Reclaim

Data permissions without lifecycle accumulate. After staff transfers, project closure, model retirement, report deprecation, or system migration, historical access often remains. The market should make every authorization expire by default. Low-sensitivity data can have longer terms; high-sensitivity data should have shorter terms; project access should follow the project lifecycle; temporary debugging access should be brief.

Before expiration, the platform should ask consumers to renew or stop use. Renewal should revalidate purpose, scope, consumer identity, and risk level. Without renewal, permissions should be reclaimed automatically and the reclaim event recorded. Stable high-frequency use may become a subscription, but subscriptions should still undergo periodic review.

### 30.3.7 Violation Handling

Governance needs consequences. Violations include use beyond declared purpose, unapproved resharing, sensitive-data export, bypassing the platform, long-term occupation of expired permissions, privacy or compliance incidents, and refusal to cooperate with audit.

Minor violations can require correction, explanation, or reapplication. Repeated violations can suspend authorization, restrict request eligibility, or notify team leaders. Serious violations should enter security-incident handling, audit, accountability, and compliance procedures. Rules should be visible in market policies, and consumers should acknowledge use limits and consequences when requesting data.

Providers and approvers also have obligations. If providers fail to maintain promised quality or approvers leave requests stuck for long periods, those failures should enter market operations review. Sharing governance constrains the whole market ecosystem, not only consumers.

### 30.3.8 Authorization Workflow Table

*Table 30-2: Authorization workflow nodes*

| Stage | Main action | Responsible role | Output |
| --- | --- | --- | --- |
| Need submission | Fill in purpose, scope, duration, and project | Data consumer | Data request |
| Automatic validation | Check identity, data level, and required fields | Platform team | Initial risk judgment |
| Business approval | Judge reasonableness and necessary scope | Approver | Business decision |
| Compliance review | Judge sensitivity, anonymization, and restrictions | Security/compliance team | Compliance controls |
| Authorization execution | Write to permission system and generate authorization ID | Platform team | Executable permission policy |
| Usage monitoring | Collect logs and detect abnormal behavior | Platform team, security/compliance team | Audit records and alerts |
| Expiration review | Renew, reduce scope, or reclaim | Consumer, approver, platform team | Permission lifecycle record |

The process can be simplified or strengthened by risk level. Low-risk data can automate more nodes; high-risk data needs fuller human review.

### 30.3.9 Section Summary

This section covered sharing, authorization, and audit. Authorization should move from table access to purpose-bound policies covering scope, duration, environment, and reshare limits. Requests, access logs, expiration, reclaim, and violation handling form the closed loop that lets data flow without losing control.

## 30.4 From Sharing to a Product Ecosystem

### 30.4.1 Sharing Is the Starting Point; Productization Is Long-Term Supply

The early goal of a data market is to let data be found and requested. If the market stops at sharing, new problems appear: many listed datasets but few dependable products, request flows without clear consumer choice, and providers with little incentive to maintain after one authorization.

Sharing means "letting others use existing data." Productization means "continuously supplying dependable data capability around stable demand." This aligns with data productization and data mesh: data is not a one-time deliverable but a product capability that must be operated over time (Dehghani 2022). A shared table may lack SLA, consumer feedback, or a roadmap. A data product must have clear definitions, quality commitments, contracts, documentation, owner, change process, and support channel. Market maturity should be measured by durable consumed products, not listing count.

### 30.4.2 Turning High-Value Shared Assets Into Data Products

Not all shared data needs productization. One-off analysis, low-frequency niche use, and exploratory data may remain lightweight. Productization is appropriate for assets with high reuse frequency, many consumers, clear business value, stable quality requirements, and sustainable ownership.

Usage logs reveal candidates. Repeated requests from multiple teams suggest horizontal reuse. High access frequency and repeated renewals suggest stable production dependency. Frequent quality feedback suggests consumers already depend on the asset but the supply capability is insufficient.

Productization usually has five steps:

1. Redefine the asset boundary.
2. Complete field descriptions, definitions, and examples.
3. Establish quality monitoring and SLA.
4. Create a data contract and change process.
5. Build a product catalog page and support workflow.

After these steps, the data is no longer a passively shared resource; it is an internal data product that can be subscribed to and reused.

### 30.4.3 Example Shared Data Product Catalog

A data product catalog is the core interface of the market. It serves both consumer discovery and governance portfolio management. A good page tells consumers what the data is, what it is suitable for, who maintains it, how quality looks, whether it can be requested, what conditions apply, who already uses it, and what recently changed. Dataset documentation research emphasizes that provenance, composition, use, limits, and maintenance information must be explicit for responsible reuse (Gebru et al. 2021).

*Table 30-3: Example shared data product catalog*

| Product name | Domain | Main content | Suitable scenarios | Sensitivity | Quality status | Request method | Owner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Unified Customer Profile | Customer | Basic attributes, lifecycle, tags | Operations analysis, customer segmentation | Medium | Normal, daily refresh | Subscribe after approval | Customer data team |
| Transaction Detail Wide Table | Transaction | Order, payment, refund, channel fields | Business analysis, finance reconciliation | High | Normal, hourly refresh | Strict approval | Transaction data team |
| Risk Event Labels | Risk | Fraud, overdue, blacklist, device risk | Risk modeling, strategy validation | High | Normal, daily refresh | Isolated environment | Risk data team |
| Product Metric Definition Pack | Product | Active use, retention, conversion, funnel metrics | Product analysis, experiment evaluation | Low | Normal, daily refresh | Self-service subscription | Product analytics team |
| Contract Master Data | Legal | Contract parties, status, term, amount | Finance audit, fulfillment analysis | Medium | Some fields under repair | Subscribe after approval | Legal data team |
| Knowledge Document Index | Knowledge | Policies, processes, FAQ, procedure documents | RAG, intelligent Q&A | Medium | Normal, weekly refresh | Access after approval | Knowledge management team |

The fields are simple, but they put the consumer's key questions in one place.

### 30.4.4 Product Tiers and Operating Strategies

Shared products should be operated by tier. A practical system classifies them as core products, key products, ordinary products, observation products, and retirement candidates.

Core products support multiple key business chains and require SLA, monitoring, change notices, dedicated maintenance, and quarterly review. Key products are important within a domain and should receive stable maintenance and active feedback collection. Ordinary products have stable but limited use and can follow standard maintenance. Observation products have unclear value and should be kept at low cost while use trends are watched. Retirement candidates have long-term nonuse, poor quality, or high risk and should enter shutdown evaluation.

This tiering connects directly to the valuation work in Chapter 29: the market provides use facts, valuation provides grading evidence, and product operations take action.

### 30.4.5 From One-Time Authorization to Long-Term Subscription

Market relationships can evolve from one-time authorization to long-term subscription. One-time authorization fits temporary analysis, short projects, experiments, and low-frequency audits. Long-term subscription fits stable reporting, production models, RAG knowledge bases, core metric services, and cross-system synchronization.

When a consumer repeatedly requests the same product with stable purpose, high access frequency, and controllable risk, the platform can recommend subscription. Subscription should include purpose, data scope, refresh frequency, SLA, change notification, quality feedback, renewal cycle, and exit mechanism. Subscription is not permanent authorization; it simplifies repeated approval into periodic review.

### 30.4.6 Feedback, Ratings, and Product Improvement

A market ecosystem requires feedback. Consumers should report quality, documentation, request process, response speed, and suitability issues. Star ratings alone are not enough; they are emotional and hard to diagnose. Useful feedback includes issue type, impact, reproduction steps, expected fix time, whether business is blocked, and whether alternatives exist.

The platform can aggregate feedback into operating metrics: average response time, unresolved issue count, quality recurrence rate, approval pass rate, consumer satisfaction, and number of reuse teams. These metrics guide provider improvement and investment. High-value products with repeated quality feedback need more governance investment. Low-value products with no use or feedback may become retirement candidates.

### 30.4.7 Section Summary

This section explained how an internal data market moves from sharing to a product ecosystem. Sharing answers whether others can use data. Productization answers whether they can use it reliably over time. High-value shared assets become sustainable products through asset-boundary definition, documentation, monitoring, contracts, catalog pages, subscription, and feedback.

## 30.5 Operating Metrics and Incentives

### 30.5.1 Why Operating Metrics Are Needed

After launch, a market should not be judged by listing count alone. A market may list thousands of entries while only a few are used. It may record many requests but fail to support business because approval is slow, quality is poor, or documentation is unclear. Operating metrics turn subjective impressions into observable facts. They help platform teams find process bottlenecks, providers identify maintainable products, governance teams detect risk, and managers allocate resources. Without observable governance feedback, data platforms easily accumulate hidden technical debt and ownerless dependencies (Sculley et al. 2015).

Metrics should cover supply, demand, efficiency, quality, risk, and value.

### 30.5.2 Supply-Side Metrics

Supply metrics describe what products exist and whether they are reusable. Examples include listed product count, active product count, percentage with owners, percentage with quality indicators, percentage with field documentation, percentage with SLA, and percentage with contracts.

Listing count is insufficient. A product without owner, quality indicators, documentation, or request rules is closer to a catalog entry than a market-consumable product. Supply metrics should also track coverage by business domain, data type, sensitivity level, and scenario. If a core business domain has few reusable products, the market has a structural supply gap.

### 30.5.3 Demand-Side Metrics

Demand metrics describe consumer behavior: search count, product views, request count, subscriptions, reuse teams, cross-domain reuse, repeated requests, and request conversion rate.

Searches with no result are a crucial signal. They may indicate missing assets or poor naming and tagging. Rejected requests should also be analyzed. Rejections may come from unreasonable purposes, excessive scope, sensitivity, absence of anonymized versions, or unclear product documentation. Demand signals should drive product construction.

### 30.5.4 Efficiency Metrics

Efficiency metrics describe process flow: average approval duration, auto-approval ratio, fast-approval ratio, permission-provisioning time, on-time reclaim rate, renewal handling time, and number of returned requests.

Approval duration should be segmented by data level and purpose type. Slow approval for low-risk data means the process is too heavy. Very fast approval for high-risk data may mean risk detection is weak. Permission provisioning time also matters; if approval is complete but access is delayed, consumers still experience the market as inefficient.

### 30.5.5 Quality and Risk Metrics

Quality metrics include quality-incident count, repair duration, SLA attainment, field-change count, consumer complaints, and recurrence rate. Risk metrics include high-sensitive requests, expired permissions, abnormal-access alerts, violation incidents, long-term unreviewed authorizations, and accesses inconsistent with declared purpose.

Quality and risk should be observed together. A high-quality but high-risk product needs strict authorization and audit. A low-risk but low-quality product should not be recommended for critical workflows.

### 30.5.6 Value Metrics and Incentives

Value metrics ask whether the market reduces duplicate construction, promotes reuse, and creates products. Examples include hours saved through reuse, duplicate procurement avoided, number of reuse teams, number of new pipelines avoided through sharing, number of shared assets upgraded into products, and downstream scenarios supported by core products.

These metrics support incentives. Providers should receive recognition for organizational value created through reuse; otherwise they bear maintenance cost without visible benefit. Consumers should also receive recognition for reusing existing products rather than rebuilding. Reuse is not laziness; it is organizational efficiency. Platform teams should be measured by process efficiency, consumer experience, and governance closure, not only listing count.

### 30.5.7 Section Summary

This section proposed an operating metric system for internal data markets. Supply, demand, efficiency, quality, risk, and value metrics together show whether the market is healthy. Continuous operation prevents the market from becoming a static directory and helps it grow into a sustainable data product ecosystem.

## 30.6 Case: Customer Risk Data Product Enters the Internal Data Market

### 30.6.1 Case Background

Imagine a financial services company with a customer risk label system. Initially, the risk team uses it internally for loan approval, anti-fraud, and overdue prediction. As the business grows, the service team wants to prioritize high-risk complaint handling, the marketing team wants to exclude unsuitable high-risk users from certain campaigns, the compliance team wants to locate historical risk events during complaints, and the data science team wants risk events for model evaluation.

If every team asks the risk system separately, the company will repeat requests, explanations, anonymization, and reviews. The enterprise therefore turns the label system into a shared data product in the internal data market named `customer_risk_profile`.

### 30.6.2 Product Listing

The risk data team defines the product boundary. `customer_risk_profile` does not expose raw rules or sensitive details. It provides customer-level risk grade, risk type, latest risk event time, label confidence, and label update time. The documentation states suitable scenarios: risk analysis, service prioritization, compliance investigation, and model evaluation. It also states unsuitable scenarios: unapproved customer marketing, external sharing, and automated refusal of customer service.

Security and compliance classify the product as highly sensitive. All access must bind to project purpose. Plaintext identity export is prohibited. Model-training use must occur in an isolated environment. The platform creates a catalog page with field documentation, quality status, refresh frequency, request conditions, owner, SLA, and change history.

### 30.6.3 Authorization and Use

The service team requests access for prioritizing high-risk complaint tickets. Approvers accept the purpose but allow only anonymized customer ID, risk grade, and risk type; detailed rule-hit records remain closed. The platform provisions API access for 180 days and requires service-ticket IDs in calls.

The marketing team requests access to exclude high-risk customers from a campaign. Compliance judges that this may affect customer rights, so the team must use an aggregated audience package rather than customer-level labels, and the campaign strategy must pass compliance review.

The data science team requests evaluation use. Approval allows use inside an isolated environment but prohibits plaintext customer identifier export.

The same data product can therefore serve several scenarios, each with different scope and purpose limits.

### 30.6.4 Audit and Feedback

After three months, platform metrics show that `customer_risk_profile` is reused by four teams, with six authorizations and three long-term subscriptions. Logs show that the service team has the highest call frequency, with several off-hours query clusters. Security review finds that these are normal night batch tasks and registers them to reduce future false alerts.

Consumer feedback shows that risk-label freshness is critical for service. The original daily update is insufficient for some high-risk complaints. The risk data team changes the high-risk event field to hourly incremental update while keeping low-risk fields daily, and records the change in the data contract. Feedback improves the product for all consumers.

### 30.6.5 From Shared Asset to Core Product

After two quarters, reuse teams increase, quality complaints decline, authorization stabilizes, and repeated data extraction and anonymization work are reduced. The data governance committee classifies `customer_risk_profile` as a core data product. This brings several changes: home-page recommendation in the market, stronger quality monitoring, automatic alerts for key fields, advance change notification to subscribers, and quarterly reports on reuse value, risk events, and improvement plans.

The case shows that an internal data market is not one-time openness. It lets data become a stable product through controlled sharing and usage evidence.

### 30.6.6 Section Summary

The customer-risk product case showed the end-to-end market process: listing, purpose declaration, differentiated authorization, audit, feedback, product tiering, and core-product operation. Sharing value comes from controlled flow, not boundaryless openness.

## 30.7 Implementation Anti-Patterns and Acceptance Checklist

### 30.7.1 Anti-Pattern 1: Turning the Market Into a Static Catalog

The most common anti-pattern is building a prettier catalog. Teams invest in names, tags, and descriptions but do not connect requests, authorization, audit, quality feedback, and product operations. Governance literature repeatedly emphasizes that policies, processes, roles, and controls must be designed together with the technical platform, or governance remains at the documentation layer (Ladley 2019; Abraham, Schneider and vom Brocke 2019).

The symptom is simple: consumers can find data but still do not know how to request it; after requesting, they still need offline approval; after access is granted, there is no purpose record; quality issues have no feedback route. A real market closes the loop from discovery to request, approval, permission, usage record, feedback, and value review.

### 30.7.2 Anti-Pattern 2: Emphasizing Control Without Improving Experience

Another anti-pattern is turning the market into a new approval gate. Every dataset requires a request, every request requires multiple approvers, every permission waits for manual provisioning, and consumers find the market slower than the old informal path. They then bypass it through private sharing, old accounts, temporary files, and copied scripts. Heavy governance can increase real risk by pushing use outside the governed path.

Good sharing governance is layered. Low-sensitive, low-risk, routine data should be self-service or fast approval. High-sensitive, cross-domain, external, model-training, and customer-outreach scenarios deserve stricter review. User experience is not the enemy of governance. Clear catalogs, standard request templates, automatic policy evaluation, transparent approval progress, automatic provisioning, and expiration reminders all improve governance quality.

### 30.7.3 Anti-Pattern 3: No Maintenance After Listing

The third anti-pattern is listing without maintenance. Providers complete the listing task, then field documentation goes stale, quality incidents are unanswered, feedback accumulates, and changes are not announced. This quickly drains market trust. If one critical project is harmed by poor data, consumers lose confidence not only in that product but in the market.

Avoid this by making owner, SLA, quality indicators, and feedback response part of listing requirements. Data without a clear owner should not be listed as a formal product. Data without a quality commitment should appear only as an observation or exploration asset. Market operators should regularly find products with no owner, no access, repeated quality incidents, or unresolved feedback and move them into remediation or retirement.

### 30.7.4 Anti-Pattern 4: Ignoring Consumer Feedback

Some markets record requests and access but not user experience. Feedback reveals problems that metrics do not: unclear field documentation, broken sample queries, ambiguous request conditions, quality indicators that do not explain business anomalies, or refresh frequency that misses real needs. If feedback is ignored, the market develops supply without trust, and consumers rebuild data privately.

Feedback should be categorized into consultation, quality issue, definition issue, permission issue, performance issue, and enhancement request, each with response targets. Repeated enhancement requests should feed the product roadmap.

### 30.7.5 Anti-Pattern 5: No Retirement Mechanism

A market that only lists and never retires will accumulate old, duplicate, unmaintained, and low-quality products. Consumers then find it harder, not easier, to locate trusted data.

Retirement must be possible and orderly. It should include impact analysis, consumer notification, alternative-product recommendation, observation period, permission reclaim, and archive strategy. Low-frequency audit value may justify downgrading a product to archive status rather than deleting it. High-risk unused products should stop receiving new authorization and be decommissioned after dependencies are cleared.

### 30.7.6 Implementation Acceptance Checklist

Use the following checklist to judge whether an internal data market has basic production capability:

1. Key data products have owner, business definition, field documentation, quality status, and request rules.
2. Consumers can complete search, request, approval-status tracking, and access provisioning within the platform.
3. Authorization binds purpose, scope, duration, and access method rather than granting permanent table-level access.
4. Highly sensitive data has anonymization, isolated environments, access logs, and anomaly alerts.
5. Permissions support expiration reminders, renewal review, and automatic reclaim.
6. The market records access frequency, request conversion, approval duration, quality feedback, and reuse-team count.
7. High-value shared assets can be upgraded into data products, and low-value or high-risk assets can be downgraded, archived, or retired.

If these questions can be answered systematically, the market has moved from a display catalog to a governance market.

### 30.7.7 Section Summary

This section described five implementation anti-patterns: static catalog, excessive control, no maintenance, ignored feedback, and no retirement. Their common root is treating the data market as a one-time system build rather than a continuous operating mechanism. A useful internal market promotes sharing, controls risk, improves consumer experience, creates data products, and allows low-value assets to exit.

## Chapter Summary

This chapter discussed internal data markets and sharing governance. Enterprises need internal data markets because duplicate collection, duplicate cleaning, unclear permissions, and opaque quality continuously create cost, risk, and fractured definitions. A market is neither an ordinary catalog nor a pure permission system. It connects data products, authorization, quality transparency, usage audit, and reuse feedback.

In role design, providers turn data into products, consumers declare purpose and least-privilege scope, approvers judge business reasonableness, platform teams automate process, and security/compliance teams define sharing boundaries. In process design, the market covers product listing, consumer discovery, request submission, policy evaluation, approval, permission provisioning, usage audit, and feedback.

In sharing governance, authorization should move from table permissions to purpose permissions, with structured policies for data scope, field scope, access method, duration, environment, and reshare limits. Access logs, expiration, reclaim, and violation handling make sharing traceable, reviewable, and controllable.

In the long term, the market should not accumulate listings. It should help high-value shared assets become sustainable data products with definitions, quality commitments, contracts, SLA, owners, and feedback. Operating metrics and incentives across supply, demand, efficiency, quality, risk, and value keep the market healthy.

Ultimately, the internal data market solves the problem of how data can flow credibly inside an organization. It upgrades data sharing from personal favors, temporary permissions, and project-by-project extraction into a governed system with catalogs, products, approval, audit, feedback, and continuous improvement.

## References

Abraham R, Schneider J, vom Brocke J (2019) Data governance: A conceptual framework, structured review, and research agenda. International Journal of Information Management 49:424-438.

Alhassan I, Sammon D, Daly M (2016) Data governance activities: an analysis of the literature. Journal of Decision Systems 25(sup1):64-75.

DAMA International (2017) DAMA-DMBOK: Data Management Body of Knowledge, 2nd Edition. Technics Publications.

Dehghani Z (2022) Data Mesh: Delivering Data-Driven Value at Scale. O'Reilly Media.

Ferraiolo D F, Kuhn D R (1992) Role-Based Access Controls. In: Proceedings of the 15th National Computer Security Conference, pp 554-563.

Gebru T, Morgenstern J, Vecchione B, Vaughan J W, Wallach H, Daume III H, Crawford K (2021) Datasheets for Datasets. Communications of the ACM 64(12):86-92.

Hu V C, Ferraiolo D, Kuhn R, Schnitzer A, Sandlin K, Miller R, Scarfone K (2014) Guide to Attribute Based Access Control (ABAC) Definition and Considerations. NIST Special Publication 800-162.

Khatri V, Brown C V (2010) Designing data governance. Communications of the ACM 53(1):148-152.

Ladley J (2019) Data Governance: How to Design, Deploy, and Sustain an Effective Data Governance Program, 2nd Edition. Academic Press.

Laney D B (2017) Infonomics: How to Monetize, Manage, and Measure Information as an Asset for Competitive Advantage. Routledge, New York.

Moody D, Walsh P (1999) Measuring the Value of Information: An Asset Valuation Approach. In: Proceedings of the 7th European Conference on Information Systems (ECIS), pp 496-512.

National Institute of Standards and Technology (2020a) Security and Privacy Controls for Information Systems and Organizations. NIST Special Publication 800-53 Revision 5.

National Institute of Standards and Technology (2020b) NIST Privacy Framework: A Tool for Improving Privacy through Enterprise Risk Management, Version 1.0.

Otto B (2011) Data Governance. Business & Information Systems Engineering 3(4):241-244.

Reis J, Housley M (2022) Fundamentals of Data Engineering. O'Reilly Media.

Sandhu R S, Coyne E J, Feinstein H L, Youman C E (1996) Role-Based Access Control Models. IEEE Computer 29(2):38-47.

Schomm F, Stahl F, Vossen G (2013) Marketplaces for data: an initial survey. ACM SIGMOD Record 42(1):15-26.

Sculley D, Holt G, Golovin D, Davydov E, Phillips T, Ebner D, Chaudhary V, Young M, Crespo J-F, Dennison D (2015) Hidden Technical Debt in Machine Learning Systems. In: Advances in Neural Information Processing Systems 28, pp 2503-2511.

Weber K, Otto B, Osterle H (2009) One Size Does Not Fit All: A Contingency Approach to Data Governance. ACM Journal of Data and Information Quality 1(1):4.

Wilkinson M D, Dumontier M, Aalbersberg I J, Appleton G, Axton M, Baak A, Blomberg N, Boiten J-W, da Silva Santos L B, Bourne P E, Bouwman J, Brookes A J, Clark T, Crosas M, Dillo I, Dumon O, Edmunds S, Evelo C T, Finkers R, Gonzalez-Beltran A, Gray A J G, Groth P, Goble C, Grethe J S, Heringa J, 't Hoen P A C, Hooft R, Kuhn T, Kok R, Kok J, Lusher S J, Martone M E, Mons A, Packer A L, Persson B, Rocca-Serra P, Roos M, van Schaik R, Sansone S-A, Schultes E, Sengstag T, Slater T, Strawn G, Swertz M A, Thompson M, van der Lei J, van Mulligen E, Velterop J, Waagmeester A, Wittenburg P, Wolstencroft K, Zhao J, Mons B (2016) The FAIR Guiding Principles for scientific data management and stewardship. Scientific Data 3:160018.
