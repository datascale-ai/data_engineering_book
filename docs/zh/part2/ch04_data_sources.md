# 第4章：数据源、采集与版权

## 摘要

本章讨论文本预训练数据工程的源头治理问题，重点回答哪些数据可以采集、如何采集以及如何证明其来源和许可边界。章节首先说明数据源选择对模型能力、版权风险和后续清洗上限的影响，随后建立开放网页、论坛问答、百科知识、代码、学术论文、书籍、企业内部数据和用户反馈数据的分类框架。接着，章节从分布式采集、异构格式解析、元数据存证和任务可靠性四个角度说明生产级采集流水线的基本要求，并进一步给出白名单、灰名单、黑名单和许可证分类机制。两个匿名化复合案例用于说明 Common Crawl 解析路线和企业内部文档版权审查中的典型风险。通过本章，读者应能够在大规模抓取或内部数据接入之前建立可审计的数据来源清单、许可判断框架和元数据记录规范，从而为后续清洗、分词和训练评估提供可靠边界。

## 关键词

数据源；数据采集；版权许可；Common Crawl；元数据存证；robots.txt；许可证分类；来源治理

## 学习目标

- 能够区分主要预训练数据源的质量价值、规模潜力和许可风险。
- 能够设计包含 robots.txt 检查、解析质量抽检和断点续传的采集流水线。
- 能够为每批数据建立来源、许可、解析器版本和处理配置等元数据记录。
- 能够使用白名单、灰名单、黑名单和许可证分类机制降低版权风险。
- 能够解释为什么源头治理决定后续清洗和训练数据质量的上限。

## 开篇场景：一个"数据够多了"的团队，为什么还是失败了

以下为匿名化复合案例，数字用于说明工程量级和风险口径。某 AI 研究院的算法团队花费数月时间，从公开的网络资源里爬取并积累了一批中文文本数据。团队原本认为数据体量足以启动一轮基座模型预训练。在信心满满地投入训练之后，首次评测结果却令人沮丧：模型在中文阅读理解和数学推理上的表现不如同规模的开源基线，更严重的是，模型频繁输出"SEO 风格"的凑字数段落，以及疑似来自某些论坛的仿网络小说片段。

问题在哪里？数据量看似够，训练过程也没有异常。复盘后，团队才发现问题的根源：候选数据中相当一部分来自 SEO 站群（内容通过自动采集和改写生成的低质量网站）和版权高度存疑的网络小说正文，真正具有实质知识密度的内容——百科、技术文档、学术摘要等——占比不足。换言之，团队采集的不是"数据"，而是"网络噪声"。模型忠实地学习了训练集的分布，表现出了符合训练数据特征的行为：输出流畅却空洞的文字。

这个案例揭示了预训练数据工程中的基本原则：**源头质量决定后续清洗能够达到的上限。** 当数据源本质上是低质量的，无论后续清洗管线多么精细，都只能降低噪声，而不能凭空补足缺失的知识密度和许可边界。

---

## 4.1 预训练语料为何先输在源头

### 4.1.1 数据源选择决定模型能力上限

预训练（Pre-training）阶段的本质，是让语言模型通过"阅读"大量文本，形成对世界的基础认知图谱——对语言规律的掌握、对事实知识的积累、对推理逻辑的隐性建模。在这个阶段，模型不是在学习一套明确的规则，而是在统计意义上捕捉训练语料的分布模式。这意味着，训练语料是什么样的，模型就倾向于产出什么样的文本。

这条因果关系有一个重要的延伸含义：**预训练数据的质量上限，就是模型能力的天花板**。一个在高质量多样化语料上预训练的模型，即使后续 SFT 数据量较少，往往也能通过指令微调达到不错的效果；而一个在劣质偏斜语料上预训练的模型，无论后续投入多少 SFT 资源，都很难突破底层认知能力的缺陷——因为 SFT 改变的是模型的"行为风格"，而不能凭空注入预训练阶段没有学到的知识结构。

### 4.1.2 三类经典错误选择

在大量失败的预训练项目复盘中，可以归纳出三类最常见的数据源选择性错误：

**偏差源头（Biased Source）**：数据主要来自特定平台、特定人群或特定风格的内容。例如，某团队将某头部内容平台作为主要语料来源——这个平台的内容固然质量较高，但其整体写作风格、话题分布和读者人群极为一致，结果导致模型在这类写作风格上表现异常流畅，但在学术写作、法律文本、技术文档等风格上能力明显偏弱，呈现出强烈的"平台腔"。

**低密度来源（Low-Density Source）**：数据量虽大，但知识密度极低。上文匿名化场景中提到的 SEO 站群是典型案例。另一个常见情形是直接使用爬虫抓取的微博/微信转发类内容——这类内容的句子本身通常语言流畅，但信息量极低，充斥着情感表达和无实质信息的碎片化短文，而这恰恰是大模型记忆和知识提取最难发力的内容类型。

**版权高风险来源（High Copyright Risk Source）**：数据量和质量都不错，但版权归属存在重大法律风险。2023 至 2024 年间，OpenAI、Google、Stability AI 等公司先后面临来自媒体机构、出版商和作者的版权诉讼，核心争议点都集中在训练数据的版权归属问题上。国内监管环境同样日趋收紧——《生成式人工智能服务管理暂行办法》明确要求训练数据合规，且数据提供者对数据版权合法性承担相应责任。对这一风险的忽视，可能在产品上线后给企业带来难以估量的法律和商业损失。

### 4.1.3 "数据越多越好"的误区

"数据越多越好"是预训练数据工程中流传最广的误区之一。这个观念在宏观层面是有一定道理的——在数据质量一定的情况下，规模确实带来能力提升（Scaling Law 的核心结论）。但它在微观执行层面极易被滥用，演变为用体积代替质量的决策依据。

FineWeb 的论文（Penedo et al. 2024）提供了一个重要观察：在相同的 Token 数量约束下，从 Common Crawl 中筛选出的高质量网页语料可以训练出更强的模型，数据清洗、去重和来源配方会显著影响最终效果。DCLM/DataComp-LM 进一步把这一问题组织成可比较的数据配方竞赛：在固定训练预算下，不同数据过滤和混合策略会导致明显不同的下游表现 (Li et al. 2024)。换言之，"少而精"在预训练数据领域可以优于"多而杂"，但必须把结论绑定到实验设置、模型规模和评测集上。

这一结论为整章的数据源选择策略奠定了基础：**数据配方（Data Recipe）的制定，应优先关注每个来源的知识密度和信息多样性，而非原始体积。**

---

## 4.2 数据源地图与配比策略

如果说第4章是一份大模型数据工程的"购物清单审计"，那么数据源地图就是这份审计的核心视图——它帮助工程师在动手采集任何数据之前，先回答一个关键问题：**我们的语料从哪里来，各占多少，质量和法律风险如何？**

![图4-1：预训练数据源分层地图](../../images/part2/pretrain_data_source_map.png)

*图4-1：预训练数据源分层地图 —— 三层分类体系按照处理复杂度、知识密度和许可风险对主流数据来源进行定位，并给出典型的配比参考区间。来源：本书自绘；Alt text：预训练数据源分层地图，展示开放网页、论坛问答、百科、代码、学术论文、书籍、企业内部数据和用户反馈数据的质量与合规位置。*

### 4.2.1 八类核心数据源全景

**开放网页（Open Web）** 是体量最大、也最难驾驭的一类来源，以 Common Crawl 为代表。Common Crawl 从 2008 年起持续抓取互联网，每月发布数十亿网页的快照，累计数据量超过 PB 级，是目前许多大规模预训练数据集的上游来源。然而，开放网页的原始数据质量差异很大——据 FineWeb 项目的统计 (Penedo et al. 2024)，Common Crawl 原始内容中，真正具备知识密度的正文内容占比有限，大量内容为广告、导航栏、SEO 垃圾、JavaScript 代码等噪声。这意味着网页数据必须经过严格清洗才能使用（见第5章）。

**论坛与问答（Forums & Q&A）** 以 Reddit、StackOverflow、知乎、Quora 等平台为代表。这类数据的独特价值在于：它是真实用户针对真实问题产生的自然语言交互，包含了大量的问题追问、答案修正和社区讨论，这对提升大模型的对话能力和"追问理解能力"有很高价值。StackOverflow 在技术领域的采用极为普遍，是 LLM 代码理解能力的重要来源之一。需要注意的是，这类数据在 2023-2024 年纷纷修改 API 条款（Reddit、Twitter/X 均关闭或收费化 API），获取难度大幅上升。

**百科与结构化知识（Encyclopedia & Structured Knowledge）** 以 Wikipedia、Wikidata、Fandom Wiki 为代表。Wikipedia 是许多预训练数据集的重要来源——其多语言、高密度、事实准确率较高的特征，使其即使在体积占比不大的情况下，也能为模型提供稳定的事实知识基础。

**代码（Code）** 以 GitHub、GitLab 的开源仓库以及 BigCode 团队的 The Stack 项目为代表。代码数据的独特价值已被广泛验证：在预训练中引入大量代码，不仅提升模型的代码生成能力，还可能对自然语言推理能力产生正向迁移，原因在于代码的结构化逻辑为模型提供了更严密的符号组织信号。The Stack 项目对数百种编程语言进行了系统整理，并提供许可证过滤后的版本（仅保留 MIT、Apache 2.0 等宽松许可证），可作为代码语料工程的优先参考来源。

**学术论文（Academic Papers）** 以 ArXiv、PubMed Central、Semantic Scholar 为代表。学术论文的知识密度极高，是提升模型在专业领域（尤其是科学、医学、数学）能力的重要来源。ArXiv 开放访问，可以通过官方 API 批量下载 PDF 并解析正文。需要注意的是，ArXiv 的许可策略正在收紧，在进行商业用途训练时需仔细核查每篇论文的具体许可声明。

**书籍（Books）** 是知识密度和语言质量最高的来源之一，对提升模型的长文理解能力、叙事连贯性和专业知识深度都有显著贡献。书籍也是版权风险最高的来源——The Pile 中的 Books3 子集（包含大量版权书籍）已引发多起诉讼，包括 Meta 的 LLaMA 被作者起诉事件。安全的做法是仅使用 Project Gutenberg 等收录版权已过期经典作品的来源，或通过正式授权渠道采购出版社的授权语料。

**企业内部数据（Enterprise Proprietary Data）** 是垂直领域大模型（行业模型）最核心的差异化来源，包括公司内部的技术文档、知识库、历史工单、合规手册、标准操作规程（SOP）等。这类数据的质量往往极高（因为是真实业务需要产出的结构化文本），且对行业模型能力有极为精准的贡献。但这类数据通常涉及商业机密和内部隐私，必须经过严格的权限审批和 PII 脱敏才能用于训练（详见 §4.4）。

**用户反馈与在线对话（User Feedback & Online Interaction）** 是大模型"数据飞轮"的核心组成部分。当模型上线运行后，用户的真实交互数据——包括有价值的追问、对不满意回答的纠错、偏好标注——会持续回流到数据管线，进入 SFT 或 RLHF 阶段的训练集。这类数据稀缺且珍贵，但也是隐私风险最高的一类，需要在收集和使用全流程中严格执行用户知情同意和数据脱敏。

### 4.2.2 数据源类型、许可与风险矩阵

在实际工程决策中，数据源的选择不能仅凭质量考量，还必须将许可风险和获取可行性一并纳入框架。以下是主要数据源的风险画像矩阵：

*表4-1：数据源类型、许可与风险矩阵。来源：本书整理，许可风险应以具体数据源条款、robots.txt、服务协议和法务审核结论为准。*

| 数据源类型 | 代表来源 | 许可类型 | 商用风险 | 知识密度 | 规模潜力 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 开放网页 | Common Crawl, RefinedWeb | CC-BY / 来源不一 | △ 中等（具体页面版权不一） | ★★☆☆☆ 低-中 | ★★★★★ PB级 |
| 论坛与问答 | Reddit, StackOverflow, 知乎 | 平台服务条款许可 | △ 中等（API限制已收紧） | ★★★☆☆ 中 | ★★★☆☆ TB级 |
| 百科全书 | Wikipedia, Wikidata | CC-BY-SA 4.0（自由） | ○ 低 | ★★★★☆ 高 | ★★☆☆☆ 数百GB |
| 开源代码 | GitHub / The Stack | MIT / Apache-2.0等（需过滤） | ○ 低（需License过滤） | ★★★★☆ 高 | ★★★★☆ 数TB |
| 学术论文 | ArXiv, PubMed | CC-BY / Open Access（部分限制） | △ 中等（需逐篇核查） | ★★★★★ 极高 | ★★★☆☆ 数百GB |
| 版权书籍 | Books3, Z-Library | 版权保护（默认不可用） | ● 高（已发生多起诉讼） | ★★★★★ 极高 | ★★★☆☆ 数百GB |
| 公版书籍 | Project Gutenberg, Archive.org | 公共版权域（无限制） | ○ 极低 | ★★★★☆ 高 | ★★☆☆☆ 数GB |
| 企业内部数据 | 知识库 / 文档系统 / 工单 | 私有（需内部授权） | ○ 极低（内部授权后） | ★★★★★ 极高 | ★☆☆☆☆ 项目相关 |
| 用户在线对话 | 产品用户反馈 / 对话日志 | 隐私协议授权 | △ 中（PII脱敏要求高） | ★★★★☆ 高 | ★★★☆☆ 项目相关 |

### 4.2.3 配比策略：从业务目标反推数据组合

数据配比（Data Mix Ratio）是预训练数据工程中最具策略性的决策之一。没有一个放之四海而皆准的"黄金配方"，因为不同的业务目标需要不同的数据组合。以下是面向四类典型业务目标的配比策略参考：

*表4-2：数据配比策略与业务目标对应矩阵。来源：本书整理，配比建议为策略框架，生产环境应通过代理模型评测和消融实验校准。*

| 业务目标 | 网页通用语料 | 代码 | 学术论文 | 书籍/百科 | 垂直领域数据 | 说明 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **通用中文基座模型** | 高 | 中 | 低-中 | 中 | 低 | 追求广泛知识覆盖，代码比例不可过低（影响推理能力） |
| **代码/技术专项模型** | 中 | 高 | 低-中 | 低-中 | 低 | 代码比例大幅提升，但需保留足够通用语言理解能力 |
| **垂直行业模型（如金融/医疗）** | 中 | 低 | 中 | 中 | 高 | 领域数据占比显著提升，通用语料保底维持通用能力 |
| **多语言base模型** | 高 | 中 | 低-中 | 低-中 | 按语言目标分配 | 网页数据中需控制各语言分布与目标语言能力需求一致 |

表4-2使用"高/中/低"而非固定百分比，是为了避免把某个项目的实验配方误读为通用规律。配比策略还需要考量**动态调整机制**：不同阶段的训练（预训练初期 vs Cooldown 阶段）应当采用不同的配比权重。越接近训练后期，越应当提高高质量精选数据（书籍、学术论文、企业数据）的权重，同时降低低质量海量数据（原始网页）的权重。LLaMA 3 技术报告披露了 15T 级训练数据和多阶段后训练流程 (Grattafiori et al. 2024)，但没有给出可直接复用的完整数据配方；生产项目仍需通过小模型消融和冻结评测集校准。

---

## 4.3 采集流水线、解析与存证

数据源策略确定之后，工程团队面临的下一个核心问题是：如何将这些分散在互联网上的数据，以高效、可靠、合规的方式纳入数据管线，并建立起每一条数据的完整权属证明链？

### 4.3.1 分布式异步采集与 robots.txt 合规

在面对千万级 URL 的增量数据源（如特定垂直领域网站群）时，单线程同步爬虫的效率无法满足工程需求。Craw4LLM 的实验报告显示，在其特定预训练与评测设置下，以预训练影响力分数替代图连通度作为爬取优先级，可以用更少 URL 达到近似训练效果；其中“21% URL 量”的结果不应脱离实验配置直接外推到所有站点和模型 (Yu et al. 2025)。工业级实践通常采用基于 `aiohttp` 或 `Scrapy` 的分布式异步采集架构。同时，为了避免引发法律纠纷和保障站点的可用性，**必须在核心调度器中强制集成 robots.txt 检查机制**。

代码清单4-1给出了一个基于 `aiohttp` 的轻量级异步并发采集框架示意。它利用 `urllib.robotparser` 在发送请求前自动校验合规性；生产环境中还应加入速率限制、审计日志、异常重试和法务维护的来源策略。

*代码清单4-1：异步并发采集与 robots.txt 校验示意代码。生产环境应补充速率限制、失败重试、审计日志和来源策略白名单。*

```python
import asyncio, aiohttp
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse

class AsyncEthicalCrawler:
    def __init__(self, user_agent="LLMDataBot/1.0"):
        self.user_agent = user_agent
        self.rp_cache = {}  # 缓存不同域名的 robot parser

    async def fetch_robots(self, session, domain):
        """异步获取并解析 robots.txt"""
        robots_url = f"https://{domain}/robots.txt"
        rp = RobotFileParser()
        try:
            async with session.get(robots_url, timeout=5) as response:
                if response.status == 200:
                    text = await response.text()
                    rp.parse(text.splitlines())
        except Exception:
            pass  # 获取失败则默认允许爬取，但生产中需谨慎
        self.rp_cache[domain] = rp

    async def fetch_url(self, session, url):
        domain = urlparse(url).netloc
        if domain not in self.rp_cache:
            await self.fetch_robots(session, domain)
        
        # 强制合规检查
        if not self.rp_cache[domain].can_fetch(self.user_agent, url):
            print(f"Skipping {url} (disallowed by robots.txt)")
            return None

        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    return await response.text()
        except Exception as e:
            print(f"Failed to fetch {url}: {str(e)}")
        return None

    async def crawl_batch(self, urls, concurrency=50):
        """基于信号量的并发控制，避免压垮源站"""
        sem = asyncio.Semaphore(concurrency)
        async with aiohttp.ClientSession(headers={"User-Agent": self.user_agent}) as session:
            async def bounded_fetch(url):
                async with sem:
                    return await self.fetch_url(session, url)
            
            tasks = [bounded_fetch(url) for url in urls]
            return await asyncio.gather(*tasks)
```

这种架构通常可以将单机爬取吞吐量提升至数百 QPS，同时在入口层阻断对 `Disallow` 路径的违规访问。

### 4.3.2 异构数据源的解析策略

不同类型的数据源需要截然不同的解析技术路线，使用错误的解析工具会导致严重的内容损失或噪声引入：

**网页（HTML/WARC）** 是最常见、也最需要专业工具处理的格式。Common Crawl 提供了三种数据格式——WARC（原始 HTTP 响应+完整 HTML）、WAT（元数据）和 WET（预提取纯文本）。许多团队在早期会直接使用 WET 文件，因为它看起来最省事——已经是纯文本了，直接用就好。这是一个非常危险的陷阱：Common Crawl 的 WET 提取使用的是通用算法，解析质量相当低劣，会保留大量导航栏、页脚、广告文字和 JavaScript 代码片段。正确的做法是从 WARC 文件出发，用高质量的正文提取库（如 Trafilatura (Barbaresi 2021)）重新解析，尽管这耗时更长，但通常能带来更稳定的正文抽取质量。不同语种、站点和解析器配置会导致有效内容保留率显著不同，因此应以抽样审阅和批次统计为准。

代码清单4-2展示了从 WARC 文件解析正文并保留来源元数据的示意流程。

*代码清单4-2：WARC 正文解析与来源元数据保留示意代码。该片段展示解析路径，生产环境应补充编码识别、异常样本隔离和解析质量抽检。*

```python
import trafilatura
from warcio.archiveiterator import ArchiveIterator
import json, gzip

def parse_warc_to_clean_text(warc_path: str) -> list[dict]:
    """
    从 WARC 文件解析高质量文本，附带完整元数据。
    推荐用于 Common Crawl WARC 文件的批量处理。
    """
    records = []
    opener = gzip.open if warc_path.endswith('.gz') else open
    with opener(warc_path, 'rb') as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type != 'response':
                continue
            url = record.rec_headers.get_header('WARC-Target-URI')
            try:
                html = record.content_stream().read().decode('utf-8', errors='ignore')
            except Exception:
                continue
            # 使用 Trafilatura 提取正文（质量远优于 WET 默认提取）
            text = trafilatura.extract(
                html, url=url,
                include_comments=False,
                favor_precision=True,
                output_format='txt'
            )
            if text and len(text) > 200:  # 过滤过短内容
                metadata = trafilatura.extract_metadata(html)
                records.append({
                    'url':        url,
                    'text':       text,
                    'title':      metadata.title if metadata else None,
                    'date':       metadata.date  if metadata else None,
                    'char_count': len(text),
                    # 版权存证字段
                    'source':     'common_crawl',
                    'warc_file':  warc_path,
                })
    return records
```

**PDF（学术论文/书籍/企业文档）** 的解析是另一个技术难点。PDF 格式并非为文本提取设计——它本质上是一套页面排版描述语言，字符、行和段落的位置由坐标决定，而非语义结构。简单的 `pdfplumber` 或 `PyMuPDF` 在处理双栏学术论文时，常常把两栏文字混排在一起，严重降低内容可用性。对于学术论文，推荐使用专门针对科学文献优化的工具（如 GROBID (Lopez 2009)、Nougat (Blecher et al. 2023)、或 Mathpix）；对于企业 PDF 文档，建议在解析后进行人工抽检，确认段落结构的提取质量。

**代码仓库（Git Repos）** 应通过克隆仓库而非 API 拉取的方式获取，确保完整性。解析时需要根据文件扩展名识别编程语言，并基于文件大小（过长的文件可能是自动生成的）、语法合法性（对 Python 可用 AST 解析验证）和许可证文件内容（MIT、Apache 2.0 等宽松许可证白名单）进行质量筛选。

### 4.3.3 元数据存证：每条数据的"出生证明"

在数据采集的同时建立可追溯的元数据档案，是整个数据治理体系的奠基石。一条数据如果没有完整的元数据，在日后的合规审计中就无法证明其来源合法——这与财务账目一样，"我记得是合法的"不能替代"我有凭证证明是合法的"。

每一批采集的数据，应当在落盘（写入对象存储）的同时，向元数据数据库写入以下标准字段。代码清单4-3给出的是示例字段，实际系统应根据数据源、授权方式和审计要求扩展。

*代码清单4-3：采集批次元数据存证字段示例。字段值为说明性样例，生产环境应按授权方式、审计要求和数据源类型扩展。*

```json
{
  "ingestion_id":    "cc-2024-10-zh-batch-0042",
  "source_name":     "common_crawl_2024_10",
  "source_url":      "s3://commoncrawl/crawl-data/CC-MAIN-2024-10/...",
  "ingestion_time":  "2024-10-15T08:23:41+08:00",
  "license_type":    "cc-crawl-mixed",           // 采集时判定的许可类型
  "license_risk":    "medium",                   // low / medium / high
  "language":        "zh",                       // fastText 识别结果 (Joulin et al. 2017)
  "raw_doc_count":   4280350,
  "raw_size_bytes":  18432000000,
  "parser_version":  "trafilatura==1.6.3",
  "filter_config":   "min_len=200,favor_precision=true",
  "s3_prefix":       "s3://my-bucket/raw/cc-2024-10-zh/",
  "team_contact":    "data-team@company.com"
}
```

![图4-2：数据采集与权属存证流程图](../../images/part2/data_ingestion_provenance_chain.png)

*图4-2：数据采集与权属存证流程——从数据源触达到最终归档，每个处理阶段均向"Provenance Ledger（权属账本）"追加元数据记录，形成完整的可审计数据血缘链路。来源：本书自绘；Alt text：数据采集与权属存证流程图，展示来源触达、采集、解析、清洗、入库和审计记录之间的链路。*

### 4.3.4 断点续传与任务可靠性

大规模数据采集任务的运行周期往往长达数天，在此期间发生节点故障、网络中断或云存储访问令牌过期是大概率事件。没有断点续传机制的采集任务，一旦中断就必须从头重跑，既浪费时间又浪费算力预算。

推荐的容错设计是将采集任务拆分为粒度适中的"检查点文件列表"（通常以 WARC 文件为单位），每处理完一个文件就在状态数据库中标记完成，任务重启后自动跳过已完成文件。这一机制结合 Ray Data 或 Spark 的 Fault Tolerance 设计，可以将大规模采集任务的人工干预需求降低到接近零的水平。

---

## 4.4 版权、许可与可追溯治理

版权问题是 LLM 数据工程中容易被低估、却在商业化和监管环境下风险最高的议题之一。部分团队会出于行业惯性跳过版权评估环节，但在监管政策日趋收紧、行业诉讼案例快速积累的背景下，这种做法已经难以支撑正式产品交付。

### 4.4.1 三类许可的判定框架

针对大模型训练数据，可以将许可性质分为三大类：

**开放许可（Open License）** 是相对安全的数据来源，包括 Creative Commons 系列协议中的 CC0（放弃版权）、CC-BY（注明来源可使用）和 CC-BY-SA（相同方式分享）。Wikipedia 采用 CC-BY-SA 4.0，Project Gutenberg 的内容为公共版权域（Public Domain）。使用这类来源时，通常需要在模型文档或技术报告中注明数据来源（Attribution），但商业使用限制较少。需要注意的是，CC-BY-NC（非商业）协议明确禁止用于商业产品，不属于"可用"范畴。

**商业许可（Commercial License）** 是通过正式授权协议采购的数据，通常来自出版社、媒体机构、数据服务商等。这类数据在合规性上最有保障，但成本较高，且协议条款千变万化（例如"仅用于某一特定模型版本"或"不得用于生成其他训练数据"等限制条款），需要法务团队逐一审查。在选择商业数据供应商时，建议优先要求供应商提供数据来源证明和版权清单，避免采购到来源不清的二次加工版权风险数据。

**灰区数据（Gray Zone Data）** 是版权状态不明确或存在争议的数据，这也是实践中最棘手的类别。典型灰区包括：爬虫抓取的网页正文（网页版权默认属于作者，但"合理使用"原则的适用性存在司法争议）、通过第三方数据集间接获取的内容（如 The Pile 的 Books3）、以及 robots.txt 中标注了禁止爬取但依然被抓取的网站内容。对于灰区数据，建议引入"风险分级"机制，由法务和数据团队联合评估，根据公司的风险承受能力决定是否纳入训练集。

### 4.4.2 版权风险防控：白/灰/黑名单机制

最有效的版权风险工程化手段是建立数据来源的**三级名单管理体系**：

**白名单（Whitelist）**：经过法务确认、许可清晰、直接可用的数据来源列表。每个来源应注明许可协议版本号、使用限制说明和上次法务审核时间。例如：Wikipedia（CC-BY-SA 4.0，可商用，需署名）、The Stack v2（MIT/Apache-2.0 过滤后，可商用）、Project Gutenberg（Public Domain，无限制）。

**灰名单（Graylist）**：许可存在争议或条款限制较多的来源，使用前需要逐案提交法务审核，并记录审核结论。常见的灰名单来源包括：其他 arXiv 论文（需逐篇核查）、平台 API 数据（受平台服务条款约束）。

**黑名单（Blacklist）**：明确不可用的来源——包括已发生诉讼的来源（如 Books3）、robots.txt 明确禁止的网站、以及任何包含明确"禁止用于 AI 训练"声明的数据。技术上，可以通过 URL 域名前缀匹配的方式，在采集管线的入口阶段自动拦截黑名单来源。代码清单4-4展示了简化实现：

*代码清单4-4：版权黑名单入口拦截示意代码。生产环境应由法务和数据治理团队维护名单，并记录命中原因与审核时间。*

```python
# 版权黑名单：在采集入口拦截禁止来源
COPYRIGHT_BLACKLIST_DOMAINS = {
    "nytimes.com",       # 已明确禁止AI训练使用
    "wsj.com",           # 明确要求付费授权
    "theguardian.com",   # 修改服务条款禁止AI训练
    # ... 持续更新，由法务团队维护
}

def is_url_allowed(url: str) -> bool:
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.lstrip('www.')
    return not any(domain.endswith(bl) for bl in COPYRIGHT_BLACKLIST_DOMAINS)
```

### 4.4.3 许可证类型自动分类

对于代码数据，许可证信息通常以 LICENSE 或 LICENSE.md 文件的形式存放在仓库根目录，可以通过规则或分类器自动识别。代码清单4-5展示了简化实现，生产系统应使用更严格的许可证解析库和法务审核流程：

*代码清单4-5：许可证类型自动分类示意代码。该片段仅用于说明规则识别思路，生产环境应使用成熟许可证解析库并保留人工复核链路。*

```python
import re

# 常见许可证关键字识别（简易版，生产中建议使用 license-expression 库）
LICENSE_PATTERNS = {
    "MIT":        r"(?i)mit\s+license",
    "Apache-2.0": r"(?i)apache\s+license.*2\.0",
    "GPL-3.0":    r"(?i)gnu\s+general\s+public\s+license.*version\s+3",
    "CC-BY-4.0":  r"(?i)creative\s+commons.*attribution.*4\.0",
    "CC-BY-NC":   r"(?i)creative\s+commons.*non.?commercial",
    "Proprietary":r"(?i)(all rights reserved|proprietary|confidential)",
}

COMMERCIAL_SAFE = {"MIT", "Apache-2.0", "CC0", "Public Domain"}

def classify_license(license_text: str) -> dict:
    for name, pattern in LICENSE_PATTERNS.items():
        if re.search(pattern, license_text):
            return {
                "license": name,
                "commercial_safe": name in COMMERCIAL_SAFE,
                "risk_level": "low" if name in COMMERCIAL_SAFE else "high"
            }
    return {"license": "Unknown", "commercial_safe": False, "risk_level": "high"}
```

---

## 4.5 案例复盘与实践建议

### 案例一：Common Crawl 中文语料接入的全流程教训（匿名化复合案例）

**项目背景**：某团队计划从 Common Crawl 某批次中提取一批高质量中文文本，作为通用中文基座模型预训练的语料主体。以下规模、耗时和比例为教学性工程估算，用于说明 WET 与 WARC 解析路线的风险差异；实际结果取决于抓取批次、语言过滤策略、解析器版本和人工抽检口径。

**T+0（决策日）**：团队初步评估了该批次 WET 文件，认为直接使用 WET 最简单——毕竟 WET 里已经是纯文本，省去了解析步骤。他们下载了一个 WET 子集并做了快速评估。

**T+3（发现问题）**：数据工程师随机抽取中文文档进行人工审阅，发现质量严重低于预期：不少文档包含大量导航栏和菜单文字（如"首页 | 关于我们 | 联系我们 | 版权声明"），也存在广告、商品描述堆砌和正文截断问题；真正完整的文章正文占比不足以支撑生产级训练。

**T+4（方案切换）**：团队决定放弃 WET 路线，改为从 WARC 文件出发，用 Trafilatura `favor_precision=True` 模式重新解析。这会增加处理时间和 CPU 成本，但可以保留更完整的 HTML 上下文供正文抽取器判断。

**T+8（重新评估）**：Trafilatura 路线的结果显著好转：人工抽检显示完整正文占比和平均文档长度均优于 WET 路线。团队据此保留 WARC 解析作为生产链路，并把抽检结果写入该数据源的解析质量基准。

**核心教训**：WET 文件是"便宜的陷阱"，只适合粗略实验，不适合生产级训练数据准备。是否切换到 WARC + 高质量解析器，应该由抽检质量收益与额外时间、CPU 成本共同决定，而不是仅凭文件获取成本判断。

基于此案例，可以给出三条可直接落地的工程建议。第一，**建立"解析质量基准"**：在任何新的数据源接入早期，可先按启动阶段经验值对随机抽样的 500-1000 条文档进行人工标注，并根据数据源异质性和错误类型扩大样本量，统计完整正文占比和平均文档长度，形成该来源的"解析质量基准"，作为后续管线迭代的评估参照。第二，**区分评估样本和生产样本**：不要用同一批快速实验的数据评估最终训练效果，实验阶段的数据处理精度往往低于生产级别；第三，**在 pipeline 中埋入质量快照**：每个处理节点（解析→过滤→去重）应当在处理完成后自动输出一份"质量快照报告"，记录当前批次的平均文档长度、短文档比例、字符集分布等统计信息，工程师无需额外手工取样即可判断节点的输出质量是否达标。这套"自动质量快照"机制，是大规模数据工程中避免"黑盒管线"的核心手段之一。



---

### 案例二：金融企业内部知识库采集的合规风险（匿名化复合案例）

**项目背景**：某金融集团决定基于内部的研报、合规手册、产品说明书等文档，训练一个内部专属的金融问答模型。数据规模约 500GB（PDF + Word 格式），覆盖近 10 年的内部文档积累。以下比例和规模用于说明风险类型，不代表特定企业公开事件。

**T+0（数据摸底）**：数据工程师拿到了集团 IT 部门提供的文档目录清单，开始批量解析 PDF 文件。工程推进顺利，短时间内完成了文档解析和初步清洗，生成一批候选训练数据。

**T+15（合规部介入）**：集团合规部在例行风险排查时，发现数据集中包含大量来自第三方机构（如监管机构官网、评级机构、外部律所）的文件副本——这些文件被历史沉积在内部 OA 系统中，但其版权并不属于集团本身。部分文件甚至包含"未经许可不得复制或传播"的声明。

**T+16（紧急暂停）**：法务团队紧急叫停了训练任务，要求对数据集进行来源审查。审查发现，部分文档存在版权归属不明或明确属于第三方的问题，必须从训练集中剔除。

**T+25（修复完成）**：数据团队对每一类文档进行了来源分类标注，建立了内部版权清单，剔除了所有版权存疑的第三方文档，并对保留的文档补充了授权证明（内部文档引用监管法规的部分，经合规确认属于"合理引用"范围）。

**核心教训**：企业内部文档并不等于版权自有的数据。在启动采集之前，应当对所有数据来源进行版权归属的系统性排查，而不是在工程完成后才请合规部介入——后者的返工成本远高于前者。推荐在采集管线的第一步就引入"来源归属检查"节点，要求每个文件标注：版权方 / 内部创作 / 第三方引用 / 来源不明，并由文档所有人（业务部门 Owner）进行确认签字。

---

## 本章小结

本章从源头质量如何约束模型能力出发，建立了预训练数据源体系的认知框架。章节构建了涵盖八类核心数据源的分层地图，并通过风险矩阵（表4-1）和配比策略矩阵（表4-2）为工程决策提供可操作的量化工具。在采集流水线部分，本章说明了直接使用 WET 的风险，给出了基于 Trafilatura 的高质量 WARC 解析实现，并建立了“每条数据都有出生证明”的元数据存证标准。版权治理部分引入白名单、灰名单、黑名单的三级管理机制，配合许可证自动分类代码，为商业化 LLM 团队提供可落地的合规工程方案。两个案例分别从技术和法律两个维度说明，源头治理是预训练数据工程的第一道质量门禁。

进入下一章，我们将在本章采集到的原始数据基础上，讨论**第5章 清洗、去重与去污染**。源头治理决定可以送入清洗管线的语料上限，而清洗管线决定哪些样本能够最终进入训练集。两章共同构成文本预训练数据工程的质量守门体系。

## 参考文献

Barbaresi A (2021) Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction. In: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics, pp 122-131.

Blecher L, Cucurull G, Scialom T, Stojnic R (2023) Nougat: Neural Optical Understanding for Academic Documents. arXiv preprint arXiv:2308.13418.

Grattafiori A, Dubey A, Jauhri A, Pandey A, Kadian A, Al-Dahle A, Letman A, Mathur A, Schelten A, Vaughan A, others (2024) The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783.


Lopez P (2009) GROBID: Combining Automatic Bibliographic Data Recognition and Term Extraction for Scholarship Publications. In: Proceedings of the 13th European Conference on Digital Libraries, pp 473-474.

Li J, Zhang Y, Yu H, Ma X, Chen Y, Jiang H, Dang K, Goyal T, Keh S, Sherborn M, others (2024) DataComp-LM: In search of the next generation of training sets for language models. arXiv preprint arXiv:2406.11794.

Penedo G, Kydlíček H, Ben Allal L, Lozhkov A, Mitchell M, Raffel C, von Werra L, Wolf T (2024) The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. arXiv preprint arXiv:2406.17557.

Yu S, Liu Z, Xiong C (2025) Craw4LLM: Efficient Web Crawling for LLM Pretraining. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics. arXiv preprint arXiv:2502.13347.
