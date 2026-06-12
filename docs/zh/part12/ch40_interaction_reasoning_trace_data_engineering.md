# 第40章：交互控制与推理轨迹数据工程

<div class="chapter-authors">Fengxin Chen；Xuan Li</div>

## 摘要

本章讨论非静态样本如何记录控制条件、状态迁移和推理预算，可控语音交互与隐式/显式推理轨迹两个专项案例。VoiceStyleControl 关注语义内容与声音风格的双通道监督，Latent-Switch-69K 关注长 CoT 压缩、latent placeholder 和 supervision mask。两者共同体现了交互式数据工程的重点：不仅要保存输入输出，还要保存控制变量、隐藏状态和可验证边界。

## 关键词

可控语音；风格控制；推理轨迹；Latent-Switch；监督掩码；交互数据工程

## 案例A：VoiceStyleControl：语义响应与声音风格控制

### 案例A.0：学习目标

通过本章学习，读者应能够：

- 解释语音交互数据为何必须在语义层之外显式记录声音条件、情绪与离散语音 token，而不能沿用纯文本对话的监督目标。
- 区分语义通道、风格通道与声学监督通道各自承担的字段责任，并理解输入侧用户状态与输出侧助手目标的分离原则。
- 掌握 S2SEmoControl 与 TTSSpeakerControl 两个子集在规模、字段结构与训练价值上的互补关系。
- 设计覆盖文本一致、音频可用、声音条件一致、情绪可感知与授权可追溯的多维样本验收规则。
- 识别语音身份、授权许可、情绪滥用、防伪溯源与隐私保护等风险，并在数据流程中加以治理。

### 案例A.1：为什么语音对话需要显式风格控制

普通文本对话样本通常由角色、上下文、用户请求和助手回答构成。只要角色边界、文本长度、安全标签和训练 mask 清楚，模型就能在文本 token 上学习输入输出映射。语音样本则多出一层文本无法替代的声学状态：采样率、时长、静音、响度、噪声、说话人身份、韵律、情绪和离散语音 token 都会影响训练结果。仅有回答文本只能说明“说了什么”，不能说明“应该怎么说”。

因此，可控语音交互数据和普通 ASR/TTS 语料的差异，首先不是字段数量更多，而是问题定义变了。ASR 关心“这段声音对应哪段文字”，普通 TTS 关心“这段文字能否被自然读出来”；可控语音交互还要关心“这段回答应当以什么声音、什么情绪、什么强度进入对话”。如果这些条件不被显式表达，模型只能把声音差异当成训练音频里的随机变化，很难在推理时稳定响应“用某种情绪说”“用某类声音说”这样的控制条件。

第一，语音对话需要把“内容”和“表达”分开。用户说了什么、助手应该回答什么，是语义层；这句话由什么声音说出、语速快慢、能量高低、停顿如何、情绪是否明显，是表达层。文本对话数据通常只要组织好语义层，语音生成数据却必须让表达层也成为训练监督的一部分。否则，同一句回答在中性、开心、害怕或愤怒状态下的差别就会被数据管线抹平。

第二，语音对话需要区分“理解用户声音”和“生成助手声音”。真实系统中，用户可能焦急、愤怒、犹豫，也可能口音很重、背景嘈杂；助手则通常需要依据产品设定保持稳定的声音条件和情绪策略。一个客服助手面对愤怒用户时不应自动变得愤怒，一个陪伴助手也不应在每轮对话中无缘由地改变音色。显式风格控制的意义，就是让数据在样本层就区分输入侧状态和输出侧目标，而不是默认二者相同。

第三，语音对话需要把情绪从“文本描述”落实到“声音表现”。开心、愤怒、害怕、中性、悲伤这些状态不只是标签，它们会体现在音高、能量、语速、停顿和韵律上。对模型来说，真正的学习目标不是记住某个情绪词，而是在给定目标表达状态时，生成与之相符的语音。也正因为如此，可控语音数据必须同时保存文本内容、目标风格说明和对应语音监督，让情绪控制能够进入生成过程。

第四，语音对话需要可复查的声学监督。文本可以直接作为 token 序列进入训练，语音则要经历音频文件、采样率、时长、响度、静音、离散语音 token 等一系列工程处理。显式风格控制不能只在旁边写一句“开心地说”，还要有一段实际语音作为目标，让模型知道这种风格条件在声学上应该如何呈现。

从产品体验看，这种边界非常关键。一个陪伴型助手可以被设计为温和、稳定、少戏剧化；一个有声书角色可以被设计为情绪更强、角色感更明显；一个客服助手则通常需要在用户愤怒时保持中性和清晰。三者都可能使用同一套语义回答能力，但它们对声音身份、情绪强度和风险边界的要求不同。如果训练样本没有显式区分这些条件，模型只能把声音风格当作音频中的随机噪声，推理时就很难稳定控制。

从数据工程看，显式风格控制还改变了样本验收方式。文本样本只要用户问题与助手回答匹配，通常就能进入候选池；语音样本则必须同时满足文本一致、音频可用、目标声音条件一致、情绪可感知和授权可追溯。任何一个维度失败，都会影响训练：文本对但声音条件错，会削弱条件控制；声音条件对但情绪错，会削弱情绪控制；情绪明显但内容危险，则会把风险行为转化为更有感染力的输出。

### 案例A.2：数据集概览：S2S 与 TTS 两个互补子集

VoiceStyleControl 由两类任务共同组成：一类是语音到语音的对话生成，另一类是文本条件下的可控语音生成。两者都服务于同一个目标：让模型能够根据语义内容、声音条件和情绪风格生成带情绪的语音，但它们提供的监督角度不同。

VoiceStyleControl 共包含 154,906 条样本。其中，S2SEmoControl 包含 20,117 条样本，占全量约 13.0%，面向 style-controllable speech-to-speech dialogue generation；TTSSpeakerControl 包含 134,789 条样本，占全量约 87.0%，面向 controllable text-to-speech generation。前者更接近真实语音助手场景：模型要理解用户说出的请求，再生成助手侧语音回答；后者更集中地训练模型根据风格文本、声音条件和情绪风格生成目标语音。

**表42-1：VoiceStyleControl 数据规模与情绪分布**

| Emotion | S2SEmoControl | TTSSpeakerControl | Total | Total ratio |
|---|---:|---:|---:|---:|
| happy | 4,050 | 38,500 | 42,550 | 27.5% |
| angry | 4,104 | 38,054 | 42,158 | 27.2% |
| fearful | 4,010 | 24,925 | 28,935 | 18.7% |
| neutral | 3,825 | 0 | 3,825 | 2.5% |
| sad | 4,128 | 33,310 | 37,438 | 24.2% |
| **Total** | **20,117** | **134,789** | **154,906** | **100.0%** |

表42-1显示，S2SEmoControl 的五类情绪接近均衡，每类约 3.8k 至 4.1k；TTSSpeakerControl 则覆盖 happy、angry、fearful、sad 四类表达性情绪，不显式包含 neutral。这个设计并不是偶然的。S2S 对话需要 neutral 作为稳定基准，否则模型容易把所有回答都学成高强度情绪表达；TTS 可控生成子集样本更多，则把容量集中在“开心地说”“愤怒地说”“有点害怕”“伤心地说”等更需要声学变化的表达上。

从记录组成看，两个子集都不是单纯的“文本 + 音频”。每条样本至少包含五类信息：任务来源与任务类型、文本侧内容、声音与情绪条件、语音生成监督、基础音频配置。这些信息共同决定一条语音样本是否能用于训练条件化、带情绪的语音生成：任务信息决定加载方式，文本内容提供语义目标，声音与情绪条件规定生成风格，语音监督提供可学习的声学目标，基础音频配置保证训练和评测能够复现。

两个子集分别承担“能力底座”和“交互落地”的角色。TTSSpeakerControl 样本量更大，直接教模型把自然语言风格描述、声音条件和情绪风格映射到目标声音；S2SEmoControl 样本量较小，但更接近真实语音助手：模型要先理解用户侧语音，再生成助手侧语音回答。联合使用时，TTS 子集提供风格生成的稳定监督，S2S 子集则把这种能力放回对话语境中，让模型学习用户声音状态和助手生成目标之间的转换。

因此，VoiceStyleControl 不能被简单理解为一个 TTS 数据集。普通 TTS 语料的核心监督是“给定文本，读出文本”；VoiceStyleControl 的核心监督是“给定语义内容和风格条件，生成符合对话目标的声音”。前者主要关心发音、自然度和音质，后者还要关心用户状态、助手声音条件、情绪选择、跨轮一致性和安全边界。数据目标一旦不同，schema、配平、切分和评测都会随之改变。

### 案例A.3：样本 schema：语义通道与风格通道分开建模

![图40-1：语义响应与风格控制双通道 schema](../../images/part12/ch40_01_dual_channel_schema.svg)

*图40-1：语义响应与风格控制双通道 schema。语义通道回答“说什么”，风格通道回答“用什么声音和情绪说”，声学监督通道把二者绑定到音频文件、speech token 和采样配置。*

图40-1展示了 VoiceStyleControl 的核心结构。语义通道负责 `query`、`answer`、`task`、`language` 等字段；风格通道负责 `query_gender`、`answer_gender`、`query_mood`、`answer_mood`、`query_id`、`answer_id` 等字段；声学监督通道负责 `query_audio_path`、`answer_audio_path`、`query_token_25hz`、`answer_token_25hz` 和 `sample_rate`。三个通道在训练记录中合并，但在构建、质检和评测时必须分开检查。

分通道建模能够定位失败来源。若模型生成的回答文本正确但音色不稳定，问题通常在风格通道或参考语音池；若声音条件正确但读错了字，问题在语义通道、ASR 反查或合成文本对齐；若音频能播放但 token 路径无法读取，问题在声学监督通道或封装 manifest。把所有信息都压成一个自由文本 prompt，虽然便于快速拼装样本，却会让后续的数据修复和实验归因变得困难。

S2SEmoControl 的记录表达了从用户侧 `(query_audio, query_text, query_gender, query_mood)` 到助手侧 `(answer_text, answer_audio, answer_gender, answer_mood)` 的映射。中文对话内容、声音条件、情绪标签、音频文件和 speech token 被绑定在同一条记录中，因此它不是“文本问答 + 附件音频”的松散组合，而是一条完整的语音交互训练样本。

```json
{
  "uuid": "1977946a067ee3442",
  "_id": "6750567505b5d5170356ae61",
  "source": "S2SEmoControl",
  "task": "S2S",
  "query": "给我讲个小故事呗。",
  "answer": "好的，让我给您编一个小故事。从前有一个非常勤奋的小夜莺...",
  "query_gender": "female",
  "answer_gender": "male",
  "query_mood": "neutral",
  "answer_mood": "neutral",
  "language": "zh",
  "sample_rate": 16000,
  "query_id": "female-neutral-1",
  "answer_id": "male-neutral-2",
  "query_token_25hz": "S2SEmoControl/.../query_token_0.ark:3121",
  "query_audio_ark": "S2SEmoControl/.../query_audio_0.ark:1024",
  "query_audio_path": "S2SEmoControl/.../1977946a06cf564f1-query.wav",
  "answer_token_25hz": "S2SEmoControl/.../answer_token_0.ark:22637",
  "answer_audio_ark": "S2SEmoControl/.../answer_audio_0.ark:8192",
  "answer_audio_path": "S2SEmoControl/.../1977946a06cf564f1-answer.wav"
}
```

这条样本中，用户说“给我讲个小故事呗。”，助手回答“好的，让我给您编一个小故事。从前有一个非常勤奋的小夜莺...”。`query_gender` 为 `female`，`answer_gender` 为 `male`；`query_mood` 和 `answer_mood` 都是 `neutral`。训练时，`query_audio_path` 和 `query_token_25hz` 可以作为语音理解输入，`query` 提供转写后的语义锚点；`answer` 是语义目标，`answer_token_25hz` 和 `answer_audio_path` 是语音生成监督；`answer_gender` 与 `answer_mood` 规定输出声音的风格条件。

TTSSpeakerControl 则把控制能力集中到 text-to-speech 形态。输入文本被拆成两部分：`text` 描述声音应该如何表达，`answer` 才是要读出的内容。例如 `text` 为“女，有点害怕，手心冒汗，声音发抖”，`answer` 为“你快跑，这里不安全”。这样的记录表明，TTS 子集不是给句子随机贴 mood，而是在构造 style-content pair：自然语言风格描述、结构化标签和待合成内容要相互支持。

```json
{
  "uuid": "c6810929-8962-4cc1-b3b5-aadd4cbb1106",
  "_id": "197b764f5a31c2-female-fearful",
  "source": "TTSSpeakerControl",
  "task": "TTS",
  "text": "女，有点害怕，手心冒汗，声音发抖",
  "answer": "你快跑，这里不安全",
  "answer_gender": "female",
  "answer_mood": "fearful",
  "language": "zh",
  "sample_rate": 16000,
  "prompt": "女，有点害怕，手心冒汗，声音发抖",
  "answer_id": "female-fearful-1",
  "answer_token_25hz": "TTSSpeakerControl/.../answer_token_0.ark:1379",
  "answer_audio_ark": "TTSSpeakerControl/.../answer_audio_0.ark:4096",
  "answer_audio_path": "TTSSpeakerControl/.../c6810929-8962-4cc1-b3b5-aadd4cbb1106-answer.wav"
}
```

综合 S2S 与 TTS 两类样本，VoiceStyleControl 的字段可以分成六层：任务标识、文本内容、声音条件、情绪条件、语音监督、基础音频配置。S2S 样本同时包含用户侧和助手侧，因此字段会区分查询侧与回答侧；TTS 样本只生成回答侧语音，因此字段更集中。`language` 固定语种，`sample_rate` 固定音频采样配置；这些基础字段是训练加载和评测复现的底层契约，不能只靠路径名或文件夹约定隐式推断。

**表42-2：说话人、情绪与采样标签字段说明**

| 标签层 | 字段 | 取值/例值 | 分布或工程要求 |
|---|---|---|---|
| 查询侧说话人 | `query_gender` | `female` / `male`，如 `female` | 按 query 侧单独统计。 |
| 回答侧声音条件 | `answer_gender` | `male` / `female` | 训练前应按回答侧 gender、mood 和参考声音条件监控配平，避免输出声音偏置。 |
| 查询侧情绪 | `query_mood` | `happy`、`angry`、`fearful`、`neutral`、`sad` | S2SEmoControl 五类接近均衡。 |
| 回答侧情绪 | `answer_mood` | 同上 | 总量以表42-1为准；TTSSpeakerControl 不显式包含 `neutral`。 |
| 语种与采样 | `language` / `sample_rate` | `zh` / `16000` | 作为加载、重采样和评测复现字段，非路径隐式推断。 |
| 参考声音引用 | `query_id` / `answer_id` | `female-neutral-1` 等 | 指向授权参考语音池中的风格实例，不暴露真实身份。 |

在 VoiceStyleControl 中，emotion 分布只是第一层配平信息。真正进入训练和评测时，样本还会沿着输入侧与输出侧拆开：`query_gender × query_mood` 描述用户语音的状态分布，`answer_gender × answer_mood` 描述助手生成语音的目标分布，参考声音 ID 则约束同一种声音条件在不同文本和情绪下的复用方式。语言和采样率看似基础，却决定了加载、重采样和音频指标是否可比。把这些轴放在一起观察，才能判断某类情绪是否只集中在某个声音条件上，某个参考音色是否过度出现在训练集和评测集中，以及模型失败究竟来自语义、声音条件还是情绪控制。

落到数据合成阶段，上述字段差异会体现为两套条件组织方式：S2SEmoControl 需要同时处理 query/answer 两侧的参考语音选择与情绪注入，TTSSpeakerControl 则把风格描述与待朗读内容拆开后再合成回答侧语音。具体合成逻辑见 42.4 第四步与第五步；本节先把字段契约固定下来。

联合 JSON Schema 按任务类型约束必填字段；生产级 manifest 还应增加枚举约束、路径存在性校验、文件哈希、授权 ID、tokenizer 名称、tokenizer 版本和 token 帧率声明。

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "VoiceStyleControlRecord",
  "type": "object",
  "required": [
    "source",
    "task",
    "answer",
    "language",
    "sample_rate",
    "answer_audio_path"
  ],
  "oneOf": [
    {
      "title": "S2SEmoControl",
      "required": [
        "query",
        "query_gender",
        "answer_gender",
        "query_mood",
        "answer_mood",
        "query_id",
        "answer_id",
        "query_audio_path",
        "answer_audio_path",
        "query_token_25hz",
        "answer_token_25hz"
      ],
      "properties": {
        "task": {
          "const": "S2S"
        }
      }
    },
    {
      "title": "TTSSpeakerControl",
      "required": [
        "text",
        "answer_gender",
        "answer_mood",
        "answer_id",
        "answer_token_25hz",
        "answer_audio_path"
      ],
      "properties": {
        "task": {
          "const": "TTS"
        }
      }
    }
  ],
  "properties": {
    "source": {
      "type": "string"
    },
    "task": {
      "enum": ["S2S", "TTS"]
    },
    "query": {
      "type": "string",
      "description": "spoken user query 的转写文本，仅 S2S 使用"
    },
    "text": {
      "type": "string",
      "description": "自然语言风格描述，仅 TTS 使用"
    },
    "answer": {
      "type": "string",
      "description": "assistant response 或待合成内容"
    },
    "query_gender": {
      "type": "string"
    },
    "answer_gender": {
      "type": "string"
    },
    "query_mood": {
      "type": "string"
    },
    "answer_mood": {
      "type": "string"
    },
    "language": {
      "type": "string"
    },
    "sample_rate": {
      "type": "integer"
    },
    "query_id": {
      "type": "string"
    },
    "answer_id": {
      "type": "string"
    },
    "query_token_25hz": {
      "type": "string"
    },
    "answer_token_25hz": {
      "type": "string"
    },
    "query_audio_ark": {
      "type": "string"
    },
    "answer_audio_ark": {
      "type": "string"
    },
    "query_audio_path": {
      "type": "string"
    },
    "answer_audio_path": {
      "type": "string"
    }
  }
}
```

联合 schema 将训练入口拆成三部分：语义输入是 `query`、`text` 或 `answer` 文本 token，风格输入是 `query_gender`、`answer_gender`、`query_mood`、`answer_mood` 与参考声音 ID，声学目标是回答侧 speech token 或音频。`answer_gender`、`answer_mood` 不能只留在离线 metadata 中，它们必须在 dataloader 中被映射为控制条件或条件文本，否则模型不会真正获得可控生成能力。

训练样本进入 dataloader 后，会从标准 schema 投影成不同任务视图。S2S 视图可以是 `query_audio + answer_gender + answer_mood -> answer_token`，也可以加入 `query` 转写作为辅助语义输入；TTS 视图可以是 `text + answer + answer_gender + answer_mood -> answer_token`。评测视图则反向固定某些字段、改变另一些字段，例如固定 `answer` 改变 `answer_mood`，或固定 `answer_mood` 改变 `answer_id`。这种“记录契约稳定、训练视图可变”的设计，服务的是可控语音生成实验，而不是额外的身份识别或声纹建模实验。

### 案例A.4：构建流水线：从文本对话到可控语音记录

![图40-2：VoiceStyleControl 数据构建流水线](../../images/part12/ch40_02_voice_style_data_pipeline.svg)

*图40-2：VoiceStyleControl 数据构建流水线。文本对话或风格内容先被赋予 speaker 与 emotion 条件，再通过授权参考语音池生成或采集音频，最后经过 token 化、质检、配平和封装。*

VoiceStyleControl 的构建可以分为七步：文本对话或风格内容生成、风格属性分配、授权参考语音池准备、语音合成或采集、离散语音标记、质检配平、封装发布。每一步都同时影响语义质量、风格质量和合规风险。

这条流水线不是简单的顺序生产线，而是一组连续的数据门禁。文本内容生成之后，要判断语义是否适合指定情绪；参考语音选择之后，要判断授权是否覆盖当前任务；语音合成之后，要判断音频、文本、声音条件和 emotion 是否同时通过。任何一步发现问题，都不应简单“带病流入下一步”，而要回到对应队列修复。否则，后续评测只能发现模型不稳定，却很难解释不稳定来自哪里。

第一步是生成或收集文本内容。S2SEmoControl 消费经过清洗的对话 JSONL，每条记录包含用户 `query` 和助手 `answer`，覆盖日常请求、情绪表达、故事、解释、提醒等场景；answer 保持自然、完整，并保留安全边界。TTSSpeakerControl 则使用 Qwen3-8B 配合情绪特定 prompt 生成 style-content pair，让风格描述与待说内容相互支持。例如 fearful 样本可以更急迫，sad 样本可以更低落，但不能把情绪标签变成危险诱导。

文本内容的验收不只看语法是否通顺，还要看情绪和语义是否相容。`fearful` 可以对应“你快跑，这里不安全”，但不应对应轻松闲聊中的夸张恐吓；`angry` 可以用于角色化表达，但不应把辱骂、威胁或歧视性内容当作情绪增强。对话生成阶段如果不设边界，随后的语音合成会把风险文本转化为更有冲击力的声音，风险会被声学表达放大。

第二步是分配风格属性。S2S 需要分别为 query 侧和 answer 侧赋予 gender 与 mood，TTS 则为回答侧赋予 gender 与 mood，并在 `text` 中写出自然语言风格描述。分配策略要同时考虑均衡和组合覆盖：均衡保证每种 emotion 都有足够样本，组合覆盖则让模型见过不同用户风格到不同助手风格的迁移。如果数据里只有同 gender、同 mood 的组合，模型很容易把输入风格和输出风格绑定在一起，削弱回答侧控制能力。

组合覆盖尤其影响 S2S 子集。用户侧 angry 并不意味着助手侧也要 angry，用户侧 fearful 也不意味着助手侧要同样 fearful。相反，很多真实产品需要助手在用户高压情绪下保持中性、清楚和可执行。数据构建时应保留足够多的跨组合样本，例如 female-angry query 对应 male-neutral answer，或 male-sad query 对应 female-neutral answer。这样模型才能学会把用户状态作为理解信号，而不是直接复制成输出风格。

第三步是准备参考语音池。VoiceStyleControl 使用多说话人、多情绪参考池，并通过 CosyVoice2 以 zero-shot voice cloning 方式合成指定风格语音。工程关键不是“克隆得越像越好”，而是“可授权、可复用、可撤回”。参考音频应记录参考声音 ID、emotion condition、采集时间、用途范围、授权状态和撤回状态；`query_id` 与 `answer_id` 只应暴露工程引用，不应包含真实姓名或可反查身份的信息。

第四步是语音合成或采集。S2S 需要分别生成 query speech 和 answer speech，并让两侧音频与文本逐条绑定；TTS 则按 `text` 和 `answer` 生成回答侧语音（具体实现见第五步示例）。合成时应固定或显式记录采样率，控制响度、静音、最大时长和文件编码，避免 dataloader 在训练时因为音频长度或格式异常而不稳定。若采用真实采集，还要额外处理环境噪声、麦克风差异、说话人疲劳和第三方背景声。

以下以 S2SEmoControl 为例，展示 schema 字段如何进入合成过程：`query_id` 与 `answer_id` 选择两侧参考语音；当 `answer_mood` 不是 `neutral` 时，情绪指令会拼入 query 侧合成文本，使输入语音携带输出风格控制意图。

```python
def build_synthesis_inputs(record):
    language = record["language"]
    query_content = record["query"]
    answer_content = record["answer"]
    answer_mood = record["answer_mood"]
    query_prompt_id, answer_prompt_id, record = select_prompt_speech(record)

    if answer_mood != "neutral":
        prompt = random.choice(INSTRUCT[language]).format(mood=answer_mood)
        record["prompt"] = prompt
        if random.random() < 0.5:
            query_content = prompt + query_content
        else:
            query_content = query_content + prompt

    return (
        record,
        PROMPT_TEXT[language][query_prompt_id],
        query_content,
        PROMPT_TEXT[language][answer_prompt_id],
        answer_content,
    )

record, q_instruct, q_content, a_instruct, a_content = build_synthesis_inputs(record)
language = record["language"]
q_tokens, q_speech = backend.compute_zeroshot_speech_token(
    q_instruct, audio_dict[language][record["query_id"]], q_content
)
a_tokens, a_speech = backend.compute_zeroshot_speech_token(
    a_instruct, audio_dict[language][record["answer_id"]], a_content
)
```

这个示例体现了 S2S 侧的关键分支：`answer_mood` 决定是否注入情绪指令，`q_tokens`、`a_tokens` 与对应波形则与 manifest 中的 `query_token_25hz`、`answer_token_25hz` 字段对接。

第五步是离散语音标记。语音生成训练需要把声学目标整理为离散 speech token，使生成任务可以被组织为序列建模问题。通用做法是对已有波形用 S3Tokenizer 等 tokenizer 编码；VoiceStyleControl 则走 CosyVoice 生成式路径——合成时同步产出 speech token 并解码为可播放音频，因此本仓库并不存在单独的「先合成、再标记」后处理步骤。S2S 记录写入 `query_token_25hz` 和 `answer_token_25hz`，TTS 记录写入回答侧 `answer_token_25hz`；帧率为 25Hz（CosyVoice2 的 `token_frame_rate`），manifest 字段名与之对应。数据发布时仍应绑定 tokenizer 名称、版本、帧率、码本配置和重建方式。训练集最怕“同名字段不同含义”：如果同一字段在不同批次中被不同帧率或不同 tokenizer 版本生成，模型会在时序长度和声学粒度上接收到混乱监督。

TTSSpeakerControl 采用另一条合成路径：`answer` 是要说出的内容，`text` 或 `prompt` 是风格描述。从数据工程角度看，关键不是展开 CosyVoice 内部 flow 与 vocoder 的全部参数，而是把一条稳定的数据流固定下来：先从记录中抽取内容和风格指令，再调用合成函数得到回答侧 token 与音频，最后把监督地址写回同一条 manifest 记录。

```python
for sample_idx, record in id2meta:
    text_content, instruction_text = extract_tts_fields(record)
    if len(text_content) > 512:
        continue

    sample_key = str(record.get("uuid") or record.get("id") or sample_idx)
    speech_token, speech_audio = compute_tts_speech_token(
        text_content, instruction_text, SPK_ID
    )
    token_offset = answer_token_writer.write(sample_idx, speech_token.tobytes())
    audio_offset = answer_audio_writer.write(sample_idx, speech_audio.tobytes())

    record["answer_token_25hz"] = f"{paths.answer_token_ark}:{token_offset}"
    record["answer_audio_ark"] = f"{paths.answer_audio_ark}:{audio_offset}"
    record["answer_audio_path"] = str(
        paths.answer_wav_dir / f"{sample_key}-answer.wav"
    )
    wavfile.write(record["answer_audio_path"], ARK_SAMPLE_RATE, speech_audio)
    write_jsonl_record(jsonlf, record)
```

这个示例对应的是“自然语言风格描述如何变成可训练语音监督”的核心链路：`instruction_text` 进入合成函数，`speech_token` 成为后续训练可以直接建模的离散目标，`speech_audio` 用于听感质检、反向 ASR 和人工抽检。token offset、audio offset 和 wav 路径被写回同一条记录后，样本才真正具备可追溯性。

第六步是质检、配平和切分。质检不应只看音频能否播放，还要检查文本与音频是否一致、目标声音条件是否匹配、emotion 是否可感知、音质是否稳定、路径是否存在、token 是否能读取。配平也不只按 emotion 总量做，还要按 `task`、`language`、`sample_rate`、参考声音 ID、文本长度和音频时长监控。切分时应按参考声音 ID 做隔离，避免同一参考音色同时出现在训练集和测试集，造成声音条件评测虚高。

第七步是封装。最终样本可以存为 JSONL、Parquet 或 Hugging Face Dataset 格式，但训练清单要保留音频路径、token 路径、哈希、授权状态和数据版本。音频文件、token ark 文件和 metadata 不应由人工命名约定松散关联，而应由 manifest 严格绑定。只有这样，样本被重合成、重标注或下架时，团队才能定位受影响的训练版本。

封装产物不只是 JSONL、Parquet 或 Hugging Face Dataset，还包括描述数据边界的数据卡。数据卡记录样本总量、子集构成、emotion 分布、gender 字段分布、参考声音 ID、语言、采样率、tokenizer 版本、授权范围和切分策略，并区分训练条件、审计元数据与公开版本中的匿名化字段。这个边界说明可以防止 `answer_id` 被误用为真实身份标签，也可以防止 `mood` 被当成无需验证的可靠事实。

### 案例A.5：质量评估与闭环修复

![图40-3：质量评估与数据飞轮闭环](../../images/part12/ch40_03_quality_loop.svg)

*图40-3：质量评估与数据飞轮闭环。自动校验、反向 ASR、风格评估和人工抽检共同形成问题样本队列，再回流到重合成、重标注、降权或剔除。*

可控语音交互数据的质量评估需要同时覆盖语义、声音、情绪、音频和安全。单独听起来“像人声”的样本并不一定合格：它可能读错文字，可能声音身份不匹配，可能情绪过强，也可能在危险场景中使用了不恰当的恐惧语气。质量系统应把自动指标与人工复核组合成闭环，问题样本进入重合成、重标注、降权或剔除队列。

质量门禁应分成“硬失败”和“软风险”。路径不存在、采样率错误、音频损坏、token 不可读、ASR 反查严重不一致，通常属于硬失败，应直接拦截。情绪强度略弱、自然度一般、声音条件听感处于边界，则可以进入软风险队列，根据任务重要性选择重合成、降权或人工复核。把所有问题都当成一票否决，会浪费可修复样本；把所有问题都放行，又会让控制信号被噪声稀释。

**表42-3：质量评估指标表**

| 评估层面 | 核心问题 | 自动指标 | 人工复核要点 | 不合格处理 |
|---|---|---|---|---|
| 语义一致性 | 回答是否回应用户意图，TTS 内容是否被正确读出 | ASR 反转写 CER/WER、语义相似度、意图命中率 | 是否答非所问、遗漏关键信息、产生危险建议 | 重写文本、重合成、剔除 |
| 声音条件一致性 | 输出是否匹配目标 `answer_gender`、`answer_mood` 和参考声音条件 | 字段一致性校验、自动/人工性别核验、参考音色听感抽检 | 是否出现目标条件错误、跨样本串音、音色过像未授权真人 | 重选参考音频、重合成、降权或隔离 |
| 情绪控制 | 目标 mood 是否被稳定表达 | 情绪分类准确率、混淆矩阵、F0/能量/语速统计 | 情绪是否过强、与语义冲突或诱导操控 | 重标注、调低强度、剔除 |
| 音频质量 | 音频能否作为生成监督 | SNR、响度、静音比例、裁剪率、MOS/NISQA | 爆音、断句、机械音、背景噪声 | 降噪、重采样、重合成 |
| 对话自然度 | S2S 回答是否自然，角色是否稳定 | 多轮连贯性评分、延迟与时长分布 | 语气是否突兀、角色不一致、风格反复跳变 | 重排、补充上下文、人工审核 |
| 安全合规 | 样本是否可授权、可追溯、可撤回 | 授权记录完整率、水印命中率、审计日志覆盖率 | 是否存在冒充、诱导、敏感身份复刻风险 | 封禁、脱敏、下架和审计 |

语义一致性可以通过反向 ASR 建立第一层自动检查。将合成音频转写回文本，计算 CER/WER，并与 `answer` 比较；对于 S2S，还要检查 answer 是否回应 query。若“你快跑，这里不安全”被合成为“你快跑，这里很安全”，音频质量再高也必须剔除。语义相似度和 LLM-as-judge 可以辅助定位问题，但在安全敏感或情绪强烈样本中仍要保留人工抽检。

声音条件一致性关注的是生成结果是否符合样本中的 `answer_gender`、`answer_mood` 和参考声音条件，而不是训练或评测一个独立的身份识别模型。对于回答侧，`answer_id` 应与 `answer_gender`、`answer_mood` 一致；对于 query 侧，`query_id` 应与用户侧标签一致。如果同一个 `answer_id` 在不同样本中表现出明显不同音色，需要回查参考池、合成参数和 token 化流程。人工听辨和自动核验只是质检手段，不改变数据集的训练目标。

情绪控制评测不能只看分类器置信度。happy 往往表现为更高能量和更快节奏，sad 可能表现为更慢语速和更低能量，fearful 可能伴随颤抖、急促或不稳定停顿，angry 可能表现为更强能量和更硬语气。但中文表达、说话人差异和内容语义都会改变声学表现，因此评测目标应是“可感知且与文本相容”，而不是把每一种情绪写成固定声学模板。

闭环修复要保留问题类型。语义错误通常回到文本生成或 ASR 反查；声音条件错误回到参考语音选择或合成参数；emotion 错误回到风格描述、情绪标签或合成模型；音质错误回到波形处理；合规错误进入隔离、下架和审计流程。每次修复都应生成新版本，而不是覆盖源文件。这样，后续模型效果变化才能追溯到数据变更，而不是变成不可解释的训练波动。

### 案例A.6：评测协议：让控制能力可比较

评测集应从训练集构造逻辑中独立出来，尤其要避免同一参考声音 ID 同时出现在训练和测试中。对于 S2SEmoControl，评测样本需要覆盖不同 query 情绪到不同 answer 情绪的组合；对于 TTSSpeakerControl，评测样本需要覆盖同一 `answer` 在不同 `text`、`answer_gender`、`answer_mood` 条件下的对比。一个有效评测协议不只问“生成声音好不好听”，还要问“同一句话在不同控制条件下是否真的不同，且不同得合理”。

评测集可以拆成三类切片。第一类是常规切片，覆盖训练集中主要任务分布，用来观察整体可用性。第二类是反事实切片，固定文本或参考声音 ID，只改变 `answer_mood` 或 `answer_gender` 条件，用来检查控制字段是否生效。第三类是安全切片，包含身份冒充、高压情绪、敏感职业、金融验证码、医疗建议等场景，用来检查模型是否会把“可控生成”误用为“可控操控”。这三类切片的结论不能混成一个总分，否则高音质样本可能掩盖高风险行为。

语义评测分为内容保真和对话相关性两层。内容保真检查 TTS 输出是否准确读出 `answer`，S2S 输出是否可被转写为与目标 answer 语义一致的文本。对话相关性检查 S2S 的 answer 是否回应 query，而不是只生成流畅但无关的句子。评测中可以组合 ASR 反转写、语义相似度、LLM-as-judge 和人工审核，但要保存判分 prompt、模型版本和人工指南，避免评测随时间漂移。

声音条件评测也要分层。结构标签层检查 `answer_gender`、`answer_mood` 与样本目标是否一致；听感层检查生成音频是否符合对应参考声音条件和情绪表达；隔离层检查模型是否过度接近未授权个体或泄露训练集中某个真实声纹。评测目标不是构造声纹相似度排名，也不是把“越像某个真实人”当成唯一优化方向，而是确认模型能否在样本条件下生成合理、合规的带情绪语音。

情绪评测需要构造反事实集合。例如固定一句中性内容，分别请求 happy、angry、fearful、sad；或固定 `answer_gender`，改变 `answer_mood`；也可以固定 `answer_mood`，改变 `answer_gender`。这种 paired evaluation 能暴露模型是否真的使用控制字段。如果所有输出只在音量上变化，而语速、停顿和韵律没有随 `answer_mood` 改变，说明模型可能只学到了浅层强度调节。

音频质量评测包含客观指标与主观评分。客观指标覆盖时长分布和自动 MOS 等；主观评分关注自然度、可懂度、情绪可信度和对话舒适度等。安全性评测则应成为发布门禁的一部分：身份冒充、敏感职业、金融验证码、医疗建议、未成年人和高压情绪诱导等场景，都要检查系统是否会在不该使用强情绪或特定音色时仍然生成输出。

评测结果还应回写到数据版本，而不是只保存在模型报告里。若某一版模型在 fearful 上情绪分类准确率高但人工舒适度低，说明数据可能把 fearful 构造成过强、过戏剧化的表达；若参考声音条件越做越像某个可识别真人而合规风险上升，说明参考语音或评测目标可能过度追求身份复刻。只有把这些结论回流到样本筛选、配比和合成策略，评测才会真正改变下一版数据。

### 案例A.7：隐私、授权与滥用风险治理

声音身份属于高度敏感的数据资产。一个人的声音包含年龄、性别、地域、情绪、健康状态和身份识别线索；在声纹识别系统中，声音甚至可以成为认证凭据。可控语音数据一旦引入 voice cloning，就必须把授权、撤回、用途限制和审计写入数据生命周期，而不是只在模型发布时补充免责声明。

**表42-4：隐私与滥用风险控制清单**

| 风险类型 | 触发场景 | 控制措施 | 审计证据 |
|---|---|---|---|
| 声音身份授权 | 参考语音来自真实说话人或可识别声音 | 采集前同意、用途限定、可撤回、授权版本号 | 授权时间、撤回记录 |
| 声音克隆防滥用 | 合成音频被用于冒充、诈骗或绕过平台检测 | 音频数字水印、声学指纹、生成来源签名、公开样例防伪标记 | 水印检测日志、指纹库版本、溯源校验记录 |
| 情绪操控 | 用恐惧、愤怒或亲密语气影响用户判断 | 高风险场景禁用强情绪、提示语审查、未成年人保护 | 人工复核单 |
| 隐私泄漏 | 音频中含姓名、电话、地址或背景说话人 | ASR 脱敏、背景声过滤、数据最小化、保留期限 | 脱敏报告、删除请求处理记录 |
| 偏见与刻板印象 | `gender` 与 `mood` 或内容长期绑定 | 分布监控、反事实样本、禁止性别刻板模板 | 分布报表、偏见评测结果 |
| 版本失控 | 样本被重合成或重标注后无法追溯 | 数据版本管理、哈希、训练集冻结 | 实验追踪编号 |

表42-4将风险治理落实为数据门禁。授权缺失的 reference 不能进入合成队列；撤回授权的 reference 必须能追溯到所有派生音频和 token；高风险情绪操控样本不能只靠训练后安全策略兜底，而要在数据构建阶段就被拦截或降权。对语音生成来说，合规不是上线前最后一层过滤，而是样本生命周期的一部分。

参考语音池是治理重点。每个 reference 都应有 `consent_id`、授权范围、采集方式、允许任务、过期时间和撤回状态。若授权只允许研究用途，样本不能进入商业模型训练；若说话人撤回授权，manifest 应能定位所有受影响的 `query_id/answer_id`、音频文件、token 文件和训练版本。对外发布时，应尽量使用不可反查真实身份的 reference ID，避免将声音 ID、文件名或路径设计成真实姓名。

声音克隆产物还应具备可验证的防伪机制。进入训练集、评测集或公开样例的合成音频，宜写入不影响听感的数字水印，或至少生成可检索的声学指纹；manifest 中同步记录生成模型、模型版本、watermark key id、`consent_id`、样本哈希和数据版本。发布前要运行水印/指纹检测，确认音频仍可溯源；经过转码、裁剪或压缩后检测失败的高风险样本，应降级为内部样本、重新合成或直接下架。这样，声音克隆不只是“有授权即可使用”，还具备事后识别、平台协查和撤回处置的证据链。

情绪控制也有滥用边界。fearful、angry 等强情绪可以提升表达力，也可能用于操控用户。客服、教育、医疗、金融等场景应限制高压情绪输出，尤其不能用恐惧语气诱导用户转账、购买、泄露验证码或作出健康决策。对于未成年人和心理脆弱人群，系统应优先使用 neutral 或温和支持性风格，并保留策略触发日志。

隐私保护还包括内容脱敏。语音样本可能含有姓名、地址、电话、账户、地理位置或背景中的第三方说话声。即使 VoiceStyleControl 主要由合成文本生成，工程流程仍应保留 ASR 脱敏、敏感词扫描、背景声检测和人工抽检。若后续引入真实用户语音反馈，用户同意、数据最小化、保留期限、删除请求和用途变更通知都必须进入平台流程。

偏见治理同样重要。若训练集中女性声音更多被绑定到 fearful 或 sad，男性声音更多被绑定到 angry，模型会学习并放大刻板印象。因此，gender 统计不能只停留在边际占比，必须进入 `query_gender`、`answer_gender` 与 `query_mood`、`answer_mood` 的交叉视图；评测集也要构造反事实样本，检查同一内容在不同 gender 下的情绪表达是否公平。

### 案例A.8：与前后章节的数据工程连接

VoiceStyleControl 继承了音视频数据工程的底层能力。第10章讨论的音频切片、ASR、降噪、说话人分离和时间对齐，进一步转化为更细的样本契约：不仅要知道一段音频对应哪段文字，还要知道它由哪个参考声音 ID、以何种 mood、在什么采样率和 token 频率下生成。普通音频管线解决“能不能对齐”，可控语音交互进一步解决“对齐后的声音能否按条件生成”。

它也连接多轮交互数据。第20章关注 Agent 记忆和多轮上下文时，角色、意图和历史状态是主要变量；当交互进入语音形态，助手人格还体现在音色与情绪稳定性上。一个多轮语音助手不能第一轮是 neutral 男声，第二轮无缘由变成 fearful 女声，第三轮又变成 angry 男声。因而 `answer_gender`、`answer_mood` 和 `answer_id` 可以成为语音 Agent 记忆的一部分，用于维持连续会话中的声音身份。

在线反馈闭环会让语音风格从离线标签走向用户体验。第23章中的点击、满意度、纠错和投诉，在语音产品里会表现为“听不清”“太急”“太凶”“不像之前的声音”“情绪不合适”等反馈。这些反馈不能直接变成训练样本，而应先进入评测队列：判断是语义错误、音质错误、风格错误还是安全策略错误，再决定重合成、重标注、调整配比或修改拒绝规则。

隐私合规章节为 VoiceStyleControl 提供边界。第36章的数据合规框架要求把授权、用途、留存和审计前置到数据生命周期；第37章的隐私保护技术则提醒我们，声音身份可以通过访问控制、联邦训练、加密存储和最小化采集降低风险。可控语音数据越强调声音条件和参考音色，越不能把合规视为附录。

在多模态生成数据工程中，VoiceStyleControl 与第48章共享同一个核心模式：把生成目标拆为内容条件与风格条件，再用结构化 schema 绑定训练监督。T2I/T2V 中的 prompt、style、motion、camera、safety tag，在语音中对应 `answer`、`answer_gender`、`answer_mood`、参考声音 ID、sample_rate 和 audio token。第十四篇项目十“端到端 LLM 数据飞轮”也可以吸收这套设计：离线构建初版语音数据，训练可控生成模型，在线收集体验反馈，回流到质检和配平，再发布下一版数据与模型。

### 案例A：小结

VoiceStyleControl 的价值不在于把语音样本简单堆到更大规模，而在于把语义响应、声音条件、情绪控制和语音生成监督放进同一条可审计记录。S2SEmoControl 提供 spoken query 到 spoken answer 的交互监督，TTSSpeakerControl 提供自然语言风格描述到目标语音的直接监督。二者合在一起，使模型既能理解用户语音，又能依据指定声音条件和情绪生成回答。

数据工程的关键工作包括：显式区分语义通道与风格通道，保留 `query_gender`、`answer_gender`、`query_mood`、`answer_mood` 等控制字段；将 `sample_rate`、音频路径、speech token 路径和 tokenizer 版本写入数据契约；用 ASR 反查、声音条件核验、情绪识别、音频质量指标和人工评审共同构建评测协议；在参考语音池和声音克隆流程中落实授权、撤回、水印和审计。

当语音交互从“能说话”走向“以可控方式说话”，数据集的边界也随之变化。每条样本都要回答四个问题：内容是否正确，声音条件是否符合目标设定，情绪是否符合控制条件，生成过程是否合规可追溯。只有这四个问题同时成立，可控语音交互数据才能成为可靠的训练资产。

## 案例B：Latent-Switch-69K：推理轨迹压缩与隐式计算槽位

### 案例B：学习目标

通过本章学习，读者应能够：

- 理解 Long-CoT 在 token 成本、显式过程监督与推理效率上的工程约束，以及为何需要被压缩。
- 掌握 solution intuition、compressed CoT、latent placeholder 与 answer mask 在 latent-then-explicit 样本中的角色。
- 设计 latent budget、student sequence 与 supervision masks 之间的 mask 不变量与一致性检查。
- 评估推理数据压缩中的答案一致性、验证充分性、压缩边界与领域偏置等风险。
- 将 latent-switch 的隐藏规划与显式验证分离思想迁移到数学、代码与复杂指令等自有数据场景。

### 案例B.0：开篇问题场景：Long-CoT 为什么还需要被压缩

第18章到第20章已经讨论 Chain-of-Thought、工具调用轨迹和 Agent 交互数据的基本形态。对推理模型而言，长思维链具有明确吸引力：模型把中间步骤写出来，训练者就能检查它是否在按某种可解释的路径解题，推理时也更容易通过自洽采样、验证器或过程奖励模型发现错误。然而，当 Long-CoT 从研究样例变成训练语料时，问题会立刻变得工程化。

第一，长 CoT 的 token 成本很高。数学、代码和科学问题中的推导往往占据输出的大部分长度，真正的最终答案只占很小一段。如果所有中间推理都以可见文本形式进入训练和推理，模型需要在大量重复、展开、试探和自我修正的文字上消耗上下文窗口、训练显存和推理时间。第二，长 CoT 并不天然等于高质量推理。有些轨迹只是把简单结论拆得很细，有些轨迹包含错误分支，有些轨迹会在最终答案正确的情况下写出冗余甚至不一致的中间解释。第三，普通 SFT 很难区分“应该被模型内化的高层解题意图”和“必须显式写给用户看的验证过程”。如果把全部 CoT 当作普通目标 token，模型学到的往往是长篇展开的写作习惯，而不是更有效的推理调度方式。

Latent-Switch-69K 正是在这个问题背景下出现的。它不是一个简单的“更短 CoT 数据集”，也不是把 Long-CoT 样本做摘要后直接用于 SFT。它服务的是 [LaTER](https://github.com/TioeAre/LaTER) 这类 latent-then-explicit reasoning 系统：模型先经过一段有边界的 latent reasoning 区间，在连续隐状态中完成高层规划和压缩思考，然后切换回可见文本，用较短的显式 CoT 做符号验证，最后生成答案。数据工程目标因此发生了变化：样本不仅要回答“答案是什么”，也要回答“哪些内容适合成为隐藏规划预算，哪些内容仍需要作为可见验证监督”。

![图40-4：Latent-Switch-69K 构建流水线图](../../images/part12/ch40_04_latent_switch_pipeline.svg)

*图40-1：Latent-Switch-69K 将 Dolci-Think-SFT-32B 的推理轨迹蒸馏为 solution intuition、压缩 CoT、latent budget、student sequence 和 mask 对齐后的 SFT 记录。*

本章承接第五篇的合成数据工程和第六篇的推理数据工程。第15章到第17章讨论如何生成、蒸馏和质检高质量训练样本，第18章讨论显式 CoT 的组织方式，第19章和第20章讨论工具与 Agent 轨迹中“中间状态”的记录方式。Latent-Switch-69K 则把这些线索推进到一个更细的层次：中间推理不一定都要以自然语言存储，数据集也可以显式为隐藏计算预留槽位。向后看，它会自然连接到第45章的后训练数据配方、第46章的 RL 推理数据工程，以及第十四篇 P06、P10、P12 中的推理飞轮项目。

### 案例B.1：数据集概览：规模、难度与领域构成

[Latent-Switch-69K 数据集](https://huggingface.co/datasets/Tioe/LATENT-SWITCH-69K)的最终训练 split 包含 69,745 条样本。每条保留样本包含一个用户问题、一个蒸馏出的 solution intuition、一段缩短后的显式 CoT、最终答案、latent-step 元数据，以及用于决定不同 token 区间监督方式的 mask。这个结构决定了它与普通 CoT/SFT 数据的差异：普通 SFT 记录通常只需要 prompt 和 assistant output，普通 CoT 数据通常只需要把 reasoning 和 answer 写在 `<think>` 或自然语言段落中；Latent-Switch-69K 还需要记录一个用于隐藏规划的预算，并把这个预算渲染成 student sequence 中的 latent placeholder。

难度分布上，数据集并没有追求完全均匀。中等难度样本占主要部分，共 45,650 条，占 65.5%；困难样本 17,428 条，占 25.0%；简单样本 6,667 条，占 9.5%。这种分布对 latent-switch 训练有明确意义。中等难度问题通常需要真实推理，不是模板化问答，但又不至于让蒸馏过程过度不稳定。困难样本提供更长、更复杂的推理链，让模型接触更高预算的隐式规划场景。简单样本则帮助模型保留短回答和直接验证的能力，避免所有样本都被塑造成长推理任务。

| 统计项 | 数值 | 占比 / 说明 |
| --- | ---: | --- |
| Total examples | 69,745 | 100.0% |
| Easy | 6,667 | 9.5% |
| Medium | 45,650 | 65.5% |
| Hard | 17,428 | 25.0% |
| Compression ratio mean | 0.612 | distilled CoT length / original CoT length |
| Compression ratio median | 0.569 | 中位样本保留约 56.9% 的显式推理长度 |
| Latent steps mean | 41.49 | 每条样本平均 latent placeholder 数量 |
| Latent steps median | 40.00 | 中位样本约 40 个 latent steps |

领域构成上，Latent-Switch-69K 明显偏向 reasoning-intensive 任务。数学问题约占 37%，代码问题约占 34%，science-oriented questions 约占 5%，剩余部分主要来自 instruction-following 和 general knowledge prompts。这个比例不是偶然的。latent-then-explicit reasoning 最需要解决的是“有高层解题计划，但不希望把所有推导都展开”的任务；数学和代码恰好具有强验证性、强步骤性和较高 token 成本。科学问题提供概念推理和多条件判断场景，而通用指令与知识类样本让模型不至于只学习到竞赛数学或代码补全的表达模式。

![图40-5：Latent-Switch-69K 数据来源与领域组成](../../images/part12/ch40_05_dataset_composition.png)

*图40-2：最终训练集包含 69,745 条样本，来源中数学、代码和精确指令类数据占比较高。*

从数据工程角度看，这里有三类统计必须同时保留。第一类是规模统计，说明训练集足够大，可以作为一个专门的 latent reasoning 监督语料，而不是少量 prompt 模板。第二类是难度统计，说明数据并非随机堆叠，而是服务于 curriculum 和 latent budget 的稳定性。第三类是领域统计，说明该数据集更适合训练和评估数学、代码、科学、复杂指令等推理任务，不应被误读为覆盖所有对话场景的通用 SFT 数据。

Latent-Switch-69K 中保留有以下字段：`dataset_name`、`source_dataset`、`record_id`、`difficulty`、`domain`、`source_cot_length`、`distilled_cot_length`、`compression_ratio`、`solution_intuition_length`、`n_latent_steps`、`assistant_cot`、`assistant_answer`、`mask_schema_version`。这些字段看似偏工程，但它们决定了后续能否解释一次训练结果是来自更短 CoT、latent budget 调整，还是来自领域比例变化。

进一步看，Latent-Switch-69K 的字段可以分成四组。第一组是来源字段，用于说明样本从哪里来、原始问题属于哪个任务族、是否来自数学、代码、科学或通用指令数据。来源字段的作用不是装饰，它决定了后续配比、去重和责任追踪。比如当模型在代码任务上变强、但在开放问答上变啰嗦时，工程师需要回到来源字段检查是不是代码样本权重过高，或者 instruction-following 样本被压缩得过短。

第二组是推理内容字段，包括 source reasoning trace、solution intuition、assistant_cot 和 assistant_answer。source trace 是蒸馏前的参考，不一定进入最终 student sequence；solution intuition 是高层计划；assistant_cot 是压缩后的显式验证链；assistant_answer 是最终答案。四者之间需要保持可追溯关系。理想情况下，审计者可以从一条训练样本反查：原始长 CoT 中哪些信息被提炼成 intuition，哪些信息留在压缩 CoT 中，答案是否和原始问题的可验证目标一致。

第三组是长度与预算字段，包括 source CoT length、distilled CoT length、insight length、compression ratio 和 `n_latent_steps`。这些字段直接服务于成本控制和预算诊断。如果某个数据版本的平均压缩率突然下降到 0.3，表面上看 token 成本更低，但它可能意味着显式验证链被压得太短。如果 `n_latent_steps` 均值突然升高，模型训练时的有效序列长度和推理时的隐藏计算成本也会随之变化。没有这些长度字段，团队很难在“效率提升”和“监督损失”之间做定量判断。

第四组是监督字段，包括 prompt mask、latent internal mask、latent boundary mask、CoT mask、answer mask 和 teacher-KL mask。它们决定同一条 token 序列在训练时被怎样解释。普通数据集的 schema 往往只关心文本字段是否存在，Latent-Switch-69K 则必须把 mask 也视为数据资产。原因很简单：同一段文本如果 mask 不同，就对应不同训练任务。一个 latent placeholder 如果被 CE 拟合，就变成普通 token；如果被 mask 掉并替换为 recurrent latent state，它才是隐藏计算槽位。

### 案例B.2：蒸馏与记录形成：从 teacher trace 到压缩推理记录

Latent-Switch-69K 的构建起点是 Dolci-Think-SFT-32B 中采样得到的推理轨迹。原始轨迹可以理解为 source reasoning traces：它们包含问题、一个或多个 assistant 输出、可能的 ground truth 或可抽取答案，以及来源和元数据。构建过程并不是直接筛选短答案，而是先把长轨迹拆解为两个互补目标：高层问题求解意图和较短的显式验证链。

下面的教学化示例展示了最简单的 source trace 抽取方式：从 Hugging Face 读取 Dolci-Think-SFT-32B，使用固定随机种子打乱并选取一批记录，再把对话整理为后续蒸馏需要的最小字段。真实 LaTER 管线中的 `sample_Dolci-Think-SFT-32B.py` 会读取本地 Parquet 分片，并使用按来源分层的 reservoir sampling，避免简单随机抽样改变不同数据源的比例。

```python
from datasets import load_dataset


def first_message(messages, role):
    return next(
        (item["content"] for item in messages if item.get("role") == role),
        "",
    )


def last_message(messages, role):
    return next(
        (item["content"] for item in reversed(messages) if item.get("role") == role),
        "",
    )


dataset = load_dataset("allenai/Dolci-Think-SFT-32B", split="train")
sample_size = min(2000, len(dataset))
sampled = dataset.shuffle(seed=42).select(range(sample_size))

source_traces = []
for row in sampled:
    messages = row.get("messages", [])
    source_traces.append(
        {
            "record_id": row.get("id"),
            "source_dataset": row.get("source", row.get("dataset", "unknown")),
            "problem": first_message(messages, "user"),
            "source_cot": last_message(messages, "assistant"),
        }
    )
```

第一阶段是提取 solution intuition。数据构建提示要求 teacher 只抽取关键洞见，不要写成短 CoT，也不要直接给最终答案。这个字段应该描述“解决这道题的高层计划”，例如应该建立什么方程、应该枚举哪类状态、代码题应该使用什么数据结构、科学题应该抓住哪条因果关系。它的颗粒度介于标签和完整推导之间：比领域标签更具体，但比逐步推理更压缩。这样做的核心价值是把 Long-CoT 中可被内化的 planning signal 提取出来，为后续 latent budget 提供依据。

第二阶段是生成压缩显式 CoT。teacher 在原始问题和 solution intuition 的条件下继续解题，输出较短的推理过程和最终答案。由于 teacher 已经拿到高层计划，它不需要重新展开全部探索过程，也不需要重复原始轨迹中的无效分支。保留样本因此包含四个主要内容：problem、intuition、compressed CoT、final answer。与普通摘要不同，compressed CoT 的目标不是“把原文变短”，而是留下足够的可见验证路径，让模型在 latent reasoning 之后仍能用文本完成符号检查。

下面的最小实现使用 OpenAI-compatible API 串联两个阶段。第一阶段只要求返回 JSON 格式的 `correct_insight`；第二阶段以问题和该 intuition 为条件继续生成，并把 API 返回的隐藏 reasoning 内容记录为 `distilled_cot`，把可见内容记录为最终答案。密钥、API 地址和 teacher model 均从环境变量读取。

```python
import asyncio
import json
import os

from openai import AsyncOpenAI


client_kwargs = {"api_key": os.environ["OPENAI_API_KEY"]}
if os.getenv("OPENAI_BASE_URL"):
    client_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]

client = AsyncOpenAI(**client_kwargs)
teacher_model = os.environ["TEACHER_MODEL"]


async def call_teacher(system_prompt, user_prompt):
    response = await client.chat.completions.create(
        model=teacher_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        extra_body={"thinking": {"type": "enabled"}},
    )
    message = response.choices[0].message
    reasoning = getattr(message, "reasoning_content", None)
    content = message.content or ""

    # Some compatible APIs place reasoning inside the visible content.
    if not reasoning and "<think>" in content and "</think>" in content:
        reasoning, content = content.split("<think>", 1)[1].split("</think>", 1)
    return (reasoning or "").strip(), content.strip()


async def distill_one(problem, source_cot):
    _, insight_json = await call_teacher(
        "Return valid JSON with one field named correct_insight. "
        "Give only the high-level solution plan, without the final answer "
        "or a complete chain of thought.",
        f"Problem:\n{problem}\n\nReference reasoning:\n{source_cot}",
    )
    intuition = json.loads(insight_json)["correct_insight"]

    distilled_cot, answer = await call_teacher(
        "Continue from the supplied solution intuition. Keep the reasoning "
        "compact, verify the key steps, and give the final answer.",
        f"Problem:\n{problem}\n\nSolution intuition:\n{intuition}",
    )
    return {
        "problem": problem,
        "solution_intuition": intuition,
        "distilled_cot": distilled_cot,
        "answer": answer,
    }


record = asyncio.run(
    distill_one(source_traces[0]["problem"], source_traces[0]["source_cot"])
)
```

![图40-6：原始 CoT、压缩 CoT 与 latent placeholder 对比](../../images/part12/ch40_06_cot_latent_comparison.svg)

*图40-3：source trace 中的大量可见推理被拆成两类信号：solution intuition 用于估计 latent budget，压缩 CoT 用于显式验证和答案监督。*

压缩率定义为：

$$
\text{compression ratio}
= \frac{\text{distilled CoT length}}{\text{original CoT length}}.
$$

最终语料的压缩率均值为 0.612，中位数为 0.569。这说明蒸馏后的可见 CoT 通常只保留原始推理长度的 57% 到 61% 左右。注意，这个数字不应被解释为“删除了四成推理信息”。更准确的理解是：一部分细节被压缩进 solution intuition 所代表的高层计划，并进一步映射到 latent placeholder 预算；另一部分必要推导仍保留在 `<think>` 中，用于显式验证和监督模型的可见推理风格。

![图40-7：原始与蒸馏后推理长度及压缩率统计](../../images/part12/ch40_07_token_compression_distribution.png)

*图40-4：本图展示了 source CoT length、distilled CoT length、insight length、ground truth length 和 compression ratio 的分布。*

样本保留标准应围绕三个问题展开。第一，source trace 是否有足够可信的最终答案。如果原始答案无法抽取、明显和 ground truth 不一致，或 teacher 后续无法稳定复现答案，样本就不适合进入最终集。第二，solution intuition 是否只表达高层计划。如果 intuition 中直接泄露答案或写成完整 CoT，它就不再适合作为 latent budget 的代理。第三，compressed CoT 是否仍能连接问题和答案。如果压缩过度，显式推理会变成几个跳跃句，模型虽然能模仿答案，却学不到从隐式规划切换到显式验证的边界。

这套蒸馏过程给数据工程团队一个重要提示：推理数据压缩不能只看 token 数。更可靠的压缩必须同时检查 intent preservation、answer consistency 和 verification sufficiency。也就是说，压缩后的样本既要保留问题求解意图，又要保留足够的可见验证路径，还要在最终答案上保持一致。

把这个流程落到工程系统中，可以拆成六个可审计阶段。第一阶段是 source trace 抽取，把 Dolci-Think-SFT-32B 中的 prompt、assistant outputs、ground truth、dataset source 和 metadata 统一成内部记录。这个阶段最重要的是保留原始上下文，而不是过早改写。因为后续如果发现 teacher 输出异常，工程师需要回到 source trace 判断错误来自原始轨迹、提示模板，还是答案抽取。

第二阶段是 high-level intuition distillation。提示词明确要求 teacher 返回 JSON，并限制 correct_insight 只能描述粗粒度计划，不提供最终答案，不写成完整 CoT。这个约束很关键，因为 intuition 的角色不是训练模型“照着这段文字说”，而是估计应该给模型多少隐藏规划空间。若 intuition 已经包含详细推导，latent budget 就会从“压缩规划复杂度”变成“复制一段不可见 CoT 的长度”，这会削弱数据设计的清晰性。

第三阶段是 compact explicit CoT generation。teacher 在原问题和 intuition 的条件下生成较短推理链，这等于把 source trace 中可公开验证的部分重新整理出来。这里需要避免两个极端：一端是 CoT 仍然过长，几乎没有压缩；另一端是只剩结论，缺少验证。较好的样本通常会留下关键等式、关键分支、关键代码不变量或最终选择依据，而删除重复铺垫、自我怀疑和无效试探。

第四阶段是 answer validation。数学题可以检查抽取答案和 ground truth 是否一致，选择题可以检查选项格式，代码题可以尽量通过测试或静态规则检查，开放问答至少要做 teacher consistency 或抽样人工复核。Latent-switch 训练比普通摘要任务更依赖答案一致性，因为答案 token 是最终监督的核心位置，错误答案会和 latent 预算、显式 CoT 一起被模型学习。

第五阶段是 sequence rendering。系统把 problem、latent placeholder、compressed CoT 和 answer 渲染为 chat-style student sequence。这个阶段需要 tokenizer contract：`<latent_think>`、`</latent_think>`、`<think>`、`</think>` 必须被稳定识别，不能在不同 tokenizer 或不同 special-token 注册方式下被拆成不可预测片段。否则，span 检查和 mask 构造都会不可靠。

第六阶段是 mask materialization。数据加载器根据 token ids 重新定位边界，构造 labels、loss weights 和各种 mask。这个阶段不宜只依赖原始字符串中的字符偏移，因为 tokenizer 改变会让字符偏移失效。更稳妥的方式是基于 token id 中的 special token 位置构造 span，并在每条样本上校验边界出现次数、顺序、答案区间和 teacher-reference 区间是否有效。

### 案例B.3：Latent budget 与 student sequence：样本如何被渲染

Latent-Switch-69K 的关键字段之一是 `n_latent_steps`。它决定了 student sequence 中 `<latent_think>` 和 `</latent_think>` 之间放置多少个 latent placeholder。论文和代码中采用的基本启发式是：如果保留的 solution intuition 含有 \(L\) 个 token，则 latent budget 约为 \(L/2\)，再受最大 latent 长度和 tokenizer 约束裁剪。最终数据中 latent steps 的均值为 41.49，中位数为 40.00。

这个预算规则有两个含义。第一，latent steps 不是随意加的 padding，而是与被压缩的高层推理内容相关。intuition 越长，说明这道题的高层规划可能越复杂，模型需要更多隐藏计算槽位。第二，latent steps 也不是越多越好。过长的 latent 区间会增加训练和推理成本，也可能让模型在隐藏状态中漂移。LaTER 的训练自由度实验曾观察到 40 到 50 步附近有较好的准确率和 token 效率折中，因此最终样本的 latent-step 分布被集中在这个范围附近。

在 student sequence 中，一条样本可以抽象写成：

$$
\texttt{<latent\_think>}~l_1,\ldots,l_m~\texttt{</latent\_think>}
~\texttt{<think>}~t_1,\ldots,t_n~\texttt{</think>}~a_1,\ldots,a_r~\texttt{<|im\_end|>}.
$$

其中 $(l_1,\dots,l_m)$ 是 latent placeholder positions，$(t_1,\dots,t_n)$ 是蒸馏后的显式 CoT tokens，$(a_1,\dots,a_r)$ 是最终答案 tokens。代码实现中，latent placeholder 可以由重复的 `latent_pad_token` 填充；但训练时这些位置不会被当作普通语言目标。模型前向过程中，placeholder 的输入 embedding 会被 latent projector 产生的 recurrent latent states 替换。换言之，这些位置在序列里有 token 边界和长度，但语义上是隐藏计算槽位。

下面的记录渲染函数对应 LaTER `preprocess.py` 中 `build_sft_record` 的核心逻辑：先用 student tokenizer 计算 solution intuition 的 token 长度，再把约 $(L/2)$ 个 latent steps 裁剪到允许范围，最后同时保存结构化字段和渲染后的 assistant sequence。示例使用 `<|endoftext|>` 作为占位 token；实际训练必须确保它与 tokenizer 和模型配置中的 `latent_pad_token` 完全一致。

```python
import os

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(os.environ["STUDENT_TOKENIZER"])


def build_sft_record(problem, intuition, distilled_cot, answer):
    intuition_tokens = tokenizer.encode(intuition, add_special_tokens=False)
    n_latent_steps = min(128, max(1, len(intuition_tokens) // 2))
    latent_pad_token = "<|endoftext|>"
    latent_placeholder = latent_pad_token * n_latent_steps

    assistant_content = (
        f"<latent_think>{latent_placeholder}</latent_think>"
        f"<think>{distilled_cot}</think>{answer}"
    )
    return {
        "messages": [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": assistant_content},
        ],
        "assistant_cot": distilled_cot,
        "assistant_answer": answer,
        "solution_intuition": intuition,
        "n_latent_steps": n_latent_steps,
        "latent_pad_token": latent_pad_token,
        "state_align_reference_messages": [
            {
                "role": "user",
                "content": f"Problem:\n{problem}\n\nSolution intuition:\n{intuition}",
            },
            {
                "role": "assistant",
                "content": f"<think>{distilled_cot}</think>{answer}",
            },
        ],
    }


sft_record = build_sft_record(
    record["problem"],
    record["solution_intuition"],
    record["distilled_cot"],
    record["answer"],
)
```

生产预处理还会依据压缩率和字段完整性过滤样本，并记录 CoT、answer 的 loss weight。更重要的是，渲染后的记录仍需交给数据加载器重新定位 special-token spans 和构造 supervision masks；仅仅拼出这段字符串，并不意味着样本已经可以安全训练。

下面是一个教学化的简化样本序列示例。它只用于说明 schema 和 mask 关系，不是数据集中某条真实训练样本。

```text
<|im_start|>user
Target Question:
某数列满足 a_1 = 2, a_{n+1} = 3a_n + 1。求 a_4。
<|im_end|>
<|im_start|>assistant
<latent_think>
<|endoftext|><|endoftext|><|endoftext|><|endoftext|>
</latent_think>
<think>
用递推式逐步计算即可。先由 a_1 得到 a_2，再得到 a_3 和 a_4。
a_2 = 3 * 2 + 1 = 7；
a_3 = 3 * 7 + 1 = 22；
a_4 = 3 * 22 + 1 = 67。
</think>
最终答案是 67。
<|im_end|>
```

这条示例里，`<latent_think>` 和 `</latent_think>` 是结构边界；中间四个 `<|endoftext|>` 只是占位符，真实样本中的数量由 `n_latent_steps` 决定；`<think>` 到 `</think>` 之间是可见压缩 CoT；之后是答案。对于训练来说，最重要的不是这段文本看起来像不像自然对话，而是每个 token 区间是否能被稳定定位。[LaTER 训练代码](https://github.com/TioeAre/LaTER)中的 `build_spans` 会检查一条样本中恰好包含一个 `<latent_think>`、一个 `</latent_think>`、一个 `<think>` 和一个 `</think>`，并保证它们满足：

```text
assistant_content_start <= latent_start < latent_end < think_start < think_end
```

这个顺序约束非常关键。如果边界 token 缺失、重复或顺序错乱，mask 就会错位，latent placeholder 可能被误当作普通答案 token，或者 answer 区间被截断。对于普通 SFT 数据，边界错一处也许只是格式问题；对于 latent-switch 数据，边界错位会直接改变训练目标。

在实际数据仓库中，student sequence 不应只作为一个长字符串保存。更稳妥的做法是同时保存结构化字段和渲染后文本。结构化字段包括 `messages`、`assistant_cot`、`assistant_answer`、`n_latent_steps`、`latent_pad_token`、`state_align_reference_messages` 等；渲染后文本则用于快速查看和兼容普通训练框架。代码中的 `LatentSFTDataset` 会优先使用结构化字段构造 token ids，只有字段缺失时才退回字符串重编码路径。这个设计反映了一个经验：latent-special-token 边界太重要，不应完全依赖已经拼好的文本。

`latent_pad_token` 也值得单独说明。它在序列中承担的是“占位”而非语义内容。若 tokenizer 已经注册该 token，加载器可以直接重复其 id；若没有注册，就只能把字符串重复后再编码，这会带来长度不确定性。对于一个普通 padding token，这种差异也许还能接受；对于 latent budget，它会改变 \(m\) 的实际 token 数，进而改变 hidden rollout 的步数。因此，发布数据集时应明确 tokenizer 版本、special token 注册方式和 latent pad token 的语义。

teacher-reference conversation 是另一条容易被忽视的序列。它不是 student sequence 的副本，而是省略 latent placeholder 后的参考对话。teacher 输入中包含原始问题和 solution intuition，assistant continuation 则是压缩 CoT 和答案。这个设计让 teacher KL 聚焦在可见推理质量和答案分布上，而不是要求 teacher 理解 student 的 latent 内部槽位。换句话说，student sequence 负责训练 latent-then-explicit 格式，teacher reference 负责提供显式验证部分的分布参考，两者服务于不同监督目标。

### 案例B.4：Supervision masks：哪些 token 参与 loss

Latent-Switch-69K 的监督设计可以概括为一句话：prompt tokens 和 latent interior placeholders 不参与普通 token-level CE，结构边界、显式 CoT、答案和结束 token 参与有针对性的监督，teacher KL 只对选定的显式 CoT 与答案位置生效。

在代码实现中，一条样本会构造多个 mask：`prompt_mask`、`latent_internal_mask`、`latent_boundary_mask`、`cot_mask`、`answer_mask`、`teacher_kl_mask`。这些 mask 不是为了可视化方便，而是直接决定不同 objective 的作用位置。普通 labels 初始化为 student token ids，然后对 prompt 区间和 latent interior 区间置为 `-100`，表示这些位置被 cross-entropy 忽略。

可以用下面的简化规则表示 CE label：

$$
y_i =
\begin{cases}
-100, & i \in \mathcal{S}_{prompt} \cup \mathcal{S}_{lat}^{int}, \\
x_i, & \text{otherwise}.
\end{cases}
$$

其中 $\mathcal{S}_{prompt}$ 表示用户 prompt 与 assistant prefix 之前的上下文位置，$\mathcal{S}_{lat}^{int}$ 表示 `<latent_think>` 和 `</latent_think>` 之间的内部 placeholder 位置。被置为 `-100` 的 token 不被普通 CE 直接拟合。这样做避免了一个错误目标：要求模型在 latent 内部位置预测某个固定文本 token。对 LaTER 来说，latent 内部位置的价值不是输出 `<|endoftext|>`，而是让模型执行若干步隐藏状态更新。

![图40-8：Supervision mask 示意图](../../images/part12/ch40_08_supervision_mask.svg)

*图40-5：prompt 与 latent interior 被普通 CE mask 掉；latent 边界、显式 CoT、答案和结束 token 由不同权重和 mask 控制。*

`latent_boundary_mask` 标出 `<latent_think>` 和 `</latent_think>` 两个边界位置。边界 token 本身仍然需要监督，因为模型必须学会什么时候进入 latent 区间，什么时候从 latent 区间退出。如果不监督边界，模型可能无法稳定切换到 `<think>`，或者在推理时生成不完整的结构。

`cot_mask` 覆盖 `<think>` 到 answer_start 之前的区间。论文训练目标中，内部显式 CoT tokens 可以使用不同权重，例如用 $\lambda_{CoT}$ 降低显式 reasoning 对总 CE 的支配程度。这样做符合数据集目标：显式 CoT 仍然重要，因为它承担验证和可解释输出；但训练不应退化为“越像长 CoT 越好”。模型还需要优先学会结构边界和最终答案行为。

`answer_mask` 覆盖 `</think>` 之后、`<|im_end|>` 之前的答案区间。答案 token 通常应保持较强监督，因为最终答案是任务正确性的主要承载位置。对于数学题，它可能是一个 boxed answer；对于选择题，它可能是 A、B、C、D；对于代码题，它可能是一段函数实现。无论 latent 区间如何设计，答案一致性都必须被严格维护。

`teacher_kl_mask` 则用于 teacher-distribution supervision。每条样本还会构造一个 teacher-reference conversation：它不包含 student 的 latent placeholder，而是把原始问题和 distilled solution intuition 合并为 teacher 输入，让 teacher 在缩短后的 `<think> ... </think>` 与答案位置提供分布参考。这样做的好处是，teacher 不需要模拟 continuous latent placeholders；它只监督显式推理和答案的 token 分布质量。

| 区间 | 示例 token | 普通 CE label | 主要 mask | 工程含义 |
| --- | --- | --- | --- | --- |
| Prompt 与 assistant prefix | user question `<\|im_start\|>assistant` | `-100` | `prompt_mask` | 作为条件，不作为输出目标 |
| Latent 起始边界 | `<latent_think>` | supervised | `latent_boundary_mask` | 学会进入 latent reasoning |
| Latent 内部槽位 | `l_1 ... l_m` | `-100` | `latent_internal_mask` | 隐藏计算槽位，不拟合占位文本 |
| Latent 结束边界 | `</latent_think>` | supervised | `latent_boundary_mask` | 学会停止 latent reasoning |
| 显式 reasoning | `<think> ... </think>` | weighted supervised | `cot_mask`、`teacher_kl_mask` | 可见验证链，可降低权重 |
| 最终答案 | answer tokens | supervised | `answer_mask`、`teacher_kl_mask` | 任务正确性核心监督 |
| 结束 token `<\|im_end\|>` | supervised | end-token weight | `answer_mask` | 保证聊天格式闭合 |

这套 mask 设计解释了为什么 Latent-Switch-69K 不能被普通数据加载器随意重编码。普通聊天数据加载器通常只关心 prompt 和 response 的分界，而 latent-switch 数据加载器必须知道 latent_start、latent_end、think_start、think_end、answer_start 和 im_end 的精确位置。只要 tokenizer 特殊 token 注册不一致，或者 string re-encoding 改变了边界 token 的位置，mask 就会失真。

更细地说，mask 还承担了训练目标之间的解耦。CE 目标负责让模型学会输出结构边界、显式推理和答案；latent internal mask 保护隐藏计算槽位，避免模型把它们当成普通文本学习；teacher KL 目标让显式 CoT 和答案更接近 teacher 的分布；halt 或 boundary 相关监督则帮助模型在合适位置结束 latent reasoning。虽然本章不复现 LaTER 的完整训练算法，但数据集必须为这些目标提供稳定接口。

对于数据工程师来说，最实用的检查不是重新推导损失函数，而是确认每条样本的 mask 是否满足几条不变量。第一，prompt 区间所有 labels 都应为 `-100`。第二，latent interior 区间所有 labels 都应为 `-100`，但 latent boundary token 不应被当作普通 prompt mask。第三，`cot_mask` 应覆盖 `<think>` 到 `</think>` 相关位置，且 answer_start 必须在 think_end 之后。第四，`answer_mask` 不应包含 `<|im_end|>`，因为结束 token 可以单独监督。第五，teacher KL mask 不应覆盖 latent interior，因为 teacher reference 本身不含这些 placeholder。

这些不变量应在数据构造和训练加载两个阶段都检查一次。构造阶段检查可以阻止坏样本入库；训练加载阶段检查可以发现 tokenizer、max_length、截断策略或配置变更带来的新问题。尤其是 max_length 截断，一旦截掉 answer 区间，样本就会只剩结构和推理，没有最终答案监督。代码中因此会在截断后重新构造 spans，并检查 answer_start 是否仍小于 im_end。

还有一个细节是显式 CoT 的权重。Latent-Switch-69K 不是要删除显式推理，而是要降低对完整长 CoT 的依赖。若 CoT 权重过高，模型会更倾向于把能力用在复现可见推理文字上；若 CoT 权重过低，模型可能只学会结构和答案，显式验证链变弱。数据侧至少要保留可配置的 `cot_loss_weight` 或等价字段，使训练者能够在不同任务上调整“可见验证”与“最终答案”的平衡。

### 案例B.5：质量控制：压缩、边界与偏置的五类风险

Latent-Switch-69K 的质量控制不只是过滤脏文本。由于它同时包含压缩推理、latent 预算和多种 mask，风险也分成多层。

| 风险类型 | 典型症状 | 影响 | 修复动作 |
| --- | --- | --- | --- |
| 压缩过度 | compressed CoT 只有结论，没有可见验证链 | 模型学不到从 latent 规划切换到显式验证的过程 | 增加 verification sufficiency 检查；拒绝跳步样本 |
| 推理断裂 | solution intuition 与 compressed CoT 使用不同解题路线 | latent budget 对应的高层计划无法支撑后续 CoT | 检查 intuition-CoT entailment；要求 teacher 重新生成 |
| 答案不一致 | source、teacher continuation、final answer 不一致 | 训练目标在答案位置冲突 | 用 ground truth、verifier 或答案抽取规则复核 |
| 边界错位 | `<latent_think>`、`<think>` 缺失、重复或顺序错误 | mask 错位，latent placeholder 被错误监督 | 在数据加载前做 span validation；错误样本隔离 |
| 领域偏置 | 数学和代码占比过高，通用指令覆盖不足 | 模型迁移到非推理任务时风格偏窄 | 记录 domain mix；按训练目标调整采样权重 |
| latent budget 异常 | `n_latent_steps` 为 0、过大或与 intuition 长度不匹配 | 隐式规划预算失真，推理成本不可控 | 对预算设上下限；监控均值、中位数和长尾 |
| teacher KL 错位 | KL mask 与 teacher reference token 不对齐 | teacher 分布监督作用到错误位置 | 保留 teacher span 校验；记录 top-k 分布版本 |

第一类风险是压缩过度。压缩率均值 0.612 和中位数 0.569 表明语料确实显著缩短了可见 CoT，但工程上不能把压缩率越低越好作为目标。如果一个样本从 1,000 个推理 token 压缩到 50 个 token，却失去关键等式、状态转移或代码不变量，那么它虽然节省 token，却破坏了监督质量。更稳妥的指标是组合式的：压缩后长度下降、答案仍一致、可见推理仍能解释最终答案。

第二类风险是推理断裂。solution intuition 是 latent budget 的来源，如果 intuition 描述的是一种路线，而 compressed CoT 实际使用另一种路线，模型就会收到不一致信号。比如 intuition 说“用动态规划”，compressed CoT 却写成贪心证明；或 intuition 说“先建立方程”，后文却直接枚举。此时 latent placeholder 的数量仍然可能合理，但它对应的高层计划已经失配。数据管线需要检查 intuition 与 CoT 的语义一致性。

第三类风险是答案不一致。推理数据中最常见的问题之一是中间链路看起来合理，但最终答案和 ground truth 不同。对 Latent-Switch-69K 来说，答案不一致更严重，因为 teacher-reference、student sequence 和 answer_mask 都会围绕最终答案构建。如果错误答案进入训练，模型不仅会学到错误结论，还可能学到错误的 latent-to-explicit 切换模式。数学和选择题可用规则验证器或答案抽取器，代码题可用单元测试，开放问答则至少需要 teacher check 或人工抽检。

第四类风险是边界与 mask 错位。`<latent_think>`、`</latent_think>`、`<think>`、`</think>` 都是结构 token，而不是普通文本装饰。数据加载器会检查它们出现次数和顺序，并据此计算 span。如果一条样本多了一个 `</think>`，普通渲染可能仍然能显示，但训练 mask 会发生错位。质量控制应把 span validation 放在数据入库前，而不是等训练报错。

第五类风险是领域偏置。Latent-Switch-69K 的 math 约 37%、code 约 34%、science 约 5%，这使它非常适合 reasoning-heavy 训练，但也意味着它不是通用助手语料的完整替代品。如果把它和普通 SFT 数据混合，应明确训练目的：是强化数学代码推理、压缩可见 CoT，还是改善所有用户问题的回答效率。不同目标对应不同采样权重和评估集。

质量控制还需要保留审计信息。建议每个数据版本至少输出四份报告：长度与压缩率报告、difficulty/domain 分布报告、span 与 mask 校验报告、答案一致性与失败样本报告。对于 latent reasoning 数据，光有最终 parquet 或 jsonl 不够；没有这些报告，训练后很难判断模型变化来自数据质量提升，还是来自无意的分布漂移。

为了让这些报告真正可用，可以为 Latent-Switch-69K 建立一套发布前验收清单。长度层面，检查 source CoT、distilled CoT、intuition、answer 和 total sequence 的分布，重点关注过短和过长样本。过短样本可能没有足够监督，过长样本可能在训练时频繁截断。压缩层面，检查压缩率的均值、中位数、分位数和极端值，确认不是某个来源数据集导致异常。

结构层面，逐条检查四个边界 token 的出现次数和顺序。任何缺失、重复、嵌套或顺序错误都应直接隔离。mask 层面，抽样渲染 token 区间，把 prompt、latent internal、latent boundary、CoT、answer 和 im_end 用不同颜色展示，确认人工理解与程序 mask 一致。对于一类新数据源，建议至少人工查看几十条样本，尤其是长数学证明、代码函数、选择题和开放问答。

语义层面，检查 intuition 是否含有最终答案，compressed CoT 是否能支持答案，answer 是否与 ground truth 或 verifier 一致。对于代码任务，应尽量区分“解释中的思路正确”和“最终代码可运行”两个层面；对于数学任务，应区分“最终数值正确”和“推导链可验证”两个层面。latent-switch 数据的目标是高层规划加显式验证，因此这两个层面都不能完全放弃。

分布层面，检查 difficulty、domain、source dataset、语言、答案格式和 token 长度的联合分布。单独看每个字段可能都正常，但组合后可能出现偏置。例如 hard 样本几乎都来自数学，code 样本几乎都使用某一种 Python 模板，instruction 样本压缩率明显低于数学样本。这些偏置不一定都要消除，但必须记录，因为后续训练和评估会受到它们影响。

版本层面，每次发布都应给出数据版本号、构建脚本版本、teacher 模型版本、tokenizer 版本、special-token contract、过滤规则和统计报告。Latent-Switch-69K 这种数据集的可复查性来自“文本加结构加配置”的组合。如果只保存最终文本，几年后很难解释为什么某个样本有 38 个 latent steps，为什么某个 CoT 被降权，为什么某些 teacher KL 位置被跳过。

### 案例B.6：与前后章节的回链：从推理数据到推理飞轮

把 Latent-Switch-69K 放回全书结构中，它的价值不在于介绍一个孤立数据集，而在于展示一种新的推理数据接口。

对第五篇来说，它延续了合成数据和蒸馏数据的核心思想。第15章强调从数据合成任务定义出发设计样本，第16章讨论蒸馏如何把强模型行为迁移到训练语料，第17章讨论质量评估和过滤。Latent-Switch-69K 把这些原则具体化为：从 teacher trace 中提取高层 solution intuition，用压缩 CoT 保留显式验证，用 mask 把不同监督目标分配到不同 token 区间。

对第六篇来说，它提供了 CoT 数据工程的下一步。第18章中的 CoT 样本通常把推理过程直接作为文本监督；第19章的工具数据强调动作、观察和结果；第20章的 Agent 数据强调状态和轨迹。Latent-Switch-69K 则说明，推理状态也可以部分存放在不可见的 latent slots 中。它不是放弃可解释性，而是把可见解释压缩到必要验证链，把探索性规划迁移到隐藏计算区间。

对第十三篇来说，它是后训练和 RL 推理数据配方的前置样板。第45章会讨论 SFT、偏好对齐和在线持续优化的数据层级；第46章会讨论 RL reasoning、verifier、候选组和奖励信号。Latent-Switch-69K 在 SFT 阶段就提前引入了结构化推理预算和 mask schema，使后续 RL 阶段可以围绕“latent budget 是否合适”“显式验证是否充分”“答案是否可验证”继续优化。

对第十四篇项目来说，它可以和 P06、P10、P12 分别形成接口。P06 的 PRM 数据关注过程步骤的评分，Latent-Switch-69K 提供了压缩显式 reasoning 和 answer 区间，适合进一步抽取可评分的 verification steps。P10 的 LLM 数据飞轮关注线上反馈和持续迭代，latent-switch 数据可以作为一种降低推理 token 成本的候选数据资产。P12 的 R1 reasoning flywheel 关注多路采样、verifier 和拒绝采样，Latent-Switch-69K 则提供了一个冷启动思路：先用蒸馏数据教模型如何在 latent planning 和 explicit verification 之间切换，再用验证器和 RL 数据进一步调整预算和答案正确性。

最后，本章的工程结论可以压缩为四点。

1. Latent reasoning 数据不是普通 CoT 数据的短版本。它必须记录隐藏规划预算、显式验证链和最终答案之间的关系。
2. `<latent_think>` 与 `<think>` 是两类不同语义的结构区间。前者提供隐式计算槽位，后者提供可见推理监督。
3. Mask 是数据 schema 的一部分，不是训练代码的附属细节。prompt、latent interior、boundary、CoT、answer 和 teacher-KL 位置必须在数据构造时就能被稳定还原。
4. 数据压缩的目标不是删掉推理，而是把高层意图、隐藏计算和显式验证重新分配到更合适的通道中。

如果一个团队只复用 Latent-Switch-69K 的文本输出，而忽略 `n_latent_steps`、latent boundary 和 supervision masks，它得到的只是一个较短的 CoT SFT 数据集。只有把压缩率、latent placeholder、student sequence 和 mask 作为同一个数据工程对象管理，这个数据集才真正体现出 latent-then-explicit reasoning 的设计思想。

### 案例B.7：复用建议：把 Latent-Switch 思路迁移到自有数据

如果团队希望在自己的数学、代码或业务推理数据上复用 Latent-Switch-69K 的思路，不建议第一步就改模型结构。更稳妥的路线是先把数据 schema 做出来。团队可以从现有 Long-CoT 样本中抽取一小批高质量问题，人工或用 teacher 模型生成 solution intuition，再生成压缩 CoT 和最终答案。随后，按 intuition 长度分配一个保守的 latent budget，例如从 \(L/3\)、\(L/2\) 和固定 32 steps 三种方案中选择一到两个版本做对照。这样做的价值在于，团队可以先观察数据是否能被稳定渲染、mask 是否正确、答案是否一致，而不必马上进入昂贵训练。

第二步是建立小规模验收集。这个验收集不需要很大，但要覆盖短题、长题、数学、代码、选择题、开放问答和格式约束题。每个样本都应该能回答三个问题：高层 intuition 是否足够表达解题计划，压缩 CoT 是否足够支撑答案，latent steps 是否与题目复杂度大致匹配。若这三点在人工抽检中经常失败，说明问题不在训练算法，而在数据构造规则还没有稳定。

第三步是把 latent-switch 数据和普通 SFT 数据分开管理。普通 SFT 数据可以只记录 prompt 和 response，latent-switch 数据则必须记录结构化字段、特殊 token contract、mask schema 和构建版本。混合训练时，也应在 manifest 中写清楚每类数据的采样权重和用途。否则，当模型出现回答变短、推理解释变弱或格式边界不稳定时，团队很难定位是 Latent-Switch 数据本身的问题，还是混合比例、tokenizer 或训练配置的问题。

第四步是谨慎解释效果。若模型使用更少 visible tokens 得到相近答案，不一定说明 latent reasoning 已经学好；它也可能只是学会直接回答。真正的验收应同时查看答案正确率、显式验证链质量、格式闭合率、latent 边界稳定性和不同预算下的 token 成本。只有这些指标一起改善，才能说明数据集确实在支持“隐式规划加显式验证”的目标。

这也是本章反复强调 schema、mask 和质量报告的原因：latent reasoning 能否落地，首先取决于数据是否把隐藏规划的接口定义清楚。
这一点尤其重要。

### 案例B：小结

Latent-Switch-69K 展示了推理数据工程的一种重要转向：从“收集更长、更详细的 CoT”转向“设计更有效的推理监督结构”。它从 Dolci-Think-SFT-32B 的推理轨迹出发，通过 teacher distillation 提取 solution intuition 和 compressed CoT，把 intuition 长度映射为 latent budget，再渲染为 `<latent_think>`、placeholder、`<think>`、answer tokens 组成的 student sequence。最终，prompt 与 latent interior 被普通 CE mask 掉，边界、显式推理、答案和结束 token 获得各自的监督，teacher KL 只对选定可见位置生效。

这种设计让数据集不再只是文本集合，而是包含结构、预算、mask 和质量报告的训练接口。对后续推理模型和 RL 数据工程而言，Latent-Switch-69K 的价值正是在这里：它把“少写一些推理”变成了一个可训练、可检查、可迭代的数据工程问题。

## 本章小结

本章围绕交互式数据与推理轨迹数据展开，分别讨论语音风格控制数据和 latent reasoning 监督数据。前者关注语义内容与风格属性如何在多轮交互中被拆分、标注和评测；后者关注显式 CoT、压缩推理、latent budget、训练 mask 与教师信号如何共同构成可学习的推理接口。二者的共同点在于，数据集已经不只是输入输出样本，而是在定义模型行为的过程结构。

面向大模型数据工程，本章强调三点：第一，交互数据需要保留状态变化、约束条件和反馈信号；第二，推理轨迹数据需要明确可见推理、隐藏规划和答案监督之间的边界；第三，任何压缩或隐式化设计都必须配套质量报告与可解释评测。只有这样，数据集才能支持从“生成更多内容”转向“训练更可控、更高效的模型行为”。

## 参考文献

An K, Chen Q, Deng C, Du Z, Gao C, Gao Z, Gu Y, He T, Hu H, Hu K, others (2024) FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs. arXiv preprint arXiv:2407.04051.

Chanfungjan (2026) VoiceStyleControl. GitHub repository. https://github.com/Chanfungjan/VoiceStyleControl.

Du Z, Chen Q, Zhang S, Hu K, Lu H, Yang Y, Hu H, Zheng S, Gu Y, Ma Z, Gao Z, Yan Z (2024) CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens. arXiv preprint arXiv:2407.05407.

Du Z, Wang Y, Chen Q, Shi X, Lv X, Zhao T, Gao Z, Yang Y, Gao C, Wang H, others (2024) CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models. arXiv preprint arXiv:2412.10117.

Mittag G, Naderi B, Chehadi A, Möller S (2021) NISQA: A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets. In: Interspeech 2021, pp 2127-2131.

Song X (2026) S3Tokenizer: Reverse Engineering of Supervised Semantic Speech Tokenizer proposed in CosyVoice. GitHub repository. https://github.com/xingchensong/S3Tokenizer.

Yang A, Li A, Yang B, Zhang B, Hui B, Zheng B, Yu B, Gao C, Huang C, Lv C, others (2025) Qwen3 Technical Report. arXiv preprint arXiv:2505.09388.

1. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., & Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022.
2. Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., & Cobbe, K. (2023). Let's Verify Step by Step. arXiv:2305.20050.
3. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629.
4. DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.
5. Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., & Steinhardt, J. (2021). Measuring Mathematical Problem Solving With the MATH Dataset. NeurIPS 2021.
