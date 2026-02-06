# 第5章：分词与序列化

---

## 本章摘要

分词（Tokenization）是连接原始文本与神经网络的桥梁。清洗后的高质量语料需要转换为模型可以理解的数字序列，才能输入到 Transformer 中进行训练。本章将深入探讨分词器的工作原理，包括 BPE、WordPiece、Unigram 三大主流算法；介绍如何为特定领域构建和扩充词表；最后讨论数据混合与课程学习策略，这些策略决定了不同类型数据在训练过程中的呈现顺序和比例。

---

## 场景引入

你的团队正在训练一个专门处理代码的大模型。使用标准的 GPT-2 分词器进行初步实验后，发现了一个奇怪的现象：模型生成的代码经常在缩进处出错，把四个空格拆成了多个不同的 token，导致缩进不一致。更糟糕的是，一些常见的编程关键字（如 `def`、`return`）被拆分成了多个子词，模型需要额外的上下文才能理解它们的含义。

经过分析，你发现问题出在分词器上。GPT-2 的分词器是在网页文本上训练的，对代码的特殊结构（如空白符、驼峰命名、特殊符号）处理得并不好。为代码任务设计专门的分词器，成为了提升模型性能的关键一步。

这个例子说明：分词器绝非可以忽视的"预处理细节"，它对模型的能力有实质性的影响。

---

## 5_1 分词器原理

分词器的核心任务是将连续的文本字符串切分为离散的 token 序列，并将每个 token 映射到一个整数 ID。这个看似简单的任务，实际上涉及到复杂的算法设计和工程权衡。

### 5_1.1 为什么需要子词分词？

在深度学习时代早期，自然语言处理通常采用词级别（Word-level）或字符级别（Character-level）的分词方式。词级别分词将每个完整的单词视为一个 token，优点是语义清晰，缺点是词表规模庞大（需要覆盖所有可能出现的单词），且无法处理未登录词（Out-of-Vocabulary, OOV）。字符级别分词将每个字符视为一个 token，优点是词表极小且没有 OOV 问题，缺点是序列过长，模型难以捕捉长程依赖。

子词分词（Subword Tokenization）是一种折中方案。它将文本切分为比单词更小、比字符更大的单元。高频词保持完整，低频词则被拆分为更小的子词单元。例如，"unhappiness" 可能被拆分为 "un" + "happi" + "ness"。这种方式既控制了词表规模，又保留了一定的语义信息，还能通过子词组合处理未见过的词汇。

![图5-1：分词粒度对比](../images/chapter2/图5_1_分词粒度对比.png)

*图5-1：分词粒度对比 —— 词级别、字符级别与子词级别的权衡*

目前主流的大语言模型几乎都采用子词分词。GPT 系列使用 BPE，BERT 使用 WordPiece，T5 和 LLaMA 使用 SentencePiece（支持 BPE 和 Unigram）。理解这些算法的原理，是进行分词器定制和优化的基础。

### 5_1.2 BPE：字节对编码

BPE（Byte Pair Encoding）最初是一种数据压缩算法，后被 Sennrich 等人在 2015 年引入到神经机器翻译中，成为最广泛使用的子词分词算法。

BPE 的核心思想非常直观：从字符级别开始，反复合并出现频率最高的相邻 token 对，直到达到预定的词表大小。具体步骤如下：

1. 将所有训练文本拆分为字符序列，每个字符作为初始 token
2. 统计所有相邻 token 对的出现频率
3. 将频率最高的 token 对合并为一个新 token
4. 重复步骤 2-3，直到词表达到目标大小

以下是一个简化的 BPE 训练实现：

```python
from collections import Counter, defaultdict

def train_bpe(corpus: list, vocab_size: int) -> dict:
    """
    训练 BPE 分词器
    
    Args:
        corpus: 训练语料列表
        vocab_size: 目标词表大小
    
    Returns:
        合并规则字典
    """
    # 初始化：将每个词拆分为字符，并添加词尾标记
    word_freqs = Counter()
    for text in corpus:
        for word in text.split():
            # 添加词尾标记 </w> 以区分词中和词尾的相同字符
            word_freqs[' '.join(list(word)) + ' </w>'] += 1
    
    merges = {}
    vocab = set()
    
    # 初始词表为所有字符
    for word in word_freqs:
        for char in word.split():
            vocab.add(char)
    
    while len(vocab) < vocab_size:
        # 统计相邻 token 对频率
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            tokens = word.split()
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] += freq
        
        if not pair_freqs:
            break
        
        # 找到频率最高的 pair
        best_pair = max(pair_freqs, key=pair_freqs.get)
        
        # 合并这个 pair
        new_token = best_pair[0] + best_pair[1]
        merges[best_pair] = new_token
        vocab.add(new_token)
        
        # 更新词频表
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            new_word = word.replace(
                best_pair[0] + ' ' + best_pair[1], 
                new_token
            )
            new_word_freqs[new_word] = freq
        word_freqs = new_word_freqs
    
    return merges

def apply_bpe(text: str, merges: dict) -> list:
    """应用 BPE 分词"""
    tokens = list(text) + ['</w>']
    
    while True:
        # 找到可以合并的 pair
        pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
        merge_pair = None
        for pair in pairs:
            if pair in merges:
                merge_pair = pair
                break
        
        if merge_pair is None:
            break
        
        # 执行合并
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == merge_pair:
                new_tokens.append(merges[merge_pair])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    
    return tokens
```

BPE 的一个重要变体是 Byte-level BPE，由 GPT-2 引入。传统 BPE 在字符级别操作，需要处理 Unicode 编码问题。Byte-level BPE 直接在字节级别操作，将每个字节映射到一个可打印字符，从而避免了编码问题，且天然支持任何语言。这也是为什么 GPT 系列模型可以处理任意语言文本的原因。

### 5_1.3 WordPiece：BERT 的选择

WordPiece 是 Google 为 BERT 开发的分词算法，与 BPE 非常相似，主要区别在于选择合并对的标准。

BPE 选择出现频率最高的 pair 进行合并。WordPiece 则选择能够最大化训练数据似然的 pair。具体来说，对于候选 pair (A, B)，WordPiece 计算合并后词表对训练数据的语言模型概率增益，选择增益最大的 pair 进行合并。

在实践中，这意味着 WordPiece 倾向于合并那些"在一起出现的概率远高于独立出现概率之积"的 pair。这个标准使得 WordPiece 对于低频但有意义的模式更加敏感。

WordPiece 的另一个特点是使用 `##` 前缀来标识非词首的子词。例如，"playing" 可能被分词为 ["play", "##ing"]。这种表示方式明确区分了子词在原词中的位置，有助于模型理解词汇结构。

```python
# WordPiece 分词示例（使用 HuggingFace tokenizers）
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

# 初始化 WordPiece 分词器
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 训练
trainer = WordPieceTrainer(
    vocab_size=30000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)
tokenizer.train(files=["corpus.txt"], trainer=trainer)

# 使用
output = tokenizer.encode("unhappiness")
print(output.tokens)  # ['un', '##happi', '##ness']
```

### 5_1.4 Unigram：概率视角的分词

Unigram 分词由 Kudo 在 2018 年提出，采用了与 BPE/WordPiece 完全不同的思路。BPE 和 WordPiece 都是自底向上的方法——从小的单元开始，逐步合并成大的单元。Unigram 则是自顶向下的方法——从一个包含所有可能子词的大词表开始，逐步删减到目标大小。

Unigram 将分词建模为一个概率问题。给定词表 V 和每个 token 的概率 P(t)，一个文本的分词结果是使得总概率最大化的切分方式：

$$P(x_1, x_2, ..., x_n) = \prod_{i=1}^{n} P(x_i)$$

训练过程使用 EM 算法：E 步骤计算当前词表下每个 token 的期望出现次数，M 步骤更新 token 概率。然后删除那些删除后对总似然影响最小的 token，直到达到目标词表大小。

Unigram 的一个独特优势是它天然支持多种分词结果的概率建模。对于一个给定的文本，可能存在多种合法的分词方式，Unigram 可以为每种方式赋予一个概率。这在某些应用场景（如语音识别中的多假设处理）中非常有用。

### 5_1.5 三种算法的对比

三种主流子词分词算法各有特点，选择时需要根据具体场景权衡。

| 算法 | 核心思想 | 优势 | 劣势 | 典型应用 |
|------|----------|------|------|----------|
| BPE | 自底向上，频率驱动合并 | 简单直观，训练快 | 贪心策略可能非最优 | GPT系列、LLaMA |
| WordPiece | 自底向上，似然驱动合并 | 对低频有意义模式敏感 | 计算复杂度较高 | BERT、DistilBERT |
| Unigram | 自顶向下，概率建模 | 理论优雅，支持多分词 | 训练较慢 | T5、mT5、ALBERT |

![图5-2：分词算法对比](../images/chapter2/图5_2_分词算法对比.png)

*图5-2：BPE、WordPiece、Unigram 三种分词算法的对比*

在实际工程中，SentencePiece 是最常用的分词工具包。它支持 BPE 和 Unigram 两种算法，提供了语言无关的预处理（不依赖空格分词），并且与主流深度学习框架无缝集成。

```python
import sentencepiece as spm

# 训练 SentencePiece 模型
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='my_tokenizer',
    vocab_size=32000,
    model_type='bpe',  # 或 'unigram'
    character_coverage=0_9995,
    num_threads=16
)

# 加载和使用
sp = spm.SentencePieceProcessor(model_file='my_tokenizer.model')
tokens = sp.encode('Hello, world!', out_type=str)
print(tokens)  # ['▁Hello', ',', '▁world', '!']
ids = sp.encode('Hello, world!')
print(ids)  # [1234, 56, 789, 10]
```

---

## 5_2 词表设计与扩充

词表（Vocabulary）是分词器的核心组成部分。词表的大小、覆盖范围和结构直接影响模型的性能和效率。

### 5_2.1 词表大小的权衡

词表大小是分词器设计中最重要的超参数之一。较大的词表意味着更多的 token 被保留为完整单元，序列长度更短，但嵌入矩阵更大，参数更多；较小的词表意味着更多的词被拆分为子词，序列长度更长，但模型参数更少。

主流大模型的词表大小通常在 32K 到 128K 之间。GPT-2 使用 50,257 的词表，LLaMA 使用 32,000，GPT-4 据报道使用约 100,000。选择词表大小时需要考虑以下因素：

**计算效率**：词表越大，嵌入层和输出层的参数越多。对于一个 d 维的模型，词表大小为 V 时，嵌入矩阵包含 V × d 个参数。当 V 从 32K 增加到 128K 时，这部分参数量增加 4 倍。

**序列长度**：词表越大，平均每个 token 覆盖的字符越多，同样的文本被分成的 token 数越少。这对于处理长文档尤为重要，因为 Transformer 的计算复杂度与序列长度的平方成正比。

**稀有词处理**：词表越大，越多的稀有词可以被保留为完整 token，减少了 UNK 和过度切分的问题。但这也意味着稀有 token 在训练中见到的样本更少，可能导致嵌入质量不佳。

```python
# 分析不同词表大小对序列长度的影响
def analyze_vocab_size_impact(text: str, vocab_sizes: list) -> dict:
    """分析词表大小对分词结果的影响"""
    import sentencepiece as spm
    
    results = {}
    for vocab_size in vocab_sizes:
        # 训练不同词表大小的分词器
        spm.SentencePieceTrainer.train(
            input='corpus.txt',
            model_prefix=f'tokenizer_{vocab_size}',
            vocab_size=vocab_size,
            model_type='bpe'
        )
        
        sp = spm.SentencePieceProcessor(model_file=f'tokenizer_{vocab_size}.model')
        tokens = sp.encode(text)
        
        results[vocab_size] = {
            'num_tokens': len(tokens),
            'chars_per_token': len(text) / len(tokens),
            'compression_ratio': len(text.encode('utf-8')) / (len(tokens) * 2)
        }
    
    return results
```

### 5_2.2 多语言词表设计

训练多语言模型时，词表设计面临额外的挑战：如何在有限的词表空间中平衡不同语言的覆盖？

一个常见的问题是"词表诅咒"（Vocabulary Curse）。如果直接在多语言语料上训练分词器，高资源语言（如英语）会占据大部分词表空间，低资源语言的覆盖严重不足。这导致低资源语言的文本被过度切分，序列长度膨胀，模型性能下降。

解决这一问题的常用策略包括：

**语料平衡**：在训练分词器之前，对不同语言的语料进行上采样或下采样，使每种语言的权重更加均衡。

**温度采样**：类似于我们在第 3 章讨论的多语言数据平衡策略，使用温度参数控制不同语言的采样概率。

**语言特定的字符覆盖**：确保每种目标语言的基本字符集都被纳入词表，即使它们的频率很低。SentencePiece 提供了 `character_coverage` 参数来控制这一点。

```python
# 多语言分词器训练示例
import sentencepiece as spm

# 使用字符覆盖率确保多语言支持
spm.SentencePieceTrainer.train(
    input='multilingual_corpus.txt',
    model_prefix='multilingual_tokenizer',
    vocab_size=64000,
    model_type='unigram',
    character_coverage=0_9999,  # 高覆盖率确保稀有字符被包含
    input_sentence_size=10000000,
    shuffle_input_sentence=True,
    # 特殊处理中日韩字符
    byte_fallback=True  # 未知字符回退到字节级别
)
```

### 5_2.3 领域特定词表扩充

在将预训练模型应用于特定领域（如医疗、法律、代码）时，经常会遇到大量领域术语被过度切分的问题。这不仅增加了序列长度，还可能影响模型对专业概念的理解。

词表扩充（Vocabulary Extension）是解决这一问题的有效手段。基本思路是：在保留原有词表的基础上，添加新的领域特定 token。

```python
from transformers import AutoTokenizer

def extend_tokenizer(base_tokenizer_name: str, 
                     domain_terms: list,
                     output_dir: str) -> None:
    """
    扩充预训练分词器的词表
    
    Args:
        base_tokenizer_name: 基础分词器名称
        domain_terms: 领域特定术语列表
        output_dir: 输出目录
    """
    # 加载基础分词器
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
    
    print(f"原始词表大小: {len(tokenizer)}")
    
    # 过滤已存在的 token
    new_tokens = []
    for term in domain_terms:
        if term not in tokenizer.get_vocab():
            new_tokens.append(term)
    
    # 添加新 token
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"添加了 {num_added} 个新 token")
    print(f"新词表大小: {len(tokenizer)}")
    
    # 保存扩充后的分词器
    tokenizer.save_pretrained(output_dir)
    
    return tokenizer

# 示例：为医疗领域扩充词表
medical_terms = [
    '冠状动脉',
    '心肌梗死',
    '动脉粥样硬化',
    'COVID-19',
    'mRNA疫苗',
    '计算机断层扫描',
    # ... 更多术语
]

tokenizer = extend_tokenizer(
    'meta-llama/Llama-2-7b',
    medical_terms,
    './medical_tokenizer'
)
```

词表扩充后，需要同步扩展模型的嵌入矩阵。新增 token 的嵌入通常初始化为随机值或现有相关 token 的平均值，然后通过继续预训练来学习有意义的表示。

```python
from transformers import AutoModelForCausalLM

def resize_model_embeddings(model_name: str, 
                            tokenizer,
                            output_dir: str) -> None:
    """调整模型嵌入层大小以匹配扩充后的词表"""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 调整嵌入层大小
    model.resize_token_embeddings(len(tokenizer))
    
    # 可选：使用相似 token 的均值初始化新嵌入
    # 这比随机初始化通常能带来更好的效果
    
    model.save_pretrained(output_dir)
```

### 5_2.4 词表设计的最佳实践

基于业界的经验，以下是词表设计的一些最佳实践：

**保留足够的特殊 token 位置**：预留一些 token ID 用于未来可能添加的特殊 token（如新的控制符、领域标记等）。许多分词器预留了 100-1000 个位置。

**确保数字和代码符号的合理切分**：数字在很多任务中很重要，但标准分词器往往对数字处理不佳。考虑将单个数字作为独立 token，或使用特殊的数字编码策略。

**测试边界情况**：在确定词表之前，测试各种边界情况：超长单词、特殊字符、混合语言文本、代码片段等。确保分词结果符合预期。

**文档化词表决策**：记录词表大小、训练语料、特殊 token 列表等信息，便于后续的模型迭代和问题排查。

---

## 5_3 数据混合与课程学习

确定了分词器之后，下一个关键问题是：如何组织和呈现训练数据？不同来源、不同质量的数据应该以怎样的比例混合？训练过程中数据的顺序是否重要？

### 5_3.1 数据混合策略

正如我们在第 3 章讨论的，高质量的预训练数据集通常混合了多种来源：网页、书籍、代码、论文、对话等。每种来源的数据量和质量都不同，简单地按原始比例混合往往不是最优的。

**静态混合**是最简单的策略：在训练开始前确定各来源的混合比例，将数据打乱后顺序训练。这种方法简单易实现，但缺乏灵活性。

```python
# 静态数据混合示例
import random

def static_mix(data_sources: dict, target_size: int) -> list:
    """
    静态混合多个数据源
    
    Args:
        data_sources: {source_name: (data_list, weight)}
        target_size: 目标数据集大小
    
    Returns:
        混合后的数据列表
    """
    mixed_data = []
    
    # 计算每个来源的采样数量
    total_weight = sum(w for _, w in data_sources.values())
    
    for source_name, (data, weight) in data_sources.items():
        num_samples = int(target_size * weight / total_weight)
        
        # 如果数据不足，重复采样
        if len(data) < num_samples:
            sampled = random.choices(data, k=num_samples)
        else:
            sampled = random.sample(data, num_samples)
        
        mixed_data.extend(sampled)
    
    random.shuffle(mixed_data)
    return mixed_data

# 使用示例
data_sources = {
    'web': (web_data, 0_6),
    'books': (book_data, 0_15),
    'code': (code_data, 0_1),
    'papers': (paper_data, 0_1),
    'wikipedia': (wiki_data, 0_05)
}

mixed = static_mix(data_sources, target_size=1000000)
```

![图5-3：数据混合策略](../images/chapter2/图5_3_数据混合策略.png)

*图5-3：静态混合与动态混合策略对比*

**动态混合**允许在训练过程中调整混合比例。一些研究表明，不同训练阶段的最优数据配比可能不同。例如，训练早期使用更多样化的数据帮助模型建立广泛的语言理解；训练后期增加高质量数据的比例以提升模型的精细能力。

```python
class DynamicDataMixer:
    """动态数据混合器"""
    
    def __init__(self, data_sources: dict, schedule: list):
        """
        初始化动态混合器
        
        Args:
            data_sources: 数据源字典
            schedule: [(step_threshold, weights_dict), ...]
                     在不同训练步数使用不同的混合权重
        """
        self.data_sources = data_sources
        self.schedule = sorted(schedule, key=lambda x: x[0])
        self.current_step = 0
    
    def get_weights(self) -> dict:
        """获取当前步数对应的权重"""
        for step_threshold, weights in reversed(self.schedule):
            if self.current_step >= step_threshold:
                return weights
        return self.schedule[0][1]
    
    def sample_batch(self, batch_size: int) -> list:
        """采样一个 batch"""
        weights = self.get_weights()
        batch = []
        
        for source_name, weight in weights.items():
            num_samples = int(batch_size * weight)
            data = self.data_sources[source_name]
            batch.extend(random.choices(data, k=num_samples))
        
        random.shuffle(batch)
        self.current_step += 1
        return batch[:batch_size]

# 使用示例：训练早期强调多样性，后期强调质量
schedule = [
    (0, {'web': 0_5, 'books': 0_2, 'code': 0_15, 'papers': 0_1, 'wiki': 0_05}),
    (100000, {'web': 0_4, 'books': 0_25, 'code': 0_15, 'papers': 0_15, 'wiki': 0_05}),
    (500000, {'web': 0_3, 'books': 0_3, 'code': 0_2, 'papers': 0_15, 'wiki': 0_05}),
]

mixer = DynamicDataMixer(data_sources, schedule)
```

### 5_3.2 课程学习

课程学习（Curriculum Learning）是一种受人类学习过程启发的训练策略。核心思想是：先让模型学习"简单"的样本，再逐渐过渡到"困难"的样本。这种策略在多项研究中被证明可以加速收敛并提升最终性能。

在预训练场景下，"简单"和"困难"可以有多种定义：

**基于长度**：短文本通常比长文本更容易学习。课程可以从短序列开始，逐渐增加序列长度。

**基于困惑度**：困惑度低的文本（语言模型更"熟悉"的文本）可以视为"简单"样本。可以使用一个预训练的小模型评估样本难度，然后按难度排序呈现给主模型。

**基于噪声水平**：高质量、低噪声的文本先呈现，然后逐渐引入质量较低但可能包含独特信息的文本。

```python
import numpy as np

class CurriculumScheduler:
    """课程学习调度器"""
    
    def __init__(self, 
                 data: list, 
                 difficulty_scores: list,
                 total_steps: int,
                 strategy: str = 'linear'):
        """
        初始化课程调度器
        
        Args:
            data: 数据列表
            difficulty_scores: 每个样本的难度分数（越高越难）
            total_steps: 总训练步数
            strategy: 课程策略 ('linear', 'sqrt', 'exp')
        """
        self.data = np.array(data)
        self.difficulty_scores = np.array(difficulty_scores)
        self.total_steps = total_steps
        self.strategy = strategy
        
        # 按难度排序
        sorted_indices = np.argsort(self.difficulty_scores)
        self.sorted_data = self.data[sorted_indices]
        self.sorted_scores = self.difficulty_scores[sorted_indices]
    
    def get_curriculum_fraction(self, current_step: int) -> float:
        """
        计算当前步数应该使用的数据比例
        
        返回值在 [0, 1] 之间，表示使用最简单的多少比例的数据
        """
        progress = current_step / self.total_steps
        
        if self.strategy == 'linear':
            return progress
        elif self.strategy == 'sqrt':
            return np.sqrt(progress)
        elif self.strategy == 'exp':
            return 1 - np.exp(-3 * progress)
        else:
            return progress
    
    def sample_batch(self, current_step: int, batch_size: int) -> list:
        """根据当前进度采样 batch"""
        fraction = self.get_curriculum_fraction(current_step)
        
        # 确定可用数据范围
        available_size = max(int(len(self.sorted_data) * fraction), batch_size)
        available_data = self.sorted_data[:available_size]
        
        # 从可用范围内随机采样
        indices = np.random.choice(len(available_data), size=batch_size, replace=True)
        return available_data[indices].tolist()
```

![图5-4：课程学习示意图](../images/chapter2/图5_4_课程学习示意图.png)

*图5-4：课程学习原理 —— 从简单样本逐渐过渡到困难样本*

### 5_3.3 数据采样与批次构建

在实际训练中，数据的组织方式对效率和效果都有影响。以下是一些重要的工程考量：

**Pack 打包策略**：为了充分利用计算资源，通常将多个短序列打包到一个固定长度的序列中。这样可以减少 padding 带来的计算浪费。关键问题是如何处理打包后的注意力掩码——不同文档之间不应该相互注意。

```python
def pack_sequences(sequences: list, max_length: int, eos_token_id: int) -> list:
    """
    将多个短序列打包到固定长度
    
    Args:
        sequences: token id 序列列表
        max_length: 目标序列长度
        eos_token_id: 序列结束符 ID
    
    Returns:
        打包后的序列列表，每个长度为 max_length
    """
    packed = []
    current_pack = []
    current_length = 0
    
    for seq in sequences:
        seq_with_eos = seq + [eos_token_id]
        
        if current_length + len(seq_with_eos) <= max_length:
            current_pack.extend(seq_with_eos)
            current_length += len(seq_with_eos)
        else:
            # 当前 pack 已满，开始新的
            if current_pack:
                # padding 到 max_length
                current_pack.extend([eos_token_id] * (max_length - current_length))
                packed.append(current_pack)
            
            current_pack = seq_with_eos
            current_length = len(seq_with_eos)
    
    # 处理最后一个 pack
    if current_pack:
        current_pack.extend([eos_token_id] * (max_length - current_length))
        packed.append(current_pack)
    
    return packed
```

**文档边界处理**：在打包序列时，需要创建一个"文档边界掩码"，确保模型在生成时不会跨越文档边界进行注意力计算。

**数据加载效率**：对于 TB 级数据集，数据加载本身可能成为瓶颈。常用的优化手段包括：预处理后以二进制格式存储（如 numpy 的 memmap）、多进程并行加载、预取（prefetch）下一批数据。

### 5_3.4 序列化与存储格式

完成分词后，需要将 token 序列以高效的格式存储，以便训练时快速读取。

**常见的存储格式**包括：

**NumPy memmap**：将 token ID 存储为 numpy 数组，使用内存映射访问。优点是简单直接，支持随机访问；缺点是不支持压缩，存储空间较大。

```python
import numpy as np

def save_as_memmap(token_ids: list, output_path: str):
    """将 token ID 列表保存为 memmap 格式"""
    arr = np.array(token_ids, dtype=np.uint16)  # 假设词表 < 65536
    fp = np.memmap(output_path, dtype='uint16', mode='w+', shape=arr.shape)
    fp[:] = arr[:]
    fp.flush()
    
def load_memmap(path: str, shape: tuple):
    """加载 memmap 格式的 token ID"""
    return np.memmap(path, dtype='uint16', mode='r', shape=shape)
```

**Arrow/Parquet**：使用 Apache Arrow 格式存储，支持压缩和高效的列式访问。HuggingFace Datasets 库内部使用这种格式。

**自定义二进制格式**：一些大型项目使用自定义的二进制格式，针对特定的访问模式优化。例如 GPT-NeoX 使用的二进制打包格式。

```python
# 使用 HuggingFace Datasets 处理分词后的数据
from datasets import Dataset

def tokenize_and_save(raw_data: list, tokenizer, output_dir: str):
    """分词并保存为 Datasets 格式"""
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=2048,
            return_attention_mask=False
        )
    
    # 创建 Dataset
    ds = Dataset.from_dict({'text': raw_data})
    
    # 分词
    tokenized_ds = ds.map(
        tokenize_function,
        batched=True,
        num_proc=16,
        remove_columns=['text']
    )
    
    # 保存
    tokenized_ds.save_to_disk(output_dir)
```

---

## 5_4 完整的数据准备流水线

将前面讨论的各个步骤串联起来，构建一个从原始文本到训练就绪数据的完整流水线。

```python
from dataclasses import dataclass
from typing import Optional
import sentencepiece as spm

@dataclass
class DataPrepConfig:
    """数据准备配置"""
    # 分词器配置
    tokenizer_path: str
    max_seq_length: int = 2048
    
    # 数据混合配置
    mix_weights: dict = None  # {source: weight}
    
    # 课程学习配置
    use_curriculum: bool = False
    curriculum_strategy: str = 'linear'
    
    # 输出配置
    pack_sequences: bool = True
    output_format: str = 'arrow'  # 'arrow', 'memmap', 'jsonl'

class DataPreparationPipeline:
    """数据准备流水线"""
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
        self.tokenizer = spm.SentencePieceProcessor(model_file=config.tokenizer_path)
    
    def tokenize_document(self, text: str) -> list:
        """分词单个文档"""
        return self.tokenizer.encode(text)
    
    def process_source(self, source_path: str, source_name: str) -> list:
        """处理单个数据源"""
        documents = self.load_documents(source_path)
        
        tokenized = []
        for doc in documents:
            tokens = self.tokenize_document(doc['text'])
            if len(tokens) > 10:  # 过滤过短的文档
                tokenized.append({
                    'input_ids': tokens,
                    'source': source_name,
                    'length': len(tokens)
                })
        
        return tokenized
    
    def mix_sources(self, sources: dict) -> list:
        """混合多个数据源"""
        mixed = []
        weights = self.config.mix_weights or {s: 1_0 for s in sources}
        total_weight = sum(weights.values())
        
        # 确定每个来源的采样数
        total_samples = sum(len(data) for data in sources.values())
        
        for source_name, data in sources.items():
            weight = weights.get(source_name, 1_0) / total_weight
            num_samples = int(total_samples * weight)
            
            if len(data) >= num_samples:
                sampled = random.sample(data, num_samples)
            else:
                sampled = random.choices(data, k=num_samples)
            
            mixed.extend(sampled)
        
        random.shuffle(mixed)
        return mixed
    
    def pack_and_save(self, data: list, output_path: str):
        """打包并保存数据"""
        if self.config.pack_sequences:
            sequences = [d['input_ids'] for d in data]
            packed = pack_sequences(
                sequences, 
                self.config.max_seq_length,
                self.tokenizer.eos_id()
            )
        else:
            packed = [d['input_ids'] for d in data]
        
        # 根据配置选择输出格式
        if self.config.output_format == 'arrow':
            self.save_as_arrow(packed, output_path)
        elif self.config.output_format == 'memmap':
            self.save_as_memmap(packed, output_path)
        else:
            self.save_as_jsonl(packed, output_path)
    
    def run(self, source_paths: dict, output_path: str):
        """运行完整流水线"""
        # 1. 处理各数据源
        sources = {}
        for source_name, path in source_paths.items():
            print(f"Processing {source_name}...")
            sources[source_name] = self.process_source(path, source_name)
        
        # 2. 混合数据
        print("Mixing data sources...")
        mixed = self.mix_sources(sources)
        
        # 3. 可选：应用课程学习排序
        if self.config.use_curriculum:
            print("Applying curriculum ordering...")
            mixed = self.apply_curriculum(mixed)
        
        # 4. 打包并保存
        print("Packing and saving...")
        self.pack_and_save(mixed, output_path)
        
        print(f"Done! Saved {len(mixed)} samples to {output_path}")
```

![图5-5：数据准备完整流水线](../images/chapter2/图5_5_数据准备完整流水线.png)

*图5-5：从原始文本到训练就绪数据的完整流水线*

---

## 5_5 本章小结

本章系统介绍了分词与数据序列化的核心技术。

在分词器原理方面，子词分词是当前大模型的主流选择，在词表大小和序列长度之间取得了良好平衡。BPE 采用频率驱动的自底向上合并策略，简单高效；WordPiece 使用似然驱动的合并标准，对低频有意义模式更敏感；Unigram 采用自顶向下的概率建模方法，理论上更优雅。SentencePiece 是最常用的工具包，支持多种算法和语言无关的处理。

在词表设计方面，词表大小需要在计算效率、序列长度和稀有词处理之间权衡，主流模型通常使用 32K-128K 的词表。多语言词表设计需要平衡不同语言的覆盖，避免"词表诅咒"。领域特定词表扩充可以改善专业术语的处理，但需要配合模型嵌入层的扩展。

在数据混合方面，静态混合简单直接，动态混合允许训练过程中调整配比。课程学习策略从简单样本开始、逐渐过渡到困难样本，可以加速收敛并提升性能。数据打包和高效的存储格式对于大规模训练至关重要。

![图5-6：本章知识结构](../images/chapter2/图5_6_本章知识结构.png)

*图5-6：第5章知识结构 —— 分词算法、词表设计、数据组织三大主题*

---

## 延伸阅读

关于分词与数据序列化的深入内容，以下资源值得参考：

SentencePiece 论文（Kudo and Richardson, 2018）介绍了语言无关的子词分词方法。BPE 论文（Sennrich et al., 2015）是将 BPE 引入 NLP 的开创性工作。Unigram 论文（Kudo, 2018）提供了子词分词的概率视角。HuggingFace Tokenizers 库文档（huggingface.co/docs/tokenizers）是实践层面的权威参考。关于课程学习，Bengio 等人的综述论文提供了全面的理论框架。

---

## 下一章预告

至此，我们完成了文本预训练数据工程的全部内容。在下一章《图文对数据处理》中，我们将进入多模态数据工程的领域。你将学习如何处理 LAION-5B 风格的图文配对数据，如何使用 img2dataset 进行高并发图像下载，以及如何构建多模态数据清洗流水线。

带着这个问题进入下一章：一张图片的"质量"应该如何定义？除了分辨率和清晰度，还有哪些维度需要考虑？
