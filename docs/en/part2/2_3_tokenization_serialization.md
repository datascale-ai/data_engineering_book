# Chapter 5: Tokenization, Serialization, and Efficient Loading (DataLoader Optimization)

---

## Chapter Summary

Tokenization is the bridge connecting raw text and neural networks. High-quality corpora after cleaning need to be converted into numeric sequences that models can understand before they can be fed into Transformers for training. This chapter delves into how tokenizers work, including the three mainstream algorithms BPE, WordPiece, and Unigram; introduces how to build and extend vocabularies for specific domains; and finally discusses data mixing and curriculum learning strategies, which determine the presentation order and proportions of different data types during training.

---

## Scenario Introduction

Your team is training a large model specialized for code. After preliminary experiments with the standard GPT-2 tokenizer, you discover a strange phenomenon: the model often makes errors at indentation in generated code, splitting four spaces into multiple different tokens, causing inconsistent indentation. Worse yet, some common programming keywords like `def` and `return` are split into multiple subwords, requiring the model to use additional context to understand their meaning.

After analysis, you find the problem lies in the tokenizer. The GPT-2 tokenizer was trained on web text and does not handle the special structure of code (such as whitespace, camelCase, special symbols) well. Designing a specialized tokenizer for code tasks has become a key step to improving model performance.

This example shows: the tokenizer is by no means a "preprocessing detail" that can be ignored—it has a substantial impact on model capabilities.

---

## 5.1 Tokenizer Principles

The core task of a tokenizer is to segment continuous text strings into discrete token sequences and map each token to an integer ID. This seemingly simple task actually involves complex algorithm design and engineering trade-offs.

### 5.1.1 Why Subword Tokenization?

In the early days of deep learning, natural language processing typically used word-level or character-level tokenization. Word-level tokenization treats each complete word as a token—the advantage is clear semantics, the disadvantage is a huge vocabulary (needing to cover all possible words) and inability to handle out-of-vocabulary (OOV) words. Character-level tokenization treats each character as a token—the advantage is a tiny vocabulary with no OOV problem, the disadvantage is excessively long sequences making it difficult for models to capture long-range dependencies.

Subword tokenization is a compromise. It segments text into units smaller than words but larger than characters. High-frequency words remain intact; low-frequency words are split into smaller subword units. For example, "unhappiness" might be split into "un" + "happi" + "ness". This approach both controls vocabulary size and retains certain semantic information, while being able to handle unseen vocabulary through subword combination.

![Figure 5-1: Tokenization Granularity Comparison](../../images/part2/图5_1_分词粒度对比.png)

*Figure 5-1: Tokenization Granularity Comparison — Trade-offs between word-level, character-level, and subword-level*

Currently, almost all mainstream large language models use subword tokenization. The GPT series uses BPE, BERT uses WordPiece, and T5 and LLaMA use SentencePiece (supporting BPE and Unigram). Understanding the principles of these algorithms is the foundation for tokenizer customization and optimization.

### 5.1.2 BPE: Byte Pair Encoding

BPE (Byte Pair Encoding) was originally a data compression algorithm, later introduced to neural machine translation by Sennrich et al. in 2015, becoming the most widely used subword tokenization algorithm.

The core idea of BPE is very intuitive: start from the character level, repeatedly merge the most frequently occurring adjacent token pairs until reaching the target vocabulary size. Specific steps:

1. Split all training text into character sequences, each character as initial token
2. Count frequency of all adjacent token pairs
3. Merge the highest-frequency token pair into a new token
4. Repeat steps 2-3 until vocabulary reaches target size

Below is a simplified BPE training implementation:

```python
from collections import Counter, defaultdict

def train_bpe(corpus: list, vocab_size: int) -> dict:
    """
    Train BPE tokenizer
    
    Args:
        corpus: Training corpus list
        vocab_size: Target vocabulary size
    
    Returns:
        Merge rules dictionary
    """
    # Initialize: split each word into characters, add end-of-word marker
    word_freqs = Counter()
    for text in corpus:
        for word in text.split():
            # Add end-of-word marker </w> to distinguish same character in middle vs end of word
            word_freqs[' '.join(list(word)) + ' </w>'] += 1
    
    merges = {}
    vocab = set()
    
    # Initial vocabulary is all characters
    for word in word_freqs:
        for char in word.split():
            vocab.add(char)
    
    while len(vocab) < vocab_size:
        # Count adjacent token pair frequencies
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            tokens = word.split()
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] += freq
        
        if not pair_freqs:
            break
        
        # Find highest-frequency pair
        best_pair = max(pair_freqs, key=pair_freqs.get)
        
        # Merge this pair
        new_token = best_pair[0] + best_pair[1]
        merges[best_pair] = new_token
        vocab.add(new_token)
        
        # Update word frequency table
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
    """Apply BPE tokenization"""
    tokens = list(text) + ['</w>']
    
    while True:
        # Find mergeable pairs
        pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
        merge_pair = None
        for pair in pairs:
            if pair in merges:
                merge_pair = pair
                break
        
        if merge_pair is None:
            break
        
        # Perform merge
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

### 5.1.6 Byte-Level BPE: Deep Dive

An important variant of BPE is **Byte-level BPE**, first introduced by GPT-2. Traditional BPE operates at the character level and needs to handle Unicode encoding issues—character sets differ enormously across languages, and some characters (such as Emoji, special symbols) may not appear in training data, leading to UNK.

Byte-level BPE operates directly at the **byte level**, mapping each byte (0-255) to a printable character, thus avoiding encoding issues and natively supporting any language. This is also why GPT series models can process text in any language.

#### Working Principles

1. **Byte encoding**: Encode input text as UTF-8 byte sequences. For example, the Chinese character "你" is encoded as 3 bytes `[0xe4, 0xbd, 0xa0]` in UTF-8.
2. **Byte mapping**: Map the 256 possible byte values to 256 printable Unicode characters. This allows standard character-level BPE algorithms to process byte sequences.
3. **BPE training and application**: Perform standard BPE algorithm on the mapped byte sequences.

#### Impact on Multilingual Text

Byte-level BPE has significantly different impacts across languages:

- **English**: ASCII characters require only 1 byte, nearly equivalent to character-level BPE.
- **Chinese**: Each Chinese character requires 3 bytes; if the vocabulary doesn't contain sufficient Chinese tokens, one character may be split into 2-3 tokens, severely affecting sequence length and computational efficiency.
- **Japanese/Korean**: Require 3 and 3-4 bytes respectively, with similar issues.

This is why the original LLaMA model had poor Chinese capabilities—its vocabulary was primarily trained on English, causing Chinese characters to be excessively segmented, inflating input sequence lengths. The solution is **Chinese vocabulary extension**, which we'll discuss in detail in Section 5.2.4.

```python
# Core implementation of Byte-level BPE byte mapping
def bytes_to_unicode():
    """
    GPT-2's byte-to-Unicode mapping
    Maps 256 byte values to printable Unicode characters
    """
    # Directly printable ASCII ranges
    bs = list(range(ord('!'), ord('~') + 1)) + \
         list(range(ord('¡'), ord('¬') + 1)) + \
         list(range(ord('®'), ord('ÿ') + 1))
    
    cs = bs[:]
    n = 0
    # Map remaining bytes to higher Unicode code points
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def analyze_byte_level_impact(text: str, tokenizer_name: str):
    """Analyze the impact of Byte-level BPE on different languages"""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokens = tokenizer.tokenize(text)
    
    # Compute compression ratio
    utf8_bytes = len(text.encode('utf-8'))
    num_tokens = len(tokens)
    chars_per_token = len(text) / num_tokens
    bytes_per_token = utf8_bytes / num_tokens
    
    print(f"Text: '{text[:50]}...'")
    print(f"Characters: {len(text)}, UTF-8 bytes: {utf8_bytes}")
    print(f"Tokens: {num_tokens}")
    print(f"Average characters per token: {chars_per_token:.2f}")
    print(f"Average bytes per token: {bytes_per_token:.2f}")
    print(f"Tokens: {tokens[:20]}")
    
    return {'num_tokens': num_tokens, 'chars_per_token': chars_per_token}
```

### 5.1.3 WordPiece: BERT's Choice

WordPiece is the tokenization algorithm developed by Google for BERT, very similar to BPE, with the main difference in the criterion for selecting merge pairs.

BPE selects the most frequently occurring pair for merging. WordPiece selects the pair that maximizes training data likelihood. Specifically, for candidate pair (A, B), WordPiece computes the language model probability gain of the merged vocabulary on training data, and selects the pair with the greatest gain for merging.

In practice, this means WordPiece tends to merge pairs whose "probability of co-occurrence is far higher than the product of independent occurrence probabilities." This criterion makes WordPiece more sensitive to low-frequency but meaningful patterns.

Another characteristic of WordPiece is using the `##` prefix to identify non-word-initial subwords. For example, "playing" might be tokenized as ["play", "##ing"]. This representation clearly distinguishes subword position in the original word, helping the model understand vocabulary structure.

```python
# WordPiece tokenization example (using HuggingFace tokenizers)
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialize WordPiece tokenizer
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Train
trainer = WordPieceTrainer(
    vocab_size=30000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)
tokenizer.train(files=["corpus.txt"], trainer=trainer)

# Use
output = tokenizer.encode("unhappiness")
print(output.tokens)  # ['un', '##happi', '##ness']
```

### 5.1.4 Unigram: A Probabilistic Perspective on Tokenization

Unigram tokenization was proposed by Kudo in 2018, adopting a completely different approach from BPE/WordPiece. BPE and WordPiece are bottom-up methods—starting from small units and gradually merging into larger units. Unigram is top-down—starting from a large vocabulary containing all possible subwords and gradually pruning to target size.

Unigram models tokenization as a probability problem. Given vocabulary V and probability P(t) for each token, the tokenization result for a text is the segmentation that maximizes total probability:

$$P(x_1, x_2, ..., x_n) = \prod_{i=1}^{n} P(x_i)$$

The training process uses the EM algorithm: E-step computes expected occurrence count of each token under current vocabulary; M-step updates token probabilities. Then delete tokens whose deletion has minimal impact on total likelihood until reaching target vocabulary size.

A unique advantage of Unigram is its natural support for probability modeling of multiple tokenization results. For a given text, there may be multiple valid segmentation ways; Unigram can assign a probability to each. This is very useful in certain application scenarios (e.g., multi-hypothesis processing in speech recognition).

### 5.1.5 Comparison of the Three Algorithms

The three mainstream subword tokenization algorithms each have their characteristics; selection requires trade-offs based on specific scenarios.

| Algorithm | Core Idea | Advantages | Disadvantages | Typical Applications |
|------|----------|------|------|----------|
| BPE | Bottom-up, frequency-driven merging | Simple and intuitive, fast training | Greedy strategy may not be optimal | GPT series, LLaMA |
| WordPiece | Bottom-up, likelihood-driven merging | Sensitive to low-frequency meaningful patterns | Higher computation complexity | BERT, DistilBERT |
| Unigram | Top-down, probability modeling | Theoretically elegant, supports multiple segmentations | Slower training | T5, mT5, ALBERT |

![Figure 5-2: Tokenization Algorithm Comparison](../../images/part2/图5_2_分词算法对比.png)

*Figure 5-2: Comparison of BPE, WordPiece, and Unigram Tokenization Algorithms*

In actual engineering, SentencePiece is the most commonly used tokenization toolkit. It supports both BPE and Unigram algorithms, provides language-agnostic preprocessing (not dependent on space-based tokenization), and integrates seamlessly with mainstream deep learning frameworks.

```python
import sentencepiece as spm

# Train SentencePiece model
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='my_tokenizer',
    vocab_size=32000,
    model_type='bpe',  # or 'unigram'
    character_coverage=0_9995,
    num_threads=16
)

# Load and use
sp = spm.SentencePieceProcessor(model_file='my_tokenizer.model')
tokens = sp.encode('Hello, world!', out_type=str)
print(tokens)  # ['▁Hello', ',', '▁world', '!']
ids = sp.encode('Hello, world!')
print(ids)  # [1234, 56, 789, 10]
```

---

## 5.2 Vocabulary Design and Extension

The vocabulary is the core component of a tokenizer. Vocabulary size, coverage, and structure directly affect model performance and efficiency.

### 5.2.1 Vocabulary Size Trade-offs

Vocabulary size is one of the most important hyperparameters in tokenizer design. A larger vocabulary means more tokens retained as complete units, shorter sequences, but a larger embedding matrix and more parameters; a smaller vocabulary means more words split into subwords, longer sequences, but fewer model parameters.

Mainstream large model vocabulary sizes typically range from 32K to 128K. GPT-2 uses 50,257, LLaMA uses 32,000, GPT-4 is reported to use approximately 100,000. When selecting vocabulary size, consider:

**Computation efficiency**: The larger the vocabulary, the more parameters in the embedding and output layers. For a d-dimensional model with vocabulary size V, the embedding matrix contains V × d parameters. When V increases from 32K to 128K, this increases 4x.

**Sequence length**: The larger the vocabulary, the more characters each token covers on average, and the fewer tokens the same text is split into. This is especially important for long documents, as Transformer computation complexity is quadratic in sequence length.

**Rare word handling**: The larger the vocabulary, the more rare words can be retained as complete tokens, reducing UNK and over-segmentation issues. But this also means rare tokens see fewer training samples, potentially leading to poor embedding quality.

```python
# Analyze impact of different vocabulary sizes on sequence length
def analyze_vocab_size_impact(text: str, vocab_sizes: list) -> dict:
    """Analyze impact of vocabulary size on tokenization results"""
    import sentencepiece as spm
    
    results = {}
    for vocab_size in vocab_sizes:
        # Train tokenizers with different vocabulary sizes
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

### 5.2.2 Multilingual Vocabulary Design

When training multilingual models, vocabulary design faces additional challenges: how to balance coverage of different languages within limited vocabulary space?

A common problem is the "Vocabulary Curse." If a tokenizer is trained directly on multilingual corpus, high-resource languages (e.g., English) will occupy most vocabulary space, while low-resource languages have severely insufficient coverage. This causes low-resource language text to be over-segmented, sequence length to inflate, and model performance to degrade.

Common strategies to address this include:

**Corpus balancing**: Before training the tokenizer, oversample or undersample corpus from different languages to make weights more balanced across languages.

**Temperature sampling**: Similar to the multilingual data balancing strategy discussed in Chapter 3, use temperature parameter to control sampling probability of different languages.

**Language-specific character coverage**: Ensure basic character sets of each target language are included in the vocabulary even if their frequency is low. SentencePiece provides the `character_coverage` parameter to control this.

```python
# Multilingual tokenizer training example
import sentencepiece as spm

# Use character coverage to ensure multilingual support
spm.SentencePieceTrainer.train(
    input='multilingual_corpus.txt',
    model_prefix='multilingual_tokenizer',
    vocab_size=64000,
    model_type='unigram',
    character_coverage=0_9999,  # High coverage ensures rare characters included
    input_sentence_size=10000000,
    shuffle_input_sentence=True,
    # Special handling for CJK characters
    byte_fallback=True  # Fall back to byte level for unknown characters
)
```

### 5.2.3 Domain-Specific Vocabulary Extension

When applying pre-trained models to specific domains (e.g., medical, legal, code), one often encounters the problem of domain terminology being over-segmented. This not only increases sequence length but may also affect the model's understanding of professional concepts.

Vocabulary extension is an effective solution. The basic idea: add new domain-specific tokens while preserving the original vocabulary.

```python
from transformers import AutoTokenizer

def extend_tokenizer(base_tokenizer_name: str, 
                     domain_terms: list,
                     output_dir: str) -> None:
    """
    Extend pre-trained tokenizer vocabulary
    
    Args:
        base_tokenizer_name: Base tokenizer name
        domain_terms: List of domain-specific terms
        output_dir: Output directory
    """
    # Load base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
    
    print(f"Original vocabulary size: {len(tokenizer)}")
    
    # Filter already existing tokens
    new_tokens = []
    for term in domain_terms:
        if term not in tokenizer.get_vocab():
            new_tokens.append(term)
    
    # Add new tokens
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added} new tokens")
    print(f"New vocabulary size: {len(tokenizer)}")
    
    # Save extended tokenizer
    tokenizer.save_pretrained(output_dir)
    
    return tokenizer

# Example: Extend vocabulary for medical domain
medical_terms = [
    '冠状动脉',
    '心肌梗死',
    '动脉粥样硬化',
    'COVID-19',
    'mRNA疫苗',
    '计算机断层扫描',
    # ... more terms
]

tokenizer = extend_tokenizer(
    'meta-llama/Llama-2-7b',
    medical_terms,
    './medical_tokenizer'
)
```

After vocabulary extension, the model's embedding matrix needs to be extended accordingly. Embeddings for new tokens are typically initialized to random values or the average of existing related tokens, then learned through continued pre-training for meaningful representations.

```python
from transformers import AutoModelForCausalLM

def resize_model_embeddings(model_name: str, 
                            tokenizer,
                            output_dir: str) -> None:
    """Resize model embedding layer to match extended vocabulary"""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Resize embedding layer
    model.resize_token_embeddings(len(tokenizer))
    
    # Optional: Initialize new embeddings with mean of similar tokens
    # This usually gives better results than random initialization
    
    model.save_pretrained(output_dir)
```

### 5.2.4 LLaMA Chinese Vocabulary Extension: Engineering Practice

In practice, extending LLaMA-type models with Chinese vocabulary is a very common engineering task. Since LLaMA's original tokenizer was trained primarily on English data, Chinese characters are excessively segmented, leading to three major issues: sequences are 2-3x longer (higher compute cost), model struggles to learn complete Chinese semantics, and context window is effectively shortened.

The complete workflow for Chinese vocabulary extension is as follows:

**Step 1: Train Chinese SentencePiece model**

First, train a dedicated Chinese BPE model on a Chinese corpus.

```python
import sentencepiece as spm

# Train Chinese SentencePiece model
spm.SentencePieceTrainer.train(
    input='chinese_corpus.txt',  # Large-scale Chinese corpus
    model_prefix='chinese_sp',
    vocab_size=20000,            # Number of Chinese tokens to add
    model_type='bpe',
    character_coverage=0.9999,   # Ensure CJK character coverage
    byte_fallback=True,
    num_threads=32
)
```

**Step 2: Merge vocabularies**

Merge new Chinese tokens into LLaMA's original vocabulary, avoiding duplicates.

```python
from transformers import LlamaTokenizer
import sentencepiece as spm

def merge_tokenizers(
    base_tokenizer_path: str,
    chinese_sp_model_path: str,
    output_path: str
):
    """Merge LLaMA tokenizer with Chinese SentencePiece model"""
    # Load base tokenizer
    base_tokenizer = LlamaTokenizer.from_pretrained(base_tokenizer_path)
    base_vocab = set(base_tokenizer.get_vocab().keys())
    
    # Load Chinese model
    chinese_sp = spm.SentencePieceProcessor(model_file=chinese_sp_model_path)
    
    # Extract new tokens (not in original vocabulary)
    new_tokens = []
    for i in range(chinese_sp.get_piece_size()):
        piece = chinese_sp.id_to_piece(i)
        if piece not in base_vocab:
            new_tokens.append(piece)
    
    # Add new tokens
    num_added = base_tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added} Chinese tokens")
    print(f"Vocabulary size: {len(base_tokenizer.get_vocab().keys())} -> {len(base_tokenizer)}")
    
    # Save
    base_tokenizer.save_pretrained(output_path)
    return base_tokenizer
```

**Step 3: Resize model embedding matrix**

After vocabulary expansion, the model's embedding and output layers must be resized accordingly.

```python
from transformers import LlamaForCausalLM
import torch

def resize_model_for_new_vocab(
    model_path: str,
    new_tokenizer,
    output_path: str
):
    """Resize model to accommodate extended vocabulary"""
    model = LlamaForCausalLM.from_pretrained(model_path)
    
    original_vocab_size = model.config.vocab_size
    new_vocab_size = len(new_tokenizer)
    
    # Resize embedding layer
    model.resize_token_embeddings(new_vocab_size)
    
    # Initialize new embeddings: use mean of existing embeddings
    with torch.no_grad():
        # Input embedding
        embed_weight = model.model.embed_tokens.weight
        mean_embed = embed_weight[:original_vocab_size].mean(dim=0)
        embed_weight[original_vocab_size:] = mean_embed
        
        # Output layer (lm_head)
        lm_head_weight = model.lm_head.weight
        mean_head = lm_head_weight[:original_vocab_size].mean(dim=0)
        lm_head_weight[original_vocab_size:] = mean_head
    
    model.save_pretrained(output_path)
    print(f"Model resized: {original_vocab_size} -> {new_vocab_size}")
```

**Step 4: Verify the effect**

```python
def verify_chinese_extension(original_tokenizer_path: str, 
                              extended_tokenizer_path: str):
    """Verify the effect of Chinese vocabulary extension"""
    from transformers import AutoTokenizer
    
    original = AutoTokenizer.from_pretrained(original_tokenizer_path)
    extended = AutoTokenizer.from_pretrained(extended_tokenizer_path)
    
    test_texts = [
        "人工智能是计算机科学的一个重要分支",
        "大语言模型的训练数据质量决定了模型的上限",
        "深度学习在自然语言处理领域取得了重大突破"
    ]
    
    for text in test_texts:
        orig_tokens = original.tokenize(text)
        ext_tokens = extended.tokenize(text)
        
        print(f"\nText: {text}")
        print(f"  Original: {len(orig_tokens)} tokens -> {orig_tokens}")
        print(f"  Extended: {len(ext_tokens)} tokens -> {ext_tokens}")
        print(f"  Compression: {len(orig_tokens)/len(ext_tokens):.1f}x")
```

Typically, after Chinese vocabulary extension, Chinese text token count is reduced by 50-70%, meaning the same context window can process 2-3x more Chinese content.

### 5.2.5 Vocabulary Design Best Practices

Based on industry experience, here are some best practices for vocabulary design:

**Reserve sufficient special token positions**: Reserve some token IDs for future special tokens (e.g., new control symbols, domain markers). Many tokenizers reserve 100-1000 positions.

**Ensure reasonable segmentation of numbers and code symbols**: Numbers are important in many tasks, but standard tokenizers often handle them poorly. Consider keeping single digits as independent tokens or using special number encoding strategies.

**Test edge cases**: Before finalizing vocabulary, test various edge cases: very long words, special characters, mixed-language text, code snippets. Ensure tokenization results meet expectations.

**Document vocabulary decisions**: Record vocabulary size, training corpus, special token list, etc., to facilitate subsequent model iteration and troubleshooting.

---

## 5.3 Data Mixing and Curriculum Learning

After determining the tokenizer, the next key question is: how to organize and present training data? In what proportions should data from different sources and qualities be mixed? Does the order of data during training matter?

### 5.3.1 Data Mixing Strategies

As discussed in Chapter 3, high-quality pre-training datasets typically mix multiple sources: web, books, code, papers, dialogue, etc. Each source has different data volume and quality; simply mixing by original proportions is often not optimal.

**Static mixing** is the simplest strategy: determine mixing proportions for each source before training begins, shuffle data, then train sequentially. This method is simple to implement but lacks flexibility.

```python
# Static data mixing example
import random

def static_mix(data_sources: dict, target_size: int) -> list:
    """
    Statically mix multiple data sources
    
    Args:
        data_sources: {source_name: (data_list, weight)}
        target_size: Target dataset size
    
    Returns:
        Mixed data list
    """
    mixed_data = []
    
    # Compute sample count for each source
    total_weight = sum(w for _, w in data_sources.values())
    
    for source_name, (data, weight) in data_sources.items():
        num_samples = int(target_size * weight / total_weight)
        
        # If insufficient data, repeat sampling
        if len(data) < num_samples:
            sampled = random.choices(data, k=num_samples)
        else:
            sampled = random.sample(data, num_samples)
        
        mixed_data.extend(sampled)
    
    random.shuffle(mixed_data)
    return mixed_data

# Usage example
data_sources = {
    'web': (web_data, 0_6),
    'books': (book_data, 0_15),
    'code': (code_data, 0_1),
    'papers': (paper_data, 0_1),
    'wikipedia': (wiki_data, 0_05)
}

mixed = static_mix(data_sources, target_size=1000000)
```

![Figure 5-3: Data Mixing Strategies](../../images/part2/图5_3_数据混合策略.png)

*Figure 5-3: Static vs. Dynamic Mixing Strategy Comparison*

**Dynamic mixing** allows adjusting mixing proportions during training. Some research suggests optimal data ratios may differ across training stages. For example, training early with more diverse data helps the model establish broad language understanding; training later with increased high-quality data proportion improves fine-grained capabilities.

```python
class DynamicDataMixer:
    """Dynamic data mixer"""
    
    def __init__(self, data_sources: dict, schedule: list):
        """
        Initialize dynamic mixer
        
        Args:
            data_sources: Data source dictionary
            schedule: [(step_threshold, weights_dict), ...]
                     Use different mixing weights at different training steps
        """
        self.data_sources = data_sources
        self.schedule = sorted(schedule, key=lambda x: x[0])
        self.current_step = 0
    
    def get_weights(self) -> dict:
        """Get weights for current step"""
        for step_threshold, weights in reversed(self.schedule):
            if self.current_step >= step_threshold:
                return weights
        return self.schedule[0][1]
    
    def sample_batch(self, batch_size: int) -> list:
        """Sample one batch"""
        weights = self.get_weights()
        batch = []
        
        for source_name, weight in weights.items():
            num_samples = int(batch_size * weight)
            data = self.data_sources[source_name]
            batch.extend(random.choices(data, k=num_samples))
        
        random.shuffle(batch)
        self.current_step += 1
        return batch[:batch_size]

# Usage example: Emphasize diversity early, quality later
schedule = [
    (0, {'web': 0_5, 'books': 0_2, 'code': 0_15, 'papers': 0_1, 'wiki': 0_05}),
    (100000, {'web': 0_4, 'books': 0_25, 'code': 0_15, 'papers': 0_15, 'wiki': 0_05}),
    (500000, {'web': 0_3, 'books': 0_3, 'code': 0_2, 'papers': 0_15, 'wiki': 0_05}),
]

mixer = DynamicDataMixer(data_sources, schedule)
```

### 5.3.2 Curriculum Learning

Curriculum learning is a training strategy inspired by human learning. The core idea: have the model learn "easy" samples first, then gradually transition to "hard" samples. This strategy has been proven in multiple studies to accelerate convergence and improve final performance.

In pre-training scenarios, "easy" and "hard" can be defined in multiple ways:

**Length-based**: Short text is usually easier to learn than long text. The curriculum can start with short sequences and gradually increase length.

**Perplexity-based**: Text with low perplexity (text the language model is more "familiar" with) can be considered "easy" samples. A small pre-trained model can be used to evaluate sample difficulty, then samples ordered by difficulty for the main model.

**Noise-level-based**: High-quality, low-noise text first, then gradually introduce lower-quality but potentially unique-information text.

```python
import numpy as np

class CurriculumScheduler:
    """Curriculum learning scheduler"""
    
    def __init__(self, 
                 data: list, 
                 difficulty_scores: list,
                 total_steps: int,
                 strategy: str = 'linear'):
        """
        Initialize curriculum scheduler
        
        Args:
            data: Data list
            difficulty_scores: Difficulty score for each sample (higher = harder)
            total_steps: Total training steps
            strategy: Curriculum strategy ('linear', 'sqrt', 'exp')
        """
        self.data = np.array(data)
        self.difficulty_scores = np.array(difficulty_scores)
        self.total_steps = total_steps
        self.strategy = strategy
        
        # Sort by difficulty
        sorted_indices = np.argsort(self.difficulty_scores)
        self.sorted_data = self.data[sorted_indices]
        self.sorted_scores = self.difficulty_scores[sorted_indices]
    
    def get_curriculum_fraction(self, current_step: int) -> float:
        """
        Compute data fraction to use at current step
        
        Return value in [0, 1], indicating proportion of easiest data to use
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
        """Sample batch according to current progress"""
        fraction = self.get_curriculum_fraction(current_step)
        
        # Determine available data range
        available_size = max(int(len(self.sorted_data) * fraction), batch_size)
        available_data = self.sorted_data[:available_size]
        
        # Random sample from available range
        indices = np.random.choice(len(available_data), size=batch_size, replace=True)
        return available_data[indices].tolist()
```

![Figure 5-4: Curriculum Learning Illustration](../../images/part2/图5_4_课程学习示意图.png)

*Figure 5-4: Curriculum Learning Principle — Gradual transition from easy to hard samples*

### 5.3.3 Data Sampling and Batch Construction

In actual training, how data is organized affects both efficiency and effectiveness. Here are some important engineering considerations:

**Pack strategy**: To fully utilize compute resources, multiple short sequences are typically packed into a fixed-length sequence. This reduces computation waste from padding. The key question is how to handle attention masks after packing—different documents should not attend to each other.

```python
def pack_sequences(sequences: list, max_length: int, eos_token_id: int) -> list:
    """
    Pack multiple short sequences to fixed length
    
    Args:
        sequences: List of token id sequences
        max_length: Target sequence length
        eos_token_id: End-of-sequence token ID
    
    Returns:
        List of packed sequences, each of length max_length
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
            # Current pack full, start new one
            if current_pack:
                # Pad to max_length
                current_pack.extend([eos_token_id] * (max_length - current_length))
                packed.append(current_pack)
            
            current_pack = seq_with_eos
            current_length = len(seq_with_eos)
    
    # Handle last pack
    if current_pack:
        current_pack.extend([eos_token_id] * (max_length - current_length))
        packed.append(current_pack)
    
    return packed
```

**Document boundary handling**: When packing sequences, need to create a "document boundary mask" to ensure the model does not perform attention across document boundaries during generation.

**Data loading efficiency**: For TB-scale datasets, data loading itself may become a bottleneck. Common optimization methods include: storing preprocessed data in binary format (e.g., numpy memmap), multi-process parallel loading, prefetching the next batch.

### 5.3.4 Serialization and Storage Formats

After tokenization, token sequences need to be stored in efficient format for fast reading during training.

**Common storage formats** include:

**NumPy memmap**: Store token IDs as numpy array, access via memory mapping. Advantage is simple and direct, supports random access; disadvantage is no compression support, larger storage.

```python
import numpy as np

def save_as_memmap(token_ids: list, output_path: str):
    """Save token ID list as memmap format"""
    arr = np.array(token_ids, dtype=np.uint16)  # Assume vocabulary < 65536
    fp = np.memmap(output_path, dtype='uint16', mode='w+', shape=arr.shape)
    fp[:] = arr[:]
    fp.flush()
    
def load_memmap(path: str, shape: tuple):
    """Load memmap format token IDs"""
    return np.memmap(path, dtype='uint16', mode='r', shape=shape)
```

**Arrow/Parquet**: Use Apache Arrow format, supports compression and efficient columnar access. HuggingFace Datasets uses this format internally.

**Custom binary format**: Some large projects use custom binary formats optimized for specific access patterns. For example, the binary packing format used by GPT-NeoX.

```python
# Use HuggingFace Datasets for tokenized data
from datasets import Dataset

def tokenize_and_save(raw_data: list, tokenizer, output_dir: str):
    """Tokenize and save as Datasets format"""
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=2048,
            return_attention_mask=False
        )
    
    # Create Dataset
    ds = Dataset.from_dict({'text': raw_data})
    
    # Tokenize
    tokenized_ds = ds.map(
        tokenize_function,
        batched=True,
        num_proc=16,
        remove_columns=['text']
    )
    
    # Save
    tokenized_ds.save_to_disk(output_dir)
```

---

## 5.4 Complete Data Preparation Pipeline

Connect the steps discussed above to build a complete pipeline from raw text to training-ready data.

```python
from dataclasses import dataclass
from typing import Optional
import sentencepiece as spm

@dataclass
class DataPrepConfig:
    """Data preparation configuration"""
    # Tokenizer config
    tokenizer_path: str
    max_seq_length: int = 2048
    
    # Data mixing config
    mix_weights: dict = None  # {source: weight}
    
    # Curriculum learning config
    use_curriculum: bool = False
    curriculum_strategy: str = 'linear'
    
    # Output config
    pack_sequences: bool = True
    output_format: str = 'arrow'  # 'arrow', 'memmap', 'jsonl'

class DataPreparationPipeline:
    """Data preparation pipeline"""
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
        self.tokenizer = spm.SentencePieceProcessor(model_file=config.tokenizer_path)
    
    def tokenize_document(self, text: str) -> list:
        """Tokenize single document"""
        return self.tokenizer.encode(text)
    
    def process_source(self, source_path: str, source_name: str) -> list:
        """Process single data source"""
        documents = self.load_documents(source_path)
        
        tokenized = []
        for doc in documents:
            tokens = self.tokenize_document(doc['text'])
            if len(tokens) > 10:  # Filter too-short documents
                tokenized.append({
                    'input_ids': tokens,
                    'source': source_name,
                    'length': len(tokens)
                })
        
        return tokenized
    
    def mix_sources(self, sources: dict) -> list:
        """Mix multiple data sources"""
        mixed = []
        weights = self.config.mix_weights or {s: 1_0 for s in sources}
        total_weight = sum(weights.values())
        
        # Determine sample count per source
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
        """Pack and save data"""
        if self.config.pack_sequences:
            sequences = [d['input_ids'] for d in data]
            packed = pack_sequences(
                sequences, 
                self.config.max_seq_length,
                self.tokenizer.eos_id()
            )
        else:
            packed = [d['input_ids'] for d in data]
        
        # Select output format based on config
        if self.config.output_format == 'arrow':
            self.save_as_arrow(packed, output_path)
        elif self.config.output_format == 'memmap':
            self.save_as_memmap(packed, output_path)
        else:
            self.save_as_jsonl(packed, output_path)
    
    def run(self, source_paths: dict, output_path: str):
        """Run complete pipeline"""
        # 1. Process each data source
        sources = {}
        for source_name, path in source_paths.items():
            print(f"Processing {source_name}...")
            sources[source_name] = self.process_source(path, source_name)
        
        # 2. Mix data
        print("Mixing data sources...")
        mixed = self.mix_sources(sources)
        
        # 3. Optional: Apply curriculum ordering
        if self.config.use_curriculum:
            print("Applying curriculum ordering...")
            mixed = self.apply_curriculum(mixed)
        
        # 4. Pack and save
        print("Packing and saving...")
        self.pack_and_save(mixed, output_path)
        
        print(f"Done! Saved {len(mixed)} samples to {output_path}")
```

![Figure 5-5: Complete Data Preparation Pipeline](../../images/part2/图5_5_数据准备完整流水线.png)

*Figure 5-5: Complete Pipeline from Raw Text to Training-Ready Data*

---

## 5.5 Chapter Summary

This chapter systematically introduced the core technologies of tokenization and data serialization.

In tokenizer principles: subword tokenization is the mainstream choice for current large models, achieving good balance between vocabulary size and sequence length. BPE uses frequency-driven bottom-up merging strategy, simple and efficient; WordPiece uses likelihood-driven merging criterion, more sensitive to low-frequency meaningful patterns; Unigram uses top-down probability modeling, theoretically more elegant. SentencePiece is the most commonly used toolkit, supporting multiple algorithms and language-agnostic processing.

In vocabulary design: vocabulary size requires trade-offs between computation efficiency, sequence length, and rare word handling; mainstream models typically use 32K-128K. Multilingual vocabulary design needs to balance coverage across languages and avoid the "vocabulary curse." Domain-specific vocabulary extension can improve professional terminology handling but requires extending the model embedding layer accordingly.

In data mixing: static mixing is simple and direct; dynamic mixing allows adjusting proportions during training. Curriculum learning strategy starts with easy samples and gradually transitions to hard ones, which can accelerate convergence and improve performance. Data packing and efficient storage formats are crucial for large-scale training.

![Figure 5-6: Chapter Knowledge Structure](../../images/part2/图5_6_本章知识结构.png)

*Figure 5-6: Chapter 5 Knowledge Structure — Three themes: Tokenization Algorithms, Vocabulary Design, Data Organization*

---

## Further Reading

For in-depth content on tokenization and data serialization, the following resources are worth referencing:

The SentencePiece paper (Kudo and Richardson, 2018) introduces language-agnostic subword tokenization. The BPE paper (Sennrich et al., 2015) is the pioneering work introducing BPE to NLP. The Unigram paper (Kudo, 2018) provides a probabilistic perspective on subword tokenization. HuggingFace Tokenizers library documentation (huggingface.co/docs/tokenizers) is the authoritative practical reference. For curriculum learning, Bengio et al.'s survey paper provides a comprehensive theoretical framework.

---

## Next Chapter Preview

With this, we have completed all content on text pre-training data engineering. In the next chapter "Image-Text Pairs Processing," we will enter the field of multimodal data engineering. You will learn how to process LAION-5B style image-text paired data, how to use img2dataset for high-concurrency image downloads, and how to build multimodal data cleaning pipelines.

Consider this question as you enter the next chapter: How should the "quality" of an image be defined? Besides resolution and clarity, what other dimensions need to be considered?
