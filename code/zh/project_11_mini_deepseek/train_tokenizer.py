from datasets import load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers

def train_large_tokenizer(dataset_path, vocab_size=150000):
    print("Loading dataset for tokenizer training...")
    ds = load_from_disk(dataset_path)
    train_ds = ds.select(range(0, len(ds), 10))
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Replace(" ", " "), 
        normalizers.NFKC()
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    def batch_iterator(batch_size=1000):
        for i in range(0, len(train_ds), batch_size):
            yield train_ds[i : i + batch_size]["text"]
            
    print(f"Training tokenizer with {vocab_size} vocab size...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    tokenizer.save("./data/mini_deepseek_tokenizer.json")
    print("Tokenizer saved.")

if __name__ == "__main__":
    train_large_tokenizer("./data/mixed_1b_dedup")
