from tokenizers import Tokenizer
from datasets import load_from_disk

SEQ_LEN = 4096

def pack_and_shuffle(dataset_path, tokenizer_path):
    print("Loading tokenizer and deduped dataset...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    ds = load_from_disk(dataset_path)
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    
    def tokenize_and_pack(examples):
        encoded = [tokenizer.encode(t).ids for t in examples['text']]
        all_tokens = []
        for ids in encoded:
            all_tokens.extend(ids)
            all_tokens.append(eot_id)
        total_length = len(all_tokens)
        total_length = (total_length // SEQ_LEN) * SEQ_LEN
        result = [all_tokens[i : i + SEQ_LEN] for i in range(0, total_length, SEQ_LEN)]
        return {"input_ids": result}

    print("Tokenizing and packing into uniform lengths...")
    packed_ds = ds.map(
        tokenize_and_pack,
        batched=True,
        batch_size=1000,
        remove_columns=ds.column_names,
        num_proc=8
    )
    print("Shuffling dataset globally...")
    packed_ds = packed_ds.shuffle(seed=42)
    print("Saving to .arrow shards...")
    packed_ds.save_to_disk("./data/mixed_1b_final_packed")

if __name__ == "__main__":
    pack_and_shuffle("./data/mixed_1b_dedup", "./data/mini_deepseek_tokenizer.json")
