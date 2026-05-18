import hashlib
from datasketch import MinHash, MinHashLSH
from datasets import load_from_disk

def get_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    tokens = [text[i:i+5] for i in range(max(1, len(text)-4))]
    for token in tokens:
        m.update(token.encode('utf8'))
    return m

def cross_source_dedup(dataset_path, threshold=0.8, num_perm=128):
    print("Loading dataset for global deduplication...")
    ds = load_from_disk(dataset_path)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique_indices = set()
    duplicates = 0
    with lsh.insertion_session() as session:
        for idx, item in enumerate(ds):
            m = get_minhash(item['text'], num_perm)
            result = lsh.query(m)
            if not result:
                session.insert(str(idx), m)
                unique_indices.add(idx)
            else:
                duplicates += 1
    print(f"Deduplication complete. Found {duplicates} duplicates.")
    return ds.select(list(unique_indices))

if __name__ == "__main__":
    ds_unique = cross_source_dedup("./data/mixed_1b_raw")
    ds_unique.save_to_disk("./data/mixed_1b_dedup")
