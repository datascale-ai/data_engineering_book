import datasets
from datasets import load_dataset, concatenate_datasets

RECIPE = {
    "HuggingFaceFW/fineweb-edu": {"split": "train", "weight": 0.40},
    "bigcode/the-stack-v2": {"split": "train", "weight": 0.25},
    "open-web-math/open-web-math": {"split": "train", "weight": 0.15},
    "togethercomputer/RedPajama-Data-1T": {"split": "train", "weight": 0.10, "name": "arxiv"},
    "m-a-p/WanJuan-1.0-Text": {"split": "train", "weight": 0.10}
}

def sample_multi_source(recipe, target_docs):
    sampled_datasets = []
    for repo_id, config in recipe.items():
        weight = config["weight"]
        num_docs = int(target_docs * weight)
        print(f"Sampling {num_docs} docs from {repo_id}...")
        ds = load_dataset(repo_id, config.get("name", "default"), split=config["split"], streaming=True)
        ds_iter = iter(ds)
        docs = []
        for _ in range(num_docs):
            try:
                item = next(ds_iter)
                text_content = item.get('text') or item.get('content')
                if text_content:
                    docs.append({"text": text_content, "source": repo_id})
            except StopIteration:
                break
        sampled_datasets.append(datasets.Dataset.from_list(docs))
    return concatenate_datasets(sampled_datasets)

if __name__ == "__main__":
    mixed_data = sample_multi_source(RECIPE, 500000)
    mixed_data.save_to_disk("./data/mixed_1b_raw")
    print("Multi-source sampling complete.")
