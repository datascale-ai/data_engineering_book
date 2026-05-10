from datasets import load_dataset
import random

def select_seeds(dataset_name="laion/laion2B-en", num_samples=5000):
    print("Loading LAION metadata...")
    ds = load_dataset(dataset_name, split="train", streaming=True)
    seeds = []
    for item in ds:
        try:
            w, h = item.get("WIDTH", 0), item.get("HEIGHT", 0)
            if w > 512 and h > 512 and 0.5 < (w/h) < 2.0:
                if len(str(item.get("TEXT", "")).split()) > 10:
                    seeds.append({
                        "url": item["URL"],
                        "original_caption": item["TEXT"]
                    })
        except:
            continue
        if len(seeds) >= num_samples:
            break
    print(f"Selected {len(seeds)} high-quality seed images.")
    return seeds

if __name__ == "__main__":
    select_seeds(num_samples=100)
