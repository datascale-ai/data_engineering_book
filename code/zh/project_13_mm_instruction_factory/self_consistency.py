def self_consistency_filter(generated_results):
    # Simulated self-consistency
    filtered = []
    for res in generated_results:
        # In reality, multiple samples would be generated and consensus checked.
        res['consistency_score'] = 1.0
        filtered.append(res)
    return filtered
