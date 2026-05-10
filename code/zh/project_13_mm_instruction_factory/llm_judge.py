def score_with_llm_judge(generated_data):
    scored_data = []
    for item in generated_data:
        word_count = len(item["response"].split())
        score = 4.5 if word_count > 50 else 3.0
        if score >= 4.0:
            item["judge_score"] = score
            scored_data.append(item)
    print(f"Filtered {len(generated_data)} down to {len(scored_data)} high-quality samples.")
    return scored_data
