# === save.py ===
import pandas as pd
import json

def save_results(results_dict, filename="results.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)

def save_predictions(prompts, best_pairs, filename="predictions.csv"):
    df = pd.DataFrame({
        "prompt": prompts,
        "generated_answer": [b for b, _ in best_pairs],
        "reference_answer": [r for _, r in best_pairs]
    })
    df.to_csv(filename, index=False)