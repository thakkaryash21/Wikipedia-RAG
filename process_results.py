import os
import json
import csv

INPUT_DIR = "./results"
OUTPUT_FILE = "combined_results.csv"

CSV_COLUMNS = [
    "Embedding Model",
    "Prompt",
    "Top k",
    "F1 Score",
    "Exact Match",
    "Faithfulness",
    "Context Recall",
    "Context Precision",
    "Answer Relevancy",
]


def main():
    rows = []

    # Loop through all JSON files in the directory
    for file_name in os.listdir(INPUT_DIR):
        if file_name.startswith("exp_"):
            file_path = os.path.join(INPUT_DIR, file_name)

            # Read the JSON file
            with open(file_path, "r") as f:
                data = json.load(f)

            # Extract values
            config = data.get("config", {})
            metrics = data.get("metrics", {})

            row = {
                "Embedding Model": config.get("embedding_model"),
                "Prompt": config.get("prompt"),
                "Top k": config.get("top_k"),
                "F1 Score": metrics.get("f1_score"),
                "Exact Match": metrics.get("exact_match"),
                "Faithfulness": metrics.get("faithfulness"),
                "Context Recall": metrics.get("context_recall"),
                "Context Precision": metrics.get("context_precision"),
                "Answer Relevancy": metrics.get("answer_relevancy"),
            }
            rows.append(row)

    # Write to CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Combined CSV created: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
