import os
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Files are saved in the same folder as this script
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "classification_results.csv"
GRAPH_FILE = PROJECT_DIR / "sentiment_comparison.png"

# Small teaching dataset. For a real project, use a larger labeled dataset.
training_texts = [
    "I love this product", "excellent quality and great service",
    "I am very happy with this purchase", "the food was delicious",
    "the experience was fantastic", "I would recommend this",
    "the product works perfectly", "great value for money",
    "I hate this product", "terrible quality and awful service",
    "I am disappointed with this purchase", "the food was disgusting",
    "the experience was horrible", "I would never recommend this",
    "the product stopped working", "a complete waste of money"
]
training_labels = [
    "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive",
    "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative"
]

nb_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])
nb_model.fit(training_texts, training_labels)


def classify_with_llm(text):
    """Uses Claude if ANTHROPIC_API_KEY is set; otherwise returns None."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None, "No API key set"

    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=20,
            messages=[{
                "role": "user",
                "content": (
                    "Classify the sentiment of this text. Reply with exactly "
                    "one word: positive or negative.\nText: " + text
                )
            }]
        )
        answer = response.content[0].text.strip().lower()
        if "positive" in answer:
            return "positive", "OK"
        if "negative" in answer:
            return "negative", "OK"
        return "unknown", "LLM returned an unexpected label"
    except Exception as exc:
        return None, f"LLM error: {exc}"


def save_result(row):
    file_exists = DATA_FILE.exists()
    with DATA_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def make_graph():
    if not DATA_FILE.exists():
        print("No saved records yet, so there is no graph to draw.")
        return

    df = pd.read_csv(DATA_FILE)
    if df.empty:
        print("No saved records yet, so there is no graph to draw.")
        return

    # Count positive and negative predictions for each model.
    labels = ["positive", "negative", "unknown/unavailable"]
    nb_values = [
        int((df["naive_bayes"].fillna("").str.lower() == "positive").sum()),
        int((df["naive_bayes"].fillna("").str.lower() == "negative").sum()),
        int(df["naive_bayes"].fillna("").str.lower().isin(["", "unknown", "unavailable"]).sum())
    ]

    if "llm" in df.columns:
        llm_series = df["llm"].fillna("").str.lower()
    else:
        llm_series = pd.Series([""] * len(df))
    llm_values = [
        int((llm_series == "positive").sum()),
        int((llm_series == "negative").sum()),
        int(llm_series.isin(["", "unknown", "unavailable"]).sum())
    ]

    x = range(len(labels))
    width = 0.36
    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], nb_values, width, label="Naive Bayes")
    plt.bar([i + width / 2 for i in x], llm_values, width, label="LLM (Claude)")
    plt.xticks(list(x), labels)
    plt.ylabel("Number of saved texts")
    plt.title("Naive Bayes vs LLM: Prediction Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_FILE, dpi=150)
    plt.close()
    print(f"Graph updated: {GRAPH_FILE}")


def main():
    print("\nTEXT CLASSIFICATION: NAIVE BAYES vs LLM")
    print("Type a sentence, or type 'quit' to stop.")
    while True:
        user_text = input("\nEnter text: ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break
        if not user_text:
            print("Please enter some text.")
            continue

        nb_prediction = nb_model.predict([user_text])[0]
        llm_prediction, llm_status = classify_with_llm(user_text)

        print("\n--- CLASSIFICATION OUTPUT ---")
        print("Input text:", user_text)
        print("Naive Bayes prediction:", nb_prediction)
        print("LLM prediction:", llm_prediction if llm_prediction else "Unavailable")
        if llm_status != "OK":
            print("LLM note:", llm_status)

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "input_text": user_text,
            "naive_bayes": nb_prediction,
            "llm": llm_prediction if llm_prediction else "unavailable"
        }
        save_result(row)
        print(f"Saved this result to: {DATA_FILE}")
        make_graph()

    print("\nFinished. Your input history is stored in classification_results.csv")
    print("The latest graph is stored in sentiment_comparison.png")


if __name__ == "__main__":
    main()
