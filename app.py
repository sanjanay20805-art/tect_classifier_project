
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# File used to store predictions
DATA_FILE = Path(__file__).parent / "classification_results.csv"

st.set_page_config(
    page_title="Text Classification Comparison",
    page_icon="📊",
    layout="wide"
)

st.title("Naive Bayes vs LLM Text Classifier")
st.write("Enter text and compare the predictions from both models.")

# Training dataset
texts = [
    "I love this product",
    "excellent quality and great service",
    "I am very happy with this purchase",
    "the food was delicious",
    "the experience was fantastic",
    "I would recommend this",
    "the product works perfectly",
    "great value for money",
    "I hate this product",
    "terrible quality and awful service",
    "I am disappointed with this purchase",
    "the food was disgusting",
    "the experience was horrible",
    "I would never recommend this",
    "the product stopped working",
    "a complete waste of money"
]

labels = [
    "Positive", "Positive", "Positive", "Positive",
    "Positive", "Positive", "Positive", "Positive",
    "Negative", "Negative", "Negative", "Negative",
    "Negative", "Negative", "Negative", "Negative"
]

# Train Naive Bayes model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("nb", MultinomialNB())
])
model.fit(texts, labels)


def classify_llm(text):
    """Use Claude API when an API key is configured."""
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        return "Unavailable (API key not set)"

    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=20,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Classify the sentiment of this text. "
                        "Reply with exactly one word: Positive or Negative.\n"
                        f"Text: {text}"
                    )
                }
            ]
        )

        answer = response.content[0].text.strip().lower()

        if "positive" in answer:
            return "Positive"
        if "negative" in answer:
            return "Negative"

        return "Unknown"

    except Exception as error:
        return f"API error: {error}"


# Input area
user_text = st.text_area(
    "Enter your text",
    placeholder="Example: The product is excellent and I love it."
)

if st.button("Classify Text", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        nb_result = model.predict([user_text])[0]
        llm_result = classify_llm(user_text)

        st.subheader("Classification Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Naive Bayes Prediction", nb_result)

        with col2:
            st.metric("LLM Prediction", llm_result)

        # Save each prediction
        row = pd.DataFrame([{
            "input_text": user_text,
            "naive_bayes": nb_result,
            "llm": llm_result
        }])

        if DATA_FILE.exists():
            old_data = pd.read_csv(DATA_FILE)
            row = pd.concat([old_data, row], ignore_index=True)

        row.to_csv(DATA_FILE, index=False)

        st.success("Input text and predictions saved successfully!")

# Display stored data
st.subheader("Stored Classification Data")

if DATA_FILE.exists():
    data = pd.read_csv(DATA_FILE)
    st.dataframe(data, use_container_width=True)

    st.download_button(
        "Download Results CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="classification_results.csv",
        mime="text/csv"
    )

    # Prepare graph from stored predictions
    valid_labels = ["Positive", "Negative"]
    nb_counts = [
        int((data["naive_bayes"] == label).sum())
        for label in valid_labels
    ]
    llm_counts = [
        int((data["llm"] == label).sum())
        for label in valid_labels
    ]

    chart_data = pd.DataFrame({
        "Sentiment": valid_labels,
        "Naive Bayes": nb_counts,
        "LLM": llm_counts
    }).set_index("Sentiment")

    st.subheader("Model Prediction Comparison")
    st.bar_chart(chart_data)

else:
    st.info("Your stored results will appear here after the first classification.")
