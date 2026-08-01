"""
engine.py
Luxury Bag Review Intelligence Platform
Core AI Engine
"""

import pandas as pd
import plotly.express as px

from transformers import pipeline
from keybert import KeyBERT


print("Loading AI models...")

# -----------------------------
# Sentiment Model
# -----------------------------
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# -----------------------------
# Keyword Model
# -----------------------------
kw_model = KeyBERT()

print("Models loaded successfully!")


# -------------------------------------------------
# Split large text into manageable chunks
# -------------------------------------------------

def split_text(text, max_length=450):

    sentences = text.split(".")

    chunks = []

    current = ""

    for sentence in sentences:

        if len(current) + len(sentence) < max_length:

            current += sentence + "."

        else:

            chunks.append(current)

            current = sentence + "."

    if current:

        chunks.append(current)

    return chunks


# -------------------------------------------------
# Analyze sentiment
# -------------------------------------------------

def analyze_text(text):

    chunks = split_text(text)

    results = []

    for chunk in chunks:

        if len(chunk.strip()) < 20:

            continue

        result = sentiment_pipeline(chunk)[0]

        results.append(result)

    return results


# -------------------------------------------------
# Overall Sentiment
# -------------------------------------------------

def overall_sentiment(results):

    if len(results) == 0:

        return "UNKNOWN", 0

    labels = [r["label"] for r in results]

    scores = [r["score"] for r in results]

    sentiment = max(set(labels), key=labels.count)

    confidence = round(sum(scores) / len(scores), 3)

    return sentiment, confidence


# -------------------------------------------------
# Keyword Extraction
# -------------------------------------------------

def extract_keywords(text):

    keywords = kw_model.extract_keywords(

        text,

        keyphrase_ngram_range=(1,2),

        stop_words="english",

        top_n=10

    )

    return [k[0] for k in keywords]


# -------------------------------------------------
# Executive Summary
# -------------------------------------------------

def executive_summary(label, confidence, keywords):

    summary = f"""
Overall Sentiment

{label}

Confidence

{confidence}

Key Discussion Topics

{", ".join(keywords[:5])}

Business Insight

Customers frequently discuss the topics above.
The overall perception appears to be {label.lower()}.
"""

    return summary


# -------------------------------------------------
# Pie Chart
# -------------------------------------------------

def sentiment_chart(results):

    labels = [r["label"] for r in results]

    df = pd.DataFrame(labels, columns=["Sentiment"])

    fig = px.pie(

        df,

        names="Sentiment",

        title="Sentiment Distribution"

    )

    return fig


# -------------------------------------------------
# Main Engine
# -------------------------------------------------

def analyze_review(review_text):

    results = analyze_text(review_text)

    label, confidence = overall_sentiment(results)

    keywords = extract_keywords(review_text)

    summary = executive_summary(

        label,

        confidence,

        keywords

    )

    chart = sentiment_chart(results)

    return summary, chart