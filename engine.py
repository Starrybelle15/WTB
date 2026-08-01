"""
engine.py
Luxury Bag Review Intelligence Platform
Core AI Engine
"""

import pandas as pd
import plotly.express as px

from transformers import pipeline
from keybert import KeyBERT

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import requests
import trafilatura

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

    if len(keywords) >= 5:
        top_keywords = ", ".join(keywords[:5])
    else:
        top_keywords = ", ".join(keywords)

    return f"""
======================================

AI BUSINESS INSIGHT REPORT

======================================

Overall Customer Sentiment

{label.upper()}

Confidence Score

{confidence:.2f}

Top Customer Discussion Topics

{top_keywords}

Business Insight

Customers generally express a {label.lower()} opinion regarding this luxury handbag.

The most frequently discussed topics indicate what customers value most and what they pay the most attention to during purchase and ownership.

Recommendation

Luxury brands should continue monitoring customer feedback to identify recurring strengths and potential improvement areas.

======================================
"""
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
# Download Review Article
# -------------------------------------------------

def extract_from_url(url):
    """
    Download and extract readable text from a webpage.
    """

    try:

        downloaded = trafilatura.fetch_url(url)

        if downloaded is None:
            return None

        text = trafilatura.extract(downloaded)

        if text is None:
            return None

        return text

    except Exception:

        return None

# -------------------------------------------------
# Main Engine
# -------------------------------------------------

def analyze_review(review_text="", review_url=""):

    # If URL is provided,
    # download article automatically

    if review_url.strip():

        extracted = extract_from_url(review_url)

        if extracted:

            review_text = extracted

    if len(review_text.strip()) == 0:

        return "No review text found.", None

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

    def create_wordcloud(text):

    wc = WordCloud(
        width=900,
        height=500,
        background_color="white"
    ).generate(text)

    fig = plt.figure(figsize=(10,5))

    plt.imshow(wc)

    plt.axis("off")

    return fig