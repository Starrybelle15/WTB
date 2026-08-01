# ==========================================
# Imports
# ==========================================
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.express as px
import gradio as gr

from transformers import pipeline
from duckduckgo_search import DDGS
import trafilatura

from keybert import KeyBERT

# ==========================================
# AI Models
# ==========================================

print("Loading AI models...")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

kw_model = KeyBERT()

print("AI models loaded successfully.")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)
# ==========================================
# Search Functions
# ==========================================
def search_reviews(query, max_results=10):
    """
    Search for DuckDuckGo for public review pages related to a luxury bag.
    """
    results = []

    with DDGS() as ddgs:
      for item in ddgs.text(
            f"{query} review",
            max_results=max_results
        ):

            results.append(item)

    return results
  
        return list(
            ddgs.text(
                f"{query} review",
                max_results=max_results
            )
        )
# ==========================================
# Text Extraction
# ==========================================
def extract_text(url):
    """
    Download and extract readable text from a webpage.
    """
     try:

        downloaded = trafilatura.fetch_url(url)

        if downloaded is None:
            return ""

        text = trafilatura.extract(downloaded)

        if text is None:
            return ""

        return text

    except Exception:

        return ""
      
# ==========================================
# Sentiment Analysis
# ==========================================
def analyze_sentiment(text):

    if len(text.strip()) == 0:

        return None

    result = sentiment_pipeline(text[:512])[0]

    return result["label"], float(result["score"])
  
# ==========================================
# Keyword Extraction
# ==========================================
kw_model = KeyBERT()

def extract_keywords(text):

    if len(text.strip()) < 50:

        return []

    keywords = kw_model.extract_keywords(

        text,

        keyphrase_ngram_range=(1,2),

        stop_words="english",

        top_n=10

    )

    return [k[0] for k in keywords]
# ==========================================
# Summary Generation
# ==========================================
def build_summary(df):

    total = len(df)

    positive = len(df[df["label"]=="positive"])

    neutral = len(df[df["label"]=="neutral"])

    negative = len(df[df["label"]=="negative"])

    summary = f"""
Reviews analyzed : {total}

Positive : {positive}

Neutral : {neutral}

Negative : {negative}

Overall customer opinion is mostly
{df['label'].mode()[0].upper()}.
"""

    return summary
# ==========================================
# Charts
# ==========================================
def build_chart(df):

    counts = df["label"].value_counts().reset_index()

    counts.columns = ["Sentiment","Count"]

    fig = px.pie(

        counts,

        values="Count",

        names="Sentiment",

        title="Sentiment Distribution"

    )

    return fig
# ==========================================
# Main Analysis Function
# ==========================================
def analyze_bag(bag_name):

    search_results = search_reviews(bag_name)

    texts = []

    sentiments = []

    for item in search_results:

        url = item.get("href") or item.get("url")

        if not url:

            continue

        text = extract_text(url)

        if len(text) < 300:

            continue

        sentiment = analyze_sentiment(text)

        if sentiment is None:

            continue

        texts.append(text)

        sentiments.append(sentiment)

    if len(sentiments) == 0:

        return (

            "No review text found.",

            None,

            ""

        )

    df = pd.DataFrame(sentiments)

    chart = build_chart(df)

    keywords = extract_keywords(" ".join(texts))

    summary = build_summary(df)

    output = f"""

Reviews Found

{len(df)}

Top Keywords

{', '.join(keywords)}

{summary}

"""

    return output, chart, df

# ==========================================
# Gradio Interface
# ==========================================
