"""
=========================================================
Luxury Bag Review Intelligence Platform
Core AI Engine

Version 2.0

Author:
Final Year Project

=========================================================
"""

# =====================================================
# Imports
# =====================================================

import re
import requests
import trafilatura

import pandas as pd
import plotly.express as px

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from transformers import pipeline

from keybert import KeyBERT



# =====================================================
# Load AI Models
# =====================================================

print("Loading AI Models...")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

kw_model = KeyBERT()

print("Models Loaded Successfully!")



# =====================================================
# Aspect Categories
# =====================================================

ASPECTS = {

    "Leather Quality":[
        "leather",
        "material",
        "soft",
        "quality",
        "craftsmanship",
        "stitching",
        "canvas"
    ],

    "Design":[
        "design",
        "beautiful",
        "style",
        "appearance",
        "look",
        "fashion",
        "classic",
        "luxury"
    ],

    "Price":[
        "price",
        "expensive",
        "budget",
        "cost",
        "afford",
        "worth",
        "value"
    ],

    "Comfort":[
        "comfort",
        "comfortable",
        "heavy",
        "strap",
        "carry",
        "weight"
    ],

    "Durability":[
        "durable",
        "scratch",
        "hardware",
        "lasting",
        "zip",
        "zipper",
        "damage"
    ],

    "Storage":[
        "space",
        "spacious",
        "storage",
        "pocket",
        "inside",
        "capacity"
    ]

}



# =====================================================
# Download Review
# =====================================================

def extract_from_url(url):

    """
    Download readable review text
    from any article page.
    """

    try:

        downloaded = trafilatura.fetch_url(url)

        if downloaded is None:
            return None

        text = trafilatura.extract(downloaded)

        return text

    except Exception:

        return None



# =====================================================
# Clean Text
# =====================================================

def clean_text(text):

    text = re.sub(r"\s+"," ",text)

    text = text.replace("\n"," ")

    return text.strip()



# =====================================================
# Split Text
# =====================================================

def split_text(text,max_length=450):

    sentences=text.split(".")

    chunks=[]

    current=""

    for sentence in sentences:

        if len(current)+len(sentence)<max_length:

            current+=sentence+"."

        else:

            chunks.append(current)

            current=sentence+"."

    if current:

        chunks.append(current)

    return chunks

    # =====================================================
# Sentiment Analysis
# =====================================================

def analyze_text(text):
    """
    Analyze the sentiment of long text by splitting it into chunks.
    Returns a list of sentiment predictions.
    """

    text = clean_text(text)

    chunks = split_text(text)

    results = []

    for chunk in chunks:

        if len(chunk.strip()) < 20:
            continue

        try:
            prediction = sentiment_pipeline(chunk)[0]
            results.append(prediction)

        except Exception:
            continue

    return results


# =====================================================
# Overall Sentiment
# =====================================================

def overall_sentiment(results):

    if not results:
        return "UNKNOWN", 0.0

    labels = [r["label"] for r in results]
    scores = [r["score"] for r in results]

    sentiment = max(set(labels), key=labels.count)

    confidence = round(sum(scores) / len(scores), 3)

    return sentiment, confidence


# =====================================================
# Luxury Score
# =====================================================

def luxury_score(label, confidence):
    """
    Generates a simple 0–100 Luxury Score.
    """

    score = confidence * 100

    if label.lower() == "positive":
        score += 5

    elif label.lower() == "negative":
        score -= 20

    score = max(0, min(100, round(score)))

    stars = "⭐" * (score // 20)

    return score, stars


# =====================================================
# Keyword Extraction
# =====================================================

def extract_keywords(text):

    text = clean_text(text)

    if len(text) < 50:
        return []

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=10
    )

    return [item[0] for item in keywords]


# =====================================================
# Aspect Detection
# =====================================================

def detect_aspects(text):

    text = text.lower()

    detected = []

    for aspect, words in ASPECTS.items():

        score = 0

        for word in words:

            if word in text:
                score += 1

        if score > 0:
            detected.append((aspect, score))

    detected.sort(key=lambda x: x[1], reverse=True)

    return detected


# =====================================================
# Review Statistics
# =====================================================

def review_statistics(text):

    words = len(text.split())

    characters = len(text)

    sentences = len(
        [s for s in text.split(".") if s.strip()]
    )

    reading_time = round(words / 200, 1)

    return {
        "words": words,
        "characters": characters,
        "sentences": sentences,
        "reading_time": reading_time
    }

# =====================================================
# AI BUSINESS INSIGHT REPORT
# =====================================================

def executive_summary(
    bag_name,
    label,
    confidence,
    keywords,
    aspects,
    stats
):

    confidence_pct = round(confidence * 100, 1)

    score, stars = luxury_score(label, confidence)

    if len(keywords) == 0:
        keywords = ["No keywords detected"]

    positive_topics = ", ".join(keywords[:3])

    negative_topics = ", ".join(keywords[3:6]) if len(keywords) > 3 else "None"

    aspect_report = ""

    if aspects:

        for aspect, value in aspects:

            aspect_report += f"• {aspect} ({value} mentions)\n"

    else:

        aspect_report = "No major discussion themes detected."

    report = f"""
========================================================

AI BUSINESS INSIGHT REPORT

========================================================

Bag Analysed

{bag_name}

--------------------------------------------------------

Overall Customer Sentiment

{label.upper()}

Confidence Score

{confidence_pct:.1f}%

Luxury Score

{score}/100 {stars}

--------------------------------------------------------

Most Discussed Topics

{positive_topics}

--------------------------------------------------------

Additional Discussion

{negative_topics}

--------------------------------------------------------

Aspect Analysis

{aspect_report}

--------------------------------------------------------

Review Statistics

Words: {stats['words']}

Characters: {stats['characters']}

Sentences: {stats['sentences']}

Estimated Reading Time:
{stats['reading_time']} minutes

--------------------------------------------------------

Business Recommendation

The analysed customer feedback indicates an overall
{label.lower()} perception of the {bag_name}.

Customers primarily discuss the themes shown above.

Luxury brands should continuously monitor online
reviews to understand changing customer preferences
and identify recurring strengths or weaknesses.

========================================================
"""

    return report


# =====================================================
# SENTIMENT PIE CHART
# =====================================================

def sentiment_chart(results):

    if len(results) == 0:

        return None

    labels = [r["label"] for r in results]

    df = pd.DataFrame(labels, columns=["Sentiment"])

    fig = px.pie(

        df,

        names="Sentiment",

        title="Customer Sentiment Distribution",

        hole=0.45

    )

    fig.update_layout(

        title_x=0.5,

        font=dict(size=16)

    )

    return fig


# =====================================================
# BAR CHART
# =====================================================

def sentiment_bar_chart(results):

    if len(results) == 0:

        return None

    labels = [r["label"] for r in results]

    df = pd.DataFrame(labels, columns=["Sentiment"])

    counts = df.value_counts().reset_index(name="Count")

    fig = px.bar(

        counts,

        x="Sentiment",

        y="Count",

        title="Sentiment Frequency"

    )

    fig.update_layout(

        title_x=0.5

    )

    return fig


# =====================================================
# WORD CLOUD
# =====================================================

def create_wordcloud(text):

    text = clean_text(text)

    wc = WordCloud(

        width=1200,

        height=600,

        background_color="white",

        colormap="viridis",

        max_words=100

    ).generate(text)

    fig = plt.figure(figsize=(12,6))

    plt.imshow(wc)

    plt.axis("off")

    plt.tight_layout()

    return fig

# =====================================================
# MAIN ANALYSIS ENGINE
# =====================================================

def analyze_review(
    bag_name,
    review_text="",
    review_url=""
):
    """
    Main function called by the Gradio app.

    Parameters
    ----------
    bag_name : str
        Name of the luxury handbag.
    review_text : str
        User pasted reviews.
    review_url : str
        Optional review article URL.

    Returns
    -------
    report
    pie_chart
    wordcloud
    statistics
    keywords
    aspects
    """

    # -----------------------------------------
    # Download article if URL supplied
    # -----------------------------------------

    if review_url.strip():

        extracted = extract_from_url(review_url)

        if extracted:
            review_text = extracted

    review_text = clean_text(review_text)

    if len(review_text) < 30:

        return (
            "No review text available.\n\nPaste customer reviews or provide a valid review URL.",
            None,
            None,
            "No statistics available.",
            "No keywords found.",
            "No aspects detected."
        )

    # -----------------------------------------
    # AI Analysis
    # -----------------------------------------

    sentiment_results = analyze_text(review_text)

    label, confidence = overall_sentiment(sentiment_results)

    keywords = extract_keywords(review_text)

    aspects = detect_aspects(review_text)

    stats = review_statistics(review_text)

    # -----------------------------------------
    # Charts
    # -----------------------------------------

    pie_chart = sentiment_chart(sentiment_results)

    wordcloud = create_wordcloud(review_text)

    # -----------------------------------------
    # Business Report
    # -----------------------------------------

    report = executive_summary(
        bag_name=bag_name,
        label=label,
        confidence=confidence,
        keywords=keywords,
        aspects=aspects,
        stats=stats
    )

    # -----------------------------------------
    # Aspect Text
    # -----------------------------------------

    if aspects:

        aspect_text = ""

        for aspect, score in aspects:

            aspect_text += f"• {aspect} ({score})\n"

    else:

        aspect_text = "No major discussion themes detected."

    # -----------------------------------------
    # Keyword Text
    # -----------------------------------------

    keyword_text = ", ".join(keywords)

    if keyword_text == "":
        keyword_text = "No keywords detected."

    # -----------------------------------------
    # Statistics Text
    # -----------------------------------------

    statistics = f"""
Words: {stats['words']}

Characters: {stats['characters']}

Sentences: {stats['sentences']}

Estimated Reading Time:
{stats['reading_time']} minutes
"""

    return (
        report,
        pie_chart,
        wordcloud,
        statistics,
        keyword_text,
        aspect_text
    )