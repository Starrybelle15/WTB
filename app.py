"""
=========================================================
Luxury Bag Review Intelligence Platform

Version 2.0

Gradio User Interface

=========================================================
"""

import gradio as gr

from engine import analyze_review


# =====================================================
# Main Function
# =====================================================

def analyze(

    bag_name,

    review_url,

    review_text

):

    if not bag_name.strip():

        return (

            "Please enter a luxury bag name.",

            "",

            "",

            "",

            None,

            None

        )

    report, chart, wordcloud, statistics, keywords, aspects = analyze_review(

        bag_name=bag_name,

        review_text=review_text,

        review_url=review_url

    )

    return (

        report,

        statistics,

        keywords,

        aspects,

        chart,

        wordcloud

    )


# =====================================================
# User Interface
# =====================================================

with gr.Blocks(

    title="Luxury Bag Review Intelligence Platform"

) as demo:

    gr.Markdown(

"""
# 👜 Luxury Bag Review Intelligence Platform

### AI-Powered Luxury Bag Review Analysis

Analyse customer reviews from pasted text or a review URL.

Generate business intelligence using Artificial Intelligence.

---
"""
    )

    with gr.Row():

        with gr.Column(scale=1):

            bag_name = gr.Textbox(

                label="Luxury Bag Name",

                placeholder="Example: Louis Vuitton Neverfull MM"

            )

            review_url = gr.Textbox(

                label="Review URL (Optional)",

                placeholder="Paste a review article URL"

            )

            review_text = gr.Textbox(

                label="Customer Reviews",

                lines=16,

                placeholder="Paste review text here..."

            )

            analyze_button = gr.Button(

                "🔍 Analyze Reviews",

                variant="primary"

            )

        with gr.Column(scale=1):

            report = gr.Textbox(

                label="AI Business Insight Report",

                lines=20

            )

            statistics = gr.Textbox(

                label="Review Statistics",

                lines=6

            )

            keywords = gr.Textbox(

                label="Keywords",

                lines=3

            )

            aspects = gr.Textbox(

                label="Aspect Analysis",

                lines=6

            )

                        chart = gr.Plot(

                label="Sentiment Distribution"

            )

            wordcloud = gr.Plot(

                label="Word Cloud"

            )

    analyze_button.click(

        fn=analyze,

        inputs=[

            bag_name,

            review_url,

            review_text

        ],

        outputs=[

            report,

            statistics,

            keywords,

            aspects,

            chart,

            wordcloud

        ]

    )

    gr.Markdown(

"""
---

## 📊 AI Features

✅ Transformer-based Sentiment Analysis

✅ Keyword Extraction (KeyBERT)

✅ Aspect Detection

✅ AI Business Insight Report

✅ Word Cloud Generation

✅ Interactive Data Visualisation

---

### Technology Stack

- Hugging Face Transformers
- CardiffNLP RoBERTa
- KeyBERT
- Plotly
- WordCloud
- Trafilatura
- Gradio

© 2026 Luxury Bag Review Intelligence Platform
"""
    )

# =====================================================
# Launch Application
# =====================================================

if __name__ == "__main__":

    demo.launch(
        share=False,
        debug=True
    )