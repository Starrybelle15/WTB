"""
Luxury Bag Review Intelligence Platform
Main Gradio Application
"""

import gradio as gr
from engine import analyze_review


# ------------------------------------------------
# Main Function
# ------------------------------------------------

def run_analysis(bag_name, review_text):

    if not review_text.strip():

        return (
            "Please paste some review text.",
            None
        )

    summary, chart = analyze_review(review_text)

    return summary, chart


# ------------------------------------------------
# Gradio UI
# ------------------------------------------------

with gr.Blocks(
    title="Luxury Bag Review Intelligence Platform"
) as demo:

    gr.Markdown(
        """
# 👜 Luxury Bag Review Intelligence Platform

Analyze customer opinions using Artificial Intelligence.

### Instructions

1. Enter the luxury bag name.
2. Paste customer reviews.
3. Click Analyze.
"""
    )

    with gr.Row():

        with gr.Column():

            bag_name = gr.Textbox(
                label="Luxury Bag Name",
                placeholder="Example: Louis Vuitton Neverfull MM"
            )

            reviews = gr.Textbox(
                label="Customer Reviews",
                lines=15,
                placeholder="Paste customer reviews here..."
            )

            analyze = gr.Button(
                "Analyze Reviews",
                variant="primary"
            )

        with gr.Column():

            summary = gr.Textbox(
                label="Executive Summary",
                lines=12
            )

            chart = gr.Plot(
                label="Sentiment Distribution"
            )

    analyze.click(
        fn=run_analysis,
        inputs=[
            bag_name,
            reviews
        ],
        outputs=[
            summary,
            chart
        ]
    )

    gr.Markdown(
        """
---

### Technology Stack

- Hugging Face Transformers
- KeyBERT
- Plotly
- Gradio

Final Year Project
"""
    )


if __name__ == "__main__":
    demo.launch()
