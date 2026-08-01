"""
Luxury Bag Review Intelligence Platform
Version 1.0
"""

import gradio as gr
from engine import analyze_review

# -------------------------------------------------------
# Main Function
# -------------------------------------------------------

def analyze(bag_name, review_url, review_text):

    # Future version:
    # if review_url is provided:
    #     download review automatically

    if review_text.strip() == "":
        return (
            "Please paste customer reviews or provide a review URL.",
            None,
        )

    summary, chart = analyze_review(

    review_text=review_text,

    review_url=review_url

)

    return summary, chart


# -------------------------------------------------------
# UI
# -------------------------------------------------------

with gr.Blocks(
    title="Luxury Bag Review Intelligence Platform"
) as demo:

    gr.Markdown(
        """
# 👜 Luxury Bag Review Intelligence Platform

### AI-powered customer review analysis for luxury handbags

Analyze online reviews and generate business insights using Artificial Intelligence.
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
                placeholder="Paste a review article URL here..."
            )

            review_text = gr.Textbox(
                label="Customer Reviews",
                lines=15,
                placeholder="Paste customer reviews here..."
            )

            analyze_button = gr.Button(
                "🔍 Analyze Reviews",
                variant="primary"
            )

        with gr.Column(scale=1):

            report = gr.Textbox(
                label="AI Business Insight Report",
                lines=15
            )

            chart = gr.Plot(
                label="Sentiment Distribution"
            )

    analyze_button.click(
        fn=analyze,
        inputs=[
            bag_name,
            review_url,
            review_text,
        ],
        outputs=[
            report,
            chart,
        ]
    )

    gr.Markdown(
        """
---

### Technology

- Hugging Face Transformers
- KeyBERT
- Plotly
- Gradio

Final Year Project
"""
    )

if __name__ == "__main__":
    demo.launch()