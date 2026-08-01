from engine import analyze_review

sample = """
The Louis Vuitton Neverfull MM is beautifully crafted and the leather feels premium.
The bag is spacious and elegant.
However, it is very expensive and the straps can become uncomfortable after long use.
Overall I love it and would recommend it.
"""

summary, chart = analyze_review(sample)

print(summary)
chart.show()
