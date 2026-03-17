"""
Extract the final answer from reasoning traces.

We look for:
"The answer is X"
or the final number in reasoning.
"""

import re

def extract_answer(text):
    """
    Extracts numerical answer from model output.
    """

    # pattern 1: "The answer is X"
    match = re.search(r"The answer is (-?\d+)", text)
    if match:
        return match.group(1)

    # fallback: last integer in text
    numbers = re.findall(r"-?\d+", text)
    if numbers:
        return numbers[-1]

    return None