import random

TEMPLATES = {
    "detailed_description": [
        "Please provide a highly detailed, comprehensive description of this image, capturing every visible element, spatial relationship, and background context.",
        "Describe this image as if you are explaining it to someone who cannot see it, ensuring no detail is left out."
    ],
    "complex_reasoning": [
        "Based on the visual evidence in the image, infer the sequence of events that likely led to this scene. Explain your reasoning step-by-step.",
        "What are the implicit relationships between the objects shown? Provide a logical deduction."
    ],
    "ocr_reading": [
        "Extract all visible text in this image and format it into a structured markdown table or list."
    ]
}

def get_random_prompt(task_type):
    return random.choice(TEMPLATES.get(task_type, TEMPLATES["detailed_description"]))
