# agents/image_insight_agent_api.py

import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def image_insight_agent_api(evidence):
    insights = []

    for idx, e in enumerate(evidence, start=1):
        if not e.get("image_path"):
            continue

        print(f"[INFO] [ImageInsightAPI] Analyzing image: {e['image_path']}")

        image_base64 = encode_image(e["image_path"])

        response = client.responses.create(
            model="gpt-4.1",
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are a medical imaging assistant. "
                                "Describe only visible findings. "
                                "Do NOT diagnose or suggest treatment."
                            )
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Describe this medical image."
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{image_base64}"
                        }
                    ]
                }
            ],
            max_output_tokens=200
        )

        # Correct way to read output
        text = response.output_text
        insights.append(f"[R{idx}-IMAGE] {text}")

    return insights
