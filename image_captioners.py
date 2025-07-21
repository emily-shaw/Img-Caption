import replicate
import base64
import os
from openai import OpenAI

# Always create a new Replicate client for each call to ensure a fresh session

def run_replicate_model(image_path, model, input_kwargs):
    client = replicate.Client()
    with open(image_path, "rb") as image_file:
        input_data = dict(input_kwargs)
        input_data["image"] = image_file  # Always set the image key
        output = client.run(model, input=input_data)
        if hasattr(output, '__iter__') and not isinstance(output, (str, bytes, list, dict)):
            return ''.join(str(part) for part in output)
        return output

def run_llava13b(image_path, prompt):
    return run_replicate_model(
        image_path,
        "yorickvp/llava-13b:a0fdc44e4f2e1f20f2bb4e27846899953ac8e66c5886c5878fa1d6b73ce009e5",
        {"prompt": prompt}
    )

def run_blip(image_path, prompt=None):
    input_kwargs = {}
    if prompt:
        input_kwargs["prompt"] = prompt
    return run_replicate_model(
        image_path,
        "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
        input_kwargs
    )

def run_moondream2(image_path, prompt=None):
    input_kwargs = {}
    if prompt:
        input_kwargs["prompt"] = prompt
    return run_replicate_model(
        image_path,
        "lucataco/moondream2:72ccb656353c348c1385df54b237eeb7bfa874bf11486cf0b9473e691b662d31",
        input_kwargs
    )

def run_gpt4_vision(image_path, prompt):
    # Create a new OpenAI client for each call to ensure a fresh session
    client = OpenAI()
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    base64_image = encode_image(image_path)
    data_url = f"data:image/jpeg;base64,{base64_image}"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            }
        ],
        max_tokens=100
    )
    return response.choices[0].message.content