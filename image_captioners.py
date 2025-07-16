import replicate

def run_replicate_model(image_path, model, input_kwargs):
    with open(image_path, "rb") as image_file:
        input_data = dict(input_kwargs)
        input_data["image"] = image_file  # Always set the image key
        output = replicate.run(model, input=input_data)
        if hasattr(output, '__iter__') and not isinstance(output, (str, bytes, list, dict)):
            return ''.join(str(part) for part in output)
        return output

def run_llava13b(image_path, prompt):
    return run_replicate_model(
        image_path,
        "yorickvp/llava-13b:a0fdc44e4f2e1f20f2bb4e27846899953ac8e66c5886c5878fa1d6b73ce009e5",
        {"prompt": prompt}
    )

def run_blip(image_path):
    return run_replicate_model(
        image_path,
        "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
        {}  # Use only required/default arguments
    )

def run_moondream2(image_path):
    return run_replicate_model(
        image_path,
        "lucataco/moondream2:72ccb656353c348c1385df54b237eeb7bfa874bf11486cf0b9473e691b662d31",
        {}  # Use only required/default arguments
    )