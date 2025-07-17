import click
import sys
import os
import zipfile
from dotenv import load_dotenv
from image_captioners import run_llava13b, run_blip, run_moondream2
import json

@click.command()
@click.argument('zipfile_path', type=click.Path(exists=True))
def main(zipfile_path):
    print(f"Received zip file: {zipfile_path}")

    output_dir = "img_storage"
    os.makedirs(output_dir, exist_ok=True)
    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    try:
        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
            found_image = False
            for file_info in zip_ref.infolist():
                filename = file_info.filename
                if filename.startswith('__MACOSX/'):
                    continue
                ext = os.path.splitext(filename)[1].lower()
                if ext in image_exts:
                    print(f"Extracting {filename}...")
                    zip_ref.extract(file_info, output_dir)
                    found_image = True
            if not found_image:
                print("No image files found in the zip archive.")
    except zipfile.BadZipFile:
        print(f"Error: '{zipfile_path}' is not a valid zip file.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
    else:
        print(f"Images extracted to {output_dir}/")

    load_dotenv()
    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        print("REPLICATE_API_TOKEN not found in environment or .env file.")
        sys.exit(1)
    
    images = [f for f in os.listdir(output_dir) if os.path.splitext(f)[1].lower() in image_exts and not f.startswith(".")]
    if not images:
        print("No images found in img_storage to process.")
        sys.exit(1)

    output_json_dir = "results"
    os.makedirs(output_json_dir, exist_ok=True)

    combos = [
        ("yorickvp_llava-13b", "What is a short and accurate caption for this image?", run_llava13b),
        ("yorickvp_llava-13b", "image captioning", run_llava13b),
        ("salesforce_blip", None, run_blip),
        ("lucataco_moondream2", None, run_moondream2),
    ]

    results_dict = {}
    for model_name, prompt, _ in combos:
        key = (model_name, prompt)
        results_dict[key] = []

    for image_name in images:
        image_path = os.path.join(output_dir, image_name)
        for model_name, prompt, func in combos:
            if prompt:
                response = func(image_path, prompt)
            else:
                response = func(image_path)
            results_dict[(model_name, prompt)].append({
                "image_name": image_name,
                "response": response
            })
            print(f"Finished {model_name} (prompt: '{prompt}') on {image_name}")
            print(f"Output: {response}\n")

    for (model_name, prompt), results in results_dict.items():
        prompt_part = prompt.replace(" ", "_").replace("?", "") if prompt else "default"
        json_name = f"{model_name}_{prompt_part}_results.json"
        json_path = os.path.join(output_json_dir, json_name)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {json_name}")

if __name__ == "__main__":
    main() 