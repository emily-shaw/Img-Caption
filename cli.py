import click
import sys
import os
import zipfile
from dotenv import load_dotenv
from image_captioners import run_llava13b, run_blip, run_moondream2, run_gpt4_vision
import json
import csv


load_dotenv()


@click.command()
@click.argument('zipfile_paths', nargs=-1, type=click.Path(exists=True))
def main(zipfile_paths):
    if not zipfile_paths:
        print("Please provide at least one zip file.")
        sys.exit(1)
    print(f"Received zip files: {zipfile_paths}")

    output_dir = "img_storage"
    os.makedirs(output_dir, exist_ok=True)
    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    zip_to_images = {}
    for zipfile_path in zipfile_paths:
        images_in_this_zip = []
        try:
            with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
                found_image = False
                for file_info in zip_ref.infolist():
                    filename = file_info.filename
                    if filename.startswith('__MACOSX/'):
                        continue
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in image_exts:
                        print(f"Extracting {filename} from {zipfile_path}...")
                        zip_ref.extract(file_info, output_dir)
                        images_in_this_zip.append(os.path.basename(filename))
                        found_image = True
                if not found_image:
                    print(f"No image files found in the zip archive {zipfile_path}.")
        except zipfile.BadZipFile:
            print(f"Error: '{zipfile_path}' is not a valid zip file.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            sys.exit(1)
        else:
            print(f"Images extracted to {output_dir}/ from {zipfile_path}")
        zip_to_images[zipfile_path] = images_in_this_zip

    # Always process all unique zip files in order
    seen = set()
    zips_to_process = []
    for z in zipfile_paths:
        if z not in seen:
            zips_to_process.append(z)
            seen.add(z)

    # Collect images from the selected zips
    images = []
    for z in zips_to_process:
        images.extend(zip_to_images.get(z, []))
    if not images:
        print(f"No images found in img_storage to process for {zips_to_process}.")
        sys.exit(1)

    output_json_dir = "results"
    os.makedirs(output_json_dir, exist_ok=True)

    combos = [
        ("yorickvp_llava-13b", "Look at the image and write a single, accurate caption that describes only the physical features(such as hair color, skin color, eye color, etc.), gender(men or women) and any face accessories (such as glasses, earrings, etc.) of the main character. Do NOT mention clothing (such as shirts, pants, dresses, etc.) or anything about the room, background, movement or environment. Only include features that are clearly visible.", run_llava13b),
        ("yorickvp_llava-13b", "image captioning", run_llava13b),
        ("salesforce_blip", "describe only the physical features and any face accessory of the main charecter. do NOT describe shirt and the room.", run_blip),
        ("lucataco_moondream2", "describe only the physical features and any face accessory of the main charecter. Don't describe clothes and the room.", run_moondream2),
        ("openai_gpt4_vision", "Look at the image and write a accurate caption that describes only the physical features(such as hair or fur color, eye color, etc.), gender(men or women) and any face accessories (such as glasses, earrings, etc.) of the main character. Do NOT mention clothing (such as shirts, pants, dresses, etc.) or anything about the room, background, movement or environment. Only include features that are clearly visible.", run_gpt4_vision),
    ]

    # For each zip, process and save results independently
    for zip_path in zips_to_process:
        zip_basename = os.path.splitext(os.path.basename(zip_path))[0]
        images = zip_to_images.get(zip_path, [])
        if not images:
            print(f"No images found in img_storage to process for {zip_path}.")
            continue

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

        # Save results with zip file name and model name only, and put the prompt at the top of the JSON file
        for (model_name, prompt), results in results_dict.items():
            prompt_short = "_".join(prompt.split()[:2]) if prompt else "none"
            json_name = f"{zip_basename}_{model_name}_{prompt_short}.json"
            json_path = os.path.join(output_json_dir, json_name)
            output_data = {
                "prompt": prompt,
                "results": results,
                "prompt_short": prompt_short
            }
            with open(json_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"Saved {json_name}")

        # Create a CSV for all captions from all models for the current zip
        all_images = sorted(images)
        model_names = []
        prompts = []
        model_to_caption = {}
        for (model_name, prompt), results in results_dict.items():
            model_names.append(model_name)
            prompts.append(prompt)
            img_to_caption = {r["image_name"]: r["response"] for r in results}
            model_to_caption[model_name] = img_to_caption
        csv_name = f"{zip_basename}_all_captions.csv"
        csv_path = os.path.join(output_json_dir, csv_name)
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # First row: zip file name, then model names
            writer.writerow([zip_basename] + model_names)
            # Second row: 'captions' in the first cell, then prompts for each model
            writer.writerow(["captions"] + prompts)
            # Each subsequent row: image name, then captions for each model
            for img in all_images:
                row = [img]
                for model_name in model_names:
                    row.append(model_to_caption[model_name].get(img, ""))
                writer.writerow(row)
        print(f"Saved {csv_name}")

if __name__ == "__main__":
    main() 