import click
import sys
import os
import zipfile
from dotenv import load_dotenv
from image_captioners import run_llava13b, run_blip, run_moondream2

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
    first_image_path = os.path.join(output_dir, images[0])

    print("Running LLaVA-13B (prompt: 'What is a short and accurate caption for this image?') on the first image in img_storage...")
    print(run_llava13b(first_image_path, "What is a short and accurate caption for this image?"))

    print("Running BLIP on the first image in img_storage...")
    print(run_blip(first_image_path))

    print("Running Moondream2 on the first image in img_storage...")
    print(run_moondream2(first_image_path))

    print("Running LLaVA-13B (prompt: 'image captioning') on the first image in img_storage...")
    print(run_llava13b(first_image_path, "image captioning"))

if __name__ == "__main__":
    main() 