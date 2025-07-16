import click
import sys
import os
import zipfile

@click.command()
@click.argument('zipfile_path', type=click.Path(exists=True))
def main(zipfile_path):
    """Extract images from a ZIP file to the img_storage folder."""
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

if __name__ == "__main__":
    main() 