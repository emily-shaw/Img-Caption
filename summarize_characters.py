import os
import json
import anthropic
import sys
from typing import List
from dotenv import load_dotenv
import csv

PROMPT_TEMPLATE = '''Analyze the image captions below and identify the primary character that appears most frequently or prominently across all images.

**Instructions:**
1. Review all captions to identify the most referenced character
2. If multiple characters appear, focus on the one mentioned most often or described as the main subject
3. Provide a clear, specific description based on the visual details mentioned
4. **Do NOT include any movement, actions, or surroundings—focus only on physical features.**
5. **Do NOT include background, objects, environment (e.g., 'wooden floor', 'sofa', 'table'), or clothing. Only describe the character's physical traits (e.g., face, hair, eyes, fur, paws, etc).**
6. **Describe the color and features of any accessories (e.g., 'glasses: dark rimmed', 'collar: red') if mentioned.**
7. **Only include features that are explicitly mentioned in the captions. Do not guess or infer features that are not present in the captions.**

**Output Requirements:**
1. **Type**: Specify the character type (man, woman, boy, girl, cat, dog, bird, etc.)
2. **Features**: 
   - For humans: Describe facial features, hair, accessories (e.g., "short black hair, dark rimmed glasses, short beard")
   - For animals: Describe coat color, markings, size, breed characteristics (e.g., "black and white fur, green eyes, white paws")
   - For other objects: Describe key visual characteristics
   - For accessories: Describe color and features (e.g., "glasses: dark rimmed", "collar: red")

**Examples:**
- Type: "man", Features: "short beard, curly hair, glasses: dark rimmed"
- Type: "cat", Features: "orange fur, white paws, green eyes, collar: red"
- Type: "woman", Features: "long brown hair, glasses: round"

**Image Captions:**
{captions_text}

**Response Format:**
Please respond with ONLY a valid JSON object in this exact format:
{{
    "type": "character_type",
    "features": "comma-separated list of features"
}}

**Important:** Ensure your response is valid JSON and includes both "type" and "features" fields. Do not include movement, actions, surroundings, background/objects like 'wooden floor', 'sofa', etc., or clothing. Only use features explicitly mentioned in the captions. Describe the color and features of accessories if present.'''

RESULTS_DIR = "results"
SUMMARY_DIR = "summaries"

def setup_directories():
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: {RESULTS_DIR} directory not found!")
        sys.exit(1)

def get_anthropic_client():
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set!")
        print("Please set it in your .env file or with: export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)

def load_captions(filepath: str) -> List[str]:
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"Warning: {filepath} does not contain a list of results")
            return []
        return [str(item.get("response", "")) for item in data if item.get("response")]
    except json.JSONDecodeError as e:
        print(f"Error reading {filepath}: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error reading {filepath}: {e}")
        return []

def call_claude(client: anthropic.Anthropic, prompt: str) -> str:
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=512,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text if response.content else ""
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return ""

def save_summary(filename: str, summary: str):
    try:
        summary_path = os.path.join(SUMMARY_DIR, filename.replace('.json', '_summary.json'))
        with open(summary_path, 'w') as f:
            json.dump({
                "summary": summary,
                "source_file": filename
            }, f, indent=2)
        print(f"✓ Saved summary to {summary_path}")
    except Exception as e:
        print(f"Error saving summary for {filename}: {e}")

def aggregate_results_to_csv():
    """
    Aggregate all result JSONs in the results/ directory into a single character_summaries.csv,
    including model_name, image_name, caption for each image, and then at the end, one row per model with only the summary columns filled.
    """
    results_dir = "results"
    summary_dir = "summaries"
    output_csv = os.path.join("summaries", "character_summaries.csv")
    image_rows = []
    summary_rows = []
    model_summaries = {}
    for fname in os.listdir(results_dir):
        if fname.endswith(".json") and not fname.endswith("_summary.json"):
            json_path = os.path.join(results_dir, fname)
            # Use rsplit to robustly remove the last _results for summary filename
            if fname.endswith("_results.json"):
                base = fname.rsplit("_results", 1)[0]
                summary_filename = f"{base}_results_summary.json"
            else:
                base = fname[:-5]
                summary_filename = f"{base}_summary.json"
            model_name = base
            summary_path = os.path.join(summary_dir, summary_filename)
            summary_type = ""
            summary_features = ""
            if os.path.exists(summary_path):
                with open(summary_path) as sf:
                    summary_data = json.load(sf)
                    summary_raw = summary_data.get("summary", "")
                    try:
                        summary_json = json.loads(summary_raw)
                        summary_type = summary_json.get("type", "")
                        summary_features = summary_json.get("features", "")
                    except Exception:
                        summary_type = ""
                        summary_features = ""
            # Store summary for this model for later (no summary_json key)
            model_summaries[model_name] = {
                "model_name": model_name,
                "image_name": "",
                "caption": "",
                "summary_type": summary_type,
                "summary_features": summary_features
            }
            # Add image rows (no summary_json key)
            with open(json_path) as jf:
                data = json.load(jf)
            for row in data:
                image_rows.append({
                    "model_name": model_name,
                    "image_name": row.get("image_name", ""),
                    "caption": row.get("response", ""),
                    "summary_type": "",
                    "summary_features": ""
                })
    # Write to CSV: first all image rows, then one summary row per model
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "model_name", "image_name", "caption", "summary_type", "summary_features"
        ])
        writer.writeheader()
        for row in image_rows:
            writer.writerow(row)
        for model_row in model_summaries.values():
            writer.writerow(model_row)
    print(f"Aggregated CSV written to {output_csv}")

def main():
    print("🚀 Starting character summarization...")
    
    setup_directories()
    client = get_anthropic_client()
    
    json_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')]
    if not json_files:
        print(f"No JSON files found in {RESULTS_DIR}")
        return
    
    print(f"Found {len(json_files)} files to process")
    
    csv_rows = []
    
    for i, filename in enumerate(json_files, 1):
        print(f"\n📁 Processing {i}/{len(json_files)}: {filename}")
        
        filepath = os.path.join(RESULTS_DIR, filename)
        captions = load_captions(filepath)
        
        if not captions:
            print(f"⚠️  No captions found in {filename}, skipping...")
            continue
        
        captions_text = "\n".join(captions)
        prompt = PROMPT_TEMPLATE.format(captions_text=captions_text)
        
        print(f"📝 Sending to Claude Sonnet 3.5...")
        
        summary = call_claude(client, prompt)
        
        if summary:
            print(f"✅ Summary for {filename}:")
            print(f"   {summary}")
            save_summary(filename, summary)
            # Try to parse summary as JSON
            try:
                summary_json = json.loads(summary)
                csv_rows.append({
                    "source_file": filename,
                    "type": summary_json.get("type", ""),
                    "features": summary_json.get("features", ""),
                    "summary": summary
                })
            except Exception:
                csv_rows.append({
                    "source_file": filename,
                    "type": "",
                    "features": "",
                    "summary": summary
                })
        else:
            print(f"❌ Failed to get summary for {filename}")
    print(f"\n🎉 Processing complete!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "aggregate":
        aggregate_results_to_csv()
    else:
        main() 