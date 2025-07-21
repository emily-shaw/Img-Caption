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
        # Always expect a dict with 'results' key
        return [str(item.get("response", "")) for item in data.get("results", []) if item.get("response")]
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

def save_summary(filename: str, summary: str, model: str, prompt: str, zipfile: str, summary_type: str, summary_features: str):
    try:
        summary_path = os.path.join(SUMMARY_DIR, filename.replace('.json', '_summary.json'))
        with open(summary_path, 'w') as f:
            json.dump({
                "summary": summary,
                "model": model,
                "prompt": prompt,
                "zipfile": zipfile,
                "summary_type": summary_type,
                "summary_features": summary_features
            }, f, indent=2)
        print(f"✓ Saved summary to {summary_path}")
    except Exception as e:
        print(f"Error saving summary for {filename}: {e}")

def write_zip_summaries_csv(summary_records):
    # Group by zipfile using a standard dict
    zip_to_records = {}
    for rec in summary_records:
        zf = rec['zipfile']
        if zf not in zip_to_records:
            zip_to_records[zf] = []
        zip_to_records[zf].append(rec)
    for zipfile, records in zip_to_records.items():
        # Sort by model name for consistency
        records = sorted(records, key=lambda r: r['model'])
        csv_path = os.path.join(SUMMARY_DIR, f"{os.path.splitext(zipfile)[0]}_summaries.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # First row: key 'model', then model names
            writer.writerow(['model'] + [r['model'] for r in records])
            # Second row: key 'prompt', then prompts
            writer.writerow(['prompt'] + [r['prompt'] for r in records])
            # Third row: key 'summary_type', then summary types
            writer.writerow(['summary_type'] + [r['summary_type'] for r in records])
            # Fourth row: key 'summary_features', then summary features
            writer.writerow(['summary_features'] + [r['summary_features'] for r in records])
        print(f"Wrote summary CSV for {zipfile} to {csv_path}")

def main():
    print("🚀 Starting character summarization...")
    
    setup_directories()
    client = get_anthropic_client()
    
    json_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')]

    print(f"Found {len(json_files)} files to process")
    
    summary_records = []
    
    for i, filename in enumerate(json_files, 1):
        print(f"\n📁 Processing {i}/{len(json_files)}: {filename}")
        
        filepath = os.path.join(RESULTS_DIR, filename)
        base = filename[:-5]
        parts = base.split('_')
        if len(parts) < 3:
            print(f"⚠️  Unexpected filename format for {filename}, skipping...")
            continue
        zipfile_name = parts[0] + '.zip'
        # Model name is all parts except the first (zipfile) and last (hash)
        model = '_'.join(parts[1:-1])
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            prompt = data.get('prompt', '')
        except Exception as e:
            print(f"⚠️  Could not load prompt from {filename}: {e}")
            prompt = ''
        captions = load_captions(filepath)
        if not captions:
            print(f"⚠️  No captions found in {filename}, skipping...")
            continue
        captions_text = "\n".join(captions)
        prompt_for_llm = PROMPT_TEMPLATE.format(captions_text=captions_text)
        
        print(f"📝 Sending to Claude Sonnet 3.5...")
        
        summary = call_claude(client, prompt_for_llm)
        summary_type = ""
        summary_features = ""
        if summary:
            print(f"✅ Summary for {filename}:")
            print(f"   {summary}")
            # Try to parse summary as JSON
            try:
                summary_json = json.loads(summary)
                summary_type = summary_json.get("type", "")
                summary_features = summary_json.get("features", "")
            except Exception:
                summary_type = ""
                summary_features = ""
            save_summary(filename, summary, model, prompt, zipfile_name, summary_type, summary_features)
            summary_records.append({
                "zipfile": zipfile_name,
                "model": model,
                "prompt": prompt,
                "summary_type": summary_type,
                "summary_features": summary_features
            })
        else:
            print(f"❌ Failed to get summary for {filename}")
    write_zip_summaries_csv(summary_records)
    print(f"\n🎉 Processing complete!")

if __name__ == "__main__":
    main() 