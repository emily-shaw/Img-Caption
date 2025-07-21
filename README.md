# Img-Caption

A toolkit for generating and summarizing image captions, focusing on extracting the primary character's physical features from batches of images. Supports multiple vision models (OpenAI GPT-4 Vision, Replicate models) and flexible batch processing.

## Features
- Run multiple image captioning models on batches of images
- Supports OpenAI GPT-4 Vision and Replicate models (LLaVA, BLIP, Moondream2)
- Summarizes the main character's physical features from all captions per model
- Aggregates all results and summaries into a single CSV for analysis
- Flexible CLI: process multiple zip files, select which zips to caption
- Output JSON files include prompt and are named by zip/model

## Directory Structure
- `img_storage/` — Images extracted from zip files for processing
- `results/` — Model captioning result JSON files (output)
- `summaries/` — Summaries and aggregated CSVs (output)
- `summarize_characters.py` — Main script for summarizing and aggregating
- `image_captioners.py`, `cli.py` — Model and CLI logic

## Setup
1. **Clone the repository**
2. **Install dependencies**
   - Python 3.9+
   - Install with [uv](https://github.com/astral-sh/uv):
     ```bash
     uv add -r requirements.txt
     # or, if using pyproject.toml:
     uv add .
     ```
3. **Set up API keys in .env**
   - Create a `.env` file in the project root with:
     ```env
     OPENAI_API_KEY=your_openai_key_here
     REPLICATE_API_TOKEN=your_replicate_token_here
     ANTHROPIC_API_KEY=your_claude_api_key_here
     ```

## Usage

### 1. Run Image Captioning on Zip Files
Extracts images from one or more zip files and runs the selected models. Results are saved in `results/`.

**Basic usage (processes only the last zip):**
```bash
uv run cli.py cat.zip chole.zip
```
This will caption only images from `chole.zip`.

**Process specific zips (comma-separated):**
```bash
uv run cli.py cat.zip chole.zip --process-zips cat.zip,chole.zip
```
This will caption images from both `cat.zip` and `chole.zip`.

- Only uncommented models in the `combos` list in `cli.py` will be run.
- Each model call uses a new session/client for isolation.
- Output files are named as `<zipname>_<modelname>.json` (e.g., `cat_openai_gpt4_vision.json`).
- The prompt used is included as a top-level field in the output JSON.

**Output JSON format:**
```json
{
  "prompt": "...the prompt used...",
  "results": [
    {"image_name": "...", "response": "..."},
    ...
  ]
}
```

### 2. Summarize Characters from Captions
This will process all JSON files in `results/`, send captions to Claude, and write summaries to `summaries/`.

```bash
uv run summarize_characters.py
```

- Each summary will be saved as a JSON file in `summaries/`, named after the input file.

### 3. Aggregate All Results to CSV
After summarization, aggregate all results and summaries into a single CSV:

```bash
uv run summarize_characters.py aggregate
```

- This creates `summaries/character_summaries.csv` with columns:
  - `model_name`, `image_name`, `caption`, `summary_type`, `summary_features`

## Model Support
- **OpenAI GPT-4 Vision**: Uses the OpenAI API, requires `OPENAI_API_KEY`.
- **Replicate models**: LLaVA, BLIP, Moondream2 (see `image_captioners.py` for details), require `REPLICATE_API_TOKEN`.
- **Anthropic Claude**: Used for summarization, requires `ANTHROPIC_API_KEY`.
- Each model call is isolated in its own session/client for reliability.

## Troubleshooting
- If you see errors about missing API keys, ensure your `.env` file is present and correct.
- If you see `Error code: 529 - overloaded`, the Claude API is temporarily unavailable. Wait and retry.
- Ensure your API keys are valid and you have internet access.
- If you see errors about zip files or images not found, check your zip file paths and contents.

## License
MIT License (add your license here if different)
