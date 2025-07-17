# Img-Caption

A toolkit for summarizing image captions and extracting the primary character's physical features from batches of image captioning results. Uses Anthropic Claude for summarization and outputs structured summaries and CSV aggregations for analysis.

## Features
- Processes batches of image captioning results (JSON format)
- Summarizes the main character's physical features from all captions per model
- Aggregates all results and summaries into a single CSV for easy analysis

## Directory Structure
- `results/` — Place your model captioning result JSON files here (input)
- `summaries/` — Summaries and aggregated CSVs will be written here (output)
- `summarize_characters.py` — Main script for summarizing and aggregating
- `image_captioners.py`, `cli.py` — (Other project scripts)

## Setup
1. **Clone the repository**
2. **Install dependencies**
   - Python 3.8+
   - Add required packages with [uv](https://github.com/astral-sh/uv):
     ```bash
     uv add -r requirements.txt
     # or, if using pyproject.toml:
     uv add .
     ```
3. **Set up Anthropic API key**
   - Create a `.env` file in the project root with:
     ```env
     ANTHROPIC_API_KEY=your_claude_api_key_here
     ```
   - Or export it in your shell:
     ```bash
     export ANTHROPIC_API_KEY=your_claude_api_key_here
     ```

## Usage

### 1. Summarize Characters from Captions
This will process all JSON files in `results/`, send captions to Claude, and write summaries to `summaries/`.

```bash
python summarize_characters.py
```

- Each summary will be saved as a JSON file in `summaries/`, named after the input file.

### 2. Aggregate All Results to CSV
After summarization, aggregate all results and summaries into a single CSV:

```bash
python summarize_characters.py aggregate
```

- This creates `summaries/character_summaries.csv` with columns:
  - `model_name`, `image_name`, `caption`, `summary_type`, `summary_features`

## Input Format
- Each file in `results/` should be a JSON list of objects, each with at least:
  - `image_name`: Name of the image
  - `response`: Caption text

## Output
- **Summaries:** One JSON summary per input file in `summaries/`
- **Aggregated CSV:** `summaries/character_summaries.csv` with all image captions and per-model character summaries

## Notes
- Only features explicitly mentioned in captions are included in summaries (no guessing/inference)
- Accessories (e.g., glasses, collar) are included if described
- Background, actions, and clothing are excluded from summaries

## Troubleshooting
- If you see `Error code: 529 - overloaded`, the Claude API is temporarily unavailable. Wait and retry.
- Ensure your API key is valid and you have internet access.

## License
MIT License (add your license here if different)
