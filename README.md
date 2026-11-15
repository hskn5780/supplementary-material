# python_scripts

Python scripts used in the study's data collection and analysis. They are provided as supplementary material for transparency and reproducibility.

## Contents

The scripts cover three steps:

1. **Prompt generation**: files starting with `generate_`
2. **Data collection** (Gemini API): files starting with `run_` (and `patch_`)
3. **Analysis**: files starting with `calculate_`

## Requirements

- Python 3.10 or higher
- Dependencies:

```bash
pip install pandas numpy scipy openpyxl google-genai python-dotenv
```

- The data collection scripts require a Gemini API key, stored in a `.env` file at the project root:

```
GEMINI_API_KEY=your_key_here
```

> The `.env` file is not included in this repository and must never be shared.

## Usage

Run the scripts from the project root, with the scripts located in a folder named `02_scripts/`. The analysis scripts can be run on the provided result files without an API key.

```bash
python 02_scripts/<script_name>.py
```
