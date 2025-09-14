# A2 - Mind2Web Implementation

This directory contains the implementation of $A^2$ for the Mind2Web benchmark.

## Getting Started

### 1. Installation

Run the following commands to setup your environment.

```bash
cd mind2web
conda create -n A2_mind2web python=3.10
conda activate A2_mind2web
pip install -r requirements.txt
```

### 2. Data Preparation 

1. Download the Mind2Web testset from the [official repository](https://github.com/OSU-NLP-Group/Mind2Web) or directly [here](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/deng_595_buckeyemail_osu_edu/EUkdc16xUC1EplDiXS1bvSEBOddFLgOyJNkWJOxdltNEGA?e=8N1D9S).
2. Extract the files and place the following directories into `./data/`:
   - `test_task/`
   - `test_website/`
   - `test_domain/`
3. Download `scores_all_data.pkl` for elements filtering from [here](https://buckeyemailosu-my.sharepoint.com/personal/deng_595_buckeyemail_osu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fdeng%5F595%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FMind2Web%2Fscores%5Fall%5Fdata%2Epkl&parent=%2Fpersonal%2Fdeng%5F595%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FMind2Web&ga=1) and save it under `./data/`

The files in `data/memory/` are sourced from [AWM](https://github.com/zorazrw/agent-workflow-memory.git).

### 3. Configure OpenAI API

In `./utils/llm.py` file, specify your OpenAI API key in line 14:

```python
self.client = OpenAI(api_key="YOUR_API_KEY")
```


## Usage

### Basic Command

Run the pipeline with the following command:

```bash
python pipeline.py --load_results 0 --load_scores 1 --if_workflow 1 --top_k_elements 5 --benchmark test_task
```


## Project Structure

```
mind2web/
├── pipeline.py              # Main experiment script
├── abs_induction.py         # Abstraction-based workflow induction
├── utils/                   # Utility functions
├── prompt/                  # Prompt templates
├── data/                    # Mind2web dataset
├── results/                 # Output results
├── abstraction/             # Generated abstractions
└── logs/                    # Execution logs
```

