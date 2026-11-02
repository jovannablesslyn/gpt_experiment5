# Experiment 5 - Legal Document Analyzer (Ollama / local LLM version)

## Setup
1. Install Ollama: https://ollama.com  (it must be running)
2. Download a model:   ollama pull llama3.1
3. pip install -r requirements.txt

## Run
    python legal_analyzer_ollama.py

Use another model:  set OLLAMA_MODEL=mistral   (Mac/Linux: export OLLAMA_MODEL=mistral)

Menu: 1 Upload/Enter Document, 2 Analyze, 3 Generate Summary, 4 Save Report, 5 Exit.
Test with sample_contract.txt. Report is saved to outputs/legal_analysis.txt.
