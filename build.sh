#!/bin/bash
set -e

# Install dependencies
pip install -r requirements.txt

# Pre-download HuggingFace model so it's cached before app starts
python -c "
from sentence_transformers import SentenceTransformer
print('Downloading BAAI/bge-base-en-v1.5...')
SentenceTransformer('BAAI/bge-base-en-v1.5')
print('Model cached successfully.')
"