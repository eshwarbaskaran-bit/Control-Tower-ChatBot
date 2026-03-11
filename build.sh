#!/bin/bash
set -e

# Install dependencies
pip install -r requirements.txt

# Pre-download MiniLM model so it's cached before app starts (~80MB, fits in 512MB)
python -c "
from sentence_transformers import SentenceTransformer
print('Downloading sentence-transformers/all-MiniLM-L6-v2...')
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('Model cached successfully.')
"