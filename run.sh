#!/bin/bash

# Set default model name and number of worker threads
MODEL_NAME="google/gemini-3-pro-preview"
WORKERS=20  # Default number of worker threads

baseURL="https://openrouter.ai/api/v1"  # Replace with actual base URL
APIkey="Your_API_KEY"          # Replace with actual API key


if [ $# -ge 1 ]; then
    MODEL_NAME=$1
fi
if [ $# -ge 2 ]; then
    WORKERS=$2
fi

echo "Running evaluation for model: ${MODEL_NAME} with workers: ${WORKERS}"
echo "Start time: $(date)"

echo "=== Running Factor Retrival tasks ==="
cd ./AutoQuant/001_Retrival/
python main.py --task='prediction' --model_name="${MODEL_NAME}" --base_url="${baseURL}" --api_key="${APIkey}" --workers="${WORKERS}"
python main.py --task='eval' --model_name="${MODEL_NAME}"

echo "=== Running SQL Generation tasks ==="
cd ../002_SQL/
python main.py --task='prediction' --model_name="${MODEL_NAME}" --base_url="${baseURL}" --api_key="${APIkey}" --workers="${WORKERS}"
python main.py --task='eval' --model_name="${MODEL_NAME}"

echo "=== Running Back Test tasks ==="
cd ../003_BackTest/
python main.py --task='prediction' --model_name="${MODEL_NAME}" --base_url="${baseURL}" --api_key="${APIkey}" --workers="${WORKERS}"
python main.py --task='eval' --model_name="${MODEL_NAME}"

echo "=== All evaluations completed for model: ${MODEL_NAME} ==="
echo "End time: $(date)"