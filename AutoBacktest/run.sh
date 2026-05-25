#!/bin/bash

# 设置默认模型名称和工作线程数
MODEL_NAME="google/gemini-3-pro-preview"
WORKERS=20  # 默认工作线程数

# 声明baseURL和APIkey变量
baseURL="https://openrouter.ai/api/v1"  # 替换为实际的基础URL
APIkey="Your_API_KEY"          # 替换为实际的API密钥


# 检查传入参数
if [ $# -ge 1 ]; then
    MODEL_NAME=$1
fi
if [ $# -ge 2 ]; then
    WORKERS=$2
fi

echo "Running evaluation for model: ${MODEL_NAME} with workers: ${WORKERS}"
echo "Start time: $(date)"

# Slot检查命令
echo "=== Running Factor Retrival tasks ==="
cd ./AutoBacktest/001_Retrival/
python main.py --task='prediction' --model_name="${MODEL_NAME}" --base_url="${baseURL}" --api_key="${APIkey}" --workers="${WORKERS}"
python main.py --task='eval' --model_name="${MODEL_NAME}"

# 找表命令
echo "=== Running SQL Generation tasks ==="
cd ../002_SQL/
python main.py --task='prediction' --model_name="${MODEL_NAME}" --base_url="${baseURL}" --api_key="${APIkey}" --workers="${WORKERS}"
python main.py --task='eval' --model_name="${MODEL_NAME}"

# Final_Prediction目录命令
echo "=== Running Back Test tasks ==="
cd ../003_BackTest/
python main.py --task='prediction' --model_name="${MODEL_NAME}" --base_url="${baseURL}" --api_key="${APIkey}" --workers="${WORKERS}"
python main.py --task='eval' --model_name="${MODEL_NAME}"

echo "=== All evaluations completed for model: ${MODEL_NAME} ==="
echo "End time: $(date)"