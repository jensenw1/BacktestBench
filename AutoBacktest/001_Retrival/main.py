import numpy as np
import pandas as pd
import re
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from sklearn import metrics
import json
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from tqdm import tqdm
import ast
from rank_bm25 import BM25Okapi
import nltk
from utils import *
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import List

def create_parser():
    parser = argparse.ArgumentParser(description='Quant')
    parser.add_argument('--task', type=str, default='prediction', help='eval or prediction')
    parser.add_argument('--model_name', type=str, default='qwen3-8b', help='LLM model name')
    parser.add_argument('--base_url', type=str, default='https://openrouter.ai/api/v1', help='LLM base url')
    parser.add_argument('--api_key', type=str, default='sk-xxxxxxxxxx', help='LLM API KEY')
    parser.add_argument('--processCount', type=int, default=1, help='The number of processes sending requests to the LLM.')
    parser.add_argument('--workers', type=int, default=10, help='Request for the number of concurrent LLMs.')
    return parser


parser = create_parser()
args = parser.parse_args()
BASE_URL = args.base_url
MODEL_NAME = args.model_name
OPENAI_API_KEY = args.api_key

def run_prediction(args, root_dir):
    CONCURRENT_WORKERS = args.workers 
    global_save_lock = threading.Lock()
    print(f'Connecting to LLM-{args.model_name}...')

    llm = ChatOpenAI(
        model_name=MODEL_NAME,         
        openai_api_key=OPENAI_API_KEY,  
        base_url=BASE_URL   
    )
    
    template_json_file = f'{root_dir}/AutoQuant/prompts/prompts.json'
    templates = load_json(template_json_file)

    json_file = f'{root_dir}/datasets/test.json'
    datas = load_json(json_file)

    factor_result_saved_file = f'{root_dir}/AutoQuant/001_Retrival/results/{MODEL_NAME} Factor Prediction Results.json'
    result_saved_file = f'{root_dir}/AutoQuant/results/{MODEL_NAME}_results.json'

    processed_queries = set()
    if os.path.exists(factor_result_saved_file):
        try:
            with open(factor_result_saved_file, 'r') as f:
                existing_data = json.load(f)
                processed_queries = {item['uuid'] for item in existing_data if 'uuid' in item}
        except json.JSONDecodeError:
            pass

    remaining_datas = [data for data in datas if data.get('uuid') not in processed_queries]

    def process_data(data):
        try:
            system_prompt = templates["retrival_system_prompt"]
            user_prompt = templates["retrival_user_prompt"]
            predict_factor_name(data = data, file_name = factor_result_saved_file , llm=llm, system_prompt=system_prompt, user_template=user_prompt, save_lock = global_save_lock)
            return True
        except Exception as e:
            print(f"Error processing uuid '{data.get('uuid', 'unknown')}': {str(e)}")
            return False

    with tqdm(total=len(remaining_datas), desc="处理进度", unit="Strategy") as pbar:
        with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
            futures = [executor.submit(process_data, data) for data in remaining_datas]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f" Error: {str(e)}")
                    continue
                pbar.update(1)

    processed_QAs = load_json(factor_result_saved_file)

    processed_uuids = {item['uuid'] for item in processed_QAs if 'uuid' in item}
    required_uuids = {data['uuid'] for data in datas if 'uuid' in data}
    if required_uuids.issubset(processed_uuids):
        processed_dict = {item['uuid']: item for item in processed_QAs}
        for data_item in datas:
            uuid = data_item.get('uuid')
            if uuid in processed_dict:
                processed_item = processed_dict[uuid]
                for key, value in processed_item.items():
                    if key != 'uuid':
                        data_item[key] = value
        save_json(datas, result_saved_file)
        print("🎉🎉🎉 Factor Retrival Task Completed!")
        return True
    else:
        return False


def run_eval(args, root_dir):
    factor_file_path = f"{root_dir}/datasets/factors.json"
    factors = load_json(factor_file_path)
    index_file_path = f"{root_dir}/datasets/indexs.json"
    indexs = load_json(index_file_path)
    all_factor_names = list(factors.keys())
    indexs_name = list(indexs.keys())
    all_factor_names.extend(indexs_name)
    nltk.download('punkt_tab')
    
    # Corpus
    corpus = all_factor_names
    # tokenized_corpus = [list(jieba.cut(doc, cut_all=True)) for doc in corpus]
    tokenized_corpus = [nltk.word_tokenize(doc.lower()) for doc in corpus]
    # Build Index
    bm25 = BM25Okapi(tokenized_corpus)

    result_saved_file = f'{root_dir}/AutoQuant/results/{MODEL_NAME}_results.json'
    datas = load_json(result_saved_file)

    predicts = []
    bm_25s = []
    trues = []

    for data in tqdm(datas):
        predict = data["predict_factors"]
        bm_25 = []  # Initialize for this iteration

        if isinstance(data['predict_factors'], list):
            # 1. Logic for Empty Prediction: Use Strategy -> Top 5
            if len(predict) == 0:
                fallback_query = data.get('strategy', '')
                print(f"⚠️ Predicted factors are empty, falling back to policy text retrieval: {fallback_query[:15]}...")
                
                # Tokenize the strategy text
                tokenized_query = nltk.word_tokenize(fallback_query.lower())
                
                # Retrieve Top 5 directly
                bm_25 = bm25.get_top_n(tokenized_query, corpus, n=5)
                
                # Option: Update 'predict' to match the found factors so metric_compute doesn't see empty
                predict = bm_25 

            # 2. Logic for Existing Predictions: Clean -> Top 1 Check
            else:
                current_bm25_list = []
                for i, pred in enumerate(predict):
                    # Cleaning strings
                    pred = str(pred).replace("[", "").replace("]", "").replace("'", "").replace('"', '').replace('\\', "").replace("\n", "")
                    predict[i] = pred # Update the cleaned string in the list

                    if pred not in all_factor_names:
                        tokenized_query = nltk.word_tokenize(pred.lower())
                        result = bm25.get_top_n(tokenized_query, corpus, n=1)
                        current_bm25_list.append(result[0])
                    else:
                        current_bm25_list.append(pred)

                predict = list(set(predict))
                bm_25 = list(set(current_bm25_list))

        predicts.append(predict)
        bm_25s.append(bm_25)
        
        true = data['factors']
        trues.append(true)
        data['predict_factors(BM25)'] = bm_25

    metric_compute(trues, predicts)
    print('2、The accuracy of after BM25.')
    metric_compute(trues, bm_25s)
    
    with open(result_saved_file, 'w', encoding='utf-8') as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)



def main():
    args = create_parser().parse_args()
    ROOT_DIRTORY = re.search(r'^(.*?BacktestBench)', os.getcwd()).group(1)
    if args.task == 'prediction':
        while not run_prediction(args, ROOT_DIRTORY):
            pass
    elif args.task == 'eval':
        run_eval(args, ROOT_DIRTORY)


if __name__ == "__main__":
    main()