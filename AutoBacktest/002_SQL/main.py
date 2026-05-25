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
from tabulate import tabulate
import os
from tqdm import tqdm
import ast
import re
import psycopg2
from psycopg2 import sql, DatabaseError
from utils import *
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor

def create_parser():
    # basic config
    parser = argparse.ArgumentParser(description='SQL Generation')
    parser.add_argument('--task', type=str, default='test', help='transform, eval or prediction')
    parser.add_argument('--model_name', type=str, default='gpt-oss-20b', help='LLM model name')
    parser.add_argument('--base_url', type=str, default='https://openrouter.ai/api/v1', help='LLM base url')
    parser.add_argument('--api_key', type=str, default='sk-xxxxxxxxxxx', help='LLM API KEY')
    parser.add_argument('--workers', type=int, default=5, help='Request for the number of concurrent LLMs.')
    parser.add_argument('--temperature', type=float, default=0.6, help='Sampling temperature for LLM generation')
    return parser


def run_prediction(args, root_dir):
    global_save_lock = threading.Lock()

    factor_file_path = f"{root_dir}/datasets/factors.json"
    factors = load_json(factor_file_path)
    index_file_path = f"{root_dir}/datasets/indexs.json"
    indexs = load_json(index_file_path)
    all_factors = {}
    all_factors.update(factors)
    all_factors.update(indexs)
    
    template_json_file = f'{root_dir}/prompts/prompts.json'
    
    result_saved_file = f'{root_dir}/results/{args.model_name}_results.json'

    
    SQL_result_saved_file = f'{root_dir}/002_SQL/results/{args.model_name}_results.json'
    
    templates = load_json(template_json_file)
    all_datas = load_json(result_saved_file)
    processed_datas = load_json(SQL_result_saved_file)
    processed_uuids = [data["uuid"] for data in processed_datas]
    
    llm = ChatOpenAI(
        model_name=args.model_name,
        openai_api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature
    )
    print(f'🟢 Connecting to LLM-{args.model_name}.........🔌')

    remaining_datas = [data for data in all_datas if data.get('uuid') not in processed_uuids]
    def process_data(data):
        try:
            system_prompt = templates['sql_system_prompt']
            user_prompt = templates['sql_user_prompt']
            predict_SQL(data=data, file_name=SQL_result_saved_file, llm=llm, system_prompt=system_prompt, user_prompt=user_prompt, factors=all_factors, save_lock = global_save_lock)
            return True
        except Exception as e:
            print(f"Error processing uuid '{data.get('uuid', 'unknown')}': {str(e)}")
            return False
    with tqdm(total=len(remaining_datas), desc="处理进度", unit="Strategy") as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_data, data) for data in remaining_datas]
            for future in futures:
                try:
                    future.result()
                    pbar.update(1)
                except Exception as e:
                    print(f"Handle exceptions: {str(e)}")
                    continue
    processed_QAs = load_json(SQL_result_saved_file)
    processed_uuids = {item['uuid'] for item in processed_QAs if 'uuid' in item}
    required_uuids = {data['uuid'] for data in all_datas if 'uuid' in data}
    if required_uuids.issubset(processed_uuids):
        processed_dict = {item['uuid']: item for item in processed_QAs}
        for data_item in all_datas:
            uuid = data_item.get('uuid')
            if uuid in processed_dict:
                processed_item = processed_dict[uuid]
                for key, value in processed_item.items():
                    if key != 'uuid': 
                        data_item[key] = value
        save_json(all_datas, result_saved_file)
        print("🎉🎉🎉 SQL Prediction Task Completed!")
        return True
    else:
        return False


def run_eval(args, root_dir):
    result_saved_file = f'{root_dir}/results/{args.model_name}_results.json'
    datas = load_json(result_saved_file)
    
    if not datas:
        print("⚠️ No data found to evaluate.")
        return

    total_samples = len(datas)
    correct_count = 0 
    unexecutable_count = 0
    executor = PostgresQueryExecutor() 
    
    print(f"🚀 Starting evaluation on {total_samples} samples...")
    

    for result in tqdm(datas, desc="Evaluating", unit="sample", ncols=100):
        try:
            # 这里的 execute_sql 返回 (preview_str, df) 或者 error_str
            res_true = executor.execute_sql(result['SQL_statement'])
            
            if isinstance(res_true, tuple):
                _, df_true = res_true
            else:
                print(f"⚠️ GT SQL failed for UUID {result.get('uuid')}: {res_true}")
                df_true = None
        except Exception as e:
            print(f"⚠️ GT Execution Critical Error: {e}")
            df_true = None

        pred_sql = result.get('predict_SQL', '')
        df_pred = None
        is_executable = False
        
        if pred_sql:
            try:
                res_pred = executor.execute_sql(pred_sql)
                if isinstance(res_pred, tuple):
                    _, df_pred = res_pred
                    is_executable = True
                else:
                    is_executable = False
            except Exception:
                is_executable = False
        
        if not is_executable:
            unexecutable_count += 1
  
        is_consistent = False
        if df_true is not None and df_pred is not None:
            try:
                if df_true.shape != df_pred.shape:
                    is_consistent = False
                    fail_reason = f"Shape Mismatch: GT {df_true.shape} vs Pred {df_pred.shape}"
                
                elif set(df_true.columns) != set(df_pred.columns):
                    is_consistent = False
                    fail_reason = f"Columns Mismatch: GT {list(df_true.columns)} vs Pred {list(df_pred.columns)}"
                    
                else:
                    cols = sorted(list(df_true.columns))
                    df_true_sorted = df_true[cols]
                    df_pred_sorted = df_pred[cols]
                    
                    df_true_sorted = df_true_sorted.sort_values(by=cols).reset_index(drop=True)
                    df_pred_sorted = df_pred_sorted.sort_values(by=cols).reset_index(drop=True)
                    
                    pd.testing.assert_frame_equal(
                        df_true_sorted, 
                        df_pred_sorted, 
                        check_dtype=False, 
                        check_index_type=False 
                    )
                    is_consistent = True

            except AssertionError:
                is_consistent = False
            except Exception as e:
                is_consistent = False
        result['predict_correctness'] = is_consistent
        
        if is_consistent:
            correct_count += 1
            
        del df_true
        del df_pred


    execution_accuracy = correct_count / total_samples if total_samples > 0 else 0
    ecr = (total_samples - unexecutable_count) / total_samples if total_samples > 0 else 0

    print("\n" + "="*40)
    print("📊 Evaluation Summary (DataFrame Strict Mode)")
    print("="*40)
    stats_table = [
        ["Metric", "Value"],
        ["Total Samples", total_samples],
        ["Correct Count", correct_count],
        ["Unexecutable", unexecutable_count],
        ["Execution Accuracy", f"{execution_accuracy:.2%}"],
        ["ECR", f"{ecr:.2%}"]
    ]
    print(tabulate(stats_table, headers="firstrow", tablefmt="fancy_grid"))
    print("="*40 + "\n")




def main():
    args = create_parser().parse_args()
    ROOT_DIRTORY = re.search(r'^(.*?AutoBacktest)', os.getcwd()).group(1)
    if args.task == 'prediction':
        while not run_prediction(args, ROOT_DIRTORY):
            pass
    elif args.task == 'eval':
        run_eval(args, ROOT_DIRTORY)

if __name__ == "__main__":
    main()