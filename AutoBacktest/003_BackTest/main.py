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
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score


def create_parser():
    # basic config
    parser = argparse.ArgumentParser(description='Back Test')
    parser.add_argument('--task', type=str, default='test', help='transform, eval or prediction')
    parser.add_argument('--model_name', type=str, default='gpt-oss-20b', help='LLM model name')
    parser.add_argument('--base_url', type=str, default='https://openrouter.ai/api/v1', help='LLM base url')
    parser.add_argument('--api_key', type=str, default='sk-xxxxxxxxx', help='LLM API KEY')
    parser.add_argument('--workers', type=int, default=20, help='Request for the number of concurrent LLMs.')
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

    
    BackTest_result_saved_file = f'{root_dir}/003_BackTest/results/{args.model_name}_results.json'
    
    templates = load_json(template_json_file)
    all_datas = load_json(result_saved_file)
    processed_datas = load_json(BackTest_result_saved_file)
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
            prompts = templates
            BackTest(data=data, file_name=BackTest_result_saved_file, llm=llm, prompts=prompts, factors=all_factors, save_lock = global_save_lock)
            return True
        except Exception as e:
            print(f"Error processing uuid '{data.get('uuid', 'unknown')}': {str(e)}")
            return False
    with tqdm(total=len(remaining_datas), desc="Processing Progress", unit="Strategy") as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_data, data) for data in remaining_datas]
            for future in as_completed(futures):
                try:
                    future.result() 
                except Exception as e:
                    print(f"Handling Exception: {str(e)}")
                finally:
                    pbar.update(1)

    processed_QAs = load_json(BackTest_result_saved_file)
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
    df = pd.DataFrame(datas)
    results_summary = {}
    total_correct_global = 0
    total_count_global = 0
    metrics_df = df[df['strategy_type'] == 'metrics_calculation'].copy()
    if not metrics_df.empty:
        correct = 0
        total = len(metrics_df)
        for _, row in metrics_df.iterrows():
            try:
                a_str = str(row['answer']).replace(',', '')
                p_str = str(row['pred_answer']).replace(',', '')
                a, p = float(a_str), float(p_str)
                
                if a == 0: 
                    is_corr = abs(p) < 1e-6
                else: 
                    d1 = abs((p - a) / a)
                    d2 = abs((p * 100 - a) / a)
                    d3 = abs((p / 100 - a) / a)
                    is_corr = min(d1, d2, d3) <= 0.001
                
                if is_corr: correct += 1
            except: 
                pass
        
        results_summary['metrics_calculation'] = {"Count": total, "Accuracy": correct/total if total > 0 else 0}
        total_correct_global += correct
        total_count_global += total
    
    # Parameter Confirmation (Intelligent Numeric Matching)
    param_df = df[df['strategy_type'] == 'parameter_confirmation'].copy()
    if not param_df.empty:
        correct_count = 0 
        for _, row in param_df.iterrows():
            ans = str(row['answer']).strip()
            pred = str(row['pred_answer']).strip()
            is_match = False
            
            if ans == pred:
                is_match = True
            else:
                try:
                    if abs(float(ans) - float(pred)) < 1e-6:
                        is_match = True
                except: pass
            
            if is_match: correct_count += 1
            
        results_summary['parameter_confirmation'] = {
            "Count": len(param_df),
            "Accuracy": correct_count / len(param_df) if len(param_df) > 0 else 0
        }
        total_correct_global += correct_count
        total_count_global += len(param_df)
    
    # 3. Stock & Strategy 
    other_cats = ["ticker_selection", "strategy_selection"]
    for cat in other_cats:
        sub_df = df[df['strategy_type'] == cat].copy()
        if not sub_df.empty:
            y_true = sub_df['answer'].astype(str).str.strip()
            y_pred = sub_df['pred_answer'].astype(str).str.strip()
            
            if cat == "strategy_selection":
                y_true = y_true.str.upper()
                y_pred = y_pred.str.upper()
            

            correct_count = (y_true == y_pred).sum()
            total = len(sub_df)
            
            results_summary[cat] = {
                "Count": total,
                "Accuracy": correct_count / total if total > 0 else 0
            }
            total_correct_global += correct_count
            total_count_global += total
    
    if total_count_global > 0:
        overall_acc = total_correct_global / total_count_global
        results_summary['Overall'] = {
            'Count': total_count_global,
            'Accuracy': overall_acc
        }

    print_table(results_summary, title="Overall Performance by Strategy Type")


    metrics_breakdown_summary = {}

    if not metrics_df.empty and 'target_function' in metrics_df.columns:
        # Group by target_function
        grouped = metrics_df.groupby('target_function')
        
        for func_name, group in grouped:
            correct = 0
            total = len(group)
            
            for _, row in group.iterrows():
                try:
                    # Cleaning strings
                    a_str = str(row['answer']).replace(',', '').strip()
                    p_str = str(row['pred_answer']).replace(',', '').strip()
                    
                    # Convert to float
                    a, p = float(a_str), float(p_str)
                    
                    if a == 0: 
                        is_corr = abs(p) < 1e-6
                    else: 
                        # Tolerance: original, x100, /100
                        d1 = abs((p - a) / a)
                        d2 = abs((p * 100 - a) / a)
                        d3 = abs((p / 100 - a) / a)
                        is_corr = min(d1, d2, d3) <= 0.001
                    
                    if is_corr: correct += 1
                except: 
                    pass # Parse failure counts as incorrect
            
            metrics_breakdown_summary[func_name] = {
                "Count": total, 
                "Accuracy": correct / total if total > 0 else 0
            }

        # Optional: Add an Overall row for just metrics_calculation
        total_metrics_count = len(metrics_df)
        total_metrics_correct = sum([res['Count'] * res['Accuracy'] for res in metrics_breakdown_summary.values()])
        
        metrics_breakdown_summary['Overall (Metrics Only)'] = {
            "Count": total_metrics_count,
            "Accuracy": total_metrics_correct / total_metrics_count if total_metrics_count > 0 else 0
        }

        # Print Breakdown Table
        print_table(metrics_breakdown_summary, title="Metrics Breakdown by Target Function")



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