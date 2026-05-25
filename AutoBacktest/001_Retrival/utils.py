import json
import os
import threading
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
import json
import json5
import re

class PredictResultStorage:
    def __init__(self, file_name='testdata.json', save_lock=None):
        self.current_data = {}
        self.file_name = file_name
        self.save_lock = save_lock or threading.Lock()
    def set_data(self, data_dict):
        self.current_data.update(data_dict)
    def save_data(self):
        with self.save_lock:
            if os.path.exists(self.file_name):
                with open(self.file_name, 'r+') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = []
                    data.append(self.current_data)
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()
            else:
                with open(self.file_name, 'w') as f:
                    json.dump([self.current_data], f, indent=4)
            self.current_data = {}

def extract_content_after_think(response):
    keyword = '</think>'
    start_index = response.find(keyword)
    if start_index == -1:
        return ''
    return response[start_index + len(keyword):]



class FactorResult(BaseModel):
    factors: List[str] = Field(
        description="A list of extracted factor names from the candidate whitelist."
    )


def clean_and_parse_factors(ai_message):

    text = ai_message.content if hasattr(ai_message, 'content') else str(ai_message)

    if "<think>" in text:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    
    if match:
        json_str = match.group(1)
    else:
        bracket_pattern = r"\{.*\}"
        match_bracket = re.search(bracket_pattern, text, re.DOTALL)
        if match_bracket:
            json_str = match_bracket.group(0)
        else:
            json_str = text

    try:
        data_dict = json5.loads(json_str)

        validated_obj = FactorResult.model_validate(data_dict)

        return validated_obj.factors
        
    except Exception as e:
        print(f"⚠️ # Failed to parse Factors: {e},\n<ai_message>: {ai_message}")
        return []

    

def predict_factor_name(data: dict, file_name: str, llm, system_prompt, user_template, save_lock):
    user_message = user_template.format(strategy_text=data['strategy'])
    result = {}
    result['uuid'] = data['uuid']
    prompt = ChatPromptTemplate.from_messages([
            ("system",system_prompt),
            ("user", user_template)
        ])
    try:
        parser = PydanticOutputParser(pydantic_object=FactorResult)
        chain = prompt | llm | clean_and_parse_factors
        response_list = chain.invoke({
                    "strategy_text": data['strategy'],
                    "format_instructions": parser.get_format_instructions()
                })
        result["predict_factors"] = response_list

    except Exception as e:
        print(f"# An exception occurred during API call or processing: {str(e)}")
        return
    saver = PredictResultStorage(file_name, save_lock)
    saver.set_data(result)
    saver.save_data()



def extract_content_after_think(response):
    keyword = '</think>'
    start_index = response.find(keyword)
    if start_index == -1:
        return ''
    return response[start_index + len(keyword):]






def metric_compute(trues: list, preds: list):
    if len(trues) != len(preds):
        return 'Input lengths not equal!'
    
    precision = 0
    precision_all = 0
    recall = 0
    recall_all = 0
    correct_samples = 0
    
    for true_label, pred_label in zip(trues, preds):
        if isinstance(true_label, str):
            true_label = [true_label]
        if isinstance(pred_label, str):
            pred_label = [pred_label]
            
        if sorted(true_label) == sorted(pred_label):
            correct_samples += 1

        for pred in pred_label:
            if pred in true_label:
                precision += 1
            precision_all += 1
        for true in true_label:
            if true in pred_label:
                recall += 1
            recall_all += 1
    
    P = precision / precision_all if precision_all > 0 else 0
    R = recall / recall_all if recall_all > 0 else 0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
    Acc = correct_samples / len(trues) if len(trues) > 0 else 0
    
    print("Evaluation Metrics:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"│ Precision (P) │ {P:.4f} │")
    print(f"│ Recall (R)    │ {R:.4f} │")
    print(f"│ F1 Score      │ {F1:.4f} │")
    print(f"│ Accuracy (Acc)│ {Acc:.4f} │")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    return P, R, F1, Acc



def load_json(file_path, default=None):
    """Safe JSON loader with fallback."""
    if not os.path.exists(file_path):
        return default if default is not None else []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return default if default is not None else []


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)