import json
import os
import psycopg2
from psycopg2 import sql, DatabaseError
import numpy as np
import threading
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from typing import Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from pydantic import BaseModel
from typing import Union, List, Tuple, Optional
from langchain_core.messages import ToolMessage
import re
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage
import ast
from typing import Optional, Any
from tools import executePostgresQuery, PostgresQueryExecutor





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



def predict_SQL(data: dict, file_name: str, llm, system_prompt, user_prompt, factors, save_lock):
    result = {}
    result['uuid'] = data['uuid']
    factor_context = ''
    for item in data['predict_factors(BM25)']:
        factor_context += f"'{item}': {factors[item]['short_code']}\n"

    user_message = user_prompt.format(
        strategy=data['strategy'],
        factors=factor_context
    )
    agent = create_react_agent(llm, tools=[executePostgresQuery])
    
    try:
        response = agent.invoke(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            }
        )
        sql_statement, full_ai_message = extract_sql_from_response(response)

        if sql_statement:
             sql_statement = sql_statement.strip()
        
        result["predict_SQL"] = sql_statement
        result['predict_SQL_message'] = full_ai_message
        
    except Exception as e:
        print(f"API call or parsing failed, save operation was skipped | Error details: {str(e)}")
        return

    saver = PredictResultStorage(file_name, save_lock)
    saver.set_data(result)
    saver.save_data()



def extract_sql_from_response(response):
    """
    Extract the final SQL statement from LangChain Agent's response.
    Supports parsing plain JSON text or JSON within Markdown code blocks.
    """
    try:
        # 1. Get the content of the last AI message
        # Different LangChain versions may have response as a dictionary or object, add compatibility here
        if isinstance(response, dict) and "messages" in response:
            ai_message = response["messages"][-1].content
        elif hasattr(response, "get") and response.get("output"):
            ai_message = response["output"]
        else:
            ai_message = response['messages'][-1].content

        # 2. Clean up any potential  tags (commonly seen in DeepSeek/R1 models)

        if "<think>" in ai_message:
            ai_message = re.split(r'</think>', ai_message, flags=re.IGNORECASE)[-1].strip()

        # 3. Attempt to extract JSON string
        # Match ```json ... ``` or {...} structures
        json_match = re.search(r'\{.*"sql_statement":.*\}', ai_message, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            return data.get("sql_statement", "").strip(), ai_message
        else:
            print("Warning: No JSON format found in AI response. Attempting raw extraction.")
            return ai_message.strip(), ai_message

    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from message: {ai_message}")
        return "", ai_message
    except Exception as e:
        print(f"Error extracting SQL: {str(e)}")
        return "", str(response)


def extract_content_after_think(response):
    keywords = ['</think>', '</seed:think>']
    for keyword in keywords:
        start_index = response.find(keyword)
        if start_index != -1:
            return response[start_index + len(keyword):]
    return ''




def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k /np.arange(n - c + 1, n + 1))

def calculate_average_pass1(data):
    total_pass1 = sum(item['pass1'] for item in data.values())
    count = len(data)
    if count == 0:
        return 0
    return total_pass1 / count

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
