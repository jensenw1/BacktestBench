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
from tools import *
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import StructuredTool



class PostgresQueryExecutor:
    def __init__(self, database='stocks', host='127.0.0.1', user='quant', password='123456', port="5432"):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.conn = None
        self.cur = None

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            self.cur = self.conn.cursor()
        except DatabaseError as e:
            raise RuntimeError(f"Database connection error for {self.database}: {e}")

    def execute_sql(self, sql_statement):
        try:
            self.connect()
            self.cur.execute(sql_statement)
            result = self.cur.fetchall()
            headers = [desc[0] for desc in self.cur.description] if self.cur.description else []
            df = pd.DataFrame(result, columns=headers)
            
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date']) 
                df = df.sort_values(by='trade_date').reset_index(drop=True)
            
            self.conn.commit()
            return df
            
        except Exception as e:
            print(f"Error executing SQL on {self.database}: {e}")
            return pd.DataFrame()
        finally:
            self.close()

    def close(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

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


# "metrics_calculation"
class ClassA_FinalAnswerFormat(BaseModel):
    answer: float = Field(
        description="Your final answer is the value of the indicator requested by the user."
    )

# "ticker_selection"
class ClassB_FinalAnswerFormat(BaseModel):
    answer: str = Field(
        description="The exact name of the company that meets the strategy criteria (e.g., yields the lowest/highest specified metric)."
    )

# "parameter_confirmation"
class ClassC_FinalAnswerFormat(BaseModel):
    answer: float = Field(
        description="The specific parameter value from the candidate list that resulted in the best strategy performance."
    )

# "strategy_selection"
class ClassD_FinalAnswerFormat(BaseModel):
    answer: str = Field(
        description="The identifier of the best performing strategy. Return ONLY the uppercase letter: 'A', 'B', 'C' or 'D'."
    )


def BackTest(data: dict, file_name: str, llm, prompts, factors, save_lock):
    strategy = data['strategy']
    strategy_type = data['strategy_type']
    SQL_statement = data['predict_SQL']
    result = {}
    result['uuid'] = data['uuid']

    parserA = JsonOutputParser(pydantic_object=ClassA_FinalAnswerFormat)
    parserB = JsonOutputParser(pydantic_object=ClassB_FinalAnswerFormat)
    parserC = JsonOutputParser(pydantic_object=ClassC_FinalAnswerFormat)
    parserD = JsonOutputParser(pydantic_object=ClassD_FinalAnswerFormat)
    factor_context = ''
    for item in data['predict_factors(BM25)']:
        factor_context += f"Factor Name: {item}\n" + f"Short Code: {factors[item]['short_code']}\n"
    
    executor = PostgresQueryExecutor()
    df = executor.execute_sql(SQL_statement)
    executor.close()
    if df is None or df.empty:
        result['pred_answer'] = 'SQL_RESULT_EMPTY_ERROE'
        result['response'] = 'SQL_RESULT_EMPTY_ERROE'
        # 保存逻辑保持不变
        saver = PredictResultStorage(file_name, save_lock)
        saver.set_data(result)
        saver.save_data()
        return
    system_prompt = prompts['backtest_system_prompt']
    user_prompt = prompts['backtest_user_prompt']
    dataframe_heads = df.head().to_markdown()
    if strategy_type == "metrics_calculation":
        format_instructions = parserA.get_format_instructions()

    elif strategy_type == "ticker_selection":
        format_instructions = parserB.get_format_instructions()

    elif strategy_type == "parameter_confirmation":
        format_instructions = parserC.get_format_instructions()

    elif strategy_type == "strategy_selection":
        format_instructions = parserD.get_format_instructions()

    else:
        pass
    formatted_system_prompt = system_prompt.format(
        dataframe_heads=dataframe_heads
    )

    formatted_user_prompt = user_prompt.format(
        strategy=strategy,
        factors=factor_context,
        format_instructions=format_instructions
    )
    try:
        df_tool = create_python_repl_tool(df)
        agent = create_react_agent(llm, tools=[df_tool])

        response = agent.invoke({
            "messages": [
                ("system", formatted_system_prompt),
                ("user", formatted_user_prompt)
            ]
        })
        if "messages" in response:
            final_message = response["messages"][-1]
            final_answer = final_message.content
        else:
            final_answer = response.get("output", "")
        result['response'] = str(response)
        extracted_answer = extract_answer_from_response(final_answer)
        
        result['pred_answer'] = extracted_answer if extracted_answer is not None else "ERROR"
    except Exception as e:
        print(f"API call or parsing failed, save operation has been skipped | Error details:{str(e)}")
        return
    # 保存逻辑保持不变
    saver = PredictResultStorage(file_name, save_lock)
    saver.set_data(result)
    saver.save_data()



def extract_answer_from_response(response: str):
    if not isinstance(response, str) or not response:
        return None
    for kw in ['</think>', '</seed:think>']:
        idx = response.find(kw)
        if idx != -1:
            response = response[idx + len(kw):]
    pattern = r'\{[^{}]*"answer"[^{}]*\}'
    matches = list(re.finditer(pattern, response, flags=re.DOTALL))
    if not matches:
        return None

    json_text = matches[-1].group(0).strip()
    try:
        data = json.loads(json_text)
    except Exception:
        return None

    if not isinstance(data, dict) or 'answer' not in data:
        return None

    answer = data['answer']
    if isinstance(answer, str):
        s = answer.strip()
        try:
            return float(s)
        except Exception:
            return answer
    return answer





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


def print_table(results_summary):
    if not results_summary: return
    res_df = pd.DataFrame(results_summary).T
    cols = ['Count', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    for c in cols: 
        if c not in res_df: res_df[c] = np.nan
    res_df = res_df[cols]

    res_df['Count'] = res_df['Count'].fillna(0).astype(int)
    for c in cols[1:]:
        res_df[c] = res_df[c].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")
        
    print("\n" + "="*80)
    print("="*80)
    print(res_df.to_markdown())
    print("="*80 + "\n")

def print_table(results_summary, title="Evaluation Results"):
    if not results_summary: return
    res_df = pd.DataFrame(results_summary).T

    cols = ['Count', 'Accuracy']
    for c in cols: 
        if c not in res_df: res_df[c] = np.nan
    res_df = res_df[cols]

    res_df['Count'] = res_df['Count'].fillna(0).astype(int)
    res_df['Accuracy'] = res_df['Accuracy'].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")
        
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)
    print(res_df.to_markdown())
    print("="*80 + "\n")