import psycopg2
import pandas as pd
from psycopg2 import DatabaseError
from langchain.tools import tool
from typing import Union, Tuple, Dict, Any
from pydantic import BaseModel, Field
import numpy as np
import sys
from langchain_core.tools import StructuredTool
import io


class PythonCodeInput(BaseModel):
    code: str = Field(description="valid python code to execute on the dataframe named 'df'. e.g. df.head()")

def create_python_repl_tool(dataframe: pd.DataFrame):
    shared_namespace = {
        "df": dataframe, 
        "pd": pd, 
        "np": np
    }

    def run_code(code: str):
    
        return execute_dataframe_code(dataframe, code, namespace=shared_namespace)

    return StructuredTool.from_function(
        func=run_code,
        name="python_repl_ast",
        description="...",
        args_schema=PythonCodeInput
    )


def execute_dataframe_code(df: pd.DataFrame, code: str, namespace: dict = None) -> str:
    """
    Execute a Python code string on the given DataFrame.
  
    Args:
        df (pd.DataFrame): Input pandas DataFrame, default variable name in code is 'df'.
        code (str): Python code string to be executed.
      
    Returns:
        str: String representation of execution result, or error message.
    """
    # 1. **Preparing execution environment, injecting dataframe into local variables**
    local_vars = {"df": df, "pd": pd, "np": np}  # **Also inject `pd` so that users can use pandas functions**
    if namespace is None:
        local_vars = {"df": df, "pd": pd, "np": np}
    else:
        local_vars = namespace

    try:
        result = eval(code, {}, local_vars)
        if result is not None:
            return f"Result of code execution: {str(result)}"
        else:
            pass
    except SyntaxError:
        # If it is not an expression (for example, containing assignments or multi-line code), execute it as a code block.
        pass
    except Exception as e:
        return f"Execution Error: {str(e)}"
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    try:
        exec(code, local_vars)
        sys.stdout = old_stdout
        
        output = redirected_output.getvalue().strip()
        if output:
            return output
        else:
            return "Code executed successfully, but produced no output. (Did you forget to print?)"
    except Exception as e:
        sys.stdout = old_stdout
        return f"Execution Error: {str(e)}"
