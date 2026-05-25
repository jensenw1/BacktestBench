import psycopg2
import pandas as pd
from psycopg2 import DatabaseError
from langchain.tools import tool
from typing import Union, Tuple, Dict, Any

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

    def execute_sql(self, sql_statement) -> Union[Tuple[str, pd.DataFrame], str]:
        try:
            self.connect()
            self.cur.execute("SET statement_timeout = 10000;") 
            self.cur.execute(sql_statement)
            if self.cur.description is None:
                self.conn.commit()
                return "Query executed successfully (no rows returned).", pd.DataFrame()

            result = self.cur.fetchall()
            headers = [desc[0] for desc in self.cur.description]
            df = pd.DataFrame(result, columns=headers)
            
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date']) 
                df = df.sort_values(by='trade_date').reset_index(drop=True)

            self.conn.commit()
            n_head, n_tail = 5, 5
            total = len(df)
            # Generate preview information (header and footer)
            preview_str = (
                f"✅ SQL Query Executed Successfully!\n"
                f"📊 Total Rows Retrieved: {total}\n\n"
                f"Here is a preview of the first {min(n_head, total)} and last {min(n_tail, total)} rows:\n\n"
                f"⬆️ [Head: First {min(n_head, total)} Rows]\n{df.head(n_head).to_string(index=False)}\n\n"
                f"⬇️ [Tail: Last {min(n_tail, total)} Rows]\n{df.tail(n_tail).to_string(index=False)}"
            )
            return preview_str, df

        except Exception as e:
            error_msg = f"Error executing SQL: {str(e)}"
            return error_msg
        finally:
            self.close()

    def close(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

@tool
def executePostgresQuery(sql_statement: str) -> Dict[str, str]:
    """
    Executes a SQL statement against the 'stocks' database for data inspection and validation.

    Use this tool to:
    1. Verify if the data requested by the user actually exists.
    2. Check table schemas, column names, and data formats (e.g., date formats).
    3. Preview a sample of the data to ensure your SQL logic is correct before performing full analysis.

    Args:
        sql_statement (str): A valid SQL SELECT statement.

    Returns:
        Dict[str, str]:
        - "status": "success" or "error".
        - "content": 
            If success: A text summary including total row count and a preview (Head & Tail) of the retrieved data. Use this to verify if the data meets the user's requirements.
            If error: The database error message.
    """
    executor = PostgresQueryExecutor()
    result = executor.execute_sql(sql_statement)

    if isinstance(result, tuple):
        preview_str, df = result
        return {
            "status": "success",
            "content": preview_str
        }
    else:
        return {
            "status": "error",
            "content": result
        }