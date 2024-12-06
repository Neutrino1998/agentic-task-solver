import ast
import pandas as pd
from langchain_core.tools import tool
# Log
from my_logger import Logger, LOG_LEVEL, LOG_PATH, LOG_FILE
# Initialize Logger
logger = Logger(name="AgentLogger", level=LOG_LEVEL, log_path=LOG_PATH, log_file=LOG_FILE)

class BlacklistCodeInterpreter:
    """
    A code interpreter that uses a blacklist to reject unsafe operations.
    Only the blacklisted operations and imports are blocked.
    """
    
    # List of blacklisted functions and modules (unsafe operations)
    BLACKLISTED_FUNCTIONS = {
        'open', 'exec', 'eval', 'compile', 'globals', 'locals', 'getattr', 'setattr', 'delattr',
        'os', 'sys', 'subprocess', 'socket', 'shutil', 'pdb', 'traceback', 'requests', 'urllib', 'json'
    }

    BLACKLISTED_MODULES = {
        'os', 'sys', 'subprocess', 'socket', 'shutil', 'requests', 'urllib', 'json'
    }

    def __init__(self):
        self.environment = {"_expression_results": {}}

    def is_safe(self, tree: ast.AST) -> bool:
        """
        Perform a security check on the abstract syntax tree (AST).
        Reject operations and imports from the blacklist.
        """
        for node in ast.walk(tree):
            # Check for blacklisted function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in self.BLACKLISTED_FUNCTIONS:
                    logger.logger.warning(f"⚠️ Unsafe function call detected: {node.func.id}")
                    return False
            # Check for blacklisted imports
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                if any(alias.name in self.BLACKLISTED_MODULES for alias in node.names):
                    logger.logger.warning(f"⚠️ Unsafe import detected: {', '.join([alias.name for alias in node.names])}")
                    return False
            # Additional checks for dangerous operations can be added here
        return True

    def clean_environment(self, env: dict) -> dict:
        """
        Clean the environment dictionary to remove non-serializable or unsafe objects.

        Args:
            env: The environment dictionary to clean.

        Returns:
            A cleaned dictionary with only serializable objects.
        """
        from tools.data_loader import generate_dataframe_schema
        clean_env = {}
        for key, value in env.items():
            if isinstance(value, (int, float, str, list, dict, bool, type(None))):
                clean_env[key] = value
            elif isinstance(value, pd.DataFrame):
                clean_env[key] = {"dataframe_schema": generate_dataframe_schema(value)}
            else:
                # Convert non-serializable objects to a string representation
                clean_env[key] = repr(value)
        return clean_env

    def execute(self, code: str, clean_env: bool = False) -> dict:
        """
        Execute the provided Python code in a controlled environment, handling success and failure.
        
        Args:
            code: The Python code to be executed.

        Returns:
            A dictionary with the result or an error message.
        """
        try:
            # Parse the code into an abstract syntax tree (AST)
            tree = ast.parse(code)
            
            # Get lines of the source code for expression tracking
            source_lines = code.splitlines()

            # Modify the AST to capture unbound expression results
            new_body = []
            for node in tree.body:
                if isinstance(node, ast.Expr):
                    # Capture the source code for this expression
                    expr_code = source_lines[node.lineno - 1].strip()
                    
                    # Append the expression result to `_expression_results` as a key-value pair
                    assign_node = ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="_expression_results", ctx=ast.Load()),
                                attr="__setitem__",
                                ctx=ast.Load()
                            ),
                            args=[
                                ast.Constant(value=expr_code),  # Key: expression as string
                                node.value  # Value: expression result
                            ],
                            keywords=[]
                        )
                    )
                    new_body.append(assign_node)
                else:
                    new_body.append(node)
            tree.body = new_body
            ast.fix_missing_locations(tree)

            # Security check
            if not self.is_safe(tree):
                return {"error": "Security check failed: Unsafe code detected."}
            
            # Compile the AST into a code object
            code_object = compile(tree, '<string>', 'exec')
            
            # Execute the code within a restricted environment
            exec(code_object, self.environment)
            
            # Pop '__builtins__' from the environment to avoid leaking it
            self.environment.pop('__builtins__', None)
            if not self.environment.get("_expression_results"):
                self.environment.pop("_expression_results", None)
            
            # Clean or return the environment based on the flag
            if clean_env:
                # Clean the environment to ensure JSON serializability
                cleaned_env = self.clean_environment(self.environment)
                return {"success": cleaned_env}
            else:
                return {"success": self.environment}
        
        except Exception as e:
            logger.logger.warning(f"Code execution error: {str(e)}")
            # Return the exception message if something went wrong
            return {"error": str(e)}

@tool
def execute_python_code(code: str, workspace: dict = {}) -> dict:
    """
    Execute Python code safely and return the result or error message.
    
    Args:
        code: Python code to execute.
    
    Returns:
        A dictionary containing the result: {"successs": {dict of all used variables}}
        or error message: {"error": "error message."}

    Example:

        .. code-block:: python

            code = "import math\\na = math.sqrt(4)\\nb = 2\\nresult = a + b"
            result = execute_python_code.invoke(code)
            # The result will be:
            result = {'success': {'math': "<module 'math' (built-in)>", 'a': 2.0, 'b': 2, 'result': 4.0}}
    """
    interpreter = BlacklistCodeInterpreter()
    return {"result": interpreter.execute(code, clean_env=True)}

class DataAnalysisInterpreter(BlacklistCodeInterpreter):
    """
    Extends BlacklistCodeInterpreter to handle pandas DataFrames for data analysis tasks.
    """

    def execute_analysis_code(
        self, 
        code: str, 
        dataframes: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Executes Python code with DataFrame inputs from a dictionary and returns analysis results.

        Parameters:
            code (str): The Python code to execute.
            dataframes (Dict[str, pd.DataFrame]): 
                Dictionary where keys are variable names and values are DataFrames.

        Returns:
            A dictionary containing the result: {"successs": {dict of all used variables}}
            or error message: {"error": "error message."}
        """
        # Load DataFrames into the environment
        if not isinstance(dataframes, dict):
            raise ValueError("dataframes must be a dictionary of DataFrame objects.")

        self.environment.update(dataframes)
        self.environment["pd"] = pd  # Add pandas to the execution environment

        # Execute the code safely
        result = self.execute(code, clean_env=False)

        return result
        

@tool
def execute_python_code_with_df(code: str, workspace: dict = {}) -> dict:
    """
    Execute Python code safely with provided DataFrames cached in workspace and return the results.

    Args:
        code (str): Python code to execute. This should reference DataFrame variables.
  
    Returns:
        A dictionary containing the result: {"successs": {dict of all used variables}}
        * Note the result will only contain schema of a dataframe! If you want to get any value,
          store them in a serializable variable like a dict.

    Example:

        .. code-block:: python
            # cached dataframe:
            dataframes = {
                "sales_data": pd.DataFrame({"region": ["North", "South"], "sales": [300, 400]}),
                "targets": pd.DataFrame({"region": ["North", "South"], "target": [350, 390]})
            }
            # your code should be like:
            code = "merged_data = pd.merge(sales_data, targets, on='region') # you will only get schema of merged_data
                    difference_dict = {} # however, you can get the value of difference_dict
                    for index, row in merged_data.iterrows():
                        region = row['region']
                        sales_diff = row['sales'] - row['target']
                        difference_dict[region] = sales_diff"
            result = execute_python_code_with_dataframes.invoke(code)
            # The result will be:
            result = {"success": {"sales_data": <pandas.DataFrame>, "targets": <pandas.DataFrame>, 
                      "difference_dict": {
                            "North": -50,
                            "South": 10
                        },...}}
    """
    from utility.manage_workspace import filter_workspace_content
    # get pd.DataFrame from workspace
    filtered_workspace = filter_workspace_content(workspace, content_type=pd.DataFrame)
    # format dataframes to be Dict[str, pd.DataFrame]
    dataframes = {key:value.get("content") for key, value in filtered_workspace.items()}
    interpreter = DataAnalysisInterpreter()
    result = interpreter.execute_analysis_code(code, dataframes)
    if "error" in result:
        # If an error occurred, log and return None
        return {"result": result, "workspace": {}}

    # Get cleaned environment to response to agent
    serializable_result = {"success": interpreter.clean_environment(result.get("success", {}))}
    # Extract DataFrame from the raw environment
    extract_dataframes = {
        key:value for key, value in result.get("success", {}).items() if isinstance(value, pd.DataFrame)
    }
    
    # Return the code execution result and update to the workspace
    for key, value in extract_dataframes.items():
        if key in workspace:
            # Update the content for existing keys
            workspace[key]['content'] = value
        else:
            # Add new keys with default metadata
            workspace[key] = {
                "content": value,
                "metadata": {}
            }
    return {"result": serializable_result, "workspace": workspace}

def update_workspace(workspace: dict, new_data: dict) -> dict:
    """
    Updates the workspace with new data. For existing keys, updates the `content`.
    For new keys, adds them with empty `metadata`.

    Args:
        workspace (dict): The original workspace dictionary.
        new_data (dict): A dictionary with new data to update the workspace.

    Returns:
        dict: The updated workspace.
    """
    for key, content in new_data.items():
        if key in workspace:
            # Update the content for existing keys
            workspace[key]['content'] = content
        else:
            # Add new keys with default metadata
            workspace[key] = {
                "content": content,
            }
    return workspace


if __name__ == "__main__":
    import json
    # =======================================================
    # Test Example
    print("="*80 + "\n> Testing execute_python_code:")
    code = """
import math
a = math.sqrt(4)
b = 2
result = a + b
    """
    result = execute_python_code.invoke(code)
    print(result)
    # -------------------------------------------------------
    print("="*80 + "\n> Testing execute_python_code safty check:")
    code = """
import os
    """
    result = execute_python_code.invoke(code)
    print(result)
    # -------------------------------------------------------
    print("="*80 + "\n> Testing execute_python_code_with_dataframes:")
    sample_dataframe_1 = pd.DataFrame({"region": ["North", "South"], "sales": [300, 400]})
    sample_dataframe_2 = pd.DataFrame({"region": ["North", "South"], "target": [350, 390]})
    sample_text = "# My Article\nThis is a sample text."
    test_workspace = {

                        "sales_data": {
                            "content": sample_dataframe_1,
                            "metadata": {}
                            },
                        "targets": {
                            "content": sample_dataframe_2,
                            "metadata": {}
                            },
                        "my_article": {
                           "content": sample_text,
                           "metadata": {}
                       }
                      }
    code = """
merged_data = pd.merge(sales_data, targets, on='region')

difference_dict = {}
for index, row in merged_data.iterrows():
    region = row['region']
    sales_diff = row['sales'] - row['target']
    difference_dict[region] = sales_diff
"""
    result = execute_python_code_with_df.invoke({"code":code, "workspace":test_workspace})
    print(json.dumps(result.get("result"), indent=4, ensure_ascii=False))