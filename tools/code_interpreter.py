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
        clean_env = {}
        for key, value in env.items():
            if isinstance(value, (int, float, str, list, dict, bool, type(None))):
                clean_env[key] = value
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
            # Return the exception message if something went wrong
            return {"error": str(e)}

@tool
def execute_python_code(code: str) -> dict:
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
            dataframes (Dict[str, pd.DataFrame]): The resulting DataFrames if successfully generated.
            None: If no DataFrame is produced.
        """
        # Load DataFrames into the environment
        if not isinstance(dataframes, dict):
            raise ValueError("dataframes must be a dictionary of DataFrame objects.")

        self.environment.update(dataframes)
        self.environment["pd"] = pd  # Add pandas to the execution environment

        # Execute the code safely
        result = self.execute(code)

        if "error" in result:
            # If an error occurred, log and return None
            logger.logger.error(f"Code execution error: {result['error']}")
            return None

        # Extract DataFrame from the cleaned environment
        cleaned_env = result.get("success", {})
        output_dataframes = {
            key:value for key, value in cleaned_env.items() if isinstance(value, pd.DataFrame)
        }

        # Return the first DataFrame found or None if none were found
        return output_dataframes if output_dataframes else None

@tool
def execute_python_code_with_dataframes(code: str, workspace: dict) -> dict:
    """
    Execute Python code safely with provided DataFrames cached in workspace and return the resulting DataFrames.

    Args:
        code (str): Python code to execute. This should reference DataFrame variables.
  
    Returns:
        None: The result will be cached into workspace.

    Example:

        .. code-block:: python
            # cached dataframe:
            dataframes = {
                "sales_data": pd.DataFrame({"region": ["North", "South"], "sales": [300, 400]}),
                "targets": pd.DataFrame({"region": ["North", "South"], "target": [350, 390]})
            }
            # your code should be like:
            code = "result_df = pd.merge(sales_data, targets, on='region')"
            result = execute_python_code_with_dataframes.invoke(code)
            # The result will be:
            result = {'sales_data': <pandas.DataFrame>, 'targets': <pandas.DataFrame>, 'result': <pandas.DataFrame>}
    """
    from utility.manage_workspace import filter_workspace_content
    # get pd.DataFrame from workspace
    filtered_workspace = filter_workspace_content(workspace, content_type=pd.DataFrame)
    # format dataframes to be Dict[str, pd.DataFrame]
    dataframes = {key:value.get("content") for key, value in filtered_workspace.items()}
    interpreter = DataAnalysisInterpreter()
    return interpreter.execute_analysis_code(code, dataframes)

if __name__ == "__main__":
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
result = pd.merge(sales_data, targets, on='region')
"""
    result = execute_python_code_with_dataframes.invoke({"code":code, "workspace":test_workspace})
    print(result)