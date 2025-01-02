import pandas as pd
import chardet
import os
# Log
from my_logger import Logger, LOG_LEVEL, LOG_PATH, LOG_FILE
# 初始化日志记录器
logger = Logger(name="AgentLogger", level=LOG_LEVEL, log_path=LOG_PATH, log_file=LOG_FILE)

def load_csv_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Automatically detects the encoding of a CSV file and loads it into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file to be loaded.

    Returns:
    pd.DataFrame: The loaded pandas DataFrame.
    
    If an error occurs during file loading, None is returned and an error message is printed.
    """
    try:
        # Detect file encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        
        # Use pandas to read the CSV file with the detected encoding
        dataframe = pd.read_csv(file_path, encoding=encoding)
        logger.logger.info(f"File '{os.path.basename(file_path)}' loaded with encoding: {encoding}")
        return dataframe
    except Exception as e:
        logger.logger.error(f"Error loading the file: {e}")
        return None

def generate_dataframe_schema(df: pd.DataFrame, sample_size: int = 2, if_serializable: bool = True) -> dict:
    """
    Generates a schema for a given Pandas DataFrame, including column names, data types, 
    whether each column contains any missing values, and a sample of data for each column.
    Optionally, the schema can be made JSON serializable.

    Parameters:
    df (pd.DataFrame): The Pandas DataFrame for which the schema is to be generated.
    sample_size (int): The number of sample values to include for each column's 'data_example'.
    if_serializable (bool): Whether to ensure the schema is JSON serializable.

    Returns:
    dict: A dictionary representing the schema of the DataFrame.
          If `if_serializable` is True, ensures all values in the schema are JSON serializable.
    """
    schema = {}
    
    for column in df.columns:
        # Get the column data type
        column_type = str(df[column].dtype)
        
        # Check if the column contains any missing values
        has_nulls = bool(df[column].isnull().any()) if if_serializable else df[column].isnull().any()
        
        # Sample data
        data_example = df[column].dropna().sample(
            n=min(sample_size, df[column].notnull().sum())
        ).tolist()
        
        # If JSON serializable output is required, convert non-serializable items to strings
        if if_serializable:
            data_example = [
                str(item) if not isinstance(item, (int, float, str, bool, type(None))) else item
                for item in data_example
            ]
        
        # Add column description to the schema
        schema[column] = {
            'data_type': column_type,
            'has_nulls': has_nulls,
            'data_example': data_example
        }
    
    return schema



if __name__ == "__main__":
    from my_logger import CURRENT_PATH
    import json
    # =======================================================
    # Test Example
    print("="*80 + "\n> Testing load_csv_to_dataframe:")
    file_name = 'superstore.csv'  
    file_path = os.path.join(CURRENT_PATH, 'data', 'csv', file_name)
    df = load_csv_to_dataframe(file_path)
    if df is not None:
        print(df.head())  # Print the first 5 rows of the DataFrame
        print("="*80 + "\n> Testing generate_dataframe_schema:")
        schema = generate_dataframe_schema(df, sample_size=5, if_serializable=True)
        print(json.dumps(schema, indent=4, ensure_ascii=False))