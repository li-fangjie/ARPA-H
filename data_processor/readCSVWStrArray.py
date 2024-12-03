import pandas as pd
import ast
import numpy as np
import re 
import sys

# For reading ndi track data generated from old versions of Ayberk's python data logger for NDI
def readCSVWStrArray(filePath, **kwargs):
    df = pd.read_csv(filePath, **kwargs)

    # Apply parsing to each cell in the DataFrame
    df = df.applymap(parse_complex_cell)
    # Apply the function to the specific column (e.g., the 4th column, index 3)
    df[3] = df[3].apply(parse_array_string)

    return df


# Define a function to safely parse complex strings
def parse_column(cell):
    try:
        return ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return cell  # Leave as string if it cannot be evaluated


# Function to parse complex strings in each cell
def parse_complex_cell(cell):
    try:
        parsed_cell = ast.literal_eval(cell)
        # If it's a list of arrays, convert each array in the list to a NumPy array
        if isinstance(parsed_cell, list) and all(isinstance(arr, list) for arr in parsed_cell):
            return [np.array(arr) for arr in parsed_cell]
        return parsed_cell
    except (ValueError, SyntaxError):
        return cell

def parse_array_string(array_str):
    if not isinstance(array_str, str):
        return np.array(array_str)
    # Replace 'array(...)' with just the content inside, making it evaluable
    array_str_cleaned = re.sub(r'array\((.*?)\)', r'\1', array_str)
    array_str_cleaned = array_str.replace("array(", "")
    array_str_cleaned = array_str_cleaned.replace(")", "")
    # array_str_cleaned = array_str_cleaned.replace("\n", "")
    array_str_cleaned = array_str_cleaned.replace("nan", "float('nan')")
    # Now safely evaluate the string into Python lists
    try:
        # parsed_content = ast.literal_eval(array_str_cleaned)
        parsed_content = eval(array_str_cleaned, {"float": float})
        print(parsed_content)
        # Convert each list item into a NumPy array
        return [np.array(mat) for mat in parsed_content]
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing: {e}")
        return None

