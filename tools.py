"""
All custom tools used by the GAIA multi-agent system.
Each function is decorated with @tool so smolagents can load it.
"""

from smolagents import tool
import requests, json, math, statistics, pathlib
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import sympy as sp

# ----------  Mathematical tools  ---------- #

@tool
def advanced_calculator(expr: str) -> str:
    """
    Evaluate a numerical python expression with support for numpy and math functions.
    
    Args:
        expr: The mathematical expression to evaluate (e.g., "2*3+5", "np.sqrt(16)")
        
    Returns:
        String representation of the calculated result
    """
    # security-constrained eval namespace
    safe_ns = {"np": np, "math": math, "__builtins__": {}}
    try:
        val = eval(expr, safe_ns, {})
        return str(val)
    except Exception as exc:
        return f"CALC_ERROR: {exc}"

@tool
def solve_equation(equation: str, variable: str = "x") -> str:
    """
    Solve an algebraic equation for the given variable using symbolic math.
    
    Args:
        equation: The equation to solve in format "expression1 = expression2" (e.g., "2*x + 3 = 7")
        variable: The variable to solve for (default is "x")
        
    Returns:
        Comma-separated list of solutions for the variable
    """
    try:
        sym_var = sp.symbols(variable)
        sol = sp.solve(sp.Eq(*map(sp.sympify, equation.split("="))), sym_var)
        return ", ".join(map(str, sol))
    except Exception as exc:
        return f"SOLVE_ERROR: {exc}"

@tool
def statistical_analysis(data_csv: str, column: str, metric: str = "mean") -> str:
    """
    Compute a statistical metric for a numeric column in CSV data.
    
    Args:
        data_csv: CSV data as a string
        column: Name of the column to analyze
        metric: Statistical metric to compute (mean, median, stdev, var, min, max)
        
    Returns:
        String representation of the computed statistical value
    """
    try:
        df = pd.read_csv(pd.compat.StringIO(data_csv))
        series = df[column]
        funcs = {
            "mean": series.mean,
            "median": series.median,
            "stdev": series.std,
            "var": series.var,
            "min": series.min,
            "max": series.max,
        }
        if metric not in funcs:
            return f"Unknown metric {metric}"
        return str(funcs[metric]())
    except Exception as exc:
        return f"STAT_ERROR: {exc}"

@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert between different units for length, mass, and temperature.
    
    Args:
        value: The numerical value to convert
        from_unit: Source unit (m, cm, kg, g, c, f)
        to_unit: Target unit (m, cm, kg, g, c, f)
        
    Returns:
        String representation of the converted value
    """
    # Length conversions
    length_table = {
        ("m", "cm"): 100, ("cm", "m"): 0.01,
        ("m", "mm"): 1000, ("mm", "m"): 0.001,
        ("km", "m"): 1000, ("m", "km"): 0.001
    }
    
    # Mass conversions
    mass_table = {
        ("kg", "g"): 1000, ("g", "kg"): 0.001,
        ("kg", "lb"): 2.205, ("lb", "kg"): 0.453
    }
    
    # Temperature conversions
    if from_unit == "c" and to_unit == "f":
        return str(value * 9/5 + 32)
    if from_unit == "f" and to_unit == "c":
        return str((value - 32) * 5/9)
    if from_unit == "k" and to_unit == "c":
        return str(value - 273.15)
    if from_unit == "c" and to_unit == "k":
        return str(value + 273.15)
    
    # Check length conversions
    factor = length_table.get((from_unit, to_unit))
    if factor:
        return str(value * factor)
    
    # Check mass conversions
    factor = mass_table.get((from_unit, to_unit))
    if factor:
        return str(value * factor)
    
    return f"CONV_ERROR: unsupported unit conversion from {from_unit} to {to_unit}"

# ----------  Web research tools  ---------- #

@tool
def enhanced_web_search(query: str, k: int = 5) -> str:
    """
    Perform enhanced web search using DuckDuckGo and return structured results.
    
    Args:
        query: Search query string
        k: Maximum number of search results to return
        
    Returns:
        JSON string containing search results with title, URL, and snippet
    """
    from duckduckgo_search import DDGS
    try:
        out = []
        with DDGS() as ddgs:
            for res in ddgs.text(query, max_results=k):
                out.append({"title": res["title"], "href": res["href"], "snippet": res["body"]})
        return json.dumps(out, ensure_ascii=False)
    except Exception as exc:
        return f"WEBSEARCH_ERROR: {exc}"

@tool
def visit_and_extract_webpage(url: str, max_chars: int = 1500) -> str:
    """
    Download a webpage and extract readable text content.
    
    Args:
        url: The webpage URL to visit and extract content from
        max_chars: Maximum number of characters to return from the page
        
    Returns:
        Extracted text content from the webpage, truncated to max_chars
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        html = requests.get(url, timeout=15, headers=headers).text
        soup = BeautifulSoup(html, "html.parser")
        # drop scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join(soup.stripped_strings)
        return text[:max_chars]
    except Exception as exc:
        return f"WEBPAGE_ERROR: {exc}"

# ----------  File processing tools  ---------- #

GAIA_FILES_ENDPOINT = "https://agents-course-unit4-scoring.hf.space/files"

@tool
def download_and_analyze_file(task_id: str) -> str:
    """
    Download and analyze a file associated with a GAIA task ID.
    
    Args:
        task_id: The GAIA task identifier for the file to download and analyze
        
    Returns:
        Summary of the file contents and structure
    """
    try:
        url = f"{GAIA_FILES_ENDPOINT}/{task_id}"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "csv" in ctype or task_id.endswith(".csv"):
            df = pd.read_csv(pd.compat.StringIO(r.text))
            return f"CSV with {len(df)} rows and columns {list(df.columns)[:8]}"
        if "json" in ctype or task_id.endswith(".json"):
            data = r.json()
            return f"JSON keys: {list(data)[:8]}"
        # default to text
        txt = r.text[:300]
        return f"Text file sample: {txt}"
    except Exception as exc:
        return f"FILE_ERROR: {exc}"

@tool
def process_csv_data(csv_text: str, operation: str = "head", n: int = 5) -> str:
    """
    Perform operations on CSV data provided as text.
    
    Args:
        csv_text: CSV data as a string
        operation: Operation to perform (head, describe, columns)
        n: Number of rows for head operation
        
    Returns:
        Result of the CSV operation as a string
    """
    try:
        df = pd.read_csv(pd.compat.StringIO(csv_text))
        if operation == "head":
            return df.head(n).to_csv(index=False)
        if operation == "describe":
            return df.describe().to_csv()
        if operation == "columns":
            return ", ".join(df.columns)
        return "CSV_OP_ERROR: unknown operation"
    except Exception as exc:
        return f"CSV_ERROR: {exc}"

# ----------  Utility  ---------- #

@tool
def detect_question_type(question: str) -> str:
    """
    Classify the question type to route to appropriate specialist agent.
    
    Args:
        question: The question text to analyze and classify
        
    Returns:
        Question category: 'math', 'web', 'file', or 'generic'
    """
    q = question.lower()
    
    # Enhanced patterns for better routing
    if any(w in q for w in ["calculate", "solve", "derivative", "integral", "mean", "median", "probability", "equation", "formula"]):
        return "math"
    if any(w in q for w in ["albums", "discography", "released", "published", "studio albums", "wikipedia", "biography", "born", "died"]):
        return "web"  # Enhanced for biographical/music questions
    if "http" in q or "website" in q or "search" in q:
        return "web"
    if "attached file" in q or "csv" in q or "json" in q or "task_id" in q:
        return "file"
    return "generic"
