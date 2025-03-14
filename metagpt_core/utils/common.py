import json
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from loguru import logger
from functools import partial
from pydantic_core import to_jsonable_python

def handle_unknown_serialization(x: Any) -> str:
    """For `to_jsonable_python` debug, get more detail about the x."""

    if inspect.ismethod(x):
        tip = f"Cannot serialize method '{x.__func__.__name__}' of class '{x.__self__.__class__.__name__}'"
    elif inspect.isfunction(x):
        tip = f"Cannot serialize function '{x.__name__}'"
    elif hasattr(x, "__class__"):
        tip = f"Cannot serialize instance of '{x.__class__.__name__}'"
    elif hasattr(x, "__name__"):
        tip = f"Cannot serialize class or module '{x.__name__}'"
    else:
        tip = f"Cannot serialize object of type '{type(x).__name__}'"

    raise TypeError(tip)

def read_json_file(json_file: str, encoding="utf-8") -> list[Any]:
    if not Path(json_file).exists():
        raise FileNotFoundError(f"json_file: {json_file} not exist, return []")

    with open(json_file, "r", encoding=encoding) as fin:
        try:
            data = json.load(fin)
        except Exception:
            raise ValueError(f"read json file: {json_file} failed")
    return data

def write_json_file(json_file: str, data: Any, encoding: str = "utf-8", indent: int = 4, use_fallback: bool = False):
    folder_path = Path(json_file).parent
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

    custom_default = partial(to_jsonable_python, fallback=handle_unknown_serialization if use_fallback else None)

    with open(json_file, "w", encoding=encoding) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=indent, default=custom_default)

def read_jsonl_file(jsonl_file: str, encoding="utf-8") -> list[dict]:
    if not Path(jsonl_file).exists():
        raise FileNotFoundError(f"json_file: {jsonl_file} not exist, return []")
    datas = []
    with open(jsonl_file, "r", encoding=encoding) as fin:
        try:
            for line in fin:
                data = json.loads(line)
                datas.append(data)
        except Exception:
            raise ValueError(f"read jsonl file: {jsonl_file} failed")
    return datas

def add_jsonl_file(jsonl_file: str, data: list[dict], encoding: str = None):
    folder_path = Path(jsonl_file).parent
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

    with open(jsonl_file, "a", encoding=encoding) as fout:
        for json_item in data:
            fout.write(json.dumps(json_item) + "\n")

def get_project_root():
    """Get project root directory using multiple methods"""
    # 1. Check environment variable
    project_root_env = os.getenv("PROJECT_ROOT")
    if project_root_env:
        project_root = Path(project_root_env)
        logger.info(f"Project root set from environment variable: {str(project_root)}")
        return project_root
    
    # 2. Search for marker files by traversing up from current file
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    
    # Try up to 5 directory levels
    for _ in range(5):
        for marker in (".git", ".project_root", "pyproject.toml", "setup.py", ".gitignore", "README.md"):
            if (current_dir / marker).exists():
                logger.info(f"Project root found via marker file {marker}: {str(current_dir)}")
                return current_dir
        # Move to parent directory
        parent_dir = current_dir.parent
        if parent_dir == current_dir:  # Reached filesystem root
            break
        current_dir = parent_dir
    
    # 3. Fallback to current working directory
    project_root = Path.cwd()
    logger.info(f"Could not find project markers, using current working directory: {str(project_root)}")
    return project_root

PROJECT_ROOT = get_project_root()
