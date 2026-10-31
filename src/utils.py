import inspect
import os

def get_abs_path(relative_path):
    caller_frame = inspect.stack()[1]
    caller_file = caller_frame.filename
    base_path = os.path.dirname(os.path.abspath(caller_file))
    abs_path = os.path.join(base_path, relative_path)
    path = os.path.abspath(abs_path)
    return path

def get_tmp_path(relative_path):
    return get_abs_path(os.path.join("../tmp", relative_path))

os.makedirs(get_tmp_path(""), exist_ok=True)

def prt(text, color="gray"):
    color_codes = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "reset": "\033[0m"
    }
    
    color_code = color_codes.get(color.lower(), color_codes["white"])
    reset_code = color_codes["reset"]
    
    print(f"{color_code}{text}{reset_code}")