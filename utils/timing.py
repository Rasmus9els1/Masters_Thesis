from functools import wraps
from time import time

def timed(fn):
    """
    Decorator that prints the function/method name and elapsed time,
    but returns exactly whatever the original function returns.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time()
        result = fn(*args, **kwargs)
        elapsed = time() - start
        print(f"{fn.__name__} took {elapsed:.6f} seconds")
        return result
    return wrapper