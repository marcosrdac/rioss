import functools
import time
def timer(func):
    """Print runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

def normalize01(array):
    minval = array.min()
    maxval = array.max()
    norm_array = (array - minval) / (maxval - minval)
    return(norm_array)
