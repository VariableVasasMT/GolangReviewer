import datasets
from tqdm.auto import tqdm
from contextlib import contextmanager

class CustomTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["lock_args"] = False
        super().__init__(*args, **kwargs)

@contextmanager
def use_custom_tqdm():
    original_tqdm = datasets.utils.logging.tqdm
    datasets.utils.logging.tqdm = CustomTqdm
    try:
        yield
    finally:
        datasets.utils.logging.tqdm = original_tqdm