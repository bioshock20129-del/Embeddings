from collections.abc import Callable
from json import JSONEncoder

import numpy


def save_to_file(fn: Callable, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        fn(f)


def make_pipeline(*steps):
    def wrapper(*args, **kwargs):
        result = steps[0](*args, **kwargs)
        for step in steps[1:]:
            result = step(result)
        return result

    return wrapper


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def filter_by_author(field: str, author: str, texts: list[dict[str, str]]):
    if not texts or field not in texts[0]:
        return []
    return list(map(lambda x: x.get(field), filter(lambda x: x.get("author") == author, texts)))