from collections.abc import Callable


def save_to_file(fn: Callable, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        fn(f)
