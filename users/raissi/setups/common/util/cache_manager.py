__all__ = ["cache_file"]

import importlib
import sys
import typing

from sisyphus import tk


sys.path.append("/usr/local/")
sys.path.append("/usr/local/cache-manager/")

cm = importlib.import_module("cache-manager")


def cache_file(file: typing.Union[tk.Path, str]) -> str:
    path = file.get_path() if isinstance(file, tk.Path) else file
    return cm.cacheFile(path)
