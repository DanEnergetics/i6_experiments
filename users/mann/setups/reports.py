from sisyphus.job_path import VariableNotSet
from sisyphus.delayed_ops import DelayedBase

def maybe_get(var):
    try:
        return var.get()
    except VariableNotSet:
        return ""

def eval_tree(o, f=maybe_get, condition=lambda x: isinstance(x, DelayedBase)):
    """
    Recursively traverses a structure and calls .get() on all
    existing Delayed Operations, especially Variables in the structure

    :param Any o: nested structure that may contain DelayedBase objects
    :return:
    """
    if condition(o):
        o = f(o)
    elif isinstance(o, list):
        for k in range(len(o)):
            o[k] = eval_tree(o[k], f, condition)
    elif isinstance(o, tuple):
        o = tuple(eval_tree(e, f, condition) for e in o)
    elif isinstance(o, dict):
        for k in o:
            o[k] = eval_tree(o[k], f, condition)
    return o

class SimpleValueReport:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return str(self.value)

class DescValueReport:
    def __init__(self, values: dict):
        self.values = values
    
    def __call__(self):
        keys = [str(key) for key in self.values]
        max_width = max(len(key) for key in keys)
        fmt = "{0:<%d}: {1}" % max_width
        return "\n".join(fmt.format(key, value) for key, value in self.values.items())

class GenericReport:
    def __init__(self, cols, values, fmt_percent=False, add_latex=False):
        self.values = values
        self.cols = cols
        self.fmt_percent = fmt_percent
        self._latex = add_latex
    
    def __call__(self):
        from tabulate import tabulate
        values = self.values
        table = []
        cols = self.cols
        for cp, stat in values.items():
            row = [cp]
            try:
                for key in cols:
                    v = stat[key].get()
                    if self.fmt_percent:
                        v *= 100
                    row.append(round(v, 2))
            except TypeError as e:
                print(e)
                row += [""] * len(cols)
            table.append(row)
        if self.fmt_percent:
            cols = [c + " [%]" for c in cols]
        buffer = tabulate(table, headers=["corpus"] + cols, tablefmt="presto")
        if self._latex:
            buffer += "\n" * 2
            buffer += tabulate(table, headers=["corpus"] + cols, tablefmt="latex")
        return buffer