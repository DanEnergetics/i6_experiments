from collections import UserDict
from itertools import product

class Selector(UserDict):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
  
  @classmethod
  def from_dict(cls, d):
    return cls(**d)
  
  @classmethod
  def _get_value(cls, obj, key):
    if isinstance(obj, (dict, Selector)):
      return obj.get(key, obj)
    return obj
  
  def select(self, *keys):
    res = self.data
    for key in keys:
      res = Selector(**{
        k: self._get_value(v, key) for k, v in res.items()
      })
    return res
  
  def __repr__(self):
    return f"Selector({self.data})"


class P:
    def __init__(self, *params):
        self.params = params
    
    @staticmethod
    def maybe_tuple(p):
        if isinstance(p, tuple):
            return p
        return (p,)
    
    def __iter__(self):
        return iter(self.params)
    
    def __mul__(self, other):
        return P(*(self.maybe_tuple(p) + o for p, o in product(self.params, other.params)))
    
    def __add__(self, other):
        """ Concatenate parameter lists. """
        return P(*self.params, *other.params)
