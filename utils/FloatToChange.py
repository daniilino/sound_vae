
from dataclasses import dataclass, field
from typing import List

@dataclass
class FloatToChange(float):
    max_value : float
    operations : List = field(default_factory=list)

    def __add_x__(self, base, other):
        return base + other
    
    def __mul_x__(self, base, other):
        return base * other

    def __int_x__(self, base, other):
        return int(base)

    def __add__(self, other):
        self.operations.append(('__add_x__', other))
        return self

    def __sub__(self, other):
        self.operations.append(('__add_x__', -other))
        return self

    def __neg__(self):
        self.operations.append(('__mul_x__', -1))
        return self
    
    def __pos__(self):
        return self

    def __mul__(self, other):
        self.operations.append(('__mul_x__', other))
        return self

    def __truediv__(self, other):
        self.operations.append(('__mul_x__', 1/other))
        return self

    def as_int(self):
        self.operations.append(('__int_x__', 0))
        return self

    def __call__(self, factor):
        base = self.max_value*factor
        for op, other in self.operations:
            base = getattr(self,op)(base,other)
        return base