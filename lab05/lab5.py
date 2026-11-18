import math
import time
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward

        return out

    def sum(self):
      
        out = Value(self.data, (self,), 'sum')
        def _backward():
            self.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def backward(self):
        
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __sub__(self, other): return self + (-other)
    def __neg__(self): return self * -1
    def __rsub__(self, other): return other + (-self)

if __name__ == '__main__':
    a = Value(2.0); a.label = 'a'
    b = Value(3.0); b.label = 'b'
    c = Value(-4.0); c.label = 'c'
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = d ** 2; f.label = 'f'
    L = f.relu(); L.label = 'L'
    L.sum()
    L.backward()
    print("My engine gradients:")
    print(f"a.grad: {a.grad}, b.grad: {b.grad}, c.grad: {c.grad}")

    
    ta = torch.tensor(2.0, requires_grad=True)
    tb = torch.tensor(3.0, requires_grad=True)
    tc = torch.tensor(-4.0, requires_grad=True)
    tL = ((ta * tb + tc) ** 2).relu().sum()
    tL.backward()
    print("\nPyTorch gradients:")
    print(f"a.grad: {ta.grad.item()}, b.grad: {tb.grad.item()}, c.grad: {tc.grad.item()}")

    # PyTorch аналог
    ta = torch.tensor(2.0, requires_grad=True)
    tb = torch.tensor(3.0, requires_grad=True)
    tc = torch.tensor(-4.0, requires_grad=True)
    tL = ((ta * tb + tc) ** 2).relu().sum()
    tL.backward()
    print("\nPyTorch gradients:")
    print(f"a.grad: {ta.grad.item()}, b.grad: {tb.grad.item()}, c.grad: {tc.grad.item()}")