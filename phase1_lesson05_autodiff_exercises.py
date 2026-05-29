import math


class Value:
    def __init__(self, data, children=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, n):
        out = Value(self.data**n, (self,), f"**{n}")

        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other**-1)

    def relu(self):
        out = Value(max(0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(value):
            if value not in visited:
                visited.add(value)
                for child in value._prev:
                    build_topo(child)
                topo.append(value)

        build_topo(self)

        self.grad = 1.0
        for value in reversed(topo):
            value._backward()


def gradient_check(build_expr, x_val, h=1e-7):
    x = Value(x_val)
    y = build_expr(x)
    y.backward()
    autodiff_grad = x.grad

    y_plus = build_expr(Value(x_val + h)).data
    y_minus = build_expr(Value(x_val - h)).data
    numerical_grad = (y_plus - y_minus) / (2 * h)

    return autodiff_grad, numerical_grad, abs(autodiff_grad - numerical_grad)


class Dual:
    def __init__(self, value, derivative=0.0):
        self.value = float(value)
        self.derivative = float(derivative)

    def __repr__(self):
        return f"Dual(value={self.value:.4f}, derivative={self.derivative:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(
            self.value + other.value,
            self.derivative + other.derivative,
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(
            self.value * other.value,
            self.derivative * other.value + self.value * other.derivative,
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, n):
        return Dual(
            self.value**n,
            n * (self.value ** (n - 1)) * self.derivative,
        )

    def tanh(self):
        t = math.tanh(self.value)
        return Dual(
            t,
            (1 - t**2) * self.derivative,
        )

    def relu(self):
        if self.value > 0:
            return Dual(self.value, self.derivative)
        return Dual(0.0, 0.0)


print("=== Exercise 1: __pow__ Gradient ===")

x = Value(2.0)
y = x**3
y.backward()

print(f"x = {x.data}")
print(f"y = x^3 = {y.data}")
print(f"dy/dx = {x.grad}")
print("expected dy/dx = 3*x^2 = 12")


print("\n=== Exercise 2: tanh Gradient ===")

for value in [0.0, 2.0]:
    x = Value(value)
    y = x.tanh()
    y.backward()

    expected = 1 - math.tanh(value) ** 2

    print(f"x = {value}")
    print(f"tanh(x) = {y.data:.6f}")
    print(f"autodiff tanh'(x) = {x.grad:.6f}")
    print(f"expected tanh'(x) = {expected:.6f}")
    print()


print("=== Exercise 3: Single Neuron Gradients ===")

w1 = Value(0.5)
w2 = Value(0.25)
x1 = Value(3.0)
x2 = Value(2.0)
b = Value(-1.0)

y = (w1 * x1 + w2 * x2 + b).relu()
y.backward()

print(f"y = {y.data}")
print(f"dy/dw1 = {w1.grad}")
print(f"dy/dw2 = {w2.grad}")
print(f"dy/dx1 = {x1.grad}")
print(f"dy/dx2 = {x2.grad}")
print(f"dy/db = {b.grad}")

try:
    import torch

    tw1 = torch.tensor(0.5, requires_grad=True)
    tw2 = torch.tensor(0.25, requires_grad=True)
    tx1 = torch.tensor(3.0, requires_grad=True)
    tx2 = torch.tensor(2.0, requires_grad=True)
    tb = torch.tensor(-1.0, requires_grad=True)

    ty = torch.relu(tw1 * tx1 + tw2 * tx2 + tb)
    ty.backward()

    print("\nPyTorch comparison")
    print(f"dy/dw1 = {tw1.grad.item()}")
    print(f"dy/dw2 = {tw2.grad.item()}")
    print(f"dy/dx1 = {tx1.grad.item()}")
    print(f"dy/dx2 = {tx2.grad.item()}")
    print(f"dy/db = {tb.grad.item()}")

except ImportError:
    print("PyTorch not installed; skipping comparison.")


print("\n=== Exercise 4: Forward-Mode Autodiff with Dual Numbers ===")

# Seed x with derivative 1 because we want d/dx.
x_dual = Dual(2.0, derivative=1.0)
y_dual = x_dual**3

print("Function: y = x^3 at x=2")
print(y_dual)
print("expected derivative = 12")

x_dual = Dual(0.5, derivative=1.0)
y_dual = ((x_dual**3) + (2 * x_dual) + 1).tanh()

print("\nFunction: tanh(x^3 + 2x + 1) at x=0.5")
print(y_dual)

ad, num, diff = gradient_check(
    lambda value: ((value**3) + (2 * value) + 1).tanh(),
    0.5,
)

print("\nReverse-mode and numerical check")
print(f"reverse-mode grad = {ad:.8f}")
print(f"numerical grad = {num:.8f}")
print(f"forward-mode grad = {y_dual.derivative:.8f}")
print(f"difference reverse vs numerical = {diff:.2e}")
