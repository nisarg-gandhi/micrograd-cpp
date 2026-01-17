# Micrograd-CPP: Hybrid Autograd Engine & Neural Network Library

A high-performance, scalar-valued Autograd engine written in **C++14**, featuring native **Python bindings** via `pybind11`. This project implements **Reverse-Mode Automatic Differentiation (Backpropagation)** from scratch, allowing users to build dynamic computational graphs in C++ or Python with a PyTorch-like API.

Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), this implementation bridges the gap between low-level system efficiency and high-level usability, demonstrating the internal architecture of modern Deep Learning frameworks.

## 🚀 Core Features

* **Hybrid Architecture:** Core logic runs in optimized C++, while the interface is exposed to Python for rapid experimentation.
* **Python Bindings:** Custom `pybind11` integration allows you to `import micrograd_cpp` and train models in Python with C++ speed.
* **Dynamic Computational Graph:** Nodes (`Value` objects) are dynamically allocated and linked using a DAG structure with automatic memory management (`std::shared_ptr`).
* **Reverse-Mode Autograd:** Implements the Chain Rule to propagate gradients backward from loss to parameters.
* **Topological Sort:** Automatically resolves graph dependencies for correct gradient accumulation.
* **Deep Learning Primitives:** Includes **ReLU/Tanh** activations, MSE Loss, and MLP (Multi-Layer Perceptron) architecture.

## 🛠️ Technical Stack

* **Core Language:** C++14 (Logic, Memory Management)
* **Interface:** Python 3.x (User API)
* **Binding Library:** `pybind11` (Interoperability)
* **Build System:** CMake (Cross-platform build configuration)

## 📦 Project Structure

```text
micrograd-cpp/
├── src/
│   ├── engine.hpp      # Core C++ Autograd Logic (Value, Neuron, Layer, MLP)
│   ├── bindings.cpp    # Pybind11 Glue Code (Exposes C++ to Python)
│   └── main.cpp        # Pure C++ Demo Executable (includes Graphviz export)
├── CMakeLists.txt      # Build Configuration (Fetches pybind11 automatically)
└── README.md
```

## ⚙️ Build Instructions

This project uses **CMake** to handle dependencies and compilation.

### Prerequisites
* C++ Compiler (GCC/MinGW, Clang, or MSVC)
* CMake (3.14+)
* Python 3.6+

### Building the Python Module (Windows/Linux/Mac)

```bash
# 1. Create build directory
mkdir build && cd build

# 2. Configure CMake 
# (On Windows with MinGW, use: cmake -G "MinGW Makefiles" ..)
cmake ..

# 3. Compile
cmake --build .
```

*Upon success, a `micrograd_cpp.pyd` (Windows) or `.so` (Linux/Mac) file will be generated in the build folder.*

## 💻 Usage Examples

### 1. Python API (Recommended)
You can use the engine exactly like PyTorch.

```python
import sys
import os

# Ensure the compiled module is in path (or run this script from the build folder)
import micrograd_cpp

# 1. Build a computational graph
a = micrograd_cpp.Value(2.0)
b = micrograd_cpp.Value(-3.0)
c = micrograd_cpp.Value(10.0)

# Operations are executed in C++!
d = a * b + c     # Result: 4.0

# 2. Backward Pass
d.backward()

print(f"Gradient of a: {a.grad}")  # Output: -3.0
print(f"Gradient of b: {b.grad}")  # Output: 2.0

# 3. Neural Network Training
model = micrograd_cpp.MLP(3, [4, 4, 1]) # 3 inputs, two hidden layers of 4, 1 output
inputs = [micrograd_cpp.Value(x) for x in [2.0, 3.0, -1.0]]
output = model(inputs)[0]
```

### 2. Pure C++ API
You can also run the engine entirely in C++ for maximum performance.

```cpp
#include "engine.hpp"
#include <iostream>

int main() {
    auto a = Value::create(2.0);
    auto b = Value::create(-3.0);
    
    auto d = a * b; // -6.0
    d->backward();
    
    std::cout << "a.grad: " << a->grad << std::endl; // -3.0
    return 0;
}
```

## 📊 Graph Visualization

The C++ implementation includes a utility to export the computational graph to Graphviz format, allowing you to visualize the forward pass and gradients.

<p align="center">
  <img src="graphviz (1).svg" width="100%" />
</p>

## 🧠 Solved Task: The XOR Problem

The engine has been verified by training a 2-layer MLP to solve the non-linear **XOR** problem.

**Training Results:**
```text
Step 0   | Total Loss: 3.94239 (Random Weights)
...
Step 500 | Total Loss: 0.00214 (Converged)
```

## 📚 Acknowledgements
* **Andrej Karpathy:** For the original [micrograd](https://github.com/karpathy/micrograd).
* **Pybind11:** For making C++/Python interoperability seamless.
