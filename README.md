# Micrograd-CPP: Autograd Engine & Neural Network Library

A lightweight, scalar-valued Autograd engine written in C++17. This project implements **Reverse-Mode Automatic Differentiation (Backpropagation)** from scratch, allowing for the dynamic construction of computational graphs and the training of neural networks.

Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), this implementation focuses on C++ memory management (`std::shared_ptr`), operator overloading, and strict type safety to demonstrate the fundamental machinery behind deep learning frameworks like PyTorch.

## 🚀 Core Features

* **Dynamic Computational Graph:** Nodes (`Value` objects) are dynamically allocated and linked using a Directed Acyclic Graph (DAG) structure.
* **Automatic Memory Management:** Uses `std::shared_ptr` to handle object lifetimes and graph dependencies, preventing memory leaks without a garbage collector.
* **Reverse-Mode Autograd:** Implements the Chain Rule to propagate gradients backward from the loss function to leaf nodes.
* **Topological Sort:** Automatically resolves graph dependencies to ensure correct gradient propagation order.
* **Custom Activation Functions:** Includes **ReLU** and **Tanh** non-linearities with their respective derivative logic.
* **Optimizer:** Implements basic Stochastic Gradient Descent (SGD) to train Multi-Layer Perceptrons (MLP).

## 🛠️ Build & Run

### Prerequisites
* A C++ compiler supporting C++17 (e.g., `g++`, `clang`).
* (Optional) CMake for building.

### Option 1: Direct Compilation (Fastest)
```bash
# Create the build directory
mkdir -p build

# Compile
g++ -std=c++17 src/main.cpp -o build/micrograd

# Run
./build/micrograd
```

### Option 2: Using CMake
```bash
mkdir build && cd build
cmake ..
make
./micrograd
```

## 💻 Usage Example

Here is how to define a simple mathematical expression and calculate gradients automatically using the engine:

```cpp
// 1. Define inputs
auto a = Value::create(2.0);
auto b = Value::create(-3.0);
auto c = Value::create(10.0);

// 2. Build the graph: e = (a * b) + c
auto d = a * b;      // -6.0
auto e = d + c;      // 4.0

// 3. Backward Pass (Autograd)
e->backward();

// 4. Check Gradients (de/da = b = -3.0)
std::cout << "a.grad: " << a->grad << std::endl; // Output: -3.0
```

## 🧠 Solved Task: The XOR Problem

The engine has been verified by training a 2-layer MLP to solve the non-linear **XOR** problem, which is impossible for linear classifiers.

**Architecture:**
* **Input:** 2 Neurons
* **Hidden Layer:** 2 Neurons (`Tanh` activation)
* **Output:** 1 Neuron (Linear)
* **Loss:** Mean Squared Error (MSE)

**Training Results:**
```text
Step 0   | Total Loss: 3.94239 (Random Weights)
...
Step 500 | Total Loss: 0.00214
...
Step 999 | Total Loss: 1.60237e-30 (Converged)
```
<p align="center">
  <img src="graphviz (1).svg" width="100%" />
</p>
## 📚 Acknowledgements
* **Andrej Karpathy:** For the original Python `micrograd` and his incredible educational content.
* **Standard Template Library (STL):** Used extensively (`vector`, `set`, `functional`, `memory`) to keep the codebase dependency-free.
