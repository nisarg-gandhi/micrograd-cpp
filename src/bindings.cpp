#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(micrograd_cpp, m) {
    m.doc() = "Micrograd C++ Engine with Python Bindings";

    py::class_<Value, std::shared_ptr<Value>>(m, "Value")
        .def(py::init(&Value::create))
        .def_readwrite("data", &Value::data)
        .def_readwrite("grad", &Value::grad)
        .def("backward", &Value::backward)
        .def("relu", &Value::relu)
        .def("tanh", &Value::tanh)
        
        
        .def("__add__", [](std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
            return a + b; 
        })
        
        .def("__mul__", [](std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
            return a * b; 
        })
        
        .def("__add__", [](std::shared_ptr<Value> a, double b) {
            return a + Value::create(b);
        })
        
        .def("__mul__", [](std::shared_ptr<Value> a, double b) {
            return a * Value::create(b);
        })
        
        
        .def("__radd__", [](std::shared_ptr<Value> a, double b) {
            return a + Value::create(b);
        })
        .def("__rmul__", [](std::shared_ptr<Value> a, double b) {
            return a * Value::create(b);
        })

        .def("__repr__", [](const Value &a) {
            return "<Value data=" + std::to_string(a.data) + " grad=" + std::to_string(a.grad) + ">";
        });

    py::class_<MLP>(m, "MLP")
        .def(py::init<int, std::vector<int>>())
        .def("__call__", &MLP::operator())
        .def("parameters", &MLP::parameters);
}