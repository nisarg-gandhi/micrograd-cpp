#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <memory>
#include <set>
#include <algorithm>

inline double random_uniform(double min, double max){
    double r = (double) rand() / RAND_MAX;
    return min + r * (max - min);
}

struct Value;

class Value : public std::enable_shared_from_this<Value>
{
public:
    double data;
    double grad;
    std::string op;
    std::string label;
    std::vector<std::shared_ptr<Value>> _prev;
    std::function<void()> _backward;

    Value(double val, std::vector<std::shared_ptr<Value>> children = {}): data(val), grad(0.0), _prev(children){
        _backward = [](){};
    }

    
    static std::shared_ptr<Value> create(double val){
        return std::make_shared<Value>(val);
    }

    friend std::shared_ptr<Value> operator+(std::shared_ptr<Value> a, std::shared_ptr<Value> b){
        auto out = std::make_shared<Value>(a->data + b->data, std::vector<std::shared_ptr<Value>>{a,b});
        out->op="+";
        out->_backward = [out, a, b](){
            a->grad += 1.0 * out->grad;
            b->grad += 1.0 * out->grad;
        };
        return out;
    }

    friend std::shared_ptr<Value> operator*(std::shared_ptr<Value> a, std::shared_ptr<Value> b){
        auto out = std::make_shared<Value>(a->data*b->data, std::vector<std::shared_ptr<Value>>{a,b});
        out->op="*";
        out->_backward = [out, a, b](){
            a->grad += b->data * out->grad;
            b->grad += a->data * out->grad;
        };
        return out;
    }
    
    std::shared_ptr<Value> relu(){
        double new_data = (data > 0)?data:0;
        auto out = std::make_shared<Value>(new_data, std::vector<std::shared_ptr<Value>>{shared_from_this()});
        out->op="ReLU";
        auto self = shared_from_this();
        out->_backward = [out, self](){
            if (self->data > 0){
                self->grad += 1.0*out->grad;
            }
        };
        return out;
    }

    std::shared_ptr<Value> tanh(){
        double t = std::tanh(this->data);
        auto out = std::make_shared<Value>(t, std::vector<std::shared_ptr<Value>>{shared_from_this()});
        out->op="tanh";
        auto self = shared_from_this();
        out->_backward = [out, self](){
            double y = out->data;
            self->grad += (1.0 - y*y)*out->grad;
        };
        return out;
    }

    void backward(){
        std::vector <std::shared_ptr<Value>> topo;
        std::set<Value*> visited;

        std::function<void(std::shared_ptr<Value>)> build_topo = [&](std::shared_ptr<Value>v){
            if(visited.find(v.get())!=visited.end()) return;
            visited.insert(v.get());

            for (auto child:v->_prev){
                build_topo(child);
            }
            topo.push_back(v);
        };

        build_topo(shared_from_this());

        this->grad = 1.0;
        for (auto it = topo.rbegin(); it!=topo.rend();++it){
            (*it)->_backward();
        }
    }
    
    double getData() const { return data; }
    void setData(double v) { data = v; }
    double getGrad() const { return grad; }
    void setGrad(double v) { grad = v; }
};

class Neuron {
    public:
    std::vector<std::shared_ptr<Value>> w;
    std::shared_ptr<Value> b;
    bool non_lin;
    Neuron(int nin, bool non_lin=true): non_lin(non_lin){
        for(int i = 0; i < nin; i++){
            w.push_back(Value::create(random_uniform(-1.0, 1.0)));
        }
        b = Value::create(random_uniform(-1.0, 1.0));
    }

    std::shared_ptr<Value>operator()(const std::vector<std::shared_ptr<Value>>&x){
        auto act = b;
        for(size_t i = 0; i<w.size();i++){
            act = act + (w[i]*x[i]);
        }
        return non_lin?act->tanh():act;
    }

    std::vector<std::shared_ptr<Value>> parameters(){
        std::vector <std::shared_ptr<Value>> params = w; 
        params.push_back(b);
        return params;
    }
};

class Layer {
    public:
    std::vector <Neuron> neurons;
    Layer(int nin, int nout, bool non_lin = true){
        for(int i = 0; i<nout; i++){
            neurons.emplace_back(nin, non_lin);
        }
    }

    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>&x){
        std::vector <std::shared_ptr<Value>> outs; 
        for(auto &n:neurons){
            outs.push_back(n(x));
        }
        return outs;
    }

    std::vector <std::shared_ptr<Value>> parameters(){
        std::vector<std::shared_ptr<Value>> params;
        for(auto &n:neurons){
            auto n_params = n.parameters();
            for(auto &p:n_params){
                params.push_back(p);
            }
        }
        return params;
    }
};

class MLP {
    public:
    std::vector <Layer> layers;
    MLP(int nin, std::vector<int> nouts){
        int sz = nouts.size();
        for(int i = 0;i<sz;i++){
            layers.emplace_back(i==0?nin:nouts[i-1], nouts[i], i!=sz-1);
        }
    }
    std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>> x){
        for(auto& layer:layers){
            x = layer(x);
        }
        return x;
    }
    std::vector <std::shared_ptr<Value>> parameters(){
        std::vector<std::shared_ptr<Value>> params;
        for(auto& layer:layers){
            auto l_params = layer.parameters();
            for(auto&p :l_params){
                params.push_back(p);
            }
        }
        return params;
    }
};