#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <memory>
#include <set>
using namespace std;

double random_uniform(double min, double max){
    double r = (double) rand() / RAND_MAX;
    return min + r * (max - min);
}
struct Value;

class Value : public enable_shared_from_this<Value>
{
public:
    double data;
    double grad;

    vector<shared_ptr<Value>> _prev;
    function<void()> _backward;

    Value(double val, vector<shared_ptr<Value>> children = {}): data(val), grad(0.0), _prev(children){
        _backward = [](){};
    }

    static shared_ptr<Value> create(double val){
        return make_shared<Value>(val);
    }

    friend shared_ptr<Value> operator+(shared_ptr<Value> a, shared_ptr<Value> b){
        auto out = make_shared<Value>(a->data + b->data, vector<shared_ptr<Value>>{a,b});

        out->_backward = [out, a, b](){
            a->grad += 1.0 * out->grad;
            b->grad += 1.0 * out->grad;
        };
        
        return out;
    }

    friend shared_ptr<Value> operator*(shared_ptr<Value> a, shared_ptr<Value> b){
        auto out = make_shared<Value>(a->data*b->data, vector<shared_ptr<Value>>{a,b});

        out->_backward = [out, a, b](){
            a->grad += b->data * out->grad;
            b->grad += a->data * out->grad;
        };

        return out;
    }
    
    shared_ptr<Value> relu(){
        double new_data = (data > 0)?data:0;
        auto out = make_shared<Value>(new_data, vector<shared_ptr<Value>>{shared_from_this()});

        auto self = shared_from_this();
        out->_backward = [out, self](){
            if (self->data > 0){
                self->grad += 1.0*out->grad;
            }
        };
        return out;
    }
    shared_ptr<Value> tanh(){
        double t = std::tanh(this->data);
        auto out = make_shared<Value>(t, vector<shared_ptr<Value>>{shared_from_this()});

        auto self = shared_from_this();
        out->_backward = [out, self](){
            double y = out->data;
            self->grad += (1.0 - y*y)*out->grad;
        };
        return out;
    }


    void backward(){
        vector <shared_ptr<Value>> topo;
        set<Value*> visited;

        function<void(shared_ptr<Value>)> build_topo = [&](shared_ptr<Value>v){
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
    void print(){
        cout << "Value(data=" << data << ", grad=" << grad << ")" << endl;
    }
};

class Neuron {
    public:
    vector<shared_ptr<Value>> w;
    shared_ptr<Value> b;
    bool non_lin;
    Neuron(int nin, bool non_lin=true): non_lin(non_lin){
        for(int i = 0; i < nin; i++){
            w.push_back(Value::create(random_uniform(-1.0, 1.0)));
        }
        b = Value::create(random_uniform(-1.0, 1.0));
    }

    shared_ptr<Value>operator()(const vector<shared_ptr<Value>>&x){
        auto act = b;
        for(size_t i = 0; i<w.size();i++){
            act = act + (w[i]*x[i]);
        }
        return non_lin?act->tanh():act;
    }

    vector<shared_ptr<Value>> parameters(){
        vector <shared_ptr<Value>> params = w; 
        params.push_back(b);
        return params;
    }
};

class Layer {
    public:
    vector <Neuron> neurons;
    Layer(int nin, int nout, bool non_lin = true){
        for(int i = 0; i<nout; i++){
            neurons.emplace_back(nin, non_lin);
        }
    }

    vector<shared_ptr<Value>> operator()(const vector<shared_ptr<Value>>&x){
        vector <shared_ptr<Value>> outs; 
        for(auto &n:neurons){
            outs.push_back(n(x));
        }
        return outs;
    }

    vector <shared_ptr<Value>> parameters(){
        vector<shared_ptr<Value>> params;
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
    vector <Layer> layers;
    MLP(int nin, vector<int> nouts){
        int sz = nouts.size();
        for(int i = 0;i<sz;i++){
            layers.emplace_back(i==0?nin:nouts[i-1], nouts[i], i!=sz-1);
        }
    }
    vector<shared_ptr<Value>> operator()(vector<shared_ptr<Value>> x){
        for(auto& layer:layers){
            x = layer(x);
        }
        return x;
    }
    vector <shared_ptr<Value>> parameters(){
        vector<shared_ptr<Value>> params;
        for(auto& layer:layers){
            auto l_params = layer.parameters();
            for(auto&p :l_params){
                params.push_back(p);
            }
        }
        return params;
    }
};
int main(){
    srand(time(0));
    vector<vector<double>> X = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    };
    vector<double> Y = {1.0, -1.0, -1.0, 1.0};

    MLP model(3, {4, 4, 1});

    for(int k=0; k<200; k++) {

        vector<shared_ptr<Value>> ypred;
        for(auto& row : X) {
            vector<shared_ptr<Value>> inputs;
            for(double d : row) inputs.push_back(Value::create(d));
            ypred.push_back(model(inputs)[0]); 
        }
        auto loss = Value::create(0.0);
        for(size_t i=0; i<Y.size(); i++) {
            auto diff = ypred[i] + Value::create(-Y[i]);
            loss = loss + (diff * diff);
        }

        auto params = model.parameters();
        for(auto& p : params) p->grad = 0.0;

        loss->backward();

        for(auto& p : params) {
            p->data -= 0.05 * p->grad;
        }

        if (k % 10 == 0)
            cout << "Step " << k << " loss: " << loss->data << endl;
    }
    
}