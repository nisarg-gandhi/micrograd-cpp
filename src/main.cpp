#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <memory>
#include <set>
using namespace std;

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

int main(){
    
    srand(time(0)); 
    auto random_weight = []() {
        return (rand() % 2000 / 1000.0) - 1.0; 
    };

    auto w1 = Value::create(random_weight()); 
    auto w2 = Value::create(random_weight()); 
    auto b1 = Value::create(random_weight());
    
    auto w3 = Value::create(random_weight()); 
    auto w4 = Value::create(random_weight()); 
    auto b2 = Value::create(random_weight());
    
    auto w5 = Value::create(random_weight()); 
    auto w6 = Value::create(random_weight()); 
    auto b3 = Value::create(random_weight());
    
    cout << "Initialized with random weights..." << endl;
    
    for (int k = 0; k < 1000; k++) {
        
        cout << "Step " << k << " ----------------" << endl;
        double total_loss = 0;
        double inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
        double targets[4]   = {0,    1,    1,    0};
        
        for(int i=0; i<4; i++) {
            
            auto x1 = Value::create(inputs[i][0]);
            auto x2 = Value::create(inputs[i][1]);
            auto y_target = targets[i];
            
            auto n1 = ((w1*x1) + (w2*x2) + b1)->tanh();

            auto n2 = ((w3*x1) + (w4*x2) + b2)->tanh();

            auto score = (w5*n1) + (w6*n2) + b3;
            
            auto diff = score + Value::create(-y_target); // score - target
            auto loss = diff * diff;
            
            total_loss += loss->data;
            w1->grad=0; w2->grad=0; b1->grad=0;
            w3->grad=0; w4->grad=0; b2->grad=0;
            w5->grad=0; w6->grad=0; b3->grad=0;
            
            loss->backward();
            double lr = 0.05;
            w1->data -= lr * w1->grad; w2->data -= lr * w2->grad; b1->data -= lr * b1->grad;
            w3->data -= lr * w3->grad; w4->data -= lr * w4->grad; b2->data -= lr * b2->grad;
            w5->data -= lr * w5->grad; w6->data -= lr * w6->grad; b3->data -= lr * b3->grad;
        }
        cout << "Total Loss: " << total_loss << endl;
    }
}