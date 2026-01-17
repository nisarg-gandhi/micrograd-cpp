#include <iostream>
#include <vector>
#include <fstream>
#include "engine.hpp"
using namespace std;

void draw_graph(shared_ptr<Value>root, string filename){
    ofstream file(filename);
    file << "digraph G{" << endl;
    file << "rankdir=LR;" << endl;

    set <Value*> visited;
    function<void(shared_ptr<Value>)> build = [&](shared_ptr<Value> v){
        if (visited.find(v.get()) != visited.end()) return;
        visited.insert(v.get());

        size_t id = (size_t )v.get();

        file << " " << id << " [shape=record, label=\"{ " << (v->label.empty() ? "" : v->label + " | ") << "data " << v->data << " | " << "grad " << v->grad << " }\"];" << endl;

        if (!v->_prev.empty()){
            size_t op_id = id + 1;
            file << " " << op_id << "[label=\"" << v->op << "\", shape=circle];" << endl;
            file << "  " << op_id << " -> " << id << ";" << endl;
            for(auto child : v->_prev){
                file << "  " << (size_t)child.get() << " -> " << op_id << ";" << endl;
                build(child);
            }
        }

    };
    build(root);

    file << "}" << endl;
    file.close();
    cout << "Graph exported to " << filename << endl;
}

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

    return 0;
}