//
// Created by dell on 3/24/21.
//
#include <string>
#include "NvInfer.h"
#include <map>
#include <iostream>
#include <fstream>
#include <cassert>
using namespace nvinfer1;
using namespace std;
class Utils{
private:
    static Utils instance;
public:
    static Utils getInstance(){
        return instance;
    }
    map<string, Weights> loadWeights(string file)
    {
        cout << "Loading weights: " << file << endl;
        map<string, Weights> weightMap;

        // Open weights file
        ifstream input(file);
        assert(input.is_open() && "Unable to load weight file.");

        // Read number of weight blobs
        int32_t count;
        input >> count;
        assert(count > 0 && "Invalid weight map file.");

        while (count--)
        {
            Weights wt{DataType::kFLOAT, nullptr, 0};
            uint32_t size;

            // Read name and type of blob
            string name;
            input >> name >> dec >> size;
            cout<<name<<" "<<dec<<" "<<size<<endl;
            wt.type = DataType::kFLOAT;

            // Load blob
            uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> hex >> val[x];
            }
            wt.values = val;

            wt.count = size;
            weightMap[name] = wt;
        }

        return weightMap;
    }
};
