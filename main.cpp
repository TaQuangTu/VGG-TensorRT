#include "NvInfer.h"
#include <iostream>
#include <string>
#include <map>
#include "src/Utils.cpp"
#include "src/Logger.cpp"
#include "cuda_runtime_api.h"

using namespace nvinfer1;
using namespace std;
Logger logger;

int main() {
    int batchSize = 1;
    const char *inputName = "data";
    const char *outputName = "prob";
    DataType dataType = DataType::kFLOAT;
    static const int inputH = 224, inputW = 224;

    map<string, Weights> weightMap = Utils::getInstance().loadWeights("../Weights/vgg.wts");
    IBuilder *builder = createInferBuilder(logger);
    IBuilderConfig *config = builder->createBuilderConfig();
    INetworkDefinition *network = builder->createNetworkV2(0U);

    ITensor *data = network->addInput(inputName, dataType, Dims3{3, inputH, inputW});
    IConvolutionLayer *conv1 = network->addConvolutionNd(*data, 64, DimsHW{3, 3}, weightMap["features.0.weight"],
                                                         weightMap["features.0.bias"]);
    conv1->setPaddingNd(DimsHW{1, 1});
    IActivationLayer *relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    IPoolingLayer *pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool1->setStrideNd(DimsHW{2, 2});

    conv1 = network->addConvolutionNd(*pool1->getOutput(0), 128, DimsHW{3, 3}, weightMap["features.3.weight"],
                                      weightMap["features.3.bias"]);
    conv1->setPaddingNd(DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool1->setStrideNd(DimsHW{2, 2});

    conv1 = network->addConvolutionNd(*pool1->getOutput(0), 256, DimsHW{3, 3}, weightMap["features.6.weight"],
                                      weightMap["features.6.bias"]);
    conv1->setPaddingNd(DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    conv1 = network->addConvolutionNd(*relu1->getOutput(0), 256, DimsHW{3, 3}, weightMap["features.8.weight"],
                                      weightMap["features.8.bias"]);
    conv1->setPaddingNd(DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool1->setStrideNd(DimsHW{2, 2});

    conv1 = network->addConvolutionNd(*pool1->getOutput(0), 512, DimsHW{3, 3}, weightMap["features.11.weight"],
                                      weightMap["features.11.bias"]);
    conv1->setPaddingNd(DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    conv1 = network->addConvolutionNd(*relu1->getOutput(0), 512, DimsHW{3, 3}, weightMap["features.13.weight"],
                                      weightMap["features.13.bias"]);
    conv1->setPaddingNd(DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool1->setStrideNd(DimsHW{2, 2});

    conv1 = network->addConvolutionNd(*pool1->getOutput(0), 512, DimsHW{3, 3}, weightMap["features.16.weight"],
                                      weightMap["features.16.bias"]);
    conv1->setPaddingNd(DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    conv1 = network->addConvolutionNd(*relu1->getOutput(0), 512, DimsHW{3, 3}, weightMap["features.18.weight"],
                                      weightMap["features.18.bias"]);
    conv1->setPaddingNd(DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool1->setStrideNd(DimsHW{2, 2});
    IFullyConnectedLayer *fc1 = network->addFullyConnected(*pool1->getOutput(0), 4096, weightMap["classifier.0.weight"],
                                                           weightMap["classifier.0.bias"]);
    relu1 = network->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
    fc1 = network->addFullyConnected(*relu1->getOutput(0), 4096, weightMap["classifier.3.weight"],
                                     weightMap["classifier.3.bias"]);
    relu1 = network->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
    fc1 = network->addFullyConnected(*relu1->getOutput(0), 1000, weightMap["classifier.6.weight"],
                                     weightMap["classifier.6.bias"]);

    IActivationLayer *sigmoid = network->addActivation(*fc1->getOutput(0), ActivationType::kSIGMOID);
    sigmoid->getOutput(0)->setName(outputName);
    network->markOutput(*sigmoid->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(batchSize);
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    IExecutionContext *context = engine->createExecutionContext();

    //inference
    void **buffers = new void *[2];
    void *inputBuffer;
    void *outputBuffer;
    cudaError status1 = cudaMalloc(&inputBuffer, batchSize * 3 * inputH * inputH * sizeof(float));
    cudaError status2 = cudaMalloc(&outputBuffer, batchSize * 1000 * sizeof(float));
    int inputIndex = engine->getBindingIndex(inputName);
    int outputIndex = engine->getBindingIndex(outputName);
    buffers[inputIndex] = inputBuffer;
    buffers[outputIndex] = outputBuffer;
    //creat mock image
    int size = 3 * inputW * inputH;
    float *image = new float[3 * inputH * inputW];
    for (int i = 0; i < size; i++) {
        image[i] = rand()%256;
    }
    cudaError status3 = cudaMemcpy(inputBuffer, image, size * sizeof(float), cudaMemcpyHostToDevice);
    bool done = context->execute(batchSize, buffers);
    if (done) {
        cout << "prediction done" << endl;
        //copy prediction from GPU to host
        float *probs = new float[1000];
        cudaError statusCopy =  cudaMemcpy(probs, outputBuffer, 1000 * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 1000; i++) {
            cout << probs[i] << endl;
        }
    } else {
        cout << "prediction is not done" << endl;
    }

    return 0;
}