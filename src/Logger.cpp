//
// Created by dell on 3/24/21.
//

#include "NvInferRuntimeCommon.h"
using namespace nvinfer1;
class Logger: public ILogger{
public:
    Logger(ILogger::Severity severity = ILogger::Severity::kINTERNAL_ERROR){
        //do nothing
    }
    void log(ILogger::Severity severity, const char *msg) override{

    }
};