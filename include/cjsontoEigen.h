#pragma once
#include "cJSON.h"
#include <string>
#include <Eigen/Eigen>
using Eigen::ArrayXXf;
using Eigen::VectorXf;
using Eigen::MatrixXf;

class cjsontoEigen {
public:
    static double cjsontodouble(cJSON *cjson,std::string name);
    static int cjsontoint(cJSON* cjson,std::string name);
    static bool cjsontobool(cJSON* cjson,std::string name);
    static ArrayXXf cjsontoArrayXXf(cJSON* cjson,std::string name);
    static VectorXf cjsontoVectorXf(cJSON* cjson,std::string name);

    static MatrixXf cjsontoMatrixXf(cJSON* cjson,std::string name);
};


