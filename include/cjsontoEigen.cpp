//
// Created by tallg on 1/13/2026.
//

#include "cjsontoEigen.h"

#include <iostream>

double cjsontoEigen::cjsontodouble(cJSON* cjson,std::string name) {
    double value = 0;
    cJSON* value_json = cJSON_GetObjectItem(cjson,name.c_str());
    if(value_json == nullptr) {
        std::cerr << "value " << name << " could not be found" << std::endl;
    }
    value = value_json->valuedouble;
    return value;
}
int cjsontoEigen::cjsontoint(cJSON* cjson,std::string name) {
    int value = 0;
    cJSON* value_json = cJSON_GetObjectItem(cjson,name.c_str());
    if(value_json == nullptr) {
        std::cerr << "value " << name << " could not be found" << std::endl;
        return -1;
    }
    value = value_json->valueint;
    return value;
}
bool cjsontoEigen::cjsontobool(cJSON* cjson,std::string name) {
    bool value = false;
    cJSON* value_json = cJSON_GetObjectItem(cjson,name.c_str());
    if(value_json == nullptr) {
        std::cerr << "value " << name << " could not be found" << std::endl;
        return false;
    }
    value = cJSON_IsTrue(value_json)==1;
    return value;
}
ArrayXXf cjsontoEigen::cjsontoArrayXXf(cJSON* cjson,std::string name) {
    ArrayXXf array;

    return array;
}

VectorXf cjsontoEigen::cjsontoVectorXf(cJSON* cjson,std::string name) {
    VectorXf vector;

    return vector;
}
MatrixXf cjsontoEigen::cjsontoMatrixXf(cJSON* cjson,std::string name) {
    cJSON *item;
    cJSON *inner_list;
    const cJSON *list = cJSON_GetObjectItemCaseSensitive(cjson, name.c_str());
    if (!cJSON_IsArray(list)){

    }
    inner_list = cJSON_GetArrayItem(list,0);

    int list_size = cJSON_GetArraySize(list);

    if (list_size==1){

    }
    int inner_list_size = cJSON_GetArraySize(inner_list);

    MatrixXf matrix(list_size,inner_list_size);
    for (int i=0; i<list_size;i++){
        inner_list = cJSON_GetArrayItem(list,i);
        for (int j=0; j<inner_list_size;j++){
            item = cJSON_GetArrayItem(inner_list,j);
            matrix(i,j) = static_cast<float>(item->valuedouble);
        }
    }
    return matrix;
}