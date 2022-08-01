#pragma once

#include "defaults.hpp"

namespace MachineLearning {

DataType dotProduct(const vector<DataType> &a, const vector<DataType> &b,
                    const DataType scale = 1);
void multiplyByScalar(vector<DataType> &a, DataType value);

DataType getSummation(const vector<DataType> &a, const DataType scale = 1);
void increaseByScalar(vector<DataType> &a, DataType value);

} // namespace MachineLearning
