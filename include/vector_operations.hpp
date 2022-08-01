#pragma once

#include <vector>

namespace MachineLearning {

using std::size_t;
using std::vector;

typedef long double DataType;

DataType dotProduct(const vector<DataType> &a, const vector<DataType> &b,
                    const DataType scale = 1);
void mul(vector<DataType> &a, DataType value);

DataType sum(const vector<DataType> &a, const DataType scale = 1);
void add(vector<DataType> &a, DataType value);

} // namespace MachineLearning
