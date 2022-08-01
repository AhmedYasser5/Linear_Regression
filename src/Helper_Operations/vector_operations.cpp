#include "Helper_Operations/vector_operations.hpp"
#include <cassert>

using namespace MachineLearning;

void MachineLearning::multiplyByScalar(vector<DataType> &a, DataType value) {
  size_t n = a.size();
  for (size_t i = 0; i < n; i++)
    a[i] *= value;
}

DataType MachineLearning::dotProduct(const vector<DataType> &a,
                                     const vector<DataType> &b,
                                     const DataType scale) {
  size_t n = a.size();
  assert(n == b.size());
  DataType result = 0;
  for (size_t i = 0; i < n; i++)
    result += scale * a[i] * b[i];
  return result;
}

DataType MachineLearning::getSummation(const vector<DataType> &a,
                                       const DataType scale) {
  size_t n = a.size();
  DataType result = 0;
  for (size_t i = 0; i < n; i++)
    result += a[i] * scale;
  return result;
}

void MachineLearning::increaseByScalar(vector<DataType> &a, DataType value) {
  size_t n = a.size();
  for (size_t i = 0; i < n; i++)
    a[i] += value;
}
