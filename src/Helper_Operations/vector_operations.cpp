#include "Helper_Operations/vector_operations.hpp"
#include <stdexcept>

using namespace MachineLearning;

static const size_t maxIterationsPerThread = 1e7;

void MachineLearning::multiplyByScalar(vector<DataType> &vec, DataType value) {
  size_t size = vec.size();
  for (size_t i = 0; i < size; i++)
    vec[i] *= value;
}

DataType MachineLearning::dotProduct(const vector<DataType> &vec1,
                                     const vector<DataType> &vec2,
                                     const DataType scale) {
  size_t size = vec1.size();
  if (size != vec2.size())
    throw std::invalid_argument("Vector shapes do not match");

  DataType result = 0;
  for (size_t i = 0; i < size; i++)
    result += scale * vec1[i] * vec2[i];
  return result;
}

DataType MachineLearning::getSummation(const vector<DataType> &vec,
                                       const DataType scale) {
  size_t size = vec.size();
  DataType result = 0;
  for (size_t i = 0; i < size; i++)
    result += scale * vec[i];
  return result;
}

void MachineLearning::increaseByScalar(vector<DataType> &vec, DataType value) {
  size_t size = vec.size();
  for (size_t i = 0; i < size; i++)
    vec[i] += value;
}
