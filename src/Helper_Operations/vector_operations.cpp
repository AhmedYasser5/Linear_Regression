#include "Helper_Operations/vector_operations.hpp"
#include "Helper_Operations/concurrency_operations.hpp"
#include <cassert>
#include <stdexcept>

using namespace MachineLearning;

static const size_t maxIterationsPerThread = 1e7;

void MachineLearning::multiplyByScalar(vector<DataType> &vec, DataType value) {
  size_t size = vec.size();
  ConcurrentLoops driver(0, size, maxIterationsPerThread);
  auto loopBody = [](size_t i, vector<DataType> &vec,
                     const DataType &value) -> void { vec[i] *= value; };
  driver.initiateLoopsWithoutReturns(loopBody, std::ref(vec), std::cref(value));
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
    result += vec[i] * scale;
  return result;
}

void MachineLearning::increaseByScalar(vector<DataType> &vec, DataType value) {
  size_t size = vec.size();
  ConcurrentLoops driver(0, size, maxIterationsPerThread);
  auto loopBody = [](size_t i, vector<DataType> &vec,
                     const DataType &value) -> void { vec[i] += value; };
  driver.initiateLoopsWithoutReturns(loopBody, std::ref(vec), std::cref(value));
}
