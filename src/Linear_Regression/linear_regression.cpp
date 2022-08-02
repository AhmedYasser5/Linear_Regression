#include "Linear_Regression/linear_regression.hpp"
#include "Helper_Operations/concurrency_operations.hpp"
#include "Helper_Operations/vector_operations.hpp"
#include <cmath>
#include <stdexcept>

using namespace MachineLearning;

static const size_t maxIterationsPerThreadForFeatures = 2;
static const size_t maxIterationsPerThreadForDataSize = 1e7;

LinearRegression::LinearRegression(const DataType &alpha) : alpha(alpha) {}

vector<vector<DataType>>
LinearRegression::normalize(const vector<vector<DataType>> &x) {
  size_t features = x.front().size(), dataSize = x.size();
  vector<vector<DataType>> normalizedX(features, vector<DataType>(dataSize));

  ConcurrentLoops driver(0, dataSize, maxIterationsPerThreadForDataSize);
  driver.initiateLoopsWithoutReturns([&](size_t i) -> void {
    for (size_t j = 0; j < features; j++)
      normalizedX[j][i] = x[i][j];
  });

  avg.assign(features, 0);
  std_dev.assign(features, 0);

  driver.finish = features;
  driver.maxIterationsPerThread = maxIterationsPerThreadForFeatures;
  driver.initiateLoopsWithoutReturns([&](size_t i) -> void {
    avg[i] = getSummation(normalizedX[i], 1.0 / dataSize);
    increaseByScalar(normalizedX[i], -avg[i]);

    std_dev[i] =
        sqrt(dotProduct(normalizedX[i], normalizedX[i], 1.0 / dataSize));
    multiplyByScalar(normalizedX[i], 1 / std_dev[i]);
  });

  return normalizedX;
}

vector<DataType> LinearRegression::normalize(const vector<DataType> &x) const {
  size_t features = x.size();
  if (features != avg.size())
    throw std::logic_error("The module has not been trained yet");

  vector<DataType> res;
  res.reserve(features);
  for (size_t i = 0; i < features; i++)
    res.emplace_back((x[i] - avg[i]) / std_dev[i]);
  return res;
}

bool LinearRegression::processData(const vector<vector<DataType>> &dotX,
                                   const vector<DataType> &sumX,
                                   const DataType &sumY,
                                   const vector<DataType> &sumXY) {
  size_t features = sumX.size();
  weights.assign(features, 0);
  base = 0;

  vector<DataType> newWeights(features);
  bool converged = false;
  while (!converged) {
    converged = true;

    for (size_t i = 0; i < features; i++) {
      newWeights[i] = weights[i] - (dotProduct(weights, dotX[i]) +
                                    base * sumX[i] - sumXY[i]);
      if (std::abs(weights[i] - newWeights[i]) > EPSILON)
        converged = false;
    }

    DataType newBase = base - (dotProduct(weights, sumX) + alpha * base - sumY);

    if (std::abs(base - newBase) > EPSILON)
      converged = false;

    weights = newWeights;
    base = newBase;
  }
  return true;
}

bool LinearRegression::train(const vector<vector<DataType>> &x,
                             const vector<DataType> &y) {
  auto normalizedX = normalize(x);
  size_t features = normalizedX.size(), dataSize = y.size();

  vector<vector<DataType>> dotX(features, vector<DataType>(features));
  vector<DataType> sumX(features), sumXY(features);

  ConcurrentLoops driver(0, features, maxIterationsPerThreadForFeatures);
  driver.initiateLoopsWithoutReturns([&](size_t i) -> void {
    DataType scale = alpha / dataSize;

    sumX[i] = getSummation(normalizedX[i], scale);
    sumXY[i] = dotProduct(normalizedX[i], y, scale);

    for (size_t j = i; j < features; j++)
      dotX[i][j] = dotProduct(normalizedX[i], normalizedX[j], scale);
  });

  for (size_t i = 0; i < features; i++)
    for (size_t j = 0; j < i; j++)
      dotX[i][j] = dotX[j][i];

  DataType sumY = getSummation(y, alpha / dataSize);

  return processData(dotX, sumX, sumY, sumXY);
}

DataType LinearRegression::predict(const vector<DataType> &x) const {
  return dotProduct(weights, normalize(x)) + base;
}
