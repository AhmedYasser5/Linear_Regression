#include "Linear_Regression/linear_regression.hpp"
#include "Helper_Operations/concurrency_operations.hpp"
#include "Helper_Operations/vector_operations.hpp"
#include <cmath>

using namespace MachineLearning;

LinearRegression::LinearRegression(const DataType &alpha) : alpha(alpha) {}

static void initialize(const size_t &i, vector<vector<DataType>> &X,
                       const vector<vector<DataType>> &x,
                       const size_t &features) {
  for (size_t j = 0; j < features; j++)
    X[j][i] = x[i][j];
}

static void getStats(const size_t &i, vector<DataType> &avg,
                     vector<vector<DataType>> &X, vector<DataType> &std_dev,
                     const size_t &DataType_size) {
  avg[i] = getSummation(X[i], 1.0 / DataType_size);
  increaseByScalar(X[i], -avg[i]);

  std_dev[i] = sqrt(dotProduct(X[i], X[i], 1.0 / DataType_size));
  multiplyByScalar(X[i], 1 / std_dev[i]);
}

vector<vector<DataType>>
LinearRegression::transform(const vector<vector<DataType>> &x) {
  size_t features = x[0].size(), DataType_size = x.size();
  vector<vector<DataType>> X(features, vector<DataType>(DataType_size));

  runLoopConcurrently(0, DataType_size, initialize, std::ref(X), std::cref(x),
                      std::cref(features));

  avg.resize(features);
  std_dev.resize(features);

  runLoopConcurrently(0, features, getStats, std::ref(avg), std::ref(X),
                      std::ref(std_dev), std::cref(DataType_size));

  return X;
}

vector<DataType> LinearRegression::normalize(const vector<DataType> &x) const {
  size_t features = x.size();
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
  weights.resize(features);
  std::fill(weights.begin(), weights.end(), 0);
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

static void precalculate(const size_t &i, const size_t &features,
                         const vector<vector<DataType>> &X,
                         const vector<DataType> &y, vector<DataType> &sumX,
                         vector<DataType> &sumXY,
                         vector<vector<DataType>> &dotX,
                         const DataType &multiplicand) {
  sumX[i] = getSummation(X[i], multiplicand);
  sumXY[i] = dotProduct(X[i], y, multiplicand);

  for (size_t j = i; j < features; j++)
    dotX[i][j] = dotProduct(X[i], X[j], multiplicand);
}

bool LinearRegression::train(const vector<vector<DataType>> &x,
                             const vector<DataType> &y) {
  auto X = transform(x);
  size_t features = X.size(), DataType_size = y.size();

  vector<vector<DataType>> dotX(features, vector<DataType>(features));
  vector<DataType> sumX(features), sumXY(features);

  runLoopConcurrently(0, features, precalculate, std::cref(features),
                      std::cref(X), std::cref(y), std::ref(sumX),
                      std::ref(sumXY), std::ref(dotX), alpha / DataType_size);

  for (size_t i = 0; i < features; i++)
    for (size_t j = 0; j < i; j++)
      dotX[i][j] = dotX[j][i];

  DataType sumY = getSummation(y, alpha / DataType_size);

  return processData(dotX, sumX, sumY, sumXY);
}

DataType LinearRegression::predict(const vector<DataType> &x) const {
  return dotProduct(weights, normalize(x)) + base;
}
